"""
ContextOS Core Engine — The heart of Prometheus.

Orchestrates the complete context lifecycle:
  Load → Validate → Enrich → Inject → Monitor

Inspired by: Haystack pipeline design, OpenClaw context system,
n8n workflow patterns, and Dify production architecture.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import yaml
from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


# ============================================================
# Configuration Models
# ============================================================

class ContextMode(str, Enum):
    STRICT = "strict"       # All context validated, reject invalid
    LENIENT = "lenient"     # Best-effort, skip invalid
    ADAPTIVE = "adaptive"   # Auto-adjust based on confidence


class ContextSource(BaseModel):
    """A source of context data — file, API, database, or agent output."""
    name: str
    source_type: str  # file, api, db, agent, inline
    path: Optional[str] = None
    content: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    priority: int = 0  # Higher = more important
    ttl_seconds: Optional[int] = None  # Time-to-live


class ContextEntry(BaseModel):
    """A single governed context entry with provenance."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:12])
    key: str
    value: Any
    source: Optional[ContextSource] = None
    version: int = 1
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    tags: List[str] = Field(default_factory=list)
    confidence: float = 1.0  # 0.0 - 1.0
    validated: bool = False


class ContextConfig(BaseModel):
    """Global configuration for the context engine."""
    mode: ContextMode = ContextMode.ADAPTIVE
    max_context_tokens: int = 128000
    enable_validation: bool = True
    enable_compression: bool = True
    enable_memory_sync: bool = True
    enable_kg_enrichment: bool = True
    workspace_dir: Optional[Path] = None
    skill_dirs: List[Path] = Field(default_factory=list)
    auto_reload: bool = True
    context_file: str = "prometheus.context.yaml"
    version: str = "1.0.0"


# ============================================================
# Context Pipeline (Inspired by Haystack)
# ============================================================

class ContextPipelineStage:
    """Base class for pipeline stages. Haystack-inspired composable pipeline."""

    def __init__(self, name: str):
        self.name = name

    async def process(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    def __repr__(self):
        return f"<PipelineStage: {self.name}>"


class LoadStage(ContextPipelineStage):
    """Load context from all registered sources."""

    def __init__(self, engine: "ContextEngine"):
        super().__init__("load")
        self.engine = engine

    async def process(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        loaded = {}
        for source in self.engine.sources:
            try:
                if source.content:
                    loaded[source.name] = source.content
                elif source.path and os.path.exists(source.path):
                    loaded[source.name] = Path(source.path).read_text(encoding="utf-8")
            except Exception as e:
                logger.warning(f"Failed to load source '{source.name}': {e}")
        ctx["_loaded"] = loaded
        ctx["_load_count"] = len(loaded)
        return ctx


class ValidateStage(ContextPipelineStage):
    """Validate context entries against rules and schemas."""

    def __init__(self, engine: "ContextEngine"):
        super().__init__("validate")
        self.engine = engine

    async def process(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        if not self.engine.config.enable_validation:
            ctx["_validated_count"] = 0
            return ctx

        valid_count = 0
        for entry_id, entry in list(self.engine.store.items()):
            try:
                # Check TTL
                if entry.source and entry.source.ttl_seconds:
                    age = time.time() - entry.updated_at
                    if age > entry.source.ttl_seconds:
                        del self.engine.store[entry_id]
                        continue
                # Validate schema if defined
                schema = self.engine.schemas.get(entry.key)
                if schema:
                    schema(entry.value)  # raises on invalid
                entry.validated = True
                valid_count += 1
            except Exception as e:
                if self.engine.config.mode == ContextMode.STRICT:
                    raise
                logger.debug(f"Validation failed for '{entry.key}': {e}")
                entry.confidence *= 0.5

        ctx["_validated_count"] = valid_count
        return ctx


class EnrichStage(ContextPipelineStage):
    """Enrich context with knowledge graph and memory data."""

    def __init__(self, engine: "ContextEngine"):
        super().__init__("enrich")
        self.engine = engine

    async def process(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        enrichments = {}

        # Knowledge graph enrichment
        if self.engine.config.enable_kg_enrichment and self.engine.knowledge_graph:
            for entry in self.engine.store.values():
                related = self.engine.knowledge_graph.find_related(entry.key, top_k=3)
                if related:
                    enrichments[f"kg_{entry.key}"] = related

        # Memory layer enrichment
        if self.engine.config.enable_memory_sync and self.engine.memory_layer:
            memories = self.engine.memory_layer.recall(
                query=" ".join(e.value for e in list(self.engine.store.values())[:5] if isinstance(e.value, str)),
                limit=5,
            )
            if memories:
                enrichments["memory_context"] = memories

        ctx["_enrichments"] = enrichments
        return ctx


class InjectStage(ContextPipelineStage):
    """Inject context into the target system (agent, LLM, etc.)."""

    def __init__(self, engine: "ContextEngine"):
        super().__init__("inject")
        self.engine = engine

    async def process(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        context_str = self.engine._serialize_context()
        ctx["_injected_context"] = context_str
        ctx["_injected_token_count"] = len(context_str.split())
        return ctx


# ============================================================
# Core Context Engine
# ============================================================

class ContextEngine:
    """
    The central context engine of Prometheus.

    Manages the complete context lifecycle and coordinates all subsystems:
    - Skill Registry
    - Memory Layer
    - Knowledge Graph
    - Token Compressor
    - Context Governor

    Usage:
        engine = ContextEngine(ContextConfig(workspace_dir=Path("./my_project")))
        await engine.initialize()
        context = await engine.resolve("What are the main security risks?")
    """

    def __init__(self, config: Optional[ContextConfig] = None):
        self.config = config or ContextConfig()
        self.store: Dict[str, ContextEntry] = {}
        self.sources: List[ContextSource] = []
        self.schemas: Dict[str, Callable] = {}
        self.hooks: Dict[str, List[Callable]] = {
            "before_load": [],
            "after_load": [],
            "before_validate": [],
            "after_validate": [],
            "before_enrich": [],
            "after_enrich": [],
            "before_inject": [],
            "after_inject": [],
        }

        # Sub-systems (lazy init)
        self.skill_registry = None
        self.memory_layer = None
        self.knowledge_graph = None
        self.compressor = None
        self.governor = None

        # Pipeline
        self.pipeline = [
            LoadStage(self),
            ValidateStage(self),
            EnrichStage(self),
            InjectStage(self),
        ]

    async def initialize(self) -> "ContextEngine":
        """Initialize all subsystems and load persistent context."""
        # Initialize sub-systems
        from prometheus.skill import SkillRegistry
        from prometheus.memory import MemoryLayer, MemoryConfig
        from prometheus.knowledge import KnowledgeGraph, GraphConfig
        from prometheus.compressor import TokenCompressor
        from prometheus.context_gov import ContextGovernor

        self.skill_registry = SkillRegistry(self)
        self.memory_layer = MemoryLayer(MemoryConfig())
        self.knowledge_graph = KnowledgeGraph(GraphConfig())
        self.compressor = TokenCompressor()
        self.governor = ContextGovernor(self)

        # Load persisted context
        if self.config.workspace_dir:
            context_path = self.config.workspace_dir / self.config.context_file
            if context_path.exists():
                await self._load_context_file(context_path)

        # Auto-load skills
        await self.skill_registry.auto_discover(self.config.skill_dirs)

        logger.info(f"Prometheus ContextOS v{self.config.version} initialized")
        logger.info(f"  Mode: {self.config.mode.value}")
        logger.info(f"  Max context tokens: {self.config.max_context_tokens}")
        logger.info(f"  Skills loaded: {len(self.skill_registry.skills)}")
        return self

    # ============================================================
    # Context CRUD
    # ============================================================

    def set(self, key: str, value: Any, source: Optional[ContextSource] = None,
            tags: Optional[List[str]] = None, confidence: float = 1.0) -> ContextEntry:
        """Set a context entry with governance."""
        existing = self._find_by_key(key)
        if existing:
            existing.value = value
            existing.version += 1
            existing.updated_at = time.time()
            existing.confidence = confidence
            if tags:
                existing.tags = tags
            if hasattr(self, 'governor') and self.governor:
                self.governor.record_change("update", existing)
            return existing

        entry = ContextEntry(key=key, value=value, source=source, tags=tags or [], confidence=confidence)
        self.store[entry.id] = entry
        if hasattr(self, 'governor') and self.governor:
            self.governor.record_change("create", entry)
        return entry

    def get(self, key: str) -> Optional[Any]:
        """Get a context value by key."""
        entry = self._find_by_key(key)
        return entry.value if entry else None

    def get_entry(self, key: str) -> Optional[ContextEntry]:
        """Get a full context entry by key."""
        return self._find_by_key(key)

    def delete(self, key: str) -> bool:
        """Delete a context entry."""
        entry = self._find_by_key(key)
        if entry:
            if hasattr(self, 'governor') and self.governor:
                self.governor.record_change("delete", entry)
            del self.store[entry.id]
            return True
        return False

    def remove(self, key: str) -> bool:
        """Alias for delete()."""
        return self.delete(key)

    def list(self, tag: Optional[str] = None) -> List[ContextEntry]:
        """List context entries, optionally filtered by tag."""
        entries = list(self.store.values())
        if tag:
            entries = [e for e in entries if tag in e.tags]
        return sorted(entries, key=lambda e: e.updated_at, reverse=True)

    def register_source(self, source: ContextSource):
        """Register a context data source."""
        self.sources.append(source)
        self.sources.sort(key=lambda s: s.priority, reverse=True)

    def register_schema(self, key: str, validator_fn: Callable):
        """Register a schema validator for a context key."""
        self.schemas[key] = validator_fn

    def register_hook(self, event: str, callback: Callable):
        """Register a lifecycle hook."""
        if event in self.hooks:
            self.hooks[event].append(callback)

    # ============================================================
    # Context Resolution (Main API)
    # ============================================================

    async def resolve(self, query: str, additional_context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Resolve the full context for a given query.
        This is the main entry point — runs the complete pipeline.
        """
        ctx = {"query": query, **(additional_context or {})}

        # Run pipeline
        for stage in self.pipeline:
            await self._fire_hooks(f"before_{stage.name}", ctx)
            ctx = await stage.process(ctx)
            await self._fire_hooks(f"after_{stage.name}", ctx)

        # Compress if enabled
        if self.config.enable_compression and ctx.get("_injected_context"):
            compressed = self.compressor.compress(ctx["_injected_context"])
            ctx["_compressed_context"] = compressed
            ctx["_compression_ratio"] = self.compressor.last_ratio

        # Add skill suggestions
        ctx["_suggested_skills"] = self.skill_registry.suggest(query)

        return ctx

    async def ask(self, query: str, model_adapter: Optional[Any] = None) -> str:
        """
        Ask a question with full context resolution and LLM integration.
        If a model_adapter is provided, routes through the LLM.
        """
        ctx = await self.resolve(query)
        context_str = ctx.get("_compressed_context") or ctx.get("_injected_context", "")

        if model_adapter:
            return await model_adapter.generate(query, context=context_str)

        return context_str

    # ============================================================
    # Persistence
    # ============================================================

    async def save(self):
        """Persist the context store to disk."""
        if not self.config.workspace_dir:
            return
        context_path = self.config.workspace_dir / self.config.context_file
        data = {
            "version": self.config.version,
            "entries": {eid: entry.model_dump() for eid, entry in self.store.items()},
            "sources": [s.model_dump() for s in self.sources],
            "governance_log": self.governor.export_log(),
        }
        context_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    # ============================================================
    # Internal Helpers
    # ============================================================

    def _find_by_key(self, key: str) -> Optional[ContextEntry]:
        for entry in self.store.values():
            if entry.key == key:
                return entry
        return None

    def _serialize_context(self) -> str:
        """Serialize all context entries into a structured string for LLM injection."""
        parts = []
        for entry in sorted(self.store.values(), key=lambda e: (e.source.priority if e.source else 0), reverse=True):
            source_tag = f" [src: {entry.source.name}]" if entry.source else ""
            confidence_tag = f" (confidence: {entry.confidence:.0%})" if entry.confidence < 1.0 else ""
            parts.append(f"[{entry.key}]{source_tag}{confidence_tag}\n{entry.value}")
        return "\n\n---\n\n".join(parts)

    async def _load_context_file(self, path: Path):
        """Load persisted context from a file."""
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            for eid, raw in data.get("entries", {}).items():
                self.store[eid] = ContextEntry(**raw)
            for raw in data.get("sources", []):
                self.sources.append(ContextSource(**raw))
            logger.info(f"Loaded {len(self.store)} context entries from {path}")
        except Exception as e:
            logger.warning(f"Failed to load context file: {e}")

    async def _fire_hooks(self, event: str, ctx: Dict[str, Any]):
        for hook in self.hooks.get(event, []):
            try:
                if asyncio.iscoroutinefunction(hook):
                    await hook(ctx)
                else:
                    hook(ctx)
            except Exception as e:
                logger.debug(f"Hook '{event}' error: {e}")


# ============================================================
# Utility
# ============================================================

@dataclass
class EngineStats:
    """Runtime statistics for the context engine."""
    total_entries: int = 0
    validated_entries: int = 0
    total_sources: int = 0
    skills_loaded: int = 0
    memory_entries: int = 0
    graph_nodes: int = 0
    compression_ratio: float = 0.0
    last_resolve_ms: float = 0.0
    uptime_seconds: float = 0.0
