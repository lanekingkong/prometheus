"""
Persistent Memory Layer — Dual-engine (vector + graph) AI memory.

Inspired by:
- Cognee (topoteretes/cognee): 16K stars, vector + graph dual-engine
- Haystack Memory components
- LangChain Memory patterns

Architecture:
  Short-term (session cache) + Long-term (persistent vector + graph)
  Automatic sync, semantic recall, and relationship-aware retrieval.

Four-core API (Cognee-inspired):
  remember() → recall() → forget() → improve()
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ============================================================
# Configuration
# ============================================================

class MemoryMode(str, Enum):
    EPHEMERAL = "ephemeral"     # Session-only, no persistence
    PERSISTENT = "persistent"   # Full persistence with graph + vector
    HYBRID = "hybrid"           # Fast session cache + lazy persistence


class MemoryConfig(BaseModel):
    """Configuration for the memory layer."""
    mode: MemoryMode = MemoryMode.HYBRID
    session_ttl_minutes: int = 60  # Session memory expiry
    max_session_entries: int = 1000
    max_longterm_entries: int = 100000
    vector_dim: int = 1536
    enable_auto_sync: bool = True
    sync_interval_seconds: int = 30
    storage_dir: Optional[Path] = None
    graph_backend: str = "memory"  # "memory" | "neo4j" | "networkx"
    vector_backend: str = "memory"  # "memory" | "chromadb" | "qdrant"


# ============================================================
# Memory Entry Models
# ============================================================

class MemoryEntry(BaseModel):
    """A single memory entry with metadata."""
    id: str = Field(default_factory=lambda: hashlib.sha256(str(time.time_ns()).encode()).hexdigest()[:16])
    content: str
    memory_type: str = "fact"  # fact, event, preference, skill, relationship
    source: str = "user"
    timestamp: float = Field(default_factory=time.time)
    importance: float = 0.5  # 0.0 - 1.0, affects retention priority
    access_count: int = 0
    last_accessed: float = Field(default_factory=time.time)
    tags: List[str] = Field(default_factory=list)
    embedding: Optional[List[float]] = None  # Vector embedding
    relations: List[Dict[str, str]] = Field(default_factory=list)  # [{"target": id, "type": "related_to"}]
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def touch(self):
        """Update access metadata."""
        self.access_count += 1
        self.last_accessed = time.time()


class MemoryQuery(BaseModel):
    """A structured memory query."""
    text: str = ""
    memory_type: Optional[str] = None
    tags: Optional[List[str]] = None
    time_range: Optional[Tuple[float, float]] = None
    importance_threshold: float = 0.0
    limit: int = 10
    include_relations: bool = True


# ============================================================
# Vector Store (Simple in-memory implementation)
# ============================================================

class SimpleVectorStore:
    """In-memory cosine similarity vector store."""

    def __init__(self, dim: int = 1536):
        self.dim = dim
        self.vectors: Dict[str, List[float]] = {}
        self._id_to_idx: Dict[str, int] = {}
        self._idx_to_id: Dict[int, str] = {}
        self._matrix = None  # Lazy numpy matrix

    def add(self, entry_id: str, vector: List[float]):
        """Add or update a vector."""
        self.vectors[entry_id] = vector

    def remove(self, entry_id: str):
        """Remove a vector."""
        self.vectors.pop(entry_id, None)

    def search(self, query_vector: List[float], top_k: int = 10) -> List[Tuple[str, float]]:
        """Search by cosine similarity."""
        if not self.vectors:
            return []

        try:
            import numpy as np
        except ImportError:
            # Fallback: manual cosine similarity
            results = []
            for eid, vec in self.vectors.items():
                sim = self._cosine_similarity(query_vector, vec)
                results.append((eid, sim))
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:top_k]

        ids = list(self.vectors.keys())
        matrix = np.array([self.vectors[eid] for eid in ids])
        query = np.array(query_vector)

        # Normalize
        matrix_norm = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10)
        query_norm = query / (np.linalg.norm(query) + 1e-10)

        similarities = np.dot(matrix_norm, query_norm)
        top_indices = np.argsort(similarities)[::-1][:top_k]

        return [(ids[i], float(similarities[i])) for i in top_indices]

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = (sum(x * x for x in a)) ** 0.5
        norm_b = (sum(y * y for y in b)) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def __len__(self):
        return len(self.vectors)


# ============================================================
# Relationship Graph (Simple in-memory)
# ============================================================

class SimpleGraphStore:
    """In-memory relationship graph for memory connections."""

    def __init__(self):
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.edges: List[Tuple[str, str, str]] = []  # (source, target, relation_type)

    def add_node(self, node_id: str, data: Dict[str, Any] = None):
        self.nodes[node_id] = data or {}

    def add_edge(self, source: str, target: str, relation_type: str = "related_to"):
        self.edges.append((source, target, relation_type))

    def remove_node(self, node_id: str):
        self.nodes.pop(node_id, None)
        self.edges = [(s, t, r) for s, t, r in self.edges if s != node_id and t != node_id]

    def get_neighbors(self, node_id: str, depth: int = 1) -> List[str]:
        """Get neighbor node IDs up to a certain depth."""
        neighbors = set()
        current = {node_id}

        for _ in range(depth):
            next_level = set()
            for s, t, _ in self.edges:
                if s in current and t not in neighbors:
                    next_level.add(t)
                if t in current and s not in neighbors:
                    next_level.add(s)
            neighbors.update(next_level)
            current = next_level

        return list(neighbors)

    def find_path(self, source: str, target: str, max_depth: int = 3) -> Optional[List[str]]:
        """Find the shortest path between two nodes (BFS)."""
        if source not in self.nodes or target not in self.nodes:
            return None

        from collections import deque
        queue = deque([(source, [source])])
        visited = {source}

        while queue:
            node, path = queue.popleft()
            if len(path) > max_depth:
                continue

            for s, t, _ in self.edges:
                neighbor = t if s == node else (s if t == node else None)
                if neighbor and neighbor not in visited:
                    if neighbor == target:
                        return path + [neighbor]
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        return None


# ============================================================
# Memory Layer — Main Class
# ============================================================

class MemoryLayer:
    """
    Dual-engine persistent memory for AI agents.

    Provides:
    - remember(): Store memories with auto-embedding
    - recall(): Semantic + relational retrieval
    - forget(): Targeted or bulk memory removal
    - improve(): Feedback-driven memory refinement

    The four core APIs that make AI agents truly stateful.
    """

    def __init__(self, config: Optional[MemoryConfig] = None):
        self.config = config or MemoryConfig()

        # Session (short-term) memory
        self.session: Dict[str, MemoryEntry] = {}

        # Long-term memory stores
        self.longterm: Dict[str, MemoryEntry] = {}
        self.vector_store = SimpleVectorStore(dim=self.config.vector_dim)
        self.graph_store = SimpleGraphStore()

        # Embedding function (pluggable)
        self._embed_fn: Optional[callable] = None

    # ============================================================
    # Core API: remember()
    # ============================================================

    async def remember(self, content: str,
                       memory_type: str = "fact",
                       source: str = "user",
                       importance: float = 0.5,
                       tags: Optional[List[str]] = None,
                       metadata: Optional[Dict[str, Any]] = None) -> MemoryEntry:
        """
        Store a new memory. Auto-generates embedding and indexes.

        Args:
            content: The memory content text
            memory_type: fact | event | preference | skill | relationship
            source: Where the memory came from
            importance: Retention priority (0.0-1.0)
            tags: Categorization tags
            metadata: Additional structured data

        Returns:
            The created MemoryEntry
        """
        entry = MemoryEntry(
            content=content,
            memory_type=memory_type,
            source=source,
            importance=importance,
            tags=tags or [],
            metadata=metadata or {},
        )

        # Generate embedding
        embedding = await self._embed(content)
        if embedding:
            entry.embedding = embedding
            self.vector_store.add(entry.id, embedding)

        # Store
        self.longterm[entry.id] = entry
        self.session[entry.id] = entry
        self.graph_store.add_node(entry.id, {"type": memory_type, "importance": importance})

        # Auto-prune if over limit
        if len(self.longterm) > self.config.max_longterm_entries:
            await self._prune()

        logger.debug(f"Remembered: [{memory_type}] {content[:80]}...")
        return entry

    # ============================================================
    # Core API: recall()
    # ============================================================

    async def recall(self, query: str,
                     limit: int = 10,
                     memory_type: Optional[str] = None,
                     tags: Optional[List[str]] = None,
                     include_relations: bool = True) -> List[MemoryEntry]:
        """
        Recall memories relevant to a query.

        Uses hybrid search: semantic (vector) + relational (graph) + keyword.

        Args:
            query: The search query
            limit: Max results
            memory_type: Filter by type
            tags: Filter by tags
            include_relations: Also return relationally connected memories

        Returns:
            Ranked list of MemoryEntry objects
        """
        results = []

        # 1. Session memory (fast path)
        session_matches = self._keyword_search(query, self.session.values(), limit=limit, memory_type=memory_type, tags=tags)

        # 2. Vector search (semantic)
        query_embedding = await self._embed(query)
        if query_embedding:
            vector_results = self.vector_store.search(query_embedding, top_k=limit * 2)
            for eid, score in vector_results:
                if eid in self.longterm:
                    entry = self.longterm[eid]
                    # Apply filters
                    if memory_type and entry.memory_type != memory_type:
                        continue
                    if tags and not any(t in entry.tags for t in tags):
                        continue
                    results.append((entry, score))

        # 3. Keyword fallback for entries without embeddings
        keyword_ids = {e.id for e in session_matches}
        for entry in session_matches:
            if entry.id not in {e.id for e, _ in results}:
                results.append((entry, 0.5))  # Lower confidence for keyword-only

        # Sort by score, importance, and recency
        results.sort(key=lambda x: (
            x[1] * 0.5 + x[0].importance * 0.3 + min(x[0].access_count * 0.0, 0.2)),
            reverse=True,
        )

        selected = [entry for entry, _ in results[:limit]]

        # 4. Relation expansion
        if include_relations:
            related = set()
            for entry in selected:
                neighbors = self.graph_store.get_neighbors(entry.id, depth=1)
                for nid in neighbors:
                    if nid in self.longterm and nid not in {e.id for e in selected}:
                        related.add(self.longterm[nid])
            # Add top related entries
            rel_list = sorted(related, key=lambda e: e.importance, reverse=True)[:limit // 2]
            selected.extend(rel_list)

        # Update access metadata
        for entry in selected:
            entry.touch()
            # Promote to session
            self.session[entry.id] = entry

        return selected[:limit]

    # ============================================================
    # Core API: forget()
    # ============================================================

    async def forget(self, entry_id: Optional[str] = None,
                     memory_type: Optional[str] = None,
                     tags: Optional[List[str]] = None,
                     older_than_days: Optional[int] = None) -> int:
        """
        Forget (delete) memories.

        Args:
            entry_id: Specific memory to delete
            memory_type: Delete all of a type
            tags: Delete all with specific tags
            older_than_days: Delete memories older than N days

        Returns:
            Number of entries deleted
        """
        to_delete = set()

        if entry_id:
            to_delete.add(entry_id)
        else:
            now = time.time()
            for eid, entry in self.longterm.items():
                if memory_type and entry.memory_type != memory_type:
                    continue
                if tags and not any(t in entry.tags for t in tags):
                    continue
                if older_than_days:
                    age_days = (now - entry.timestamp) / 86400
                    if age_days < older_than_days:
                        continue
                to_delete.add(eid)

        deleted = 0
        for eid in to_delete:
            if eid in self.longterm:
                del self.longterm[eid]
                deleted += 1
            self.session.pop(eid, None)
            self.vector_store.remove(eid)
            self.graph_store.remove_node(eid)

        logger.info(f"Forgot {deleted} memories")
        return deleted

    # ============================================================
    # Core API: improve()
    # ============================================================

    async def improve(self, entry_id: str, feedback: Dict[str, Any]):
        """
        Improve a memory based on feedback.

        Feedback can include:
        - importance: Adjust importance score
        - tags: Add or remove tags
        - content: Update content (with versioning)
        - relation: Link to another memory
        - metadata: Update metadata fields
        """
        if entry_id not in self.longterm:
            raise KeyError(f"Memory '{entry_id}' not found")

        entry = self.longterm[entry_id]

        if "importance" in feedback:
            entry.importance = max(0.0, min(1.0, feedback["importance"]))
        if "tags" in feedback:
            entry.tags = feedback["tags"]
        if "content" in feedback:
            entry.content = feedback["content"]
            # Re-embed new content
            new_embedding = await self._embed(entry.content)
            if new_embedding:
                entry.embedding = new_embedding
                self.vector_store.add(entry.id, new_embedding)
        if "relation" in feedback:
            target_id = feedback["relation"]["target"]
            rel_type = feedback["relation"].get("type", "related_to")
            entry.relations.append({"target": target_id, "type": rel_type})
            self.graph_store.add_edge(entry_id, target_id, rel_type)
        if "metadata" in feedback:
            entry.metadata.update(feedback["metadata"])

        logger.debug(f"Improved memory '{entry_id}'")

    # ============================================================
    # Utility Methods
    # ============================================================

    async def create_relation(self, source_id: str, target_id: str, relation_type: str = "related_to"):
        """Create a relationship between two memories."""
        if source_id in self.longterm and target_id in self.longterm:
            self.longterm[source_id].relations.append({"target": target_id, "type": relation_type})
            self.graph_store.add_edge(source_id, target_id, relation_type)

    async def find_related(self, entry_id: str, depth: int = 1) -> List[MemoryEntry]:
        """Find memories related to a given entry through the graph."""
        neighbor_ids = self.graph_store.get_neighbors(entry_id, depth=depth)
        return [self.longterm[nid] for nid in neighbor_ids if nid in self.longterm]

    def get_stats(self) -> Dict[str, Any]:
        """Get memory system statistics."""
        return {
            "session_entries": len(self.session),
            "longterm_entries": len(self.longterm),
            "vector_entries": len(self.vector_store),
            "graph_nodes": len(self.graph_store.nodes),
            "graph_edges": len(self.graph_store.edges),
            "mode": self.config.mode.value,
        }

    async def clear_session(self):
        """Clear session (short-term) memory only."""
        self.session.clear()
        logger.info("Session memory cleared")

    async def clear_all(self):
        """Clear all memory (CAUTION)."""
        self.session.clear()
        self.longterm.clear()
        self.vector_store = SimpleVectorStore(dim=self.config.vector_dim)
        self.graph_store = SimpleGraphStore()
        logger.warning("All memory cleared!")

    # ============================================================
    # Internal Helpers
    # ============================================================

    async def _embed(self, text: str) -> Optional[List[float]]:
        """Generate a vector embedding for text. Uses simple TF-IDF-like fallback."""
        if self._embed_fn:
            try:
                return await self._embed_fn(text)
            except Exception as e:
                logger.debug(f"Embedding failed, using fallback: {e}")

        # Fallback: simple hash-based pseudo-embedding (for demo/dev)
        # In production, replace with OpenAI/Cohere/HuggingFace embeddings
        return self._simple_hash_embed(text)

    def _simple_hash_embed(self, text: str) -> List[float]:
        """Generate a simple pseudo-embedding from text for demo purposes."""
        import struct

        # Use multiple hash functions for pseudo-dimensionality
        dim = self.config.vector_dim
        vec = [0.0] * dim

        words = text.lower().split()
        for i, word in enumerate(words):
            h = hashlib.md5(word.encode()).digest()
            idx = struct.unpack("I", h[:4])[0] % dim
            vec[idx] += 1.0

        # Normalize
        norm = (sum(v * v for v in vec)) ** 0.5
        if norm > 0:
            vec = [v / norm for v in vec]

        return vec

    def _keyword_search(self, query: str, entries: List[MemoryEntry],
                        limit: int = 10, memory_type: Optional[str] = None,
                        tags: Optional[List[str]] = None) -> List[MemoryEntry]:
        """Simple keyword-based search."""
        query_words = set(query.lower().split())
        scored = []

        for entry in entries:
            if memory_type and entry.memory_type != memory_type:
                continue
            if tags and not any(t in entry.tags for t in tags):
                continue

            content_words = set(entry.content.lower().split())
            score = len(query_words & content_words) + (1 if any(w in entry.content.lower() for w in query_words) else 0)

            # Exact phrase bonus
            if query.lower() in entry.content.lower():
                score += 5

            if score > 0:
                scored.append((entry, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [entry for entry, _ in scored[:limit]]

    async def _prune(self):
        """Remove least important memories to stay under limit."""
        entries = sorted(self.longterm.values(), key=lambda e: (e.importance, e.last_accessed))
        to_remove = entries[: max(1, len(entries) - self.config.max_longterm_entries + 500)]
        for entry in to_remove:
            await self.forget(entry_id=entry.id)
