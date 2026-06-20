"""
Skill Ecosystem — Composable, shareable AI capabilities.

Markdown-driven skill system inspired by:
- OpenClaw Skills System (AgentSkills protocol)
- mattpocock/skills (composable skill modules)
- Claude Code .claude/skills pattern
- Deer-Flow2 Skill全家桶

Skills are defined as SKILL.md files with YAML front matter
and Markdown behavioral instructions. They are auto-discovered,
hot-loaded, and composable like LEGO blocks.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, TYPE_CHECKING

import yaml
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from prometheus.core import ContextEngine

logger = logging.getLogger(__name__)


# ============================================================
# Skill Models
# ============================================================

class SkillMeta(BaseModel):
    """YAML front matter for a skill definition."""
    name: str
    version: str = "1.0.0"
    description: str = ""
    author: str = ""
    license: str = "MIT"
    tags: List[str] = Field(default_factory=list)
    requires: List[str] = Field(default_factory=list)  # Dependency skill names
    provides: List[str] = Field(default_factory=list)   # Capabilities provided
    models: List[str] = Field(default_factory=list)     # Compatible models
    environment: Dict[str, str] = Field(default_factory=dict)  # Env vars
    binaries: List[str] = Field(default_factory=list)   # Required binaries
    install: Optional[str] = None  # Installation command


class Skill(BaseModel):
    """A complete skill definition — metadata + instructions."""
    meta: SkillMeta
    instructions: str  # Markdown behavioral instructions
    source_path: Optional[Path] = None
    loaded_at: float = 0.0
    usage_count: int = 0

    @property
    def id(self) -> str:
        """Unique skill identifier."""
        return f"{self.meta.name}@{self.meta.version}"

    @property
    def content_hash(self) -> str:
        """Content-based hash for change detection."""
        return hashlib.sha256(
            f"{self.meta.name}{self.meta.version}{self.instructions}".encode()
        ).hexdigest()[:16]

    def to_prompt(self) -> str:
        """Convert skill to LLM-injectable prompt format."""
        lines = [
            f"## Skill: {self.meta.name} (v{self.meta.version})",
            f"Description: {self.meta.description}",
            f"Tags: {', '.join(self.meta.tags)}" if self.meta.tags else "",
            "",
            self.instructions,
        ]
        return "\n".join(lines)

    def matches(self, query: str) -> bool:
        """Check if this skill matches a query based on name, description, and tags."""
        query_lower = query.lower()
        searchable = f"{self.meta.name} {self.meta.description} {' '.join(self.meta.tags)}".lower()
        # Check for exact name match or keyword overlap
        if self.meta.name.lower() in query_lower:
            return True
        keywords = set(query_lower.split())
        skill_words = set(searchable.split())
        overlap = keywords & skill_words
        return len(overlap) >= 2


# ============================================================
# Skill Loader — Parses SKILL.md files
# ============================================================

class SkillLoader:
    """
    Loads skills from SKILL.md files.

    Format:
        ---
        name: my-skill
        version: 1.0.0
        description: Does amazing things
        ---

        # My Skill
        You are an expert at X. When asked to do Y, follow these steps...

    Compatible with OpenClaw/AgentSkills protocol and .claude/skills pattern.
    """

    SKILL_FILE = "SKILL.md"
    YAML_PATTERN = re.compile(r'^---\s*\n(.*?)\n---\s*\n', re.DOTALL)

    @classmethod
    def parse(cls, content: str, source_path: Optional[Path] = None) -> Optional[Skill]:
        """Parse a SKILL.md content string into a Skill object."""
        match = cls.YAML_PATTERN.match(content)
        if not match:
            logger.debug(f"No YAML front matter found in skill file: {source_path}")
            return None

        yaml_str = match.group(1)
        instructions = content[match.end():].strip()

        try:
            meta_dict = yaml.safe_load(yaml_str)
            if not isinstance(meta_dict, dict) or "name" not in meta_dict:
                return None

            meta = SkillMeta(**meta_dict)
            return Skill(
                meta=meta,
                instructions=instructions,
                source_path=source_path,
                loaded_at=__import__("time").time(),
            )
        except Exception as e:
            logger.warning(f"Failed to parse skill at {source_path}: {e}")
            return None

    @classmethod
    def load_from_dir(cls, directory: Path) -> List[Skill]:
        """Recursively discover and load all skills from a directory."""
        skills = []
        if not directory.exists():
            return skills

        for skill_file in directory.rglob(cls.SKILL_FILE):
            try:
                content = skill_file.read_text(encoding="utf-8")
                skill = cls.parse(content, skill_file)
                if skill:
                    skills.append(skill)
            except Exception as e:
                logger.debug(f"Error loading {skill_file}: {e}")

        return skills

    @classmethod
    def create_skill_template(cls, name: str, description: str = "",
                              instructions: str = "") -> str:
        """Generate a SKILL.md template for new skill creation."""
        return f"""---
name: {name}
version: 1.0.0
description: {description}
author: ""
tags: []
requires: []
provides: []
---

# {name.replace('-', ' ').title()} Skill

{instructions or 'Describe what this skill does and how the AI should behave.'}

## Instructions

1. Step one
2. Step two
3. Step three

## Examples

### Example 1
[Provide an example usage]

## Constraints

- Constraint one
- Constraint two
"""


# ============================================================
# Skill Registry — Central skill management
# ============================================================

class SkillRegistry:
    """
    Central registry for all skills.

    Manages:
    - Skill discovery and loading
    - Dependency resolution (topological sort)
    - Skill suggestion based on queries
    - Hot-reload on file changes
    - Skill marketplace integration

    Inspired by: OpenClaw Plugin Registry, Dify Plugin System,
    mattpocock/skills composability model.
    """

    def __init__(self, engine: "ContextEngine"):
        self.engine = engine
        self.skills: Dict[str, Skill] = {}
        self._by_tag: Dict[str, List[str]] = {}  # tag -> skill IDs
        self._dependency_graph: Dict[str, Set[str]] = {}
        self._loaded_dirs: Set[Path] = set()

    async def auto_discover(self, dirs: List[Path]):
        """Auto-discover skills from configured directories."""
        for directory in dirs:
            if directory in self._loaded_dirs:
                continue
            skills = SkillLoader.load_from_dir(directory)
            for skill in skills:
                self.register(skill, hot_reload=True)
            self._loaded_dirs.add(directory)

        self._build_dependency_graph()
        logger.info(f"Auto-discovered {len(self.skills)} skills from {len(dirs)} directories")

    def register(self, skill: Skill, hot_reload: bool = False):
        """Register a skill in the registry."""
        skill_id = skill.id
        is_update = skill_id in self.skills

        self.skills[skill_id] = skill

        # Update tag index
        if is_update:
            self._remove_from_tag_index(skill_id)
        self._add_to_tag_index(skill_id, skill.meta.tags)

        action = "hot-reloaded" if hot_reload else ("updated" if is_update else "registered")
        logger.info(f"Skill '{skill_id}' {action} ({len(skill.instructions)} chars)")

    def unregister(self, skill_id: str):
        """Remove a skill from the registry."""
        if skill_id in self.skills:
            skill = self.skills.pop(skill_id)
            self._remove_from_tag_index(skill_id)
            logger.info(f"Skill '{skill_id}' unregistered")

    def get(self, skill_id: str) -> Optional[Skill]:
        """Get a skill by its full ID (name@version)."""
        return self.skills.get(skill_id)

    def find_by_name(self, name: str) -> List[Skill]:
        """Find all versions of a skill by name."""
        return [s for s in self.skills.values() if s.meta.name == name]

    def find_by_tag(self, tag: str) -> List[Skill]:
        """Find skills by tag."""
        skill_ids = self._by_tag.get(tag, set())
        return [self.skills[sid] for sid in skill_ids if sid in self.skills]

    def suggest(self, query: str, top_k: int = 5) -> List[Skill]:
        """
        Suggest relevant skills for a query.

        Uses keyword matching against skill names, descriptions, and tags.
        For production, this can be upgraded to semantic search via embeddings.
        """
        scored: List[tuple] = []
        query_lower = query.lower()
        query_words = set(query_lower.split())

        for skill in self.skills.values():
            score = 0
            searchable = f"{skill.meta.name} {skill.meta.description} {' '.join(skill.meta.tags)}".lower()
            skill_words = set(searchable.split())

            # Exact name match = high score
            if skill.meta.name.lower() in query_lower:
                score += 10

            # Tag matches
            tag_matches = sum(1 for t in skill.meta.tags if t.lower() in query_lower)
            score += tag_matches * 3

            # Keyword overlap
            overlap = len(query_words & skill_words)
            score += overlap

            # Usage boost
            score += min(skill.usage_count * 0.5, 5)

            if score > 0:
                scored.append((score, skill))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [skill for _, skill in scored[:top_k]]

    def resolve_dependencies(self, skill: Skill) -> List[Skill]:
        """
        Resolve all dependencies for a skill (topological order).
        Returns the complete skill chain needed to run this skill.
        """
        resolved = []
        visited = set()

        def _resolve(s: Skill):
            if s.id in visited:
                return
            visited.add(s.id)
            for dep_name in s.meta.requires:
                dep_skills = self.find_by_name(dep_name)
                if dep_skills:
                    _resolve(dep_skills[0])  # Use latest version
            resolved.append(s)

        _resolve(skill)
        return resolved

    def get_full_prompt(self, skill_ids: Optional[List[str]] = None,
                        query: Optional[str] = None) -> str:
        """
        Generate the full skill prompt for injection into LLM context.
        Includes all requested skills (or suggested skills for a query)
        with their complete instructions.
        """
        skills_to_use: List[Skill] = []

        if skill_ids:
            for sid in skill_ids:
                if sid in self.skills:
                    skills_to_use.append(self.skills[sid])
                    # Resolve dependencies
                    deps = self.resolve_dependencies(self.skills[sid])
                    for dep in deps:
                        if dep not in skills_to_use:
                            skills_to_use.append(dep)
        elif query:
            skills_to_use = self.suggest(query, top_k=3)

        if not skills_to_use:
            return ""

        # Deduplicate
        seen = set()
        unique_skills = []
        for s in skills_to_use:
            if s.id not in seen:
                seen.add(s.id)
                unique_skills.append(s)
                s.usage_count += 1

        prompts = [s.to_prompt() for s in unique_skills]
        return "\n\n---\n\n".join(prompts)

    def list_all(self) -> List[Dict[str, Any]]:
        """List all registered skills with metadata."""
        return [
            {
                "id": s.id,
                "name": s.meta.name,
                "version": s.meta.version,
                "description": s.meta.description,
                "tags": s.meta.tags,
                "requires": s.meta.requires,
                "provides": s.meta.provides,
                "usage_count": s.usage_count,
                "source": str(s.source_path) if s.source_path else None,
            }
            for s in sorted(self.skills.values(), key=lambda x: x.usage_count, reverse=True)
        ]

    # ============================================================
    # Internal Helpers
    # ============================================================

    def _add_to_tag_index(self, skill_id: str, tags: List[str]):
        for tag in tags:
            if tag not in self._by_tag:
                self._by_tag[tag] = set()
            self._by_tag[tag].add(skill_id)

    def _remove_from_tag_index(self, skill_id: str):
        for tag, skill_set in list(self._by_tag.items()):
            skill_set.discard(skill_id)
            if not skill_set:
                del self._by_tag[tag]

    def _build_dependency_graph(self):
        """Build the skill dependency graph for topological sorting."""
        self._dependency_graph.clear()
        for sid, skill in self.skills.items():
            self._dependency_graph[sid] = set()
            for dep_name in skill.meta.requires:
                dep_skills = self.find_by_name(dep_name)
                if dep_skills:
                    self._dependency_graph[sid].add(dep_skills[0].id)


# ============================================================
# Skill Marketplace (Future)
# ============================================================

class SkillMarketplace:
    """
    Skill sharing and discovery platform.

    Planned features:
    - Publish skills to a community registry
    - Search and install skills from remote
    - Version management and updates
    - Rating and review system
    """

    REGISTRY_URL = "https://skills.prometheus.dev"

    def __init__(self, registry: SkillRegistry):
        self.registry = registry

    async def search_remote(self, query: str) -> List[Dict]:
        """Search the remote skill marketplace."""
        # TODO: Implement remote registry search
        raise NotImplementedError("Remote marketplace coming in v1.1")

    async def publish(self, skill_id: str):
        """Publish a local skill to the marketplace."""
        # TODO: Implement skill publishing
        raise NotImplementedError("Remote marketplace coming in v1.1")

    async def install(self, skill_name: str, version: str = "latest"):
        """Install a skill from the marketplace."""
        # TODO: Implement skill installation
        raise NotImplementedError("Remote marketplace coming in v1.1")
