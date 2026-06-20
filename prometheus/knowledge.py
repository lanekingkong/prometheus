"""
Knowledge Graph Engine — Entity-relationship intelligence.

Inspired by:
- colbymchenry/codegraph (48K stars, code knowledge graph)
- Cognee (knowledge graph + vector fusion)
- Haystack Knowledge Graph components

Provides:
- Entity extraction from text
- Relationship discovery
- Graph traversal and reasoning
- Hybrid vector + graph search
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ============================================================
# Configuration
# ============================================================

class GraphConfig(BaseModel):
    """Configuration for the knowledge graph engine."""
    max_nodes: int = 500000
    max_edges_per_node: int = 100
    enable_auto_extraction: bool = True
    extraction_batch_size: int = 50
    similarity_threshold: float = 0.6
    enable_deduplication: bool = True
    storage_path: Optional[Path] = None
    relationship_types: List[str] = Field(default_factory=lambda: [
        "is_a", "has_a", "part_of", "depends_on", "causes",
        "related_to", "same_as", "conflicts_with", "follows",
        "influences", "creates", "uses", "implements", "extends",
    ])


# ============================================================
# Graph Entity Models
# ============================================================

class Entity(BaseModel):
    """A knowledge graph entity (node)."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:16])
    name: str
    entity_type: str = "concept"  # concept, person, organization, location, event, artifact, code_entity
    description: str = ""
    properties: Dict[str, Any] = Field(default_factory=dict)
    confidence: float = 1.0
    sources: List[str] = Field(default_factory=list)
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    access_count: int = 0

    def touch(self):
        self.access_count += 1
        self.updated_at = time.time()


class Relation(BaseModel):
    """A relationship between two entities (edge)."""
    id: str = Field(default_factory=lambda: hashlib.sha256(str(time.time_ns()).encode()).hexdigest()[:16])
    source_id: str
    target_id: str
    relation_type: str = "related_to"
    weight: float = 1.0
    evidence: str = ""
    confidence: float = 1.0
    created_at: float = Field(default_factory=time.time)


class GraphQuery(BaseModel):
    """A structured query against the knowledge graph."""
    text: str = ""
    entity_name: Optional[str] = None
    entity_type: Optional[str] = None
    relation_type: Optional[str] = None
    max_depth: int = 2
    limit: int = 20


class GraphResult(BaseModel):
    """Result from a graph query."""
    entities: List[Entity] = Field(default_factory=list)
    relations: List[Relation] = Field(default_factory=list)
    query_time_ms: float = 0.0
    total_nodes_searched: int = 0


# ============================================================
# Knowledge Graph Engine
# ============================================================

class KnowledgeGraph:
    """
    Knowledge graph engine with entity extraction and relationship discovery.

    Built for:
    - Code understanding (codegraph-inspired)
    - Document knowledge mapping
    - Multi-source entity resolution
    - Hybrid graph + vector search
    """

    def __init__(self, config: Optional[GraphConfig] = None):
        self.config = config or GraphConfig()
        self.entities: Dict[str, Entity] = {}
        self.relations: List[Relation] = []

        # Indexes for fast lookup
        self._by_name: Dict[str, List[str]] = defaultdict(list)  # name -> [entity_ids]
        self._by_type: Dict[str, Set[str]] = defaultdict(set)     # type -> {entity_ids}
        self._adjacency: Dict[str, List[Tuple[str, str, float]]] = defaultdict(list)  # source -> [(target, type, weight)]

        # Entity extraction patterns
        self._extraction_patterns = self._init_patterns()

    # ============================================================
    # Entity CRUD
    # ============================================================

    def add_entity(self, name: str, entity_type: str = "concept",
                   description: str = "", properties: Optional[Dict] = None,
                   confidence: float = 1.0, sources: Optional[List[str]] = None) -> Entity:
        """Add or update an entity in the knowledge graph."""
        # Check for duplicates
        if self.config.enable_deduplication:
            existing = self._find_duplicate(name, entity_type)
            if existing:
                existing.description = description or existing.description
                existing.confidence = max(existing.confidence, confidence)
                if properties:
                    existing.properties.update(properties)
                if sources:
                    existing.sources = list(set(existing.sources + sources))
                existing.touch()
                return existing

        entity = Entity(
            name=name,
            entity_type=entity_type,
            description=description,
            properties=properties or {},
            confidence=confidence,
            sources=sources or [],
        )

        self.entities[entity.id] = entity
        self._by_name[name.lower()].append(entity.id)
        self._by_type[entity_type].add(entity.id)

        if len(self.entities) > self.config.max_nodes:
            self._prune_least_used()

        return entity

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get an entity by ID."""
        entity = self.entities.get(entity_id)
        if entity:
            entity.touch()
        return entity

    def find_by_name(self, name: str, entity_type: Optional[str] = None) -> List[Entity]:
        """Find entities by name (case-insensitive)."""
        entity_ids = self._by_name.get(name.lower(), [])
        entities = [self.entities[eid] for eid in entity_ids if eid in self.entities]
        if entity_type:
            entities = [e for e in entities if e.entity_type == entity_type]
        return entities

    def find_by_type(self, entity_type: str) -> List[Entity]:
        """Find all entities of a given type."""
        entity_ids = self._by_type.get(entity_type, set())
        return [self.entities[eid] for eid in entity_ids if eid in self.entities]

    def remove_entity(self, entity_id: str):
        """Remove an entity and all its relations."""
        entity = self.entities.pop(entity_id, None)
        if not entity:
            return

        # Clean indexes
        self._by_name[entity.name.lower()].remove(entity_id)
        self._by_type[entity.entity_type].discard(entity_id)

        # Remove relations
        self.relations = [
            r for r in self.relations
            if r.source_id != entity_id and r.target_id != entity_id
        ]
        self._adjacency.pop(entity_id, None)

    # ============================================================
    # Relation Management
    # ============================================================

    def add_relation(self, source_id: str, target_id: str,
                     relation_type: str = "related_to",
                     weight: float = 1.0,
                     confidence: float = 1.0,
                     evidence: str = "") -> Optional[Relation]:
        """Add a relationship between two entities."""
        if source_id not in self.entities or target_id not in self.entities:
            logger.debug(f"Relation requires both entities to exist: {source_id} -> {target_id}")
            return None

        # Check for duplicate edges
        for r in self.relations:
            if r.source_id == source_id and r.target_id == target_id and r.relation_type == relation_type:
                r.weight = max(r.weight, weight)
                r.confidence = max(r.confidence, confidence)
                return r

        relation = Relation(
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            weight=weight,
            confidence=confidence,
            evidence=evidence,
        )
        self.relations.append(relation)
        self._adjacency[source_id].append((target_id, relation_type, weight))

        # Bidirectional adjacency for undirected traversal
        if relation_type in ["related_to", "same_as", "conflicts_with"]:
            self._adjacency[target_id].append((source_id, relation_type, weight))

        return relation

    def get_relations(self, entity_id: str, relation_type: Optional[str] = None,
                      direction: str = "both") -> List[Relation]:
        """Get relations for an entity. direction: 'outgoing', 'incoming', 'both'."""
        result = []
        for r in self.relations:
            if relation_type and r.relation_type != relation_type:
                continue
            if direction in ("outgoing", "both") and r.source_id == entity_id:
                result.append(r)
            if direction in ("incoming", "both") and r.target_id == entity_id:
                result.append(r)
        return result

    # ============================================================
    # Query & Traversal
    # ============================================================

    def query(self, query: GraphQuery) -> GraphResult:
        """Execute a structured graph query."""
        start_time = time.time()
        result = GraphResult()
        visited = set()

        # Resolve starting entities
        start_entities = []
        if query.entity_name:
            start_entities = self.find_by_name(query.entity_name, query.entity_type)
        elif query.entity_type:
            start_entities = self.find_by_type(query.entity_type)
        elif query.text:
            start_entities = self._text_search(query.text, limit=query.limit)

        # BFS traversal
        from collections import deque
        for start_entity in start_entities:
            queue = deque([(start_entity.id, 0)])
            while queue:
                current_id, depth = queue.popleft()
                if current_id in visited or depth > query.max_depth:
                    continue
                visited.add(current_id)

                entity = self.entities.get(current_id)
                if entity:
                    result.entities.append(entity)

                for target_id, rel_type, weight in self._adjacency.get(current_id, []):
                    if target_id not in visited:
                        if not query.relation_type or rel_type == query.relation_type:
                            queue.append((target_id, depth + 1))
                            # Find and add the relation
                            for r in self.relations:
                                if r.source_id == current_id and r.target_id == target_id:
                                    result.relations.append(r)
                                    break

                    if len(result.entities) >= query.limit:
                        break

        result.query_time_ms = (time.time() - start_time) * 1000
        result.total_nodes_searched = len(visited)
        return result

    def find_related(self, entity_name: str, top_k: int = 5,
                     relation_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Find entities related to a given entity name. Convenience method."""
        entities = self.find_by_name(entity_name)
        if not entities:
            return []

        target = entities[0]
        relations = self.get_relations(target.id)

        results = []
        for r in relations:
            if relation_types and r.relation_type not in relation_types:
                continue
            other_id = r.target_id if r.source_id == target.id else r.source_id
            other = self.entities.get(other_id)
            if other:
                results.append({
                    "entity": other.name,
                    "type": r.relation_type,
                    "weight": r.weight,
                    "confidence": r.confidence,
                    "description": other.description,
                })

        results.sort(key=lambda x: (x["weight"], x["confidence"]), reverse=True)
        return results[:top_k]

    def find_path(self, source_name: str, target_name: str,
                  max_depth: int = 4) -> Optional[List[Dict]]:
        """Find the shortest path between two named entities."""
        src_entities = self.find_by_name(source_name)
        tgt_entities = self.find_by_name(target_name)
        if not src_entities or not tgt_entities:
            return None

        src_id = src_entities[0].id
        tgt_id = tgt_entities[0].id

        from collections import deque
        queue = deque([(src_id, [])])
        visited = {src_id}

        while queue:
            current, path = queue.popleft()
            if len(path) >= max_depth:
                continue

            for target_id, rel_type, weight in self._adjacency.get(current, []):
                if target_id in visited:
                    continue

                new_path = path + [{
                    "from": self.entities[current].name,
                    "to": self.entities[target_id].name if target_id in self.entities else target_id,
                    "relation": rel_type,
                    "weight": weight,
                }]

                if target_id == tgt_id:
                    return new_path

                visited.add(target_id)
                queue.append((target_id, new_path))

        return None

    # ============================================================
    # Text Extraction (Simple Rule-based, for demo)
    # ============================================================

    def extract_from_text(self, text: str, source: str = "text") -> List[Entity]:
        """
        Extract entities from raw text.

        Uses simple rule-based extraction for demonstration.
        In production, this would use NER models (spaCy, GLiNER, etc.).
        """
        extracted = []

        for pattern_name, pattern in self._extraction_patterns.items():
            matches = pattern.findall(text)
            for match in matches:
                entity_text = match if isinstance(match, str) else match[0]
                entity_text = entity_text.strip()
                if len(entity_text) < 3 or len(entity_text) > 100:
                    continue

                entity_type_map = {
                    "code_entity": "code_entity",
                    "organization": "organization",
                    "person": "person",
                    "date": "event",
                    "url": "artifact",
                    "email": "artifact",
                }
                etype = entity_type_map.get(pattern_name, "concept")

                entity = self.add_entity(
                    name=entity_text,
                    entity_type=etype,
                    sources=[source],
                )
                extracted.append(entity)

        return extracted

    def extract_from_code(self, code: str, language: str = "python",
                          source: str = "code") -> List[Entity]:
        """
        Extract entities from source code.

        Identifies: functions, classes, imports, variables, decorators.
        Inspired by codegraph's code understanding approach.
        """
        extracted = []

        # Python extraction
        if language == "python":
            # Functions
            func_pattern = re.compile(r'def\s+(\w+)\s*\(')
            for func_name in func_pattern.findall(code):
                entity = self.add_entity(
                    name=func_name,
                    entity_type="code_entity",
                    description=f"Python function '{func_name}'",
                    properties={"language": language, "code_type": "function"},
                    sources=[source],
                )
                extracted.append(entity)

            # Classes
            class_pattern = re.compile(r'class\s+(\w+)[:(]')
            for class_name in class_pattern.findall(code):
                entity = self.add_entity(
                    name=class_name,
                    entity_type="code_entity",
                    description=f"Python class '{class_name}'",
                    properties={"language": language, "code_type": "class"},
                    sources=[source],
                )
                extracted.append(entity)

            # Imports
            import_pattern = re.compile(r'(?:from\s+(\S+)\s+import|import\s+(\S+))')
            for match in import_pattern.findall(code):
                module = match[0] or match[1]
                if module:
                    entity = self.add_entity(
                        name=module,
                        entity_type="code_entity",
                        description=f"Python module '{module}'",
                        properties={"language": language, "code_type": "module"},
                        sources=[source],
                    )
                    extracted.append(entity)

        return extracted

    # ============================================================
    # Serialization
    # ============================================================

    def export(self) -> Dict[str, Any]:
        """Export the knowledge graph to a serializable format."""
        return {
            "entities": {eid: e.model_dump() for eid, e in self.entities.items()},
            "relations": [r.model_dump() for r in self.relations],
            "config": self.config.model_dump(),
        }

    def import_data(self, data: Dict[str, Any]):
        """Import a previously exported knowledge graph."""
        for eid, raw in data.get("entities", {}).items():
            self.entities[eid] = Entity(**raw)
            entity = self.entities[eid]
            self._by_name[entity.name.lower()].append(eid)
            self._by_type[entity.entity_type].add(eid)

        for raw in data.get("relations", []):
            r = Relation(**raw)
            self.relations.append(r)
            self._adjacency[r.source_id].append((r.target_id, r.relation_type, r.weight))

    def get_stats(self) -> Dict[str, Any]:
        """Get knowledge graph statistics."""
        return {
            "total_entities": len(self.entities),
            "total_relations": len(self.relations),
            "entity_types": {t: len(ids) for t, ids in self._by_type.items()},
            "relation_types": dict(self._count_relation_types()),
            "avg_degree": len(self.relations) / max(len(self.entities), 1),
        }

    # ============================================================
    # Internal Helpers
    # ============================================================

    def _init_patterns(self) -> Dict[str, re.Pattern]:
        """Initialize entity extraction patterns."""
        return {
            "code_entity": re.compile(
                r'\b([A-Z][a-z]+(?:[A-Z][a-z]+)+)\b|'  # CamelCase
                r'\b([a-z]+(?:_[a-z]+)+)\b'             # snake_case
            ),
            "organization": re.compile(
                r'\b([A-Z][a-z]*(?:\s(?:Inc|Corp|LLC|Ltd|Co|Corporation|Company|Technologies|Labs|AI|Software|Systems)))\.?\b'
            ),
            "person": re.compile(
                r'\b(?:Dr\.|Mr\.|Mrs\.|Ms\.|Prof\.)?\s?[A-Z][a-z]+\s[A-Z][a-z]+\b'
            ),
            "date": re.compile(
                r'\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b|'
                r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s\d{1,2},?\s\d{4}\b'
            ),
            "url": re.compile(
                r'https?://[^\s<>"\']+|www\.[^\s<>"\']+'
            ),
            "email": re.compile(
                r'[\w.+-]+@[\w-]+\.[\w.-]+'
            ),
        }

    def _find_duplicate(self, name: str, entity_type: str) -> Optional[Entity]:
        """Find duplicate entities by name and type."""
        existing_ids = self._by_name.get(name.lower(), [])
        for eid in existing_ids:
            if eid in self.entities:
                entity = self.entities[eid]
                if entity.entity_type == entity_type:
                    return entity
        return None

    def _text_search(self, text: str, limit: int = 20) -> List[Entity]:
        """Search entities by text (simple keyword match)."""
        text_lower = text.lower()
        scored = []
        for entity in self.entities.values():
            score = 0
            if text_lower in entity.name.lower():
                score += 10
            if text_lower in entity.description.lower():
                score += 5
            # Property matching
            for key, val in entity.properties.items():
                if isinstance(val, str) and text_lower in val.lower():
                    score += 2
            if score > 0:
                scored.append((entity, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [e for e, _ in scored[:limit]]

    def _count_relation_types(self):
        """Count relations by type."""
        counts = defaultdict(int)
        for r in self.relations:
            counts[r.relation_type] += 1
        return dict(counts)

    def _prune_least_used(self):
        """Remove least-accessed entities to stay under max_nodes."""
        entities = sorted(self.entities.values(), key=lambda e: (e.access_count, e.updated_at))
        to_remove = entities[: max(1, len(entities) - self.config.max_nodes + 1000)]
        for entity in to_remove:
            self.remove_entity(entity.id)
