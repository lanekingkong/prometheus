"""
Tests for Prometheus Knowledge Graph Engine.
"""

import pytest
from prometheus.knowledge import KnowledgeGraph, GraphConfig, GraphQuery


class TestKnowledgeGraph:
    """Test the knowledge graph engine."""

    def test_add_entity(self):
        """Test adding entities."""
        kg = KnowledgeGraph()
        entity = kg.add_entity(
            name="Python",
            entity_type="programming_language",
            description="A high-level programming language",
        )

        assert entity.name == "Python"
        assert entity.entity_type == "programming_language"
        assert entity.id in kg.entities

        stats = kg.get_stats()
        assert stats["total_entities"] == 1

    def test_add_duplicate_entity(self):
        """Test that duplicates are handled correctly."""
        kg = KnowledgeGraph()

        e1 = kg.add_entity("Python", entity_type="programming_language", description="First desc")
        e2 = kg.add_entity("Python", entity_type="programming_language", description="Second desc")

        # Should be the same entity (deduplicated)
        assert e1.id == e2.id
        assert kg.get_stats()["total_entities"] == 1

    def test_find_by_name(self):
        """Test finding entities by name."""
        kg = KnowledgeGraph()

        kg.add_entity("Python", entity_type="programming_language")
        kg.add_entity("JavaScript", entity_type="programming_language")
        kg.add_entity("React", entity_type="framework")
        kg.add_entity("Python", entity_type="snake", description="A reptile")

        results = kg.find_by_name("Python")
        assert len(results) == 2  # Two entities named Python (different types)

        results = kg.find_by_name("Python", entity_type="programming_language")
        assert len(results) == 1

    def test_find_by_type(self):
        """Test finding entities by type."""
        kg = KnowledgeGraph()

        kg.add_entity("Python", entity_type="programming_language")
        kg.add_entity("JavaScript", entity_type="programming_language")
        kg.add_entity("React", entity_type="framework")
        kg.add_entity("Vue", entity_type="framework")

        langs = kg.find_by_type("programming_language")
        assert len(langs) == 2

        frameworks = kg.find_by_type("framework")
        assert len(frameworks) == 2

    def test_add_relation(self):
        """Test adding relations between entities."""
        kg = KnowledgeGraph()

        python = kg.add_entity("Python", entity_type="language")
        django = kg.add_entity("Django", entity_type="framework")

        relation = kg.add_relation(
            python.id,
            django.id,
            relation_type="has_a",
            weight=0.9,
        )

        assert relation is not None
        assert relation.relation_type == "has_a"

        stats = kg.get_stats()
        assert stats["total_relations"] == 1

    def test_add_relation_nonexistent(self):
        """Test adding a relation with non-existent entities."""
        kg = KnowledgeGraph()

        result = kg.add_relation("nonexistent1", "nonexistent2", "related_to")
        assert result is None  # Should fail gracefully

    def test_find_related(self):
        """Test finding related entities."""
        kg = KnowledgeGraph()

        python = kg.add_entity("Python", entity_type="language")
        django = kg.add_entity("Django", entity_type="framework")
        flask = kg.add_entity("Flask", entity_type="framework")
        numpy = kg.add_entity("NumPy", entity_type="library")

        kg.add_relation(python.id, django.id, "has_a")
        kg.add_relation(python.id, flask.id, "has_a")
        kg.add_relation(python.id, numpy.id, "has_a")

        related = kg.find_related("Python", top_k=3)
        assert len(related) == 3
        assert any(r["entity"] == "Django" for r in related)

    def test_graph_query_basic(self):
        """Test basic graph queries."""
        kg = KnowledgeGraph()

        python = kg.add_entity("Python", entity_type="language")
        django = kg.add_entity("Django", entity_type="framework")
        flask = kg.add_entity("Flask", entity_type="framework")

        kg.add_relation(python.id, django.id, "has_a")
        kg.add_relation(python.id, flask.id, "has_a")

        query = GraphQuery(entity_name="Python", max_depth=1)
        result = kg.query(query)

        assert len(result.entities) >= 1
        assert result.total_nodes_searched >= 1

    def test_graph_query_by_type(self):
        """Test graph queries filtered by type."""
        kg = KnowledgeGraph()

        kg.add_entity("Python", entity_type="language")
        kg.add_entity("Django", entity_type="framework")
        kg.add_entity("Flask", entity_type="framework")
        kg.add_entity("JavaScript", entity_type="language")

        query = GraphQuery(entity_type="framework", max_depth=0)
        result = kg.query(query)

        assert len(result.entities) == 2
        assert all(e.entity_type == "framework" for e in result.entities)

    def test_find_path(self):
        """Test finding paths between entities."""
        kg = KnowledgeGraph()

        a = kg.add_entity("A", entity_type="test")
        b = kg.add_entity("B", entity_type="test")
        c = kg.add_entity("C", entity_type="test")

        kg.add_relation(a.id, b.id, "depends_on")
        kg.add_relation(b.id, c.id, "depends_on")

        path = kg.find_path("A", "C", max_depth=3)
        assert path is not None
        assert len(path) == 2  # A -> B, B -> C

    def test_find_path_nonexistent(self):
        """Test path finding with non-existent entities."""
        kg = KnowledgeGraph()
        path = kg.find_path("X", "Y")
        assert path is None

    def test_remove_entity(self):
        """Test removing an entity and its relations."""
        kg = KnowledgeGraph()

        e1 = kg.add_entity("E1", entity_type="test")
        e2 = kg.add_entity("E2", entity_type="test")

        kg.add_relation(e1.id, e2.id, "related_to")

        kg.remove_entity(e1.id)

        stats = kg.get_stats()
        assert stats["total_entities"] == 1  # Only E2 remains
        assert stats["total_relations"] == 0  # Relation removed

    def test_extract_from_text(self):
        """Test entity extraction from text."""
        kg = KnowledgeGraph()

        text = "Apple Inc. announced a new iPhone on 2026-06-20. Tim Cook presented at WWDC. Check https://apple.com for details."

        entities = kg.extract_from_text(text, source="test")
        assert len(entities) >= 1  # Should extract at least one entity

    def test_extract_from_code(self):
        """Test entity extraction from code."""
        kg = KnowledgeGraph()

        code = """
import os
from typing import List, Optional

class Database:
    def connect(self, url: str) -> bool:
        return True

def main():
    db = Database()
    db.connect("postgres://localhost")

if __name__ == "__main__":
    main()
"""
        entities = kg.extract_from_code(code, language="python", source="test")

        # Should extract class, function, and imports
        assert len(entities) >= 1

    def test_serialization(self):
        """Test export and import."""
        kg = KnowledgeGraph()

        e1 = kg.add_entity("E1", entity_type="test")
        e2 = kg.add_entity("E2", entity_type="test")
        kg.add_relation(e1.id, e2.id, "related_to")

        exported = kg.export()

        # Create new graph and import
        kg2 = KnowledgeGraph()
        kg2.import_data(exported)

        assert kg2.get_stats()["total_entities"] == 2
        assert kg2.get_stats()["total_relations"] == 1


class TestGraphQuery:
    """Test GraphQuery model."""

    def test_defaults(self):
        query = GraphQuery()
        assert query.max_depth == 2
        assert query.limit == 20

    def test_text_search(self):
        query = GraphQuery(text="Python framework", max_depth=1, limit=5)
        assert query.text == "Python framework"
        assert query.limit == 5
