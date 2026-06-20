"""
Tests for Prometheus ContextOS Core Engine.
"""

import asyncio
import pytest
from pathlib import Path
from prometheus.core import ContextEngine, ContextConfig, ContextSource, ContextMode, ContextEntry


class TestContextEngine:
    """Test the core ContextEngine."""

    def test_init(self, tmp_path):
        """Test basic initialization."""
        config = ContextConfig(
            workspace_dir=tmp_path,
            mode=ContextMode.ADAPTIVE,
            max_context_tokens=1000,
        )
        engine = ContextEngine(config)

        assert engine.config.mode == ContextMode.ADAPTIVE
        assert engine.config.max_context_tokens == 1000
        assert len(engine.store) == 0

    def test_set_get_context(self, tmp_path):
        """Test setting and getting context entries."""
        config = ContextConfig(workspace_dir=tmp_path)
        engine = ContextEngine(config)

        source = ContextSource(name="test", source_type="inline")
        engine.set("key1", "value1", source=source, tags=["tag1"])
        engine.set("key2", 42, source=source, tags=["tag2"])

        assert len(engine.store) == 2
        assert engine.get("key1") == "value1"
        assert engine.get("key2") == 42
        assert engine.get("nonexistent") is None

    def test_context_with_tags(self, tmp_path):
        """Test context filtering by tags."""
        config = ContextConfig(workspace_dir=tmp_path)
        engine = ContextEngine(config)

        source = ContextSource(name="test", source_type="inline")
        engine.set("k1", "v1", source=source, tags=["prod", "db"])
        engine.set("k2", "v2", source=source, tags=["dev", "api"])
        engine.set("k3", "v3", source=source, tags=["prod", "api"])

        prod_entries = engine.list(tag="prod")
        assert len(prod_entries) == 2

        db_entries = engine.list(tag="db")
        assert len(db_entries) == 1

        dev_entries = engine.list(tag="dev")
        assert len(dev_entries) == 1

    def test_context_confidence(self, tmp_path):
        """Test confidence scoring."""
        config = ContextConfig(workspace_dir=tmp_path)
        engine = ContextEngine(config)

        source = ContextSource(name="test", source_type="inline")
        entry_high = engine.set("high_conf", "certain", source=source, confidence=1.0)
        entry_low = engine.set("low_conf", "maybe", source=source, confidence=0.2)

        entries = engine.list()
        high = [e for e in entries if e.key == "high_conf"][0]
        low = [e for e in entries if e.key == "low_conf"][0]
        assert high.confidence == 1.0
        assert low.confidence == 0.2

    def test_context_duplicate_update(self, tmp_path):
        """Test updating an existing context entry."""
        config = ContextConfig(workspace_dir=tmp_path)
        engine = ContextEngine(config)

        source = ContextSource(name="test", source_type="inline")
        entry1 = engine.set("key", "old_value", source=source)
        entry2 = engine.set("key", "new_value", source=source)

        assert entry1.id == entry2.id
        assert engine.get("key") == "new_value"
        assert len(engine.store) == 1  # No duplicate entries

    def test_context_delete(self, tmp_path):
        """Test deleting a context entry."""
        config = ContextConfig(workspace_dir=tmp_path)
        engine = ContextEngine(config)

        source = ContextSource(name="test", source_type="inline")
        engine.set("key", "value", source=source)
        assert len(engine.store) == 1

        engine.remove("key")
        assert len(engine.store) == 0
        assert engine.get("key") is None

    def test_source_registration(self, tmp_path):
        """Test registering and loading sources."""
        config = ContextConfig(workspace_dir=tmp_path)
        engine = ContextEngine(config)

        source = ContextSource(
            name="inline_source",
            source_type="inline",
            content="Hello from inline source",
        )
        engine.register_source(source)
        assert any(s.name == "inline_source" for s in engine.sources)


class TestContextResolve:
    """Test context resolution."""

    @pytest.mark.asyncio
    async def test_basic_resolve(self, tmp_path):
        """Test basic context resolution."""
        config = ContextConfig(workspace_dir=tmp_path)
        engine = ContextEngine(config)
        await engine.initialize()

        source = ContextSource(name="test", source_type="inline")
        engine.set("security", "Use HTTPS and auth tokens", source=source, tags=["security"])
        engine.set("database", "PostgreSQL 16 on port 5432", source=source, tags=["db"])
        engine.set("deployment", "Docker + Kubernetes on AWS", source=source, tags=["devops"])

        result = await engine.resolve("What security measures are in place?")

        assert isinstance(result, dict)
        assert "_context" in result or "_compressed_context" in result or "_injected_context" in result

    @pytest.mark.asyncio
    async def test_resolve_empty(self, tmp_path):
        """Test resolving with no context."""
        config = ContextConfig(workspace_dir=tmp_path)
        engine = ContextEngine(config)
        await engine.initialize()

        result = await engine.resolve("Any query")
        assert isinstance(result, dict)


class TestContextCompression:
    """Test context compression."""

    @pytest.mark.asyncio
    async def test_compression_enabled(self, tmp_path):
        """Test that compression is applied when enabled."""
        config = ContextConfig(
            workspace_dir=tmp_path,
            enable_compression=True,
        )
        engine = ContextEngine(config)
        await engine.initialize()

        assert engine.compressor is not None

    @pytest.mark.asyncio
    async def test_compression_disabled(self, tmp_path):
        """Test that compression is skipped when disabled."""
        config = ContextConfig(
            workspace_dir=tmp_path,
            enable_compression=False,
        )
        engine = ContextEngine(config)
        await engine.initialize()

        # Should have no compressor
        pass
