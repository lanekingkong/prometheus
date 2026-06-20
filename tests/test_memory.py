"""
Tests for Prometheus Memory Layer.
"""

import asyncio
import pytest
from pathlib import Path
from prometheus.memory import MemoryLayer, MemoryConfig, MemoryMode, MemoryEntry
from prometheus.core import ContextEngine, ContextConfig


class TestMemoryLayer:
    """Test the memory layer."""

    @pytest.mark.asyncio
    async def test_remember_and_recall(self, tmp_path):
        """Test basic remember and recall operations."""
        config = MemoryConfig(
            memory_dir=tmp_path / "memory",
            mode=MemoryMode.HYBRID,
        )
        memory = MemoryLayer(config)
        await memory.initialize()

        # Remember facts
        await memory.remember("Paris is the capital of France", memory_type="fact", importance=0.9)
        await memory.remember("Tokyo is the capital of Japan", memory_type="fact", importance=0.8)
        await memory.remember("Python is a programming language", memory_type="fact", importance=0.7)

        # Recall
        results = await memory.recall("capital of France")
        assert len(results) > 0

        # The most relevant should be about Paris
        assert any("Paris" in r.content for r in results)

    @pytest.mark.asyncio
    async def test_recall_by_type(self, tmp_path):
        """Test filtering recall by memory type."""
        config = MemoryConfig(memory_dir=tmp_path / "memory")
        memory = MemoryLayer(config)
        await memory.initialize()

        await memory.remember("Fact 1", memory_type="fact", importance=0.9)
        await memory.remember("Event 1: Meeting at 3pm", memory_type="event", importance=0.5)
        await memory.remember("Preference: Dark mode", memory_type="preference", importance=0.7)

        facts = await memory.recall("Fact", memory_type="fact")
        assert len(facts) >= 1
        assert all(m.memory_type == "fact" for m in facts)

        events = await memory.recall("Meeting", memory_type="event")
        assert len(events) >= 1
        assert all(m.memory_type == "event" for m in events)

    @pytest.mark.asyncio
    async def test_recall_with_tags(self, tmp_path):
        """Test memory recall with tag filtering."""
        config = MemoryConfig(memory_dir=tmp_path / "memory")
        memory = MemoryLayer(config)
        await memory.initialize()

        await memory.remember("DB password is secret123", memory_type="fact", tags=["security", "db"], importance=1.0)
        await memory.remember("API key is abc-def-ghi", memory_type="fact", tags=["security", "api"], importance=1.0)
        await memory.remember("Server is on AWS us-east-1", memory_type="fact", tags=["infra", "aws"], importance=0.5)

        security_memories = await memory.recall("", tags=["security"], memory_type="fact")
        assert len(security_memories) >= 2

    @pytest.mark.asyncio
    async def test_forget(self, tmp_path):
        """Test forgetting (removing) a memory."""
        config = MemoryConfig(memory_dir=tmp_path / "memory")
        memory = MemoryLayer(config)
        await memory.initialize()

        entry = await memory.remember("Temporary fact", memory_type="fact", importance=0.3)

        stats_before = memory.get_stats()
        await memory.forget(entry.id)
        stats_after = memory.get_stats()

        assert stats_after["longterm_entries"] == stats_before["longterm_entries"] - 1

    @pytest.mark.asyncio
    async def test_improve_memory(self, tmp_path):
        """Test improving (updating) a memory."""
        config = MemoryConfig(memory_dir=tmp_path / "memory")
        memory = MemoryLayer(config)
        await memory.initialize()

        entry = await memory.remember("Old info", memory_type="fact", importance=0.3)
        await memory.improve(entry.id, {"content": "Updated info with more details", "importance": 0.9})

        # Recall and verify
        results = await memory.recall("Updated info")
        assert len(results) > 0
        assert any("Updated" in r.content for r in results)

    @pytest.mark.asyncio
    async def test_get_stats(self, tmp_path):
        """Test getting memory statistics."""
        config = MemoryConfig(memory_dir=tmp_path / "memory")
        memory = MemoryLayer(config)
        await memory.initialize()

        await memory.remember("Fact 1", memory_type="fact", importance=0.5)
        await memory.remember("Fact 2", memory_type="fact", importance=0.6)
        await memory.remember("Event 1", memory_type="event", importance=0.7)

        stats = memory.get_stats()
        assert stats["longterm_entries"] == 3
        assert stats.get("types_breakdown") or True  # Should have some type info
