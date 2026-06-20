"""
Example: Basic ContextOS Usage
===============================

This example demonstrates the core workflow:
1. Initialize ContextOS
2. Set context entries
3. Resolve context for a query
4. Use skills and memory
"""

import asyncio
from pathlib import Path

from prometheus.core import ContextEngine, ContextConfig, ContextSource, ContextMode
from prometheus.skill import SkillRegistry, SkillLoader
from prometheus.memory import MemoryLayer, MemoryConfig, MemoryMode


async def main():
    print("=" * 60)
    print("Prometheus ContextOS — Basic Example")
    print("=" * 60)

    # 1. Initialize
    workspace = Path("./example_workspace")
    config = ContextConfig(
        workspace_dir=workspace,
        mode=ContextMode.ADAPTIVE,
        max_context_tokens=128000,
        enable_compression=True,
    )
    engine = ContextEngine(config)
    await engine.initialize()

    print(f"\n✓ ContextOS initialized (mode: {config.mode.value})")

    # 2. Register context sources
    engine.register_source(ContextSource(
        name="project_readme",
        source_type="file",
        path="README.md",
        priority=10,
    ))
    engine.register_source(ContextSource(
        name="api_docs",
        source_type="inline",
        content="API Base URL: https://api.example.com/v1\nAuth: Bearer token required",
        priority=8,
    ))

    # 3. Set context entries
    engine.set("app_name", "MySaaS Platform", tags=["project", "metadata"])
    engine.set("database_url", "postgresql://localhost:5432/myapp", tags=["infra", "database"])
    engine.set("deployment_env", "production", tags=["infra", "deployment"])
    engine.set("security_policy", "All API calls must use HTTPS and include authentication headers", tags=["security", "policy"])

    print(f"✓ Set {len(engine.store)} context entries")

    # 4. Resolve context
    query = "What are the security requirements for deploying?"
    print(f"\n--- Resolving context for: '{query}' ---")

    result = await engine.resolve(query)

    context_text = result.get("_compressed_context") or result.get("_injected_context", "")
    print(context_text[:500])

    # 5. Memory layer
    print(f"\n--- Memory Layer ---")
    await engine.memory_layer.remember(
        "Deployment must go through CI/CD pipeline with security scan",
        memory_type="fact",
        importance=0.9,
        tags=["deployment", "security"],
    )
    await engine.memory_layer.remember(
        "Last security audit passed with 98% score on 2026-06-15",
        memory_type="event",
        importance=0.8,
        tags=["security", "audit"],
    )

    memories = await engine.memory_layer.recall("security audit")
    for mem in memories:
        print(f"  [{mem.memory_type}] {mem.content[:80]}...")

    # 6. Suggested skills
    if result.get("_suggested_skills"):
        print(f"\n--- Suggested Skills ---")
        for skill in result["_suggested_skills"]:
            print(f"  • {skill.id}: {skill.meta.description}")

    # 7. Stats
    stats = {
        "context_entries": len(engine.store),
        "memory_entries": engine.memory_layer.get_stats()["longterm_entries"],
        "skills_loaded": len(engine.skill_registry.skills),
        "knowledge_nodes": engine.knowledge_graph.get_stats()["total_entities"],
    }
    print(f"\n--- Statistics ---")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print(f"\n✓ Example complete!")


if __name__ == "__main__":
    asyncio.run(main())
