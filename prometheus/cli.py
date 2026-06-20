"""
Prometheus CLI — Command-line interface for ContextOS.

Usage:
    prometheus init          Initialize a new Prometheus workspace
    prometheus skill add     Add a skill to the registry
    prometheus skill list    List all registered skills
    prometheus ctx set       Set a context entry
    prometheus ctx get       Get a context entry
    prometheus ctx list      List all context entries
    prometheus mem remember  Store a memory
    prometheus mem recall    Search memories
    prometheus agent run     Execute an agent orchestration
    prometheus health        Run context health check
    prometheus serve         Start the API server
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import click

from prometheus.core import ContextEngine, ContextConfig, ContextSource, ContextMode
from prometheus.skill import SkillRegistry, SkillLoader
from prometheus.memory import MemoryLayer, MemoryConfig, MemoryMode
from prometheus.knowledge import KnowledgeGraph, GraphConfig
from prometheus.orchestrator import (
    AgentOrchestrator, ExecutionMode, OrchestrationPlan,
    create_researcher_agent, create_analyst_agent,
    create_coder_agent, create_reviewer_agent,
)
from prometheus.context_gov import ContextGovernor, ContextHealthAnalyzer

# ============================================================
# Logging setup
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("prometheus.cli")


# ============================================================
# Shared state
# ============================================================

_engine: Optional[ContextEngine] = None


def get_engine() -> ContextEngine:
    global _engine
    if _engine is None:
        workspace = Path.cwd()
        config = ContextConfig(
            workspace_dir=workspace,
            skill_dirs=[
                workspace / "skills",
                workspace / ".prometheus" / "skills",
            ],
        )
        _engine = ContextEngine(config)
    return _engine


# ============================================================
# CLI Group
# ============================================================

@click.group()
@click.version_option(version="1.0.0", prog_name="Prometheus ContextOS")
def cli():
    """Prometheus — Universal AI Context Operating System.

    Solve AI context debt with governed context, composable skills,
    persistent memory, and multi-agent orchestration.
    """
    pass


# ============================================================
# init command
# ============================================================

@cli.command()
@click.option("--dir", "-d", default=".", help="Target directory")
@click.option("--force", "-f", is_flag=True, help="Overwrite existing config")
def init(dir: str, force: bool):
    """Initialize a new Prometheus workspace."""
    workspace = Path(dir).resolve()
    workspace.mkdir(parents=True, exist_ok=True)

    # Create directories
    (workspace / "skills").mkdir(exist_ok=True)
    (workspace / ".prometheus" / "skills").mkdir(parents=True, exist_ok=True)
    (workspace / "memory").mkdir(exist_ok=True)

    # Create config
    config_path = workspace / "prometheus.context.yaml"
    if config_path.exists() and not force:
        click.echo(f"Workspace already initialized at {workspace}")
        click.echo("Use --force to overwrite")
        return

    import yaml
    config_data = {
        "version": "1.0.0",
        "mode": "adaptive",
        "max_context_tokens": 128000,
        "skill_dirs": ["skills", ".prometheus/skills"],
        "memory": {"mode": "hybrid", "max_longterm_entries": 100000},
    }
    config_path.write_text(yaml.dump(config_data, default_flow_style=False))

    # Create example skill
    example_skill = SkillLoader.create_skill_template(
        name="example-skill",
        description="An example skill to get started with Prometheus",
        instructions=(
            "You are an example skill. When asked about Prometheus, "
            "explain that it is a Context Operating System that solves AI context debt."
        ),
    )
    (workspace / "skills" / "example-skill").mkdir(exist_ok=True)
    (workspace / "skills" / "example-skill" / "SKILL.md").write_text(example_skill)

    click.echo(f"✓ Prometheus workspace initialized at {workspace}")
    click.echo(f"  Config: {config_path}")
    click.echo(f"  Skills: {workspace / 'skills'}")
    click.echo(f"  Created example skill: example-skill")


# ============================================================
# skill commands
# ============================================================

@cli.group()
def skill():
    """Manage skills in the registry."""
    pass


@skill.command("add")
@click.argument("path", type=click.Path(exists=True))
def skill_add(path: str):
    """Add a skill from a SKILL.md file or directory."""
    engine = get_engine()
    asyncio.run(engine.initialize())

    skill_path = Path(path)
    if skill_path.is_dir():
        skills = SkillLoader.load_from_dir(skill_path)
        for s in skills:
            engine.skill_registry.register(s)
        click.echo(f"Added {len(skills)} skills from {skill_path}")
    else:
        content = skill_path.read_text(encoding="utf-8")
        skill_obj = SkillLoader.parse(content, skill_path)
        if skill_obj:
            engine.skill_registry.register(skill_obj)
            click.echo(f"Added skill: {skill_obj.id}")
        else:
            click.echo("Failed to parse skill file", err=True)


@skill.command("list")
@click.option("--tag", "-t", help="Filter by tag")
@click.option("--json", "json_output", is_flag=True, help="Output as JSON")
def skill_list(tag: str, json_output: bool):
    """List all registered skills."""
    engine = get_engine()
    asyncio.run(engine.initialize())

    if tag:
        skills = engine.skill_registry.find_by_tag(tag)
    else:
        skills = list(engine.skill_registry.skills.values())

    if json_output:
        click.echo(json.dumps(engine.skill_registry.list_all(), indent=2))
    else:
        click.echo(f"\n{'ID':<40} {'Version':<10} {'Description'}")
        click.echo("-" * 80)
        for s in sorted(skills, key=lambda x: x.usage_count, reverse=True):
            desc = s.meta.description[:50] + "..." if len(s.meta.description) > 50 else s.meta.description
            click.echo(f"{s.id:<40} {s.meta.version:<10} {desc}")
        click.echo(f"\nTotal: {len(skills)} skills")


@skill.command("suggest")
@click.argument("query")
def skill_suggest(query: str):
    """Suggest skills for a query."""
    engine = get_engine()
    asyncio.run(engine.initialize())

    suggestions = engine.skill_registry.suggest(query, top_k=5)
    if suggestions:
        click.echo(f"\nSuggested skills for '{query}':")
        for s in suggestions:
            click.echo(f"  • {s.id}: {s.meta.description}")
    else:
        click.echo(f"No skills found for '{query}'")


# ============================================================
# ctx commands
# ============================================================

@cli.group()
def ctx():
    """Manage context entries."""
    pass


@ctx.command("set")
@click.argument("key")
@click.argument("value")
@click.option("--tag", "-t", multiple=True, help="Tags")
@click.option("--source", "-s", default="cli", help="Source identifier")
@click.option("--confidence", "-c", type=float, default=1.0, help="Confidence 0.0-1.0")
def ctx_set(key: str, value: str, tag: tuple, source: str, confidence: float):
    """Set a context entry."""
    engine = get_engine()
    asyncio.run(engine.initialize())

    src = ContextSource(name=source, source_type="inline")
    entry = engine.set(key, value, source=src, tags=list(tag), confidence=confidence)
    click.echo(f"Set context: {entry.key} = {str(entry.value)[:80]}...")


@ctx.command("get")
@click.argument("key")
def ctx_get(key: str):
    """Get a context entry."""
    engine = get_engine()
    asyncio.run(engine.initialize())

    value = engine.get(key)
    if value is not None:
        click.echo(value)
    else:
        click.echo(f"Context key '{key}' not found", err=True)


@ctx.command("list")
@click.option("--tag", "-t", help="Filter by tag")
@click.option("--json", "json_output", is_flag=True, help="Output as JSON")
def ctx_list(tag: str, json_output: bool):
    """List all context entries."""
    engine = get_engine()
    asyncio.run(engine.initialize())

    entries = engine.list(tag=tag)

    if json_output:
        data = [e.model_dump() for e in entries]
        click.echo(json.dumps(data, indent=2))
    else:
        click.echo(f"\n{'Key':<30} {'Confidence':<12} {'Tags':<20} {'Updated'}")
        click.echo("-" * 85)
        for e in entries:
            import datetime
            updated = datetime.datetime.fromtimestamp(e.updated_at).strftime("%Y-%m-%d %H:%M")
            tags_str = ", ".join(e.tags[:3])
            click.echo(f"{e.key:<30} {e.confidence:<12.2f} {tags_str:<20} {updated}")
        click.echo(f"\nTotal: {len(entries)} entries")


@ctx.command("resolve")
@click.argument("query")
@click.option("--format", "-f", "output_format", type=click.Choice(["text", "json"]), default="text")
def ctx_resolve(query: str, output_format: str):
    """Resolve context for a query."""
    engine = get_engine()
    asyncio.run(engine.initialize())

    result = asyncio.run(engine.resolve(query))

    if output_format == "json":
        click.echo(json.dumps(result, indent=2, default=str))
    else:
        context_text = result.get("_compressed_context") or result.get("_injected_context", "")
        click.echo(context_text)

        if result.get("_suggested_skills"):
            click.echo("\n--- Suggested Skills ---")
            for s in result["_suggested_skills"]:
                click.echo(f"  • {s.id}")


# ============================================================
# mem commands
# ============================================================

@cli.group()
def mem():
    """Manage persistent memory."""
    pass


@mem.command("remember")
@click.argument("content")
@click.option("--type", "-t", "memory_type", default="fact",
              type=click.Choice(["fact", "event", "preference", "skill", "relationship"]))
@click.option("--importance", "-i", type=float, default=0.5)
@click.option("--tag", multiple=True)
def mem_remember(content: str, memory_type: str, importance: float, tag: tuple):
    """Store a memory."""
    engine = get_engine()
    asyncio.run(engine.initialize())

    entry = asyncio.run(engine.memory_layer.remember(
        content=content,
        memory_type=memory_type,
        importance=importance,
        tags=list(tag),
    ))
    click.echo(f"Remembered [{entry.memory_type}]: {entry.content[:80]}...")


@mem.command("recall")
@click.argument("query")
@click.option("--limit", "-l", type=int, default=10)
@click.option("--type", "-t", "memory_type",
              type=click.Choice(["fact", "event", "preference", "skill", "relationship"]))
def mem_recall(query: str, limit: int, memory_type: str):
    """Recall memories matching a query."""
    engine = get_engine()
    asyncio.run(engine.initialize())

    results = asyncio.run(engine.memory_layer.recall(
        query=query,
        limit=limit,
        memory_type=memory_type,
    ))

    if results:
        click.echo(f"\nMemories for '{query}':")
        for i, entry in enumerate(results, 1):
            click.echo(f"  {i}. [{entry.memory_type}] {entry.content[:100]}")
    else:
        click.echo(f"No memories found for '{query}'")


@mem.command("stats")
def mem_stats():
    """Show memory statistics."""
    engine = get_engine()
    asyncio.run(engine.initialize())

    stats = engine.memory_layer.get_stats()
    click.echo(json.dumps(stats, indent=2))


# ============================================================
# agent commands
# ============================================================

@cli.group()
def agent():
    """Manage multi-agent orchestration."""
    pass


@agent.command("run")
@click.argument("goal")
@click.option("--mode", "-m", type=click.Choice(["sequential", "parallel", "dag"]), default="dag")
def agent_run(goal: str, mode: str):
    """Execute a multi-agent task."""
    engine = get_engine()
    asyncio.run(engine.initialize())

    orch = AgentOrchestrator(engine)

    # Register built-in agents
    orch.register_agent(create_researcher_agent("Researcher"))
    orch.register_agent(create_analyst_agent("Analyst"))
    orch.register_agent(create_coder_agent("Coder"))
    orch.register_agent(create_reviewer_agent("Reviewer"))

    # Create plan
    plan = OrchestrationPlan(
        name=f"Task: {goal[:50]}",
        description=goal,
        mode=ExecutionMode(mode),
    )

    # Decompose goal into tasks
    tasks = orch.decompose(goal)
    plan.tasks = tasks

    click.echo(f"Orchestrating {len(tasks)} tasks in {mode} mode...")

    # Validate
    valid, issues = orch.validate_plan(plan)
    if not valid:
        click.echo("Plan validation failed:")
        for issue in issues:
            click.echo(f"  • {issue}")
        return

    # Execute
    results = asyncio.run(orch.execute(plan))

    click.echo(f"\nResults: {sum(1 for r in results if r.success)}/{len(results)} succeeded")
    for result in results:
        status = "✓" if result.success else "✗"
        click.echo(f"  {status} Task {result.task_id}: {result.output[:80] if result.output else result.error}")


@agent.command("list")
def agent_list():
    """List registered agents."""
    engine = get_engine()
    asyncio.run(engine.initialize())

    orch = AgentOrchestrator(engine)

    # Register built-in agents for demo
    orch.register_agent(create_researcher_agent("Researcher"))
    orch.register_agent(create_analyst_agent("Analyst"))
    orch.register_agent(create_coder_agent("Coder"))
    orch.register_agent(create_reviewer_agent("Reviewer"))

    click.echo(f"\n{'ID':<16} {'Role':<14} {'Name':<20} {'Description'}")
    click.echo("-" * 85)
    for agent in orch.agents.values():
        desc = agent.description[:50] + "..." if len(agent.description) > 50 else agent.description
        click.echo(f"{agent.id:<16} {agent.role.value:<14} {agent.name:<20} {desc}")
    click.echo(f"\nTotal: {len(orch.agents)} agents")


# ============================================================
# health command
# ============================================================

@cli.command()
def health():
    """Run context health check."""
    engine = get_engine()
    asyncio.run(engine.initialize())

    analyzer = ContextHealthAnalyzer(engine)
    analysis = analyzer.analyze()

    click.echo(f"\nContext Health Report")
    click.echo(f"{'=' * 50}")
    click.echo(f"  Health Score: {analysis['health_score']:.0%}")
    click.echo(f"  Total Entries: {analysis['total_entries']}")

    if analysis["issues"]:
        click.echo(f"\n  Issues ({len(analysis['issues'])}):")
        for issue in analysis["issues"]:
            severity_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(issue["severity"], "⚪")
            click.echo(f"    {severity_icon} [{issue['type']}] {issue['message']}")

    if analysis["recommendations"]:
        click.echo(f"\n  Recommendations:")
        for rec in analysis["recommendations"]:
            click.echo(f"    • {rec}")

    if not analysis["issues"]:
        click.echo(f"\n  ✓ Context is healthy!")


# ============================================================
# serve command (placeholder for API server)
# ============================================================

@cli.command()
@click.option("--host", "-h", default="127.0.0.1", help="Host to bind to")
@click.option("--port", "-p", default=8765, help="Port to bind to")
def serve(host: str, port: int):
    """Start the Prometheus API server."""
    click.echo(f"Starting Prometheus API server on {host}:{port}...")
    click.echo("API server coming in future release.")
    click.echo("For now, use the CLI commands to interact with ContextOS.")


# ============================================================
# Entry point
# ============================================================

def main():
    """Main entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
