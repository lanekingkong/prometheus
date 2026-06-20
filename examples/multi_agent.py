"""
Example: Multi-Agent Orchestration
==================================

Demonstrates how to:
1. Define specialized agents
2. Decompose a goal into tasks
3. Execute with dependency-aware orchestration
"""

import asyncio

from prometheus.core import ContextEngine, ContextConfig
from prometheus.orchestrator import (
    AgentOrchestrator, ExecutionMode, OrchestrationPlan,
    create_researcher_agent, create_analyst_agent,
    create_coder_agent, create_reviewer_agent,
)


async def main():
    print("=" * 60)
    print("Prometheus Multi-Agent Orchestration")
    print("=" * 60)

    # Initialize engine
    config = ContextConfig()
    engine = ContextEngine(config)
    await engine.initialize()

    # Create orchestrator
    orchestrator = AgentOrchestrator(engine)

    # Register specialized agents
    orchestrator.register_agent(create_researcher_agent("ResearchBot"))
    orchestrator.register_agent(create_analyst_agent("AnalysisBot"))
    orchestrator.register_agent(create_coder_agent("CodeBot"))
    orchestrator.register_agent(create_reviewer_agent("ReviewBot"))

    print(f"\n✓ Registered {len(orchestrator.agents)} agents")

    # Define the goal
    goal = "Research the best Python ORM for PostgreSQL, analyze trade-offs, and implement a database connection module"

    print(f"\n--- Goal ---")
    print(f"  {goal}")

    # Decompose into tasks
    tasks = orchestrator.decompose(goal)
    print(f"\n--- Decomposed into {len(tasks)} tasks ---")
    for i, task in enumerate(tasks, 1):
        agent = orchestrator.agents.get(task.agent_id or "")
        agent_name = agent.name if agent else "unassigned"
        deps = ", ".join(task.dependencies[:2]) or "none"
        print(f"  {i}. [{agent_name}] {task.description[:60]}... (deps: {deps})")

    # Create execution plan
    plan = OrchestrationPlan(
        name="ORM Selection & Implementation",
        description=goal,
        mode=ExecutionMode.DAG,
        tasks=tasks,
        max_parallel=2,
    )

    # Validate
    valid, issues = orchestrator.validate_plan(plan)
    if not valid:
        print(f"\n✗ Plan validation failed: {issues}")
        return
    print(f"\n✓ Plan validated")

    # Execute
    print(f"\n--- Executing (DAG mode, max 2 parallel) ---")
    results = await orchestrator.execute(plan)

    # Results
    success = sum(1 for r in results if r.success)
    print(f"\n--- Results: {success}/{len(results)} succeeded ---")
    for result in results:
        status = "✓" if result.success else "✗"
        time_str = f"{result.execution_time_ms:.0f}ms"
        print(f"  {status} [{time_str}] {result.output[:80] if result.output else result.error}")

    print(f"\n✓ Orchestration complete!")


if __name__ == "__main__":
    asyncio.run(main())
