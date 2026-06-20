"""
Multi-Agent Orchestrator — Coordinate specialized agents with shared context.

Inspired by:
- CrewAI (12K stars, minimalist multi-agent)
- Deer-Flow2 (35K stars, modular multi-agent with LangGraph)
- Microsoft Agent Framework (MAF)
- LangGraph (state-graph orchestration)

Architecture:
  Agent definition → Task decomposition → Parallel/Sequential execution → Result synthesis
  All agents share the same ContextOS for consistent context governance.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from prometheus.core import ContextEngine

logger = logging.getLogger(__name__)


# ============================================================
# Agent Models
# ============================================================

class AgentRole(str, Enum):
    RESEARCHER = "researcher"
    ANALYST = "analyst"
    WRITER = "writer"
    CODER = "coder"
    REVIEWER = "reviewer"
    EXECUTOR = "executor"
    COORDINATOR = "coordinator"
    CUSTOM = "custom"


class AgentCapability(BaseModel):
    """What an agent can do."""
    name: str
    description: str = ""
    skills: List[str] = Field(default_factory=list)  # Skill names
    tools: List[str] = Field(default_factory=list)    # Tool names
    priority: int = 0


class AgentDefinition(BaseModel):
    """Definition of a specialized agent."""
    id: str = Field(default_factory=lambda: hashlib.sha256(str(time.time_ns()).encode()).hexdigest()[:12])
    name: str
    role: AgentRole = AgentRole.CUSTOM
    description: str = ""
    capabilities: List[AgentCapability] = Field(default_factory=list)
    skills: List[str] = Field(default_factory=list)  # Skill IDs
    system_prompt: str = ""
    model: str = "default"
    temperature: float = 0.7
    max_iterations: int = 10
    timeout_seconds: int = 300

    @property
    def full_description(self) -> str:
        """Generate a full description for the orchestrator."""
        caps = ", ".join([c.name for c in self.capabilities])
        return f"[{self.role.value}] {self.name}: {self.description}. Capabilities: {caps}"


class Task(BaseModel):
    """A task to be executed by an agent."""
    id: str = Field(default_factory=lambda: hashlib.sha256(str(time.time_ns()).encode()).hexdigest()[:12])
    description: str
    agent_id: Optional[str] = None  # Assigned agent ID
    agent_role: Optional[AgentRole] = None  # Role requirement
    dependencies: List[str] = Field(default_factory=list)  # Task IDs that must complete first
    expected_output: str = ""
    context: Dict[str, Any] = Field(default_factory=dict)
    priority: int = 0
    status: str = "pending"  # pending, assigned, running, completed, failed

    class Config:
        use_enum_values = True


class TaskResult(BaseModel):
    """Result of a completed task."""
    task_id: str
    agent_id: str
    success: bool
    output: str = ""
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ============================================================
# Orchestration Plan
# ============================================================

class ExecutionMode(str, Enum):
    SEQUENTIAL = "sequential"   # One after another
    PARALLEL = "parallel"       # All at once (no deps)
    DAG = "dag"                 # Dependency-aware execution


class OrchestrationPlan(BaseModel):
    """A complete orchestration plan."""
    id: str = Field(default_factory=lambda: hashlib.sha256(str(time.time_ns()).encode()).hexdigest()[:16])
    name: str
    description: str = ""
    mode: ExecutionMode = ExecutionMode.DAG
    agents: List[AgentDefinition] = Field(default_factory=list)
    tasks: List[Task] = Field(default_factory=list)
    shared_context: Dict[str, Any] = Field(default_factory=dict)
    max_parallel: int = 4
    retry_failed: bool = True
    max_retries: int = 2


# ============================================================
# Orchestrator
# ============================================================

class AgentOrchestrator:
    """
    Multi-agent orchestrator with shared context.

    Coordinates multiple specialized agents, managing:
    - Task decomposition and assignment
    - Dependency resolution (DAG execution)
    - Context sharing across agents
    - Result synthesis and validation
    """

    def __init__(self, engine: Optional["ContextEngine"] = None):
        self.engine = engine
        self.agents: Dict[str, AgentDefinition] = {}
        self._agent_callbacks: Dict[str, List[Callable]] = {}

    # ============================================================
    # Agent Management
    # ============================================================

    def register_agent(self, agent: AgentDefinition):
        """Register an agent with the orchestrator."""
        self.agents[agent.id] = agent
        logger.info(f"Agent registered: {agent.full_description}")

    def unregister_agent(self, agent_id: str):
        """Remove an agent from the orchestrator."""
        self.agents.pop(agent_id, None)

    def find_agent_by_role(self, role: AgentRole) -> List[AgentDefinition]:
        """Find agents matching a role."""
        return [a for a in self.agents.values() if a.role == role]

    def find_agent_by_capability(self, capability_name: str) -> List[AgentDefinition]:
        """Find agents with a specific capability."""
        return [
            a for a in self.agents.values()
            if any(c.name == capability_name for c in a.capabilities)
        ]

    def on_agent_event(self, agent_id: str, callback: Callable):
        """Register an event callback for an agent."""
        if agent_id not in self._agent_callbacks:
            self._agent_callbacks[agent_id] = []
        self._agent_callbacks[agent_id].append(callback)

    # ============================================================
    # Task Decomposition
    # ============================================================

    def decompose(self, goal: str, available_agents: Optional[List[str]] = None) -> List[Task]:
        """
        Decompose a high-level goal into executable tasks.

        Uses agent capabilities to determine optimal task breakdown.
        For complex goals, this creates a dependency chain.

        In production, this would use an LLM to decompose.
        Here we use a rule-based approach based on agent roles.
        """
        tasks = []
        agents = [self.agents[aid] for aid in (available_agents or []) if aid in self.agents]
        if not agents:
            agents = list(self.agents.values())

        # Simple decomposition by keywords
        goal_lower = goal.lower()

        # Phase 1: Research/Understand
        if any(kw in goal_lower for kw in ["research", "find", "search", "understand", "analyze"]):
            researchers = self.find_agent_by_role(AgentRole.RESEARCHER)
            if researchers:
                tasks.append(Task(
                    description=f"Research and gather information: {goal}",
                    agent_id=researchers[0].id,
                    agent_role=AgentRole.RESEARCHER,
                    priority=10,
                ))

        # Phase 2: Analyze
        if any(kw in goal_lower for kw in ["analyze", "review", "audit", "check", "validate"]):
            # Depends on research if it exists
            deps = [tasks[-1].id] if tasks else []
            analysts = self.find_agent_by_role(AgentRole.ANALYST)
            if analysts:
                tasks.append(Task(
                    description=f"Analyze and evaluate: {goal}",
                    agent_id=analysts[0].id,
                    agent_role=AgentRole.ANALYST,
                    dependencies=deps,
                    priority=8,
                ))

        # Phase 3: Code/Build
        if any(kw in goal_lower for kw in ["code", "build", "create", "implement", "develop", "write"]):
            deps = [tasks[-1].id] if tasks else []
            coders = self.find_agent_by_role(AgentRole.CODER)
            if coders:
                tasks.append(Task(
                    description=f"Implement solution: {goal}",
                    agent_id=coders[0].id,
                    agent_role=AgentRole.CODER,
                    dependencies=deps,
                    priority=7,
                ))

        # Phase 4: Review
        if any(kw in goal_lower for kw in ["review", "test", "quality", "check"]):
            deps = [tasks[-1].id] if tasks else []
            reviewers = self.find_agent_by_role(AgentRole.REVIEWER)
            if reviewers:
                tasks.append(Task(
                    description=f"Review and verify: {goal}",
                    agent_id=reviewers[0].id,
                    agent_role=AgentRole.REVIEWER,
                    dependencies=deps,
                    priority=6,
                ))

        # If no specific phases matched, create a generic task
        if not tasks:
            # Assign to the most general-purpose agent
            coordinators = self.find_agent_by_role(AgentRole.COORDINATOR)
            target_agent = coordinators[0] if coordinators else (agents[0] if agents else None)
            if target_agent:
                tasks.append(Task(
                    description=goal,
                    agent_id=target_agent.id,
                    agent_role=target_agent.role,
                    priority=5,
                ))

        # Auto-assign unassigned tasks
        for task in tasks:
            if not task.agent_id and task.agent_role:
                candidates = self.find_agent_by_role(task.agent_role)
                if candidates:
                    task.agent_id = candidates[0].id

        return tasks

    # ============================================================
    # Execution Engine
    # ============================================================

    async def execute(self, plan: OrchestrationPlan,
                      agent_executor: Optional[Callable] = None) -> List[TaskResult]:
        """
        Execute an orchestration plan.

        Args:
            plan: The orchestration plan to execute
            agent_executor: Optional function to execute individual agent tasks.
                           Signature: async def executor(agent: AgentDefinition, task: Task) -> str
                           If not provided, runs in simulation mode.
        """
        results: List[TaskResult] = []
        completed: Set[str] = set()
        failed: Dict[str, int] = {}  # task_id -> retry count

        if plan.mode == ExecutionMode.SEQUENTIAL:
            # Sequential execution
            for task in plan.tasks:
                result = await self._execute_task(task, agent_executor)
                results.append(result)
                if result.success:
                    completed.add(task.id)
                elif plan.retry_failed:
                    # Retry logic
                    while failed.get(task.id, 0) < plan.max_retries:
                        failed[task.id] = failed.get(task.id, 0) + 1
                        result = await self._execute_task(task, agent_executor)
                        results.append(result)
                        if result.success:
                            completed.add(task.id)
                            break

        elif plan.mode == ExecutionMode.PARALLEL:
            # Parallel execution (no dependencies)
            tasks_coros = [self._execute_task(t, agent_executor) for t in plan.tasks]
            batch_results = await asyncio.gather(*tasks_coros, return_exceptions=True)
            for task, result in zip(plan.tasks, batch_results):
                if isinstance(result, Exception):
                    results.append(TaskResult(task_id=task.id, agent_id=task.agent_id or "", success=False, error=str(result)))
                else:
                    results.append(result)
                    if result.success:
                        completed.add(task.id)

        else:  # DAG mode
            # Dependency-aware execution
            pending = list(plan.tasks)
            in_flight: Dict[str, asyncio.Task] = {}

            while pending or in_flight:
                # Find tasks ready to execute (all deps satisfied)
                ready = []
                still_pending = []
                for task in pending:
                    if all(dep in completed for dep in task.dependencies):
                        ready.append(task)
                    else:
                        still_pending.append(task)
                pending = still_pending

                # Execute ready tasks (up to max_parallel)
                to_execute = ready[:plan.max_parallel - len(in_flight)]
                for task in to_execute:
                    in_flight[task.id] = asyncio.create_task(
                        self._execute_task(task, agent_executor)
                    )

                if not in_flight:
                    if pending:
                        # Deadlock — some tasks have unmet dependencies
                        stuck = [t.id for t in pending]
                        logger.error(f"Orchestration deadlock! Stuck tasks: {stuck}")
                        for task in pending:
                            results.append(TaskResult(
                                task_id=task.id,
                                agent_id=task.agent_id or "",
                                success=False,
                                error=f"Dependency deadlock. Missing: {[d for d in task.dependencies if d not in completed]}"
                            ))
                        break
                    break

                # Wait for at least one to complete
                done, _ = await asyncio.wait(
                    in_flight.values(),
                    return_when=asyncio.FIRST_COMPLETED,
                )

                for done_task in done:
                    result = done_task.result()
                    results.append(result)
                    if result.success:
                        completed.add(result.task_id)
                    elif plan.retry_failed and failed.get(result.task_id, 0) < plan.max_retries:
                        failed[result.task_id] = failed.get(result.task_id, 0) + 1
                        # Re-add to pending for retry
                        original = next((t for t in plan.tasks if t.id == result.task_id), None)
                        if original:
                            pending.append(original)

                    # Remove from in_flight
                    task_id = next((tid for tid, t in in_flight.items() if t == done_task), None)
                    if task_id:
                        del in_flight[task_id]

        # Synthesize results
        success_count = sum(1 for r in results if r.success)
        logger.info(f"Orchestration complete: {success_count}/{len(plan.tasks)} tasks succeeded")

        return results

    async def _execute_task(self, task: Task,
                            agent_executor: Optional[Callable] = None) -> TaskResult:
        """Execute a single task with the assigned agent."""
        agent = self.agents.get(task.agent_id or "")
        if not agent:
            return TaskResult(
                task_id=task.id, agent_id=task.agent_id or "",
                success=False, error=f"Agent '{task.agent_id}' not found"
            )

        start_time = time.time()
        task.status = "running"

        try:
            # Fire agent callbacks
            for callback in self._agent_callbacks.get(agent.id, []):
                try:
                    callback("task_start", task)
                except Exception:
                    pass

            if agent_executor:
                output = await agent_executor(agent, task)
            else:
                # Simulation mode
                output = await self._simulate_execution(agent, task)

            elapsed = (time.time() - start_time) * 1000
            task.status = "completed"

            # Fire completion callbacks
            for callback in self._agent_callbacks.get(agent.id, []):
                try:
                    callback("task_complete", task)
                except Exception:
                    pass

            return TaskResult(
                task_id=task.id,
                agent_id=agent.id,
                success=True,
                output=output,
                execution_time_ms=elapsed,
            )

        except asyncio.TimeoutError:
            task.status = "failed"
            return TaskResult(
                task_id=task.id, agent_id=agent.id,
                success=False,
                error=f"Timeout after {agent.timeout_seconds}s",
                execution_time_ms=agent.timeout_seconds * 1000,
            )
        except Exception as e:
            task.status = "failed"
            return TaskResult(
                task_id=task.id, agent_id=agent.id,
                success=False, error=str(e),
                execution_time_ms=(time.time() - start_time) * 1000,
            )

    async def _simulate_execution(self, agent: AgentDefinition, task: Task) -> str:
        """Simulate agent execution (for testing/demo)."""
        await asyncio.sleep(0.1)  # Simulate processing time
        return (
            f"[{agent.name}] Executed task: {task.description}\n"
            f"Role: {agent.role.value}\n"
            f"Skills: {', '.join(agent.skills)}\n"
            f"Result: Task completed successfully in simulation mode."
        )

    # ============================================================
    # Plan Management
    # ============================================================

    def create_plan(self, name: str, description: str = "",
                    mode: ExecutionMode = ExecutionMode.DAG) -> OrchestrationPlan:
        """Create a new orchestration plan."""
        return OrchestrationPlan(
            name=name,
            description=description,
            mode=mode,
        )

    def add_task_to_plan(self, plan: OrchestrationPlan, task: Task):
        """Add a task to an existing plan."""
        plan.tasks.append(task)

    def validate_plan(self, plan: OrchestrationPlan) -> Tuple[bool, List[str]]:
        """Validate an orchestration plan for correctness."""
        issues = []

        # Check all tasks have agents
        for task in plan.tasks:
            if not task.agent_id:
                issues.append(f"Task '{task.id}' has no assigned agent")

        # Check agent existence
        for task in plan.tasks:
            if task.agent_id and task.agent_id not in self.agents:
                issues.append(f"Task '{task.id}' references unknown agent '{task.agent_id}'")

        # Check dependency validity
        all_task_ids = {t.id for t in plan.tasks}
        for task in plan.tasks:
            for dep_id in task.dependencies:
                if dep_id not in all_task_ids:
                    issues.append(f"Task '{task.id}' depends on unknown task '{dep_id}'")

        # Check for cycles
        if self._has_cycle(plan.tasks):
            issues.append("Plan contains a dependency cycle!")

        return len(issues) == 0, issues

    def _has_cycle(self, tasks: List[Task]) -> bool:
        """Check if task dependencies contain a cycle (DFS)."""
        visited = set()
        rec_stack = set()

        def dfs(task_id):
            visited.add(task_id)
            rec_stack.add(task_id)
            task = next((t for t in tasks if t.id == task_id), None)
            if task:
                for dep_id in task.dependencies:
                    if dep_id not in visited:
                        if dfs(dep_id):
                            return True
                    elif dep_id in rec_stack:
                        return True
            rec_stack.discard(task_id)
            return False

        for task in tasks:
            if task.id not in visited:
                if dfs(task.id):
                    return True
        return False


# ============================================================
# Built-in Agent Templates
# ============================================================

def create_researcher_agent(name: str = "Researcher") -> AgentDefinition:
    """Create a pre-configured researcher agent."""
    return AgentDefinition(
        name=name,
        role=AgentRole.RESEARCHER,
        description="Gathers and synthesizes information from multiple sources.",
        capabilities=[
            AgentCapability(name="web_search", description="Search the web for information"),
            AgentCapability(name="document_analysis", description="Analyze documents and extract insights"),
            AgentCapability(name="data_collection", description="Collect and organize data"),
        ],
        system_prompt="You are an expert researcher. Gather comprehensive information, cite sources, and organize findings clearly.",
    )


def create_analyst_agent(name: str = "Analyst") -> AgentDefinition:
    """Create a pre-configured analyst agent."""
    return AgentDefinition(
        name=name,
        role=AgentRole.ANALYST,
        description="Analyzes data and provides insights and recommendations.",
        capabilities=[
            AgentCapability(name="data_analysis", description="Statistical and qualitative analysis"),
            AgentCapability(name="pattern_recognition", description="Identify patterns and trends"),
            AgentCapability(name="risk_assessment", description="Evaluate risks and opportunities"),
        ],
        system_prompt="You are an expert analyst. Evaluate information critically, identify patterns, and provide actionable insights.",
    )


def create_coder_agent(name: str = "Coder") -> AgentDefinition:
    """Create a pre-configured coding agent."""
    return AgentDefinition(
        name=name,
        role=AgentRole.CODER,
        description="Writes, refactors, and debugs code.",
        capabilities=[
            AgentCapability(name="code_generation", description="Generate production-quality code"),
            AgentCapability(name="code_review", description="Review code for quality and security"),
            AgentCapability(name="debugging", description="Identify and fix bugs"),
        ],
        system_prompt="You are an expert software engineer. Write clean, efficient, well-documented code following best practices.",
    )


def create_reviewer_agent(name: str = "Reviewer") -> AgentDefinition:
    """Create a pre-configured reviewer agent."""
    return AgentDefinition(
        name=name,
        role=AgentRole.REVIEWER,
        description="Reviews outputs for quality, correctness, and completeness.",
        capabilities=[
            AgentCapability(name="quality_assurance", description="Ensure output quality standards"),
            AgentCapability(name="fact_checking", description="Verify factual accuracy"),
            AgentCapability(name="completeness_check", description="Ensure all requirements are met"),
        ],
        system_prompt="You are an expert reviewer. Critically examine outputs for quality, accuracy, and completeness. Be thorough and constructive.",
    )
