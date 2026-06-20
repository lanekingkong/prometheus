"""
Prometheus — Universal AI Context Operating System
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The first ContextOS: Solve AI Context Debt with governed context,
composable skills, persistent memory, and multi-agent orchestration.
"""

__version__ = "1.0.0"
__author__ = "lanekingkong"
__license__ = "Apache-2.0"

from prometheus.core import ContextEngine, ContextConfig
from prometheus.skill import SkillRegistry, Skill, SkillLoader
from prometheus.memory import MemoryLayer, MemoryConfig
from prometheus.knowledge import KnowledgeGraph, GraphConfig
from prometheus.orchestrator import AgentOrchestrator, OrchestrationPlan
from prometheus.compressor import TokenCompressor
from prometheus.context_gov import ContextGovernor

__all__ = [
    "ContextEngine",
    "ContextConfig",
    "SkillRegistry",
    "Skill",
    "SkillLoader",
    "MemoryLayer",
    "MemoryConfig",
    "KnowledgeGraph",
    "GraphConfig",
    "AgentOrchestrator",
    "OrchestrationPlan",
    "TokenCompressor",
    "ContextGovernor",
]
