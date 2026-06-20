"""
Context Governance — Version, validate, audit, and trace every context change.

Inspired by:
- Enterprise context engineering best practices (Packmind, Tessl, Ruler)
- AI context debt research (MIT NANDA, Atlan)
- Data governance frameworks

Core governance principles:
1. Every context change is versioned
2. Every context entry has provenance
3. Schema validation prevents garbage-in-garbage-out
4. Audit trail enables trust and debugging
5. Policy enforcement prevents context drift
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from prometheus.core import ContextEngine, ContextEntry

logger = logging.getLogger(__name__)


# ============================================================
# Models
# ============================================================

class ChangeType(str, Enum):
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    VALIDATE = "validate"
    INVALIDATE = "invalidate"
    IMPORT = "import"
    EXPORT = "export"


class GovernancePolicy(BaseModel):
    """A policy that governs context operations."""
    name: str
    description: str = ""
    rules: List[str] = Field(default_factory=list)  # Python expressions that evaluate to bool
    applies_to: List[str] = Field(default_factory=list)  # Context keys (empty = all)
    action: str = "warn"  # warn | block | log_only
    enabled: bool = True


class AuditEntry(BaseModel):
    """A single entry in the governance audit log."""
    id: str = Field(default_factory=lambda: hashlib.sha256(str(time.time_ns()).encode()).hexdigest()[:16])
    timestamp: float = Field(default_factory=time.time)
    change_type: ChangeType
    entity_type: str = "context_entry"
    entity_id: str = ""
    entity_key: str = ""
    previous_value: Optional[Any] = None
    new_value: Optional[Any] = None
    source: str = ""
    user: str = "system"
    policy_violations: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class GovernanceStats(BaseModel):
    """Governance statistics."""
    total_changes: int = 0
    changes_by_type: Dict[str, int] = Field(default_factory=dict)
    total_violations: int = 0
    policies_active: int = 0
    last_change_at: Optional[float] = None
    context_health_score: float = 1.0  # 0.0-1.0


# ============================================================
# Context Governor
# ============================================================

class ContextGovernor:
    """
    Context governance layer.

    Enforces policies, maintains audit trails, and ensures
    context integrity across the entire ContextOS.

    This is what prevents AI context debt by ensuring every
    context change is validated, versioned, and traceable.
    """

    def __init__(self, engine: "ContextEngine"):
        self.engine = engine
        self.policies: Dict[str, GovernancePolicy] = {}
        self.audit_log: List[AuditEntry] = []
        self._violation_counts: Dict[str, int] = defaultdict(int)
        self._change_counts: Dict[str, int] = defaultdict(int)
        self.start_time = time.time()

        # Register default policies
        self._register_default_policies()

    # ============================================================
    # Policy Management
    # ============================================================

    def register_policy(self, policy: GovernancePolicy):
        """Register a governance policy."""
        self.policies[policy.name] = policy
        logger.info(f"Policy registered: {policy.name} (action={policy.action})")

    def enable_policy(self, name: str):
        """Enable a policy."""
        if name in self.policies:
            self.policies[name].enabled = True

    def disable_policy(self, name: str):
        """Disable a policy."""
        if name in self.policies:
            self.policies[name].enabled = False

    def check_policy(self, context_entry: "ContextEntry", change_type: ChangeType) -> List[str]:
        """
        Check a context change against all applicable policies.

        Returns:
            List of violated policy names.
        """
        violations = []
        for policy in self.policies.values():
            if not policy.enabled:
                continue
            if policy.applies_to and context_entry.key not in policy.applies_to:
                continue

            # Evaluate each rule
            for rule in policy.rules:
                try:
                    result = self._evaluate_rule(rule, context_entry)
                    if not result:
                        violations.append(policy.name)
                        self._violation_counts[policy.name] += 1

                        if policy.action == "block":
                            raise GovernanceError(f"Policy '{policy.name}' blocked the operation")
                        elif policy.action == "warn":
                            logger.warning(
                                f"Policy violation: {policy.name} for key '{context_entry.key}'"
                            )
                except Exception as e:
                    if "block" in str(e).lower():
                        raise
                    logger.debug(f"Rule evaluation error for '{policy.name}': {e}")

        return violations

    # ============================================================
    # Audit Trail
    # ============================================================

    def record_change(self, change_type: ChangeType, entry: "ContextEntry",
                      previous_value: Any = None, source: str = "",
                      metadata: Optional[Dict] = None):
        """Record a context change in the audit log."""
        # Check policies
        violations = self.check_policy(entry, change_type)

        audit = AuditEntry(
            change_type=change_type,
            entity_id=entry.id,
            entity_key=entry.key,
            previous_value=previous_value,
            new_value=entry.value if change_type != ChangeType.DELETE else None,
            source=source,
            policy_violations=violations,
            metadata=metadata or {},
        )
        self.audit_log.append(audit)
        self._change_counts[change_type.value] += 1

    def get_audit_trail(self, entity_key: Optional[str] = None,
                        limit: int = 100) -> List[AuditEntry]:
        """Get the audit trail, optionally filtered by entity key."""
        entries = self.audit_log
        if entity_key:
            entries = [e for e in entries if e.entity_key == entity_key]
        return sorted(entries, key=lambda e: e.timestamp, reverse=True)[:limit]

    def get_change_history(self, entity_key: str) -> List[Dict[str, Any]]:
        """Get the full change history for a specific context key."""
        entries = self.get_audit_trail(entity_key=entity_key)
        return [
            {
                "time": e.timestamp,
                "type": e.change_type.value,
                "from": e.previous_value,
                "to": e.new_value,
                "violations": e.policy_violations,
            }
            for e in entries
        ]

    def get_violations(self, policy_name: Optional[str] = None) -> List[AuditEntry]:
        """Get all entries with policy violations."""
        entries = [e for e in self.audit_log if e.policy_violations]
        if policy_name:
            entries = [e for e in entries if policy_name in e.policy_violations]
        return entries

    # ============================================================
    # Health & Statistics
    # ============================================================

    def health_check(self) -> GovernanceStats:
        """Perform a governance health check."""
        total_changes = len(self.audit_log)
        total_violations = sum(1 for e in self.audit_log if e.policy_violations)

        # Context health score
        # Factors: violation rate, validation coverage, consistency
        violation_rate = total_violations / max(total_changes, 1)
        validated_rate = len([
            e for e in self.engine.store.values() if e.validated
        ]) / max(len(self.engine.store), 1)

        health = 1.0 - (violation_rate * 0.5) - ((1 - validated_rate) * 0.3)
        health = max(0.0, min(1.0, health))

        return GovernanceStats(
            total_changes=total_changes,
            changes_by_type=dict(self._change_counts),
            total_violations=total_violations,
            policies_active=sum(1 for p in self.policies.values() if p.enabled),
            last_change_at=self.audit_log[-1].timestamp if self.audit_log else None,
            context_health_score=health,
        )

    def export_log(self) -> List[Dict[str, Any]]:
        """Export the audit log for persistence."""
        return [entry.model_dump() for entry in self.audit_log]

    def import_log(self, data: List[Dict[str, Any]]):
        """Import a previously exported audit log."""
        for raw in data:
            self.audit_log.append(AuditEntry(**raw))

    # ============================================================
    # Internal Helpers
    # ============================================================

    def _register_default_policies(self):
        """Register default governance policies."""
        # Policy: No empty values in critical keys
        self.register_policy(GovernancePolicy(
            name="no_empty_critical",
            description="Prevent empty values in critical context keys",
            rules=["len(str(value).strip()) > 0"],
            action="warn",
        ))

        # Policy: Maximum value length
        self.register_policy(GovernancePolicy(
            name="max_value_length",
            description="Warn when context values exceed reasonable length",
            rules=["len(str(value)) < 50000"],
            action="warn",
        ))

        # Policy: Confidence threshold
        self.register_policy(GovernancePolicy(
            name="confidence_threshold",
            description="Warn when context entries have low confidence",
            rules=["confidence >= 0.3"],
            action="warn",
        ))

        # Policy: Tag requirements
        self.register_policy(GovernancePolicy(
            name="tag_requirement",
            description="Ensure important context entries have tags",
            rules=["len(tags) > 0"],
            action="log_only",
        ))

    def _evaluate_rule(self, rule: str, entry: "ContextEntry") -> bool:
        """
        Safely evaluate a policy rule against a context entry.

        Rules are Python expressions that can reference:
        - value: the entry's value
        - key: the entry's key
        - tags: the entry's tags
        - confidence: the entry's confidence
        """
        safe_globals = {
            "__builtins__": {
                "len": len, "str": str, "int": int, "float": float,
                "bool": bool, "list": list, "dict": dict, "True": True,
                "False": False, "None": None, "isinstance": isinstance,
                "min": min, "max": max, "sum": sum, "abs": abs,
                "any": any, "all": all,
            }
        }
        safe_locals = {
            "value": entry.value,
            "key": entry.key,
            "tags": entry.tags,
            "confidence": entry.confidence,
        }

        try:
            return bool(eval(rule, safe_globals, safe_locals))
        except Exception as e:
            logger.debug(f"Rule evaluation failed: '{rule}' — {e}")
            return True  # Don't block on evaluation errors


class GovernanceError(Exception):
    """Raised when a governance policy blocks an operation."""
    pass


# ============================================================
# Context Health Analyzer
# ============================================================

class ContextHealthAnalyzer:
    """
    Analyzes context health and identifies debt.

    Inspired by:
    - AI Context Debt research
    - Packmind's context quality metrics
    - Atlan's 5 failure mode analysis
    """

    def __init__(self, engine: "ContextEngine"):
        self.engine = engine
        self.governor = engine.governor

    def analyze(self) -> Dict[str, Any]:
        """Run a comprehensive context health analysis."""
        entries = list(self.engine.store.values())
        gov_stats = self.governor.health_check() if self.governor else None

        analysis = {
            "total_entries": len(entries),
            "health_score": gov_stats.context_health_score if gov_stats else 0.5,
            "issues": [],
            "recommendations": [],
        }

        # Check 1: Stale entries (not updated in 30 days)
        now = time.time()
        stale = [e for e in entries if (now - e.updated_at) > 30 * 86400]
        if stale:
            analysis["issues"].append({
                "type": "stale_entries",
                "count": len(stale),
                "severity": "medium",
                "message": f"{len(stale)} entries haven't been updated in 30+ days",
            })

        # Check 2: Unvalidated entries
        unvalidated = [e for e in entries if not e.validated]
        if unvalidated:
            analysis["issues"].append({
                "type": "unvalidated_entries",
                "count": len(unvalidated),
                "severity": "high",
                "message": f"{len(unvalidated)} entries have not been validated",
            })

        # Check 3: Low confidence entries
        low_conf = [e for e in entries if e.confidence < 0.5]
        if low_conf:
            analysis["issues"].append({
                "type": "low_confidence",
                "count": len(low_conf),
                "severity": "medium",
                "message": f"{len(low_conf)} entries have confidence below 0.5",
            })

        # Check 4: Untagged entries
        untagged = [e for e in entries if not e.tags]
        if untagged:
            analysis["issues"].append({
                "type": "untagged",
                "count": len(untagged),
                "severity": "low",
                "message": f"{len(untagged)} entries have no tags",
            })

        # Generate recommendations
        if stale:
            analysis["recommendations"].append("Review and refresh stale context entries")
        if unvalidated:
            analysis["recommendations"].append("Run validation on unvalidated entries")
        if low_conf:
            analysis["recommendations"].append("Verify or remove low-confidence entries")

        return analysis
