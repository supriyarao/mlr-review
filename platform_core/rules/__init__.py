"""Per-tenant business rules: a registry of versioned, tenant-authored checks.

Why this exists: without it, a tenant can tune *how strict* a review is -- thresholds, severity
mappings, escalation tiers -- but cannot change *what gets checked* without an engineering change.
Real compliance teams own banned-term lists, mandatory disclaimers, trademark conventions, a code of
practice and a brand style guide, and they edit them far more often than anyone ships code. So rules
are data: authored, versioned, tested and activated by their owners.

This package is **platform** code and knows nothing about the MLR use case. It evaluates rules
against generic text and segments, and returns neutral :class:`RuleViolation` objects. Mapping a
violation onto a use case's finding schema is the use case's job.
"""

from platform_core.rules.schema import (
    Rule,
    RuleExample,
    RuleInput,
    RuleScope,
    RuleViolation,
    Segment,
)

__all__ = [
    "Rule",
    "RuleExample",
    "RuleInput",
    "RuleScope",
    "RuleViolation",
    "Segment",
]
