"""Use-case-agnostic eval harness.

Runs any compiled graph over any golden set and computes generic metrics. Use-case-specific
metrics (for MLR: blocker recall, false escalation rate) are supplied by the caller.
"""
