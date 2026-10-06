"""Tools the MLR use case registers with the platform's tool registry.

Importing this package registers every tool. ``mlr.graph`` imports it, so a graph build is enough
to make the tools available -- a node never has to remember to import the module it calls.
"""

from __future__ import annotations

from mlr.tools import fair_balance_validator

__all__ = ["fair_balance_validator"]
