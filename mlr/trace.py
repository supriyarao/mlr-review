"""Print each node's input and output state, for watching a review actually happen.

Why a module rather than ``print`` calls in the nodes: a node's job is to review content, and
debugging output scattered through fifteen of them is noise that then has to be removed. Tracing is a
cross-cutting concern, every node already passes through one wrapper in :func:`mlr.graph.build`, and
putting it there means it is uniform, can be turned off, and cannot be half-removed later.

**Off unless asked for.** ``MLR_TRACE=1`` gives a compact summary per node; ``MLR_TRACE=full`` prints
whole values, which for a document or a finding list is long but sometimes exactly what you want.

Output is assembled into one string and written under a lock, because the eight parallel branches would
otherwise interleave line by line into something unreadable.

Deterministic code. No model involved.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from collections.abc import Callable
from typing import Any

_lock = threading.Lock()

# Keys printed first, because they identify the run and are what you scan for.
_IDENTITY = ("run_id", "tenant_id", "product_id", "label_version", "audience", "channel", "status")

# How much of a long value to show in compact mode.
_SNIPPET = 72

BOLD, DIM, RESET = "\033[1m", "\033[2m", "\033[0m"
CYAN, GREEN, AMBER, RED = "\033[36m", "\033[32m", "\033[33m", "\033[31m"


def mode() -> str:
    """``""`` (off), ``"compact"`` or ``"full"``."""
    value = os.environ.get("MLR_TRACE", "").strip().lower()
    if value in ("", "0", "false", "no", "off"):
        return ""
    return "full" if value in ("full", "2", "verbose") else "compact"


def enabled() -> bool:
    return bool(mode())


def _colour(text: str, colour: str) -> str:
    return text if not sys.stdout.isatty() else f"{colour}{text}{RESET}"


def describe(key: str, value: Any, *, full: bool) -> str:
    """One line for one state key.

    Compact by choice: a review's state holds the whole submitted document and every finding, so
    printing it verbatim per node buries the thing you were looking for. What is shown instead is the
    shape -- how many claims, how many findings, which gates -- because that is what changes from node
    to node and what tells you whether a stage did its job.
    """
    if full:
        return f"{key}={value!r}"

    if value is None:
        return f"{key}=None"

    if key == "document":
        return f"{key}={len(str(value))} chars"

    if key == "doc_meta" and isinstance(value, dict):
        sections = value.get("sections") or {}
        return f"{key}={len(sections)} sections"

    if key == "findings" or key == "final_findings":
        items = list(value or [])
        blockers = sum(1 for f in items if getattr(f, "is_blocker", False))
        return f"{key}={len(items)} ({blockers} blocker)"

    if key == "gate_results":
        items = list(value or [])
        failed = [g.gate for g in items if not getattr(g, "passed", True)]
        tail = f", failed: {', '.join(failed)}" if failed else ""
        return f"{key}={len(items)}{tail}"

    if key == "claims":
        return f"{key}={len(list(value or []))}"

    if key == "rules_applied":
        items = list(value or [])
        return f"{key}={len(items)} ({', '.join(getattr(r, 'id', '?') for r in items[:4])}" + (
            ", ...)" if len(items) > 4 else ")"
        )

    if key == "escalation" and value is not None:
        codes = ", ".join(getattr(value, "reason_codes", lambda: [])())
        return f"{key}=tier={getattr(value, 'tier', '?')} reasons=[{codes}]"

    if key == "cost" and value is not None:
        return (
            f"{key}=${getattr(value, 'cost_usd', 0):.5f} {getattr(value, 'model_calls', 0)} call(s)"
        )

    if isinstance(value, (list, tuple, set)):
        return f"{key}={len(value)} item(s)"

    if isinstance(value, dict):
        return f"{key}={{{len(value)} key(s)}}"

    text = str(value)
    if len(text) > _SNIPPET:
        return f"{key}={text[:_SNIPPET]}..."
    return f"{key}={text}"


def carries_content(key: str, value: Any) -> bool:
    """Is this worth printing on the way in?

    The incoming state repeats the same identity fields at every node, so printing all of it nineteen
    times buries the few values that actually moved. Empty collections, blank strings and a zero cost
    are dropped: a node that was handed no findings is better shown by their absence from the line than
    by "findings=0" among eight other zeros. ``MLR_TRACE=full`` prints everything.
    """
    if key in ("run_id", "status"):
        return True
    if value is None or value == "" or value == 0:
        return False
    if isinstance(value, (list, tuple, set, dict)) and not value:
        return False
    # A zero-cost RunCost is the default, not information.
    return not (key == "cost" and not getattr(value, "model_calls", 0))


def _lines(state: dict[str, Any], *, full: bool, keys: list[str] | None = None) -> list[str]:
    """Render a state (or a state update) as lines, identity fields first."""
    wanted = keys if keys is not None else list(state)
    if keys is None and not full:
        wanted = [k for k in wanted if carries_content(k, state.get(k))]
    ordered = [k for k in _IDENTITY if k in wanted] + [
        k for k in sorted(wanted) if k not in _IDENTITY
    ]
    return [describe(key, state.get(key), full=full) for key in ordered]


def _wrap(items: list[str], *, width: int = 3) -> list[str]:
    """Group short descriptions onto shared lines so a trace stays scannable."""
    rows, row = [], []
    for item in items:
        row.append(item)
        if len(row) >= width or len(" ".join(row)) > 100:
            rows.append("  ".join(row))
            row = []
    if row:
        rows.append("  ".join(row))
    return rows


def traced(name: str, fn: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap a node so its input state and its returned update are printed.

    The *update* is printed, not the whole state after the node. A node returns only the keys it
    changed, and that is the interesting part -- printing the full state twice per node makes the diff
    something you have to find by eye.
    """

    def node(state: Any, *args: Any, **kwargs: Any) -> Any:
        trace_mode = mode()
        if not trace_mode:
            return fn(state, *args, **kwargs)

        full = trace_mode == "full"
        incoming = dict(state) if isinstance(state, dict) else {}
        started = time.perf_counter()

        error: BaseException | None = None
        update: Any = None
        try:
            update = fn(state, *args, **kwargs)
            return update
        except BaseException as exc:  # noqa: BLE001 - re-raised below; the trace must show the failure
            error = exc
            raise
        finally:
            elapsed = (time.perf_counter() - started) * 1000
            block = [
                _colour(f"┌─ {name} ", BOLD + CYAN) + _colour("─" * max(2, 72 - len(name)), DIM),
            ]
            for row in _wrap(_lines(incoming, full=full)):
                block.append(_colour("│ IN   ", DIM) + row)

            if error is not None:
                block.append(
                    _colour("│ RAISED ", RED) + f"{type(error).__name__}: {str(error)[:120]}"
                )
            elif isinstance(update, dict) and update:
                for row in _wrap(_lines(update, full=full, keys=list(update))):
                    block.append(_colour("│ OUT  ", GREEN) + row)
            else:
                block.append(_colour("│ OUT  ", AMBER) + "(no state change)")

            block.append(_colour(f"└─ {elapsed:.0f}ms", DIM))
            with _lock:
                print("\n".join(block), flush=True)

    node.__name__ = getattr(fn, "__name__", name)
    node.__doc__ = getattr(fn, "__doc__", None)
    return node


def banner() -> None:
    """Printed once when a traced run starts, so the output is self-explaining."""
    if not enabled():
        return
    with _lock:
        print(
            _colour(
                f"\nMLR_TRACE={mode()}: printing each node's input state and the update it returns. "
                f"Use MLR_TRACE=full for whole values.\n",
                DIM,
            ),
            flush=True,
        )
