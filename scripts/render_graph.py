"""Render the compiled graph to ``docs/graph.mmd``.

Generated from the compiled graph rather than drawn by hand, so the diagram in the README and on the
UI home page cannot drift from the code it documents. Run with ``make graph``.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TARGET = REPO_ROOT / "docs" / "graph.mmd"


def main() -> int:
    from mlr.graph import checkpointer, compile_with
    from platform_core.runtime import render_mermaid

    with checkpointer() as saver:
        mermaid = render_mermaid(compile_with(saver))

    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(mermaid, encoding="utf-8")
    print(f"Wrote {TARGET.relative_to(REPO_ROOT)} ({len(mermaid.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
