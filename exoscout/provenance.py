"""A tiny provenance / audit log.

Every tool call appends a record here so the eventual agent (and the human
reading the brief) can trace exactly which tool produced which claim. This is
the seed of the "every numeric claim traced to a tool output" evaluation goal.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class ProvenanceEntry:
    tool: str
    summary: str
    source: str
    ok: bool
    detail: dict = field(default_factory=dict)
    ts: float = field(default_factory=time.time)


class Provenance:
    def __init__(self) -> None:
        self.entries: list[ProvenanceEntry] = []

    def record(self, tool: str, summary: str, source: str, ok: bool = True, **detail) -> None:
        self.entries.append(
            ProvenanceEntry(tool=tool, summary=summary, source=source, ok=ok, detail=detail)
        )

    def as_rows(self) -> list[dict]:
        return [
            {
                "tool": e.tool,
                "ok": "OK" if e.ok else "FAIL",
                "summary": e.summary,
                "source": e.source,
            }
            for e in self.entries
        ]
