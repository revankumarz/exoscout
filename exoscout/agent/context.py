"""Shared agent context.

Tool outputs can be large (light-curve arrays are thousands of points). We keep
the *full* results here for the UI, but hand the LLM only compact summaries -
so the model reasons over evidence without drowning in raw data.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from exoscout.provenance import Provenance
from exoscout.target import Target


@dataclass
class AgentContext:
    target: Target
    max_period: float = 15.0
    max_sectors: int | None = None
    full: dict = field(default_factory=dict)          # tool_name -> full result dict
    prov: Provenance = field(default_factory=Provenance)
    trace: list = field(default_factory=list)         # readable step-by-step record

    def store(self, tool: str, full_result: dict) -> None:
        self.full[tool] = full_result

    def log_step(self, kind: str, text: str, tool: str | None = None, data: dict | None = None) -> None:
        self.trace.append({"kind": kind, "tool": tool, "text": text, "data": data or {}})
