"""Rationale trace.

T = {(k, j, r_{kj}^{(r)})}   (Sec. 4.6 of methodology.tex)

The auditable output that closes the interpretability gap left open by
purely metric-driven soft-clustering approaches: every boundary in the
final decomposition is accompanied by a human-readable justification.
"""

import json
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple


@dataclass
class RationaleTrace:
    entries: List[Dict] = field(default_factory=list)

    def add_round(
        self,
        round_idx: int,
        rationale: Dict[Tuple[str, str], str],
        assignment: Dict[str, Set[str]],
    ) -> None:
        for cls, services in assignment.items():
            for svc in services:
                self.entries.append(
                    {
                        "round": round_idx,
                        "class": cls,
                        "service": svc,
                        "rationale": rationale.get((cls, svc), ""),
                    }
                )

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.entries, f, indent=2)

    def to_markdown(self) -> str:
        lines = ["# VeriArch decomposition rationale trace", ""]
        for e in self.entries:
            lines.append(
                f"- **Round {e['round']}** - `{e['class']}` -> `{e['service']}`: "
                f"{e['rationale']}"
            )
        return "\n".join(lines)
