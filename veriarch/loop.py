"""Iterative agentic decomposition (Algorithm 1 of methodology.tex).

Alternates the architect agent and the critic until VSCORE clears the
acceptance threshold tau, no weak assignments remain, or the round
budget R_max is exhausted -- in which case the highest-scoring proposal
observed is returned instead.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Set

from .architect import ArchitectAgent, ArchitectOutput
from .critic import Critic, CriticResult
from .trace import RationaleTrace

OnRoundFn = Callable[[int, ArchitectOutput, CriticResult], None]


@dataclass
class VeriArchResult:
    assignment: Dict[str, Set[str]]
    trace: RationaleTrace
    final_vscore: float
    converged: bool
    rounds_run: int


def run_veriarch(
    descriptors: Dict[str, str],
    architect: ArchitectAgent,
    critic: Critic,
    constraints: List[str],
    tau: float = 0.75,
    max_rounds: int = 5,
    on_round: Optional[OnRoundFn] = None,
) -> VeriArchResult:
    """Runs Algorithm 1. If on_round is given, it's called after every
    round with (round_index, architect_output, critic_result) -- this is
    how callers such as server.py stream live progress to a UI without
    waiting for the whole loop to finish."""
    trace = RationaleTrace()
    critique = ""
    best = None

    for r in range(max_rounds):
        output = architect.propose(descriptors, critique, constraints)
        if output is None:
            continue
        result = critic.score(output.assignment)
        trace.add_round(r, output.rationale, output.assignment)

        if on_round is not None:
            on_round(r, output, result)

        if best is None or result.vscore > best[1].vscore:
            best = (output, result)

        if result.vscore >= tau or not result.weak_assignments:
            return VeriArchResult(
                assignment=output.assignment,
                trace=trace,
                final_vscore=result.vscore,
                converged=True,
                rounds_run=r + 1,
            )
        critique = result.critique_text

    best_output, best_result = best
    return VeriArchResult(
        assignment=best_output.assignment,
        trace=trace,
        final_vscore=best_result.vscore,
        converged=False,
        rounds_run=max_rounds,
    )
