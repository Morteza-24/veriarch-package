"""Synthetic smoke test for the architect-critic loop.

Exercises Critic and run_veriarch end-to-end on a small synthetic
class graph, with a stubbed ArchitectAgent that mimics an LLM
converging over a couple of rounds. Requires only numpy: no
Anthropic API key, torch, or transformers needed, so this can run in
any environment as a fast correctness check on the loop, VSCORE, and
rationale trace logic before wiring up real dependencies.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from veriarch.architect import ArchitectOutput  # noqa: E402
from veriarch.critic import Critic  # noqa: E402
from veriarch.loop import run_veriarch  # noqa: E402


class StubArchitect:
    """Mimics an LLM architect agent: starts with one bad assignment,
    then "revises" in round 2 after seeing the critique, exactly like a
    real Claude call would after reading the critic's flagged classes."""

    def __init__(self):
        self.calls = 0

    def propose(self, descriptors, critique, constraints):
        self.calls += 1
        if self.calls == 1:
            # Round 1: misplace "PaymentGateway" far from its structural
            # neighbors on purpose, to trigger a weak-assignment flag.
            assignment = {
                "OrderController": {"orders"},
                "OrderRepository": {"orders"},
                "PaymentGateway": {"inventory"},  # bad: unrelated service
                "InventoryService": {"inventory"},
                "InventoryRepository": {"inventory"},
            }
        else:
            # Round 2: architect "reads" the critique and fixes it.
            assignment = {
                "OrderController": {"orders"},
                "OrderRepository": {"orders"},
                "PaymentGateway": {"orders", "inventory"},  # cross-cutting, overlap
                "InventoryService": {"inventory"},
                "InventoryRepository": {"inventory"},
            }

        rationale = {
            (cls, svc): f"Round {self.calls} placement of {cls} in {svc}."
            for cls, services in assignment.items()
            for svc in services
        }
        return ArchitectOutput(assignment=assignment, rationale=rationale)


def build_synthetic_graph():
    names = [
        "OrderController",
        "OrderRepository",
        "PaymentGateway",
        "InventoryService",
        "InventoryRepository",
    ]
    idx = {n: i for i, n in enumerate(names)}
    y = len(names)
    s_str = np.zeros((y, y))

    def link(a, b, w=1.0):
        s_str[idx[a], idx[b]] = w
        s_str[idx[b], idx[a]] = w

    link("OrderController", "OrderRepository", 0.9)
    link("OrderController", "PaymentGateway", 0.8)
    link("InventoryService", "InventoryRepository", 0.9)
    link("PaymentGateway", "InventoryService", 0.6)

    # Semantic embeddings: two tight clusters plus PaymentGateway sitting
    # between them (mirrors the "overlapping cross-cutting class" motif
    # from the Mo2oM paper this method builds on).
    rng = np.random.default_rng(0)
    base_orders = rng.normal(0, 0.05, size=8)
    base_inventory = rng.normal(2, 0.05, size=8)
    embeddings = np.stack(
        [
            base_orders,  # OrderController
            base_orders + rng.normal(0, 0.02, size=8),  # OrderRepository
            (base_orders + base_inventory) / 2,  # PaymentGateway (cross-cutting)
            base_inventory,  # InventoryService
            base_inventory + rng.normal(0, 0.02, size=8),  # InventoryRepository
        ]
    )
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms

    # Reference prior M: what Mo2oM's NOCD fusion would have produced --
    # PaymentGateway genuinely overlapping both services.
    service_names = ["orders", "inventory"]
    prior_m = np.array(
        [
            [0.9, 0.05],  # OrderController
            [0.85, 0.05],  # OrderRepository
            [0.55, 0.55],  # PaymentGateway: overlap
            [0.05, 0.9],  # InventoryService
            [0.05, 0.85],  # InventoryRepository
        ]
    )
    return names, s_str, embeddings, prior_m, service_names


def main():
    names, s_str, embeddings, prior_m, service_names = build_synthetic_graph()
    # gamma=0.75: a single-service assignment for a class the prior M
    # considers a 50/50 overlap only reaches ~0.71 cosine agreement, so
    # it gets flagged; a genuine overlapping assignment reaches 1.0.
    critic = Critic(names, s_str, embeddings, prior_m, service_names, beta=0.5, gamma=0.75)
    architect = StubArchitect()

    descriptors = {n: f"stub descriptor for {n}" for n in names}
    # tau set unreachably high on purpose: this isolates the "no weak
    # assignments remain" convergence path (Algorithm 1's second stopping
    # condition) so the test exercises the revision behavior itself,
    # independent of the exact VSCORE magnitude.
    result = run_veriarch(
        descriptors, architect, critic, constraints=[], tau=10.0, max_rounds=5
    )

    print(f"converged={result.converged} rounds_run={result.rounds_run} "
          f"vscore={result.final_vscore:.3f}")
    print("final assignment:")
    for cls, services in sorted(result.assignment.items()):
        print(f"  {cls}: {sorted(services)}")

    assert result.rounds_run == 2, "expected the stub architect to need exactly 2 rounds"
    assert result.converged, "expected the loop to converge after the architect's revision"
    assert result.assignment["PaymentGateway"] == {"orders", "inventory"}, (
        "expected PaymentGateway to end up overlapping both services, "
        "matching Mo2oM's overlapping-class motivation"
    )
    print("\nOK: synthetic architect-critic loop test passed.")


if __name__ == "__main__":
    main()
