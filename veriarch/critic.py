"""The structural-semantic critic V.

Scores a candidate proposal P^{(r)} against S^str, E-hat, and the Mo2oM
reference prior M (Sec. 4.4 of methodology.tex): cohesion/coupling
(Eq. 5-6 in that section's local numbering), prior agreement (Eq. 7),
the composite VSCORE (Eq. 8), and the weak-assignment flags W (Eq. 9)
that seed the critique returned to the architect agent.
"""

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import numpy as np


@dataclass
class CriticResult:
    vscore: float
    weak_assignments: List[Tuple[str, str]]
    critique_text: str


class Critic:
    def __init__(
        self,
        names: List[str],
        s_str: np.ndarray,
        embeddings: np.ndarray,
        prior_m: np.ndarray,
        service_names: List[str],
        beta: float = 0.5,
        gamma: float = 0.2,
    ):
        self.names = names
        self.idx = {n: i for i, n in enumerate(names)}
        self.s_str = s_str
        self.embeddings = embeddings
        self.prior_m = prior_m
        self.service_names = service_names
        self.service_idx = {s: j for j, s in enumerate(service_names)}
        self.beta = beta
        self.gamma = gamma

    def score(self, assignment: Dict[str, Set[str]]) -> CriticResult:
        members = self._service_members(assignment)
        coh_str, coh_sem, coup = self._cohesion_coupling(members)
        agree = self._prior_agreement(assignment)

        cohesion = 0.5 * coh_str + 0.5 * coh_sem
        mean_agree = float(np.mean(list(agree.values()))) if agree else 0.0
        vscore = cohesion - coup + self.beta * mean_agree

        weak = self._flag_weak(assignment, agree)
        critique = self._build_critique(weak, agree)
        return CriticResult(vscore=float(vscore), weak_assignments=weak, critique_text=critique)

    def _service_members(self, assignment: Dict[str, Set[str]]) -> Dict[str, List[int]]:
        members: Dict[str, List[int]] = {s: [] for s in self.service_names}
        for cls, services in assignment.items():
            if cls not in self.idx:
                continue
            for svc in services:
                members.setdefault(svc, []).append(self.idx[cls])
        return members

    def _cohesion_coupling(
        self, members: Dict[str, List[int]]
    ) -> Tuple[float, float, float]:
        coh_str_vals, coh_sem_vals = [], []
        for idxs in members.values():
            if len(idxs) < 2:
                continue
            pairs = [(a, b) for i, a in enumerate(idxs) for b in idxs[i + 1:]]
            if not pairs:
                continue
            coh_str_vals.append(np.mean([self.s_str[i, j] for i, j in pairs]))
            coh_sem_vals.append(
                np.mean([float(self.embeddings[i] @ self.embeddings[j]) for i, j in pairs])
            )

        coh_str = float(np.mean(coh_str_vals)) if coh_str_vals else 0.0
        coh_sem = float(np.mean(coh_sem_vals)) if coh_sem_vals else 0.0

        coup_vals = []
        service_list = list(members.items())
        for a in range(len(service_list)):
            _, idx_a = service_list[a]
            for b in range(a + 1, len(service_list)):
                _, idx_b = service_list[b]
                if not idx_a or not idx_b:
                    continue
                pair_vals = [self.s_str[i, j] for i in idx_a for j in idx_b]
                if pair_vals:
                    coup_vals.append(np.mean(pair_vals))
        coup = float(np.mean(coup_vals)) if coup_vals else 0.0

        return coh_str, coh_sem, coup

    def _prior_agreement(self, assignment: Dict[str, Set[str]]) -> Dict[str, float]:
        agree: Dict[str, float] = {}
        num_services = len(self.service_names)
        for cls, services in assignment.items():
            if cls not in self.idx:
                continue
            i = self.idx[cls]
            p_row = np.zeros(num_services)
            for svc in services:
                if svc in self.service_idx:
                    p_row[self.service_idx[svc]] = 1.0
            m_row = self.prior_m[i]
            denom = np.linalg.norm(p_row) * np.linalg.norm(m_row)
            agree[cls] = float(p_row @ m_row / denom) if denom > 0 else 0.0
        return agree

    def _flag_weak(
        self, assignment: Dict[str, Set[str]], agree: Dict[str, float]
    ) -> List[Tuple[str, str]]:
        weak = []
        for cls, services in assignment.items():
            if agree.get(cls, 1.0) < self.gamma:
                for svc in services:
                    weak.append((cls, svc))
        return weak

    def _build_critique(
        self, weak: List[Tuple[str, str]], agree: Dict[str, float]
    ) -> str:
        if not weak:
            return ""
        lines = []
        for cls, svc in weak:
            i = self.idx[cls]
            row = self.s_str[i]
            nn_name = self.names[int(np.argmax(row))] if row.max() > 0 else "none"
            lines.append(
                f"- {cls} is assigned to {svc}, but its structural/semantic "
                f"evidence (agreement={agree.get(cls, 0.0):.2f}) disagrees "
                f"with this placement; its closest structural neighbor is "
                f"{nn_name}. Reconsider this assignment or justify the "
                f"divergence explicitly."
            )
        return "\n".join(lines)
