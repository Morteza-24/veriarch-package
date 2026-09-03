"""Human-constrained revision (Sec. 4.5 of methodology.tex).

Constraints H are enforced primarily through the architect agent's
prompt (they are inlined as a hard filter it must satisfy). This module
provides a best-effort post-hoc validator for the most common pattern,
so the orchestration loop can detect and report violations even if the
architect agent fails to respect a constraint.
"""

from typing import Dict, List, Set


def validate_constraints(
    assignment: Dict[str, Set[str]], constraints: List[str]
) -> List[str]:
    violations = []
    for c in constraints:
        if "must not share a service" in c:
            names = [n.strip() for n in c.replace("must not share a service", "").split(" and ")]
            if len(names) == 2:
                a, b = names
                a_services = assignment.get(a, set())
                b_services = assignment.get(b, set())
                if a_services & b_services:
                    violations.append(c)
    return violations
