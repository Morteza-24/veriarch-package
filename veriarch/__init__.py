"""VeriArch: agentic microservice extraction with verifiable, critic-grounded
reasoning.

This package implements the methodology described in the VeriArch
methodology note (methodology.tex): an LLM architect agent proposes a
microservice decomposition with natural-language rationale, a
structural-semantic critic scores the proposal against Mo2oM-style
cohesion/coupling/prior-agreement signals, and the two iterate until
convergence or a round limit, producing a final decomposition plus an
auditable rationale trace.
"""

from .config import VeriArchConfig  # noqa: F401
