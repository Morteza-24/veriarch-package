"""Global configuration for VeriArch.

Every field here corresponds to a symbol in methodology.tex:
    alpha  -- structural/semantic fusion weight for M         (Eq. 1)
    delta  -- structural-neighborhood threshold for N_str(C_k) (Eq. 2)
    beta   -- prior-agreement weight inside VSCORE             (Eq. 7)
    gamma  -- per-class flagging threshold for weak assignments (Eq. 8)
    tau    -- VSCORE acceptance / convergence threshold
    max_rounds -- R_max, the iteration budget for the architect-critic loop

llm_provider selects which LLMClient (see llm.py) the architect agent
and descriptor generator use: 'anthropic', 'ollama' (free, local), or
'openai_compatible' (any self-hosted OpenAI-chat-style server, e.g. a
locally served Qwen model via vLLM or LM Studio).
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class VeriArchConfig:
    num_services: int = 6
    alpha: float = 0.7
    beta: float = 0.5
    gamma: float = 0.2
    tau: float = 0.75
    delta: float = 0.15
    max_rounds: int = 5
    embedding_dim: int = 256

    llm_provider: str = "anthropic"        # 'anthropic' | 'ollama' | 'openai_compatible'
    llm_model: str = "claude-sonnet-4-6"   # e.g. 'qwen2.5:7b' for ollama
    llm_base_url: Optional[str] = None     # defaults per-provider if unset
    llm_api_key: Optional[str] = None      # unused for ollama
