"""Capability descriptor generation.

d_k = LLM_summ(t_k, w_k, N_str(C_k))   (Eq. 5 of methodology.tex)

Produces the natural-language vocabulary D = {d_k} that the architect
agent reasons over, grounding it in business capability rather than raw
tokens. Uses whatever LLMClient it's given (see llm.py), so it works
identically with Anthropic, a local Ollama model, or any
OpenAI-compatible endpoint.
"""

from typing import Dict, List

from .llm import LLMClient


def generate_capability_descriptors(
    classes: Dict[str, "object"],
    names: List[str],
    neighborhoods: Dict[str, List[str]],
    llm_client: LLMClient,
) -> Dict[str, str]:
    descriptors: Dict[str, str] = {}

    for name in names:
        info = classes[name]
        neighbor_names = neighborhoods.get(name, [])
        prompt = _build_descriptor_prompt(name, info, neighbor_names)
        text = llm_client.generate(prompt, max_tokens=80)
        descriptors[name] = _clean(text)

    return descriptors


def _build_descriptor_prompt(name: str, info, neighbor_names: List[str]) -> str:
    tokens_preview = " ".join(info.tokens[:200])
    comments_preview = " ".join(info.comments[:20])
    neighbors = ", ".join(neighbor_names[:8]) if neighbor_names else "none identified"
    return (
        "You are summarizing a Java class for microservice-decomposition "
        "reasoning. In one sentence (at most 25 words), describe the "
        "business capability this class most plausibly implements. "
        "Describe intent, not syntax. Respond with only the one-sentence "
        "description, no preamble, no quotation marks.\n\n"
        f"Class name: {name}\n"
        f"Structural neighbors (classes it depends on): {neighbors}\n"
        f"Code tokens (truncated): {tokens_preview}\n"
        f"Comments (truncated): {comments_preview}\n"
    )


def _clean(text: str) -> str:
    """Local models are more prone than Claude to add a leading
    'Sure, here is...' or wrap the answer in quotes; strip that off."""
    text = text.strip().strip('"').strip()
    for prefix in ("Sure, ", "Here is ", "Here's ", "Description: "):
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):].strip()
    return text
