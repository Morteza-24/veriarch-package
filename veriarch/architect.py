"""The architect agent A.

(P^{(r)}, R^{(r)}) = A(D, F^{(r-1)}, H)   (Eq. 6 of methodology.tex)

Reasons over capability descriptors using domain-driven-design
principles (bounded contexts, cross-cutting concerns), respecting any
hard human constraints H, and incorporating the critic's previous
critique F^{(r-1)} when revising.

Talks to whatever LLMClient it's given (see llm.py) -- Anthropic, a
local Ollama model, or any OpenAI-compatible endpoint -- so the
reasoning logic here is provider-agnostic.
"""

import json
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

from .llm import LLMClient


@dataclass
class ArchitectOutput:
    assignment: Dict[str, Set[str]]
    rationale: Dict[Tuple[str, str], str]


class ArchitectAgent:
    def __init__(self, service_names: List[str], llm_client: LLMClient):
        self.service_names = service_names
        self.llm = llm_client

    def propose(
        self, descriptors: Dict[str, str], critique: str, constraints: List[str]
    ) -> ArchitectOutput:
        prompt = self._build_prompt(descriptors, critique, constraints)
        text = self.llm.generate(prompt, max_tokens=4096)
        try:
            response = self._parse_response(text)
            return response
        except:
            print("unable to parse llm response.")
            print("-"*10)
            print(text)
            print("-"*10)

    def _build_prompt(
        self, descriptors: Dict[str, str], critique: str, constraints: List[str]
    ) -> str:
        services = ", ".join(self.service_names)
        desc_lines = "\n".join(f"- {name}: {desc}" for name, desc in descriptors.items())
        constraint_lines = (
            "\n".join(f"- {c}" for c in constraints) if constraints else "(none)"
        )
        critique_block = critique if critique else "(first proposal, no prior critique)"

        return f"""You are a software architect decomposing a monolithic
system into microservices, following domain-driven design: group classes
by business capability and bounded context, and assign a class to more
than one service only when it is a genuine cross-cutting concern.

Target services: {services}

Classes and their capability descriptors:
{desc_lines}

Hard constraints you MUST satisfy:
{constraint_lines}

Critique from the previous round (address every flagged item):
{critique_block}

Respond with ONLY a JSON object of this exact shape. Do not include any
explanation, preamble, or markdown code fences -- output must start with
{{ and end with }}, nothing else:
{{
  "assignments": [
    {{"class": "<class name>", "services": ["<service name>", "..."],
      "rationale": {{"<service name>": "<one sentence justification>"}}}}
  ]
}}
"""

    def _parse_response(self, text: str) -> ArchitectOutput:
        data = self._extract_json(text)
        assignment: Dict[str, Set[str]] = {}
        rationale: Dict[Tuple[str, str], str] = {}

        for entry in data.get("assignments", []):
            cls = entry["class"]
            services = set(entry.get("services", []))
            assignment[cls] = services
            for svc, why in entry.get("rationale", {}).items():
                rationale[(cls, svc)] = why

        return ArchitectOutput(assignment=assignment, rationale=rationale)

    @staticmethod
    def _extract_json(text: str) -> dict:
        """Locates and parses the JSON object in the model's response.

        Anthropic models reliably follow "respond with only JSON"; local
        models served via Ollama/vLLM more often wrap the JSON in prose
        or code fences despite being asked not to, so this takes the
        substring between the first '{' and the matching last '}' rather
        than assuming the whole response is clean JSON.
        """
        text = text.strip()
        first = text.find("{")
        last = text.rfind("}")
        if first == -1 or last == -1 or last < first:
            raise ValueError(f"No JSON object found in architect response:\n{text}")
        candidate = text[first : last + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Failed to parse JSON from architect response:\n{candidate}"
            ) from exc
