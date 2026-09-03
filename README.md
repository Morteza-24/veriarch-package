# VeriArch

Reference implementation of **VeriArch**: agentic microservice extraction
with verifiable, critic-grounded reasoning, as described in
`methodology.tex`. An LLM architect agent proposes a microservice
decomposition with natural-language rationale; a structural-semantic
critic scores the proposal against Mo2oM-style cohesion, coupling, and
prior-agreement signals; the two iterate until convergence, producing a
final decomposition plus an auditable rationale trace.

## Module map

| Module | Methodology section / equation |
|---|---|
| `parsing/java_parser.py` | Parsing stage (shared with Mo2oM) |
| `structural.py` | S^str and N_str, Eq. 2 |
| `semantic.py` | e-hat_k, Sec. 4.1 |
| `fusion.py` | NOCD branches + reference prior M, Eq. 1 |
| `descriptors.py` | Capability descriptors d_k, Eq. 5 |
| `architect.py` | Architect agent A, Eq. 6 |
| `critic.py` | Critic V: VSCORE, weak-assignment flags, Eq. 7-9 |
| `loop.py` | Algorithm 1: iterative refinement |
| `constraints.py` | Human-constrained revision, Sec. 4.5 |
| `trace.py` | Rationale trace T, Sec. 4.6 |
| `cli.py` | End-to-end orchestration / entry point |

## Install

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-...
```

`semantic.py` falls back to a TF-IDF + SVD embedding automatically if
`torch`/`transformers` or the UniXcoder weights are unavailable, so the
structural/fusion parts of the pipeline remain runnable without a GPU.

## Run on a real repository

```bash
python -m veriarch.cli \
  --repo /path/to/monolith \
  --num-services 6 \
  --constraint "Payment and Order must not share a service" \
  --output-dir out/
```

This writes `out/assignment.json` (the final class -> service map) and
`out/rationale_trace.md` / `.json` (the full auditable trace of every
proposal and revision).

## Test without any API key or heavy dependencies

```bash
python tests/test_loop_synthetic.py
```

This runs the Critic + `run_veriarch` loop against a small synthetic
class graph with a stubbed architect agent (no LLM calls, no torch),
checking that: a deliberately bad round-1 placement gets flagged by the
critic, the critique drives a round-2 revision, and the revised
assignment correctly reproduces an overlapping, cross-cutting class
placement consistent with the reference prior M.

## Notes / limitations

- `fusion.py` reimplements a lightweight NOCD-style GNN (Bernoulli-Poisson
  objective) rather than importing Mo2oM's exact training code, since
  that code isn't available in this environment; hyperparameters
  (`hidden_dim`, `epochs`, k-NN `k`) are conservative defaults and should
  be tuned per corpus size.
- `critic.py`'s cohesion/coupling terms average over all class pairs
  within/between services; for very large services this is O(n^2) per
  round and would benefit from sampling for large monoliths.
- `constraints.py`'s post-hoc validator only recognizes the
  `"X and Y must not share a service"` pattern textually; constraint
  enforcement is otherwise delegated to the architect agent's prompt.
- This is a research prototype, not a production build: there is no
  retry/backoff around the Anthropic API calls, and `ArchitectAgent`
  assumes the model returns well-formed JSON (a parse failure will
  raise rather than being retried).
