# VeriArch — Usage Guide

This walks through setting up, running, and interpreting VeriArch end to
end, plus how to tune it and what to do when something breaks. It
assumes you have the `veriarch.zip` package unzipped locally.

---

## 1. Setup

```bash
unzip veriarch.zip
cd veriarch
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
```

Notes:
- `torch` + `transformers` are only needed for the real UniXcoder
  embeddings. If you skip installing them (or the model download fails
  in a sandboxed environment), `semantic.py` automatically falls back to
  a TF-IDF + SVD embedding — the rest of the pipeline runs unchanged.
- `ANTHROPIC_API_KEY` is required for `descriptors.py` and
  `architect.py` (both call the Claude API directly). Nothing else in
  the pipeline needs it.

---

## 2. Sanity-check the loop before touching a real repo

Run this first — it exercises the Critic + `run_veriarch` loop on a
tiny synthetic example with a stubbed architect (no API key, no torch
required), so you can confirm your Python environment is basically
sound before spending API calls or GPU time:

```bash
python tests/test_loop_synthetic.py
```

Expected output ends with:
```
OK: synthetic architect-critic loop test passed.
```

If this fails, the problem is in your environment/imports, not your
target repository — fix this first.

---

## 3. Run on a real Java monolith

```bash
python -m veriarch.cli \
  --repo /path/to/monolith \
  --num-services 6 \
  --constraint "Payment and Order must not share a service" \
  --output-dir out/
```

Flags:
- `--repo` — path to the root of the Java monolith (it's walked
  recursively for `.java` files).
- `--num-services` — target number `N` of microservices; this is a
  design choice you set, not something inferred automatically.
- `--constraint` — repeatable; each one is passed to the architect
  agent as a hard requirement in its prompt. The only pattern the
  post-hoc validator (`constraints.py`) recognizes automatically is:
  `"<ClassA> and <ClassB> must not share a service"`. Other phrasings
  still reach the architect's prompt and usually work, but won't be
  double-checked by the validator.
- `--output-dir` — where results are written (default `veriarch_output/`).

The console prints progress through 6 stages:
```
[1/6] Parsing repository...
[2/6] Computing semantic embeddings...
[3/6] Fusing structural and semantic priors (NOCD)...
[4/6] Generating capability descriptors...
[5/6] Running the architect-critic loop...
[6/6] Converged=True after 3 rounds, VSCORE=0.812
```

`[4/6]` and `[5/6]` are the stages that call the Claude API — expect
these to take the longest and to be where API errors would surface.

---

## 4. Reading the output

Two files land in `--output-dir`:

**`assignment.json`** — the final decomposition:
```json
{
  "OrderController": ["orders"],
  "PaymentGateway": ["orders", "inventory"],
  "InventoryService": ["inventory"]
}
```
A class listed under more than one service is an intentional overlap —
the architect judged it a genuine cross-cutting concern, and the critic
didn't flag it as unsupported by the structural/semantic evidence.

**`rationale_trace.md`** (and the equivalent `.json`) — the full
auditable trace, one line per class-service assignment per round:
```
- **Round 0** - `PaymentGateway` -> `inventory`: Handles inventory
  reservation during checkout.
- **Round 1** - `PaymentGateway` -> `orders`: Also owns payment capture
  tied to order lifecycle; retained inventory link as cross-cutting.
```
This is the artifact meant for a human architect to actually read and
sanity-check before adopting the decomposition — it's the whole point
of the method, so it's worth skimming even when `Converged=True`.

If `Converged=False` in the console output, the run hit `max_rounds`
without clearing the acceptance threshold; `assignment.json` still
contains the **best**-scoring proposal seen across all rounds, not the
last one, and the trace will show every round that was tried, including
the not-fully-resolved final one.

---

## 5. Tuning

All hyperparameters live in `veriarch/config.py`
(`VeriArchConfig`). The CLI only exposes `--num-services`; to change the
others, edit the dataclass defaults or construct `VeriArchConfig(...)`
yourself if you're scripting against the library instead of the CLI.

| Parameter | Meaning | Effect of raising it |
|---|---|---|
| `alpha` | weight on semantic vs. structural prior in `M` (Eq. 1) | more weight on semantic similarity, less on call-graph coupling |
| `beta` | weight on prior-agreement inside VSCORE | critic anchors the architect more tightly to `M`, tolerates less capability-driven divergence |
| `gamma` | per-class flagging threshold | more assignments get flagged as "weak" → more revision rounds |
| `tau` | VSCORE acceptance threshold | harder to converge, more rounds used, better-scoring final result (up to `max_rounds`) |
| `delta` | structural-neighborhood threshold for descriptor prompts | larger neighborhoods included in each class's capability descriptor context |
| `max_rounds` | `R_max` | more chances to converge, more API calls per run |

If you find every run hits `max_rounds` without converging, `tau` is
probably set too high for your corpus, or `gamma` too low (over-flagging
nearly everything). Start by loosening `gamma` before touching `tau`.

---

## 6. Using it as a library instead of the CLI

If you want to inspect intermediate artifacts (e.g. `S^str`, `M`, or the
per-round `CriticResult`) rather than just the final files, call the
same functions `cli.py` does, directly:

```python
from veriarch.parsing.java_parser import parse_repository
from veriarch.structural import build_structural_matrix
from veriarch.semantic import compute_semantic_embeddings
from veriarch.fusion import build_adjacency_from_similarity, train_nocd, fuse_membership
from veriarch.descriptors import generate_capability_descriptors
from veriarch.architect import ArchitectAgent
from veriarch.critic import Critic
from veriarch.loop import run_veriarch

classes = parse_repository("/path/to/monolith")
names, s_str = build_structural_matrix(classes)
embeddings = compute_semantic_embeddings(classes, names)
# ... etc., matching cli.py's stage order
```

This is the easiest way to, for example, dump `M` to inspect which
classes Mo2oM's fusion already considered overlapping before the
architect agent even runs.

---

## 7. Common issues

- **`json.JSONDecodeError` inside `architect.py`** — the model didn't
  return valid JSON. This implementation doesn't retry automatically;
  re-run, or add retry/backoff around `ArchitectAgent.propose` if this
  happens often (see the limitations note in `README.md`).
- **Everything falls back to TF-IDF embeddings** — check that `torch`
  and `transformers` installed correctly and that
  `microsoft/unixcoder-base` is reachable from your network; the
  fallback fires silently on *any* exception in `_unixcoder_embeddings`,
  so a network block, an OOM, or a missing package all look the same
  from the outside.
- **Run is very slow** — `train_nocd` runs 200 GCN epochs per branch
  (structural + semantic) on the full adjacency matrix; for large
  monoliths (>1000 classes) this is the likely bottleneck, not the LLM
  calls. Lower `epochs` or `hidden_dim` in `fusion.py` if needed.
- **No `.java` files found** — `cli.py` raises immediately with the repo
  path it looked under; double check you pointed `--repo` at the source
  root, not a build/target directory.

---

## 8. Real, live-running GUI (not a mock)

`server.py` + `frontend/index.html` is a real local web app: a FastAPI
backend that actually runs the pipeline in a background thread per
request, and a single-file vanilla-JS frontend that polls it for live
progress. No build step, no separate frontend process — one command
serves both.

```bash
pip install -r requirements.txt   # now includes fastapi, uvicorn, pydantic
python -m uvicorn veriarch.server:app --reload --port 8765
```

Then open **http://localhost:8765** in a browser.

What you get:
- A **repository field** with live validation (counts `.java` files under
  the path via `/api/validate_repo` as you tab away from the field).
- **LLM backend selection** (Ollama / Anthropic / any OpenAI-compatible
  server) with per-provider defaults, matching `llm.py`'s `get_llm_client`.
- **Hard constraints** you can add/remove before running.
- **Advanced hyperparameters** (`alpha`, `beta`, `gamma`, `tau`, `delta`,
  `max_rounds`) in a collapsible panel, matching `config.py`.
- A **live stage tracker**, a **VSCORE-per-round chart**, and a
  **rationale trace panel** that fills in round by round as the real
  pipeline runs — not simulated. Weak/flagged assignments are
  highlighted in amber; the critic's critique text for the next round is
  shown inline.
- A **decomposition view** grouped by service, with overlapping /
  cross-cutting classes visually marked.
- A **download button** for `assignment.json` once the run completes.

### How it works

- `POST /api/runs` starts a run: it launches `pipeline.run_pipeline(...)`
  in a background `threading.Thread` and immediately returns a `run_id`.
- The frontend polls `GET /api/runs/{run_id}` roughly once a second.
  Each poll returns the full current state (stage, classes parsed,
  every round completed so far, and the final assignment + trace once
  `status` is `"done"`).
- `pipeline.py` is the same code path used by `cli.py` — both call
  `run_pipeline()` and differ only in how they consume the `progress`
  callback (the CLI prints it, the server turns it into pollable state).
  `loop.py`'s `run_veriarch` takes an `on_round` callback for this reason.
- Everything is served from one process on one origin (FastAPI serves
  `frontend/index.html` at `/`), so there's no CORS configuration to
  worry about.

### Known limitations

- **State is in-memory and single-process.** Restarting the server loses
  all run history; this is a local dev tool, not a deployed service.
- **No auth, no path sandboxing.** `repo` is an arbitrary filesystem path
  the server will read from — fine for local use, not something to
  expose on a network without adding access control first.
- **One thread per run, no queueing/cancellation.** Concurrent runs are
  possible (each gets its own thread) but there's no way to cancel a
  run in progress from the UI yet.
- I was not able to start the server and click through it end-to-end in
  the environment I built it in (no network access to install
  `fastapi`/`uvicorn`, and no browser to drive). Both `server.py` and
  `pipeline.py` pass a syntax compile check, and I manually verified
  every field name the frontend reads (`state.stage`, `round.vscore`,
  `round.rationale[].text`, etc.) against what `server.py` actually
  writes into run state — but this hasn't been exercised by a real
  request. If something doesn't line up when you first run it, the
  browser's devtools Network/Console tabs are the fastest way to see
  which field mismatched.
