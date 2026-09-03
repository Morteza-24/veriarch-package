"""VeriArch web server.

Runs the real pipeline (parsing, embeddings, NOCD fusion, descriptor
generation, and the architect-critic loop) in a background thread per
request, and exposes its progress for the frontend to poll. Serves the
static frontend at '/' so the whole app is a single process on one
origin -- no CORS configuration needed.

Run with:
    python -m uvicorn veriarch.server:app --reload --port 8765
then open http://localhost:8765
"""

import threading
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from .config import VeriArchConfig
from .pipeline import run_pipeline

app = FastAPI(title="VeriArch")

RUNS: Dict[str, Dict[str, Any]] = {}
LOCK = threading.Lock()


class RunRequest(BaseModel):
    repo: str
    num_services: int = 6
    constraints: List[str] = []
    llm_provider: str = "ollama"
    llm_model: str = "qwen2.5:7b"
    llm_base_url: Optional[str] = None
    llm_api_key: Optional[str] = None
    alpha: float = 0.7
    beta: float = 0.5
    gamma: float = 0.2
    tau: float = 0.75
    delta: float = 0.15
    max_rounds: int = 5


def _new_state(run_id: str) -> Dict[str, Any]:
    return {
        "id": run_id,
        "status": "running",  # running | done | error
        "stage": "parsing",
        "num_classes": None,
        "rounds": [],
        "assignment": None,
        "trace": None,
        "converged": None,
        "rounds_run": None,
        "final_vscore": None,
        "error": None,
    }


def _worker(run_id: str, req: RunRequest) -> None:
    def progress(update: Dict[str, Any]) -> None:
        with LOCK:
            state = RUNS[run_id]
            state["stage"] = update.get("stage", state["stage"])
            if update.get("stage") == "parsing" and update.get("done"):
                state["num_classes"] = update.get("num_classes")
            if update.get("stage") == "loop" and "round" in update:
                state["rounds"].append(
                    {
                        "round": update["round"],
                        "vscore": update["vscore"],
                        "weak": update["weak"],
                        "assignment": update["assignment"],
                        "rationale": update["rationale"],
                        "critique": update["critique"],
                    }
                )
            if update.get("stage") == "done":
                state["status"] = "done"
                state["converged"] = update["converged"]
                state["rounds_run"] = update["rounds_run"]
                state["final_vscore"] = update["vscore"]
                state["assignment"] = update["assignment"]
                state["trace"] = update["trace"]

    try:
        cfg = VeriArchConfig(
            num_services=req.num_services,
            alpha=req.alpha,
            beta=req.beta,
            gamma=req.gamma,
            tau=req.tau,
            delta=req.delta,
            max_rounds=req.max_rounds,
        )
        run_pipeline(
            req.repo,
            cfg,
            req.constraints,
            req.llm_provider,
            req.llm_model,
            req.llm_base_url,
            req.llm_api_key,
            progress,
        )
    except Exception as exc:  # noqa: BLE001 - surface any failure to the UI
        with LOCK:
            RUNS[run_id]["status"] = "error"
            RUNS[run_id]["error"] = str(exc)


@app.post("/api/runs")
def start_run(req: RunRequest) -> Dict[str, str]:
    repo_path = Path(req.repo).expanduser()
    if not repo_path.exists():
        raise HTTPException(400, f"Path does not exist: {req.repo}")

    run_id = uuid.uuid4().hex[:12]
    with LOCK:
        RUNS[run_id] = _new_state(run_id)

    thread = threading.Thread(target=_worker, args=(run_id, req), daemon=True)
    thread.start()
    return {"run_id": run_id}


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> JSONResponse:
    with LOCK:
        state = RUNS.get(run_id)
        if state is None:
            raise HTTPException(404, "unknown run_id")
        return JSONResponse(dict(state))


@app.get("/api/validate_repo")
def validate_repo(path: str) -> Dict[str, Any]:
    p = Path(path).expanduser()
    if not p.exists():
        return {"exists": False, "java_files": 0}
    java_files = sum(1 for _ in p.rglob("*.java"))
    return {"exists": True, "java_files": java_files}


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    html_path = Path(__file__).resolve().parent.parent / "frontend" / "index.html"
    return html_path.read_text(encoding="utf-8")
