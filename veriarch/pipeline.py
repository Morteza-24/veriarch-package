"""End-to-end VeriArch pipeline as a single callable, reporting progress
through a callback rather than printing directly, so both cli.py and
server.py can drive the same logic -- the CLI prints the callback
events, the server turns them into pollable run state.
"""

from typing import Any, Callable, Dict, List, Optional

from .architect import ArchitectAgent
from .config import VeriArchConfig
from .critic import Critic
from .descriptors import generate_capability_descriptors
from .fusion import build_adjacency_from_similarity, fuse_membership, train_nocd
from .llm import get_llm_client
from .loop import run_veriarch
from .parsing.java_parser import parse_repository
from .semantic import compute_semantic_embeddings
from .structural import build_structural_matrix, structural_neighborhood

ProgressFn = Callable[[Dict[str, Any]], None]


def run_pipeline(
    repo: str,
    cfg: VeriArchConfig,
    constraints: List[str],
    llm_provider: str,
    llm_model: str,
    llm_base_url: Optional[str],
    llm_api_key: Optional[str],
    progress: ProgressFn,
) -> Dict[str, Any]:
    def emit(stage: str, **kw: Any) -> None:
        progress({"stage": stage, **kw})

    emit("parsing")
    classes = parse_repository(repo)
    if not classes:
        raise ValueError(f"No parsable .java classes found under {repo}")
    names, s_str = build_structural_matrix(classes)
    emit("parsing", done=True, num_classes=len(names))

    emit("semantic")
    embeddings = compute_semantic_embeddings(classes, names, dim=cfg.embedding_dim)
    emit("semantic", done=True)

    emit("fusion")
    adj_str = (s_str > 0).astype(float)
    sim_sem = embeddings @ embeddings.T
    adj_sem = build_adjacency_from_similarity(sim_sem)
    m_str = train_nocd(adj_str, s_str, cfg.num_services)
    m_sem = train_nocd(adj_sem, embeddings, cfg.num_services)
    prior_m = fuse_membership(m_str, m_sem, cfg.alpha)
    emit("fusion", done=True)

    emit("descriptors")
    neighborhoods = {
        name: [names[j] for j in structural_neighborhood(i, s_str, cfg.delta)]
        for i, name in enumerate(names)
    }
    llm_client = get_llm_client(llm_provider, llm_model, api_key=llm_api_key, base_url=llm_base_url)
    descriptors = generate_capability_descriptors(classes, names, neighborhoods, llm_client)
    emit("descriptors", done=True, descriptors=descriptors)

    emit("loop")
    service_names = [f"service_{i + 1}" for i in range(cfg.num_services)]
    architect = ArchitectAgent(service_names, llm_client)
    critic = Critic(names, s_str, embeddings, prior_m, service_names, beta=cfg.beta, gamma=cfg.gamma)

    def on_round(r, output, result):
        emit(
            "loop",
            round=r,
            vscore=result.vscore,
            weak=[{"class": c, "service": s} for c, s in result.weak_assignments],
            assignment={k: sorted(v) for k, v in output.assignment.items()},
            rationale=[
                {"class": c, "service": s, "text": txt}
                for (c, s), txt in output.rationale.items()
            ],
            critique=result.critique_text,
        )

    result = run_veriarch(
        descriptors, architect, critic, constraints,
        tau=cfg.tau, max_rounds=cfg.max_rounds, on_round=on_round,
    )

    final = {
        "assignment": {k: sorted(v) for k, v in result.assignment.items()},
        "trace": result.trace.entries,
        "converged": result.converged,
        "rounds_run": result.rounds_run,
        "vscore": result.final_vscore,
    }
    emit("done", **final)
    return final
