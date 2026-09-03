"""End-to-end VeriArch pipeline, command-line entry point.

Usage:
    python -m veriarch.cli --repo /path/to/java/monolith --num-services 6 \\
        --constraint "Payment and Order must not share a service"

Requires ANTHROPIC_API_KEY in the environment if --llm-provider anthropic
is used (the default); pass --llm-provider ollama for a free, local
model instead. See README.md for the full dependency list and the
fallbacks used when optional packages are unavailable.

For a real-time browser UI instead of this CLI, run the bundled server
(see server.py / GUIDE.md) and use that instead -- it drives the same
pipeline() function and streams progress live.
"""

import argparse
import json
import os

from .config import VeriArchConfig
from .pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="VeriArch: agentic, verifiable microservice extraction"
    )
    parser.add_argument("--repo", required=True, help="Path to the monolithic Java repository")
    parser.add_argument("--num-services", type=int, default=6)
    parser.add_argument(
        "--constraint",
        action="append",
        default=[],
        help="Hard constraint, e.g. 'Payment and Order must not share a service'",
    )
    parser.add_argument("--output-dir", default="veriarch_output")
    parser.add_argument(
        "--llm-provider", default="anthropic",
        choices=["anthropic", "ollama", "openai_compatible"],
    )
    parser.add_argument("--llm-model", default=None, help="Defaults per-provider if unset")
    parser.add_argument("--llm-base-url", default=None)
    parser.add_argument("--llm-api-key", default=None)
    args = parser.parse_args()

    default_models = {
        "anthropic": "claude-sonnet-4-6",
        "ollama": "qwen2.5:7b",
        "openai_compatible": "Qwen2.5-7B-Instruct",
    }
    llm_model = args.llm_model or default_models[args.llm_provider]

    cfg = VeriArchConfig(num_services=args.num_services, llm_provider=args.llm_provider, llm_model=llm_model)
    os.makedirs(args.output_dir, exist_ok=True)

    stage_labels = {
        "parsing": "[1/5] Parsing repository",
        "semantic": "[2/5] Computing semantic embeddings",
        "fusion": "[3/5] Fusing structural and semantic priors (NOCD)",
        "descriptors": "[4/5] Generating capability descriptors",
        "loop": "[5/5] Running the architect-critic loop",
    }
    printed_stage_start = set()

    def progress(update):
        stage = update["stage"]
        if stage in stage_labels and stage not in printed_stage_start:
            print(stage_labels[stage] + "...")
            printed_stage_start.add(stage)
        if stage == "loop" and "round" in update:
            print(
                f"    round {update['round']}: VSCORE={update['vscore']:.3f}, "
                f"weak={len(update['weak'])}"
            )
        if stage == "done":
            print(
                f"[done] converged={update['converged']} rounds_run={update['rounds_run']} "
                f"VSCORE={update['vscore']:.3f}"
            )

    result = run_pipeline(
        args.repo, cfg, args.constraint,
        args.llm_provider, llm_model, args.llm_base_url, args.llm_api_key,
        progress,
    )

    trace_md_path = os.path.join(args.output_dir, "rationale_trace.md")
    with open(trace_md_path, "w") as f:
        lines = ["# VeriArch decomposition rationale trace", ""]
        for e in result["trace"]:
            lines.append(f"- **Round {e['round']}** - `{e['class']}` -> `{e['service']}`: {e['rationale']}")
        f.write("\n".join(lines))

    with open(os.path.join(args.output_dir, "rationale_trace.json"), "w") as f:
        json.dump(result["trace"], f, indent=2)

    assignment_path = os.path.join(args.output_dir, "assignment.json")
    with open(assignment_path, "w") as f:
        json.dump(result["assignment"], f, indent=2)

    print(f"Wrote decomposition to {assignment_path}")
    print(f"Wrote rationale trace to {trace_md_path}")


if __name__ == "__main__":
    main()
