from __future__ import annotations

from typing import Annotated, Dict, List, TypedDict

import operator
import sys
from pathlib import Path

from langgraph.graph import END, START, StateGraph

from .graphs.rpa_graph import run_rpa
from .graphs.sca_graph import run_sca_use_case
from .state import ActorResult, ScenarioResult, UseCase


def _import_send():
    # LangGraph moved Send around across versions.
    # Try the common locations.
    try:
        from langgraph.types import Send  # type: ignore

        return Send
    except Exception:
        try:
            from langgraph.graph import Send  # type: ignore

            return Send
        except Exception:
            return None


class OrchestratorState(TypedDict, total=False):
    requirement_text: List[str]
    # tasks: List[TaskItem]

    actors: List[str]
    actor_aliases: List[ActorResult]

    # RPA output
    use_cases: List[UseCase]

    # Map-reduce accumulator (Agent2 output)
    scenario_results_acc: Annotated[List[ScenarioResult], operator.add]

    # Final reduced view
    scenario_results: List[ScenarioResult]

    # Optional per-usecase reference file path for correctness evaluation
    # (keyed by UseCase.id)
    reference_spec_paths: Dict[int, str | None]

    # Optional per-usecase comparison spec path for extra evaluation
    # (keyed by UseCase.id)
    comparison_spec_paths: Dict[int, str | None]

    # Optional directory paths for auto-discovery of reference/comparison files
    reference_dir: str | None
    comparison_dir: str | None

    # Reduced view
    # merged_actors: List[str]


def plan_tasks_node(state: OrchestratorState):
    out = run_rpa(state.get("requirement_text", ""))
    return {
        "requirement_text": out.get(
            "requirement_text", state.get("requirement_text", "")
        ),
        "actors": out.get("actors", []),
        "actor_aliases": out.get("actor_aliases", []),
        "use_cases": out.get("use_cases", []),
    }


def _fuzzy_match_file(uc_name: str, dir_path: Path) -> str | None:
    """Find the best matching JSON file in dir_path for a use case name.

    Tries exact stem match first, then fuzzy substring matching.
    """
    import difflib

    if not dir_path.is_dir():
        return None

    json_files = list(dir_path.glob("*_report.json")) + list(dir_path.glob("*.json"))
    if not json_files:
        return None

    # Normalize the use case name for comparison
    uc_lower = uc_name.strip().lower().replace("_", " ").replace("-", " ")

    # Build a list of (file, stem_normalised) pairs
    candidates: list[tuple[Path, str]] = []
    for f in json_files:
        stem = f.stem.replace("_report", "").replace("_", " ").replace("-", " ").lower()
        candidates.append((f, stem))

    # Exact match first
    for f, stem in candidates:
        if stem == uc_lower:
            return str(f)

    # Fuzzy match – pick closest above 0.4 threshold
    stems = [c[1] for c in candidates]
    matches = difflib.get_close_matches(uc_lower, stems, n=1, cutoff=0.4)
    if matches:
        idx = stems.index(matches[0])
        return str(candidates[idx][0])

    return None


def collect_reference_paths_node(state: OrchestratorState):
    """Prompt for a reference file per use case (Enter => skip correctness).

    If reference_dir is provided in state, auto-discover files by fuzzy matching
    use case names to JSON files in that directory.
    This runs BEFORE the map step so parallel workers do not concurrently block on stdin.
    """

    use_cases = state.get("use_cases") or []
    mapping: Dict[int, str | None] = {}
    ref_dir = state.get("reference_dir")

    # Auto-discovery from reference_dir
    if isinstance(ref_dir, str) and ref_dir.strip():
        ref_dir_path = Path(ref_dir.strip())
        print("\n=== AUTO-DISCOVERING REFERENCE FILES ===")
        print(f"Directory: {ref_dir_path}\n")
        for uc in use_cases:
            uc_id = int(getattr(uc, "id", 0) or 0)
            uc_name = str(getattr(uc, "name", "") or "").strip()
            matched = _fuzzy_match_file(uc_name, ref_dir_path)
            mapping[uc_id] = matched
            status = matched or "<no match>"
            print(f"  [{uc_id}] {uc_name} => {status}")
        return {"reference_spec_paths": mapping}

    # Non-interactive mode: do not block.
    try:
        if not sys.stdin or not sys.stdin.isatty():
            for uc in use_cases:
                mapping[int(getattr(uc, "id", 0) or 0)] = None
            return {"reference_spec_paths": mapping}
    except Exception:
        for uc in use_cases:
            mapping[int(getattr(uc, "id", 0) or 0)] = None
        return {"reference_spec_paths": mapping}

    repo_root = Path(__file__).resolve().parents[1]

    print("\n=== REFERENCE FILES (Correctness) ===")
    print(
        "Enter a reference file path for each use case, or press Enter to skip correctness (N/A).\n"
    )

    for uc in use_cases:
        uc_id = int(getattr(uc, "id", 0) or 0)
        uc_name = str(getattr(uc, "name", "") or "").strip() or "<unnamed use case>"

        while True:
            raw = input(f"Reference file for [{uc_id}] {uc_name}: ").strip()
            if not raw:
                mapping[uc_id] = None
                break

            p = Path(raw).expanduser()
            if not p.is_absolute():
                p = (repo_root / p).resolve()

            if p.exists() and p.is_file():
                mapping[uc_id] = str(p)
                break

            print(f"  Not found: {p}")
            print("  Try again, or press Enter to skip.")

    return {"reference_spec_paths": mapping}


def collect_comparison_paths_node(state: OrchestratorState):
    """Prompt for an optional comparison spec per use case (Enter => skip).

    If comparison_dir is provided in state, auto-discover files by fuzzy matching
    use case names to JSON files in that directory.
    This runs BEFORE the map step so parallel workers do not concurrently block on stdin.
    """

    use_cases = state.get("use_cases") or []
    mapping: Dict[int, str | None] = {}
    cmp_dir = state.get("comparison_dir")

    # Auto-discovery from comparison_dir
    if isinstance(cmp_dir, str) and cmp_dir.strip():
        cmp_dir_path = Path(cmp_dir.strip())
        print("\n=== AUTO-DISCOVERING COMPARISON FILES ===")
        print(f"Directory: {cmp_dir_path}\n")
        for uc in use_cases:
            uc_id = int(getattr(uc, "id", 0) or 0)
            uc_name = str(getattr(uc, "name", "") or "").strip()
            matched = _fuzzy_match_file(uc_name, cmp_dir_path)
            mapping[uc_id] = matched
            status = matched or "<no match>"
            print(f"  [{uc_id}] {uc_name} => {status}")
        return {"comparison_spec_paths": mapping}

    # Non-interactive mode: do not block.
    try:
        if not sys.stdin or not sys.stdin.isatty():
            for uc in use_cases:
                mapping[int(getattr(uc, "id", 0) or 0)] = None
            return {"comparison_spec_paths": mapping}
    except Exception:
        for uc in use_cases:
            mapping[int(getattr(uc, "id", 0) or 0)] = None
        return {"comparison_spec_paths": mapping}

    repo_root = Path(__file__).resolve().parents[1]

    print("\n=== OPTIONAL COMPARISON SCENARIO (Scores Only) ===")
    print(
        "Optionally enter a file path to an existing use case scenario/spec JSON to score with the same rubric."
    )
    print("Press Enter to skip (comparison scores will be N/A).\n")

    for uc in use_cases:
        uc_id = int(getattr(uc, "id", 0) or 0)
        uc_name = str(getattr(uc, "name", "") or "").strip() or "<unnamed use case>"

        while True:
            raw = input(
                f"Comparison scenario/spec file for [{uc_id}] {uc_name} (optional): "
            ).strip()
            if not raw:
                mapping[uc_id] = None
                break

            p = Path(raw).expanduser()
            if not p.is_absolute():
                p = (repo_root / p).resolve()

            if p.exists() and p.is_file():
                mapping[uc_id] = str(p)
                break

            print(f"  Not found: {p}")
            print("  Try again, or press Enter to skip.")

    return {"comparison_spec_paths": mapping}


def map_to_workers(state: OrchestratorState):
    Send = _import_send()
    use_cases = state.get("use_cases") or []
    ref_paths = state.get("reference_spec_paths") or {}
    cmp_paths = state.get("comparison_spec_paths") or {}

    if Send is None:
        # Fallback: no Send available, run sequentially by routing to a single worker.
        return "sequential"

    sends = []
    for uc in use_cases:
        uc_actors = list(getattr(uc, "participating_actors", []) or [])
        uc_id = int(getattr(uc, "id", 0) or 0)
        sends.append(
            Send(
                "worker",
                {
                    "requirement_text": state.get("requirement_text", ""),
                    # IMPORTANT: pass only participating actors for this use case
                    "actors": uc_actors,
                    "use_case": uc,
                    "reference_spec_path": ref_paths.get(uc_id),
                    "comparison_spec_path": cmp_paths.get(uc_id),
                },
            )
        )
    return sends


def worker_node(state: dict):
    # State here is the payload produced by Send
    use_case = state.get("use_case")
    result = run_sca_use_case(
        use_case=use_case,
        requirement_text=state.get("requirement_text", []),
        # IMPORTANT: keep actors constrained to this use case
        actors=list(getattr(use_case, "participating_actors", []) or []),
        reference_spec_path=state.get("reference_spec_path"),
        comparison_spec_path=state.get("comparison_spec_path"),
    )
    return {"scenario_results_acc": [result]}


def sequential_worker_node(state: OrchestratorState):
    results: List[ScenarioResult] = []
    ref_paths = state.get("reference_spec_paths") or {}
    cmp_paths = state.get("comparison_spec_paths") or {}
    for uc in state.get("use_cases") or []:
        uc_id = int(getattr(uc, "id", 0) or 0)
        results.append(
            run_sca_use_case(
                use_case=uc,
                requirement_text=state.get("requirement_text", ""),
                # IMPORTANT: pass only participating actors for this use case
                actors=list(getattr(uc, "participating_actors", []) or []),
                reference_spec_path=ref_paths.get(uc_id),
                comparison_spec_path=cmp_paths.get(uc_id),
            )
        )
    return {"scenario_results_acc": results}


def reduce_node(state: OrchestratorState):
    # # Minimal reduce: keep unique actors + unique use cases.
    # merged_actors: List[str] = []
    # seen_actor = set()
    # for a in state.get("actors") or []:
    #     key = a.strip()
    #     if key and key.lower() not in seen_actor:
    #         seen_actor.add(key.lower())
    #         merged_actors.append(key)

    # uniq_use_cases: List[UseCase] = []
    # seen_uc = set()
    # for uc in state.get("use_cases") or []:
    #     k = (uc.name.strip().lower(), int(getattr(uc, "id", 0)))
    #     if k not in seen_uc:
    #         seen_uc.add(k)
    #         uniq_use_cases.append(uc)

    uniq_scenarios: List[ScenarioResult] = []
    seen_sr = set()
    for sr in state.get("scenario_results_acc") or []:
        k = (sr.use_case.name.strip().lower(), int(getattr(sr.use_case, "id", 0)))
        if k not in seen_sr:
            seen_sr.add(k)
            uniq_scenarios.append(sr)

    return {
        # "merged_actors": merged_actors,
        # "use_cases": uniq_use_cases,
        "scenario_results": uniq_scenarios,
    }


# NOTE: reduce_plan_node is redundant - rpa_graph.synonym_check_node already
# handles actor deduplication using LLM (more intelligent than simple string comparison)
# def reduce_plan_node(state: OrchestratorState):
#     """Pre-reduce right after plan_tasks: merge actors + unique use cases."""
#     merged_actors: List[str] = []
#     seen_actor = set()
#     for a in state.get("actors") or []:
#         key = a.strip()
#         if key and key.lower() not in seen_actor:
#             seen_actor.add(key.lower())
#             merged_actors.append(key)
#
#     uniq_use_cases: List[UseCase] = []
#     seen_uc = set()
#     for uc in state.get("use_cases") or []:
#         k = (uc.name.strip().lower(), int(getattr(uc, "id", 0)))
#         if k not in seen_uc:
#             seen_uc.add(k)
#             uniq_use_cases.append(uc)
#
#     return {
#         "merged_actors": merged_actors,
#         "use_cases": uniq_use_cases,
#     }


def build_main_graph():
    workflow = StateGraph(OrchestratorState)

    workflow.add_node("plan_tasks", plan_tasks_node)
    workflow.add_node("collect_references", collect_reference_paths_node)
    workflow.add_node("collect_comparisons", collect_comparison_paths_node)
    # workflow.add_node("reduce_plan", reduce_plan_node)
    workflow.add_node("worker", worker_node)
    workflow.add_node("sequential_worker", sequential_worker_node)
    workflow.add_node("reduce", reduce_node)

    workflow.add_edge(START, "plan_tasks")

    workflow.add_edge("plan_tasks", "collect_references")
    workflow.add_edge("collect_references", "collect_comparisons")

    # Map step (parallel if Send exists; otherwise go sequential)
    # NOTE: reduce_plan_node is redundant since rpa_graph already does
    # synonym checking with LLM in synonym_check_node
    workflow.add_conditional_edges(
        "collect_comparisons",
        map_to_workers,
        {
            "sequential": "sequential_worker",
            "worker": "worker",
        },
    )

    workflow.add_edge("worker", "reduce")
    workflow.add_edge("sequential_worker", "reduce")
    workflow.add_edge("reduce", END)

    return workflow.compile()
