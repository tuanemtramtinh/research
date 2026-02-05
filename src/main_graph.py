from __future__ import annotations

from typing import Annotated, List, TypedDict

import operator

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


def map_to_workers(state: OrchestratorState):
    Send = _import_send()
    use_cases = state.get("use_cases") or []

    if Send is None:
        # Fallback: no Send available, run sequentially by routing to a single worker.
        return "sequential"

    sends = []
    for uc in use_cases:
        uc_actors = list(getattr(uc, "participating_actors", []) or [])
        sends.append(
            Send(
                "worker",
                {
                    "requirement_text": state.get("requirement_text", ""),
                    # IMPORTANT: pass only participating actors for this use case
                    "actors": uc_actors,
                    "use_case": uc,
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
    )
    return {"scenario_results_acc": [result]}


def sequential_worker_node(state: OrchestratorState):
    results: List[ScenarioResult] = []
    for uc in state.get("use_cases") or []:
        results.append(
            run_sca_use_case(
                use_case=uc,
                requirement_text=state.get("requirement_text", ""),
                # IMPORTANT: pass only participating actors for this use case
                actors=list(getattr(uc, "participating_actors", []) or []),
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
    # workflow.add_node("reduce_plan", reduce_plan_node)
    workflow.add_node("worker", worker_node)
    workflow.add_node("sequential_worker", sequential_worker_node)
    workflow.add_node("reduce", reduce_node)

    workflow.add_edge(START, "plan_tasks")

    # Map step (parallel if Send exists; otherwise go sequential)
    # NOTE: reduce_plan_node is redundant since rpa_graph already does
    # synonym checking with LLM in synonym_check_node
    workflow.add_conditional_edges(
        "plan_tasks",
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
