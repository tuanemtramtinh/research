from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from ai.graphs.rpa_graph.graph import run_rpa
from ai.graphs.rpa_graph.state import UseCase
from ai.graphs.sca_graph.graph import run_sca_use_case
from ai.graphs.sca_graph.state import ScenarioResult
from ai.graphs.state import OrchestratorState
from ai.graphs.checkpoint import checkpointer


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


def review_before_sca_node(state: OrchestratorState):
    """Interrupt so the client can review / edit use-cases before SCA runs.

    step-3 hits this interrupt and returns the diagram to the client.
    step-4 resumes the graph, which then flows into run_sca.
    """
    interrupt({
        "type": "ready_for_sca",
        "use_cases": [
            uc.model_dump() if hasattr(uc, "model_dump") else uc
            for uc in (state.get("use_cases") or [])
        ],
    })
    return {}


def run_sca_node(state: OrchestratorState):
    """Run SCA for every use case in the current state."""

    requirement_text = state.get("requirement_text", "")
    if isinstance(requirement_text, list):
        requirement_text_list = requirement_text
    else:
        requirement_text_list = [requirement_text] if requirement_text else []

    raw_use_cases = state.get("use_cases") or []
    use_cases: List[UseCase] = [
        uc if isinstance(uc, UseCase)
        else UseCase(**(uc if isinstance(uc, dict) else uc.model_dump()))
        for uc in raw_use_cases
    ]

    # Resolve actors
    actors_list: List[str] = []
    if state.get("actor_aliases"):
        for a in state["actor_aliases"]:
            name = a.get("actor") if isinstance(a, dict) else getattr(a, "actor", None)
            if name:
                actors_list.append(name)
    elif state.get("actors"):
        actors_list = list(state["actors"])

    results: List[ScenarioResult] = [None] * len(use_cases)  # preserve order

    def _run_one(index: int, uc: UseCase) -> tuple:
        sr = run_sca_use_case(
            use_case=uc,
            requirement_text=requirement_text_list,
            actors=actors_list or None,
        )
        return index, sr

    with ThreadPoolExecutor(max_workers=min(len(use_cases), 8)) as pool:
        futures = {
            pool.submit(_run_one, i, uc): i
            for i, uc in enumerate(use_cases)
        }
        for future in as_completed(futures):
            idx, sr = future.result()
            results[idx] = sr

    return {"scenario_results": results}


def build_main_graph():
    workflow = StateGraph(OrchestratorState)

    workflow.add_node("plan_tasks", plan_tasks_node)
    workflow.add_node("review_before_sca", review_before_sca_node)
    workflow.add_node("run_sca", run_sca_node)

    workflow.add_edge(START, "plan_tasks")
    workflow.add_edge("plan_tasks", "review_before_sca")
    workflow.add_edge("review_before_sca", "run_sca")
    workflow.add_edge("run_sca", END)

    return workflow.compile(checkpointer=checkpointer)
