import uuid

from typing import Any, Dict, List
from fastapi import APIRouter, HTTPException
from langgraph.types import Command

from ai.graphs.main_graph import build_main_graph
from ai.graphs.rpa_graph.state import UseCase
from ai.graphs.sca_graph.state import ScenarioResult
from dtos.ActorReqDTO import ActorReqDTO
from dtos.ScaReqDTO import ScaReqDTO
from dtos.UsecaseReqDTO import UsecaseReqDTO

from helpers._to_diagram_data import _to_diagram_data
from helpers._sca_helpers import _evaluation_to_dict, _scenario_result_to_response


router = APIRouter(
    prefix="/chat",
    tags=["chat"],
)

graph = build_main_graph()


def CONFIG(thread_id: str):
    return {"configurable": {"thread_id": thread_id}}


@router.get("/")
def test():
    return "hello"


# Return actor list
@router.post("/step-1")
def get_actors(requirement_text: List[str]):

    thread_id = str(uuid.uuid4())

    result = graph.invoke(
        {
            "requirement_text": requirement_text,
            "use_cases": [],
            "scenario_results_acc": [],
        },
        config=CONFIG(thread_id=thread_id),
    )

    interrupts = result.get("__interrupt__", [])

    if interrupts:
        first = interrupts[0]

        if isinstance(first, dict):
            interrupt_value = first.get("value")
        else:
            interrupt_value = getattr(first, "value", None)

        return {
            "thread_id": thread_id,
            "interrupt": interrupt_value,
        }

    return {
        "thread_id": thread_id,
        "result": result,
    }


# Return Usecase
@router.post("/step-2")
def get_usecases(req: ActorReqDTO):
    result = graph.invoke(
        Command(resume=req.actors), config=CONFIG(thread_id=req.thread_id)
    )

    interrupts = result.get("__interrupt__", [])

    if interrupts:
        first = interrupts[0]

        if isinstance(first, dict):
            interrupt_value = first.get("value")
        else:
            interrupt_value = getattr(first, "value", None)

        return {
            "thread_id": req.thread_id,
            "interrupt": interrupt_value,
        }

    return {
        "thread_id": req.thread_id,
        "result": result,
    }


# Draw Usecase
@router.post("/step-3")
def draw_usecases(req: UsecaseReqDTO):
    result = graph.invoke(
        Command(resume=req.usecases), config=CONFIG(thread_id=req.thread_id)
    )

    print(result)

    # Get use_cases from result or request (normalize to UseCase objects)
    raw_uc = result.get("use_cases") or req.usecases
    if not raw_uc:
        return {"nodes": [], "links": []}
    use_cases = [uc if isinstance(uc, UseCase) else UseCase(**uc) for uc in raw_uc]

    # Get actors from result (actor_aliases has .actor, or actors list)
    actors: List[str] | None = None
    if result.get("actor_aliases"):
        raw = result["actor_aliases"]
        actors = [
            (a.get("actor") if isinstance(a, dict) else getattr(a, "actor", None))
            for a in raw
        ]
        actors = [a for a in actors if a]
    elif result.get("actors"):
        actors = list(result["actors"])

    return _to_diagram_data(
        use_cases=use_cases,
        actors=actors,
        system_name="System",
    )


# Generate Scenario Specs + Evaluation for all use cases from previous steps
@router.post("/step-4")
def generate_scenarios(req: ScaReqDTO):
    """Resume the main graph so the SCA node runs.

    The graph was interrupted at run_sca after step-3 completed.
    Resuming it triggers SCA evaluation for every use case.
    Returns all scenario results with scores and sub-criteria breakdowns.
    """

    thread_id = req.thread_id

    # Verify the graph is actually paused at the SCA interrupt
    snapshot = graph.get_state(CONFIG(thread_id))
    if not snapshot or not getattr(snapshot, "values", None):
        raise HTTPException(
            status_code=404,
            detail=f"No state found for thread_id '{thread_id}'. Run step-1 through step-3 first.",
        )

    # Resume the graph – run_sca_node continues and runs SCA for all use cases
    result = graph.invoke(
        Command(resume=True), config=CONFIG(thread_id=thread_id)
    )

    # Pull final state (scenario_results written by run_sca_node)
    final_snapshot = graph.get_state(CONFIG(thread_id))
    state_values = getattr(final_snapshot, "values", None) or {}
    scenario_results = state_values.get("scenario_results") or []

    results: List[Dict[str, Any]] = []
    for sr in scenario_results:
        try:
            if not isinstance(sr, ScenarioResult):
                sr = ScenarioResult(**sr) if isinstance(sr, dict) else sr
            results.append(_scenario_result_to_response(sr))
        except Exception as exc:
            results.append({"error": str(exc)})

    return {
        "thread_id": thread_id,
        "count": len(results),
        "results": results,
    }
