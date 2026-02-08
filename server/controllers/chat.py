import uuid

from typing import List
from fastapi import APIRouter
from langgraph.types import Command

from ai.graphs.main_graph import build_main_graph
from ai.graphs.rpa_graph.state import UseCase
from dtos.ActorReqDTO import ActorReqDTO
from dtos.UsecaseReqDTO import UsecaseReqDTO

from helpers._to_diagram_data import _to_diagram_data


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
