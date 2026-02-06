from typing import List
import uuid
from fastapi import APIRouter, Body
from langgraph.types import Command

from ai.graphs.main_graph import build_main_graph


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


@router.post("/step-2")
def get_usecases(thread_id: str = Body(...), resume_payload: str = Body(...)):
    print(thread_id, resume_payload)
    result = graph.invoke(
        Command(resume=resume_payload), config=CONFIG(thread_id=thread_id)
    )

    return "test"
