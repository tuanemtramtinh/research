from langgraph.types import interrupt
from ai.graphs.rpa_graph.state import GraphState, UseCase


def review_usecases_node(state: GraphState):

    payload_to_user = {
        "type": "review_usecases",
        "usecases": [uc.model_dump() for uc in state.get("use_cases", [])],
    }

    print("hello")

    decision = interrupt(payload_to_user)

    if decision is True or decision is None:
        return {}

    # Resume có thể nhận list trực tiếp (Command(resume=req.usecases)) hoặc dict {"usecases": [...]}
    raw = decision.get("usecases", decision) if isinstance(decision, dict) else decision
    if isinstance(raw, list) and raw:
        edited = [uc if isinstance(uc, UseCase) else UseCase(**uc) for uc in raw]
        return {"use_cases": edited}

    return {}
