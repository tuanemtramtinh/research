from ai.graphs.rpa_graph.state import ActorResult, GraphState
from langgraph.types import interrupt


def review_actors_node(state: GraphState):
    payload_to_user = {
        "type": "review_actors",
        "actors": [actor.model_dump() for actor in state.get("actor_results", [])],
    }

    decision = interrupt(payload_to_user)

    if decision is True or decision is None:
        return {}

    # Resume có thể nhận list trực tiếp (Command(resume=req.actors)) hoặc dict {"actors": [...]}
    raw_list = None
    if isinstance(decision, dict) and "actors" in decision:
        raw_list = decision["actors"]
    elif isinstance(decision, list) and len(decision) > 0:
        raw_list = decision

    if raw_list is not None:
        edited = [
            ActorResult(**(a if isinstance(a, dict) else a.model_dump()))
            for a in raw_list
        ]
        return {"actor_results": edited}

    return {}
