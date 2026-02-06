from ai.graphs.rpa_graph.state import ActorResult, GraphState
from langgraph.types import interrupt


def review_actors_node(state: GraphState):
    payload_to_user = {
        "type": "review_actors",
        "actors": [actor.model_dump() for actor in state.get("actor_results", [])],
    }

    decision = interrupt(payload_to_user)

    print(decision)
    # if decision is True or decision is None:
    #     return {}

    # if isinstance(decision, dict) and "actors" in decision:
    #     edited = [ActorResult(**a) for a in decision["actors"]]

    #     return {}

    return {}
