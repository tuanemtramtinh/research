from langgraph.graph import END, START, StateGraph

from ai.graphs.rpa_graph.graph import run_rpa
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


def build_main_graph():
    workflow = StateGraph(OrchestratorState)

    workflow.add_node("plan_tasks", plan_tasks_node)

    workflow.add_edge(START, "plan_tasks")
    workflow.add_edge("plan_tasks", END)

    return workflow.compile(checkpointer=checkpointer)
