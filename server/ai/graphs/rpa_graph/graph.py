import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.graph import END, START, StateGraph

from ai.graphs.rpa_graph.nodes.find_actors_node import find_actors_node
from ai.graphs.rpa_graph.nodes.find_aliases_node import find_aliases_node
from ai.graphs.rpa_graph.nodes.synonym_check_node import synonym_check_node
from ai.graphs.rpa_graph.state import GraphState
from ai.graphs.state import RpaState
from ai.graphs.rpa_graph.nodes.find_goals_node import find_goals_node
from ai.graphs.rpa_graph.nodes.refine_goals_node import refine_goals_node
from ai.graphs.rpa_graph.nodes.grouping_node import grouping_node
from ai.graphs.rpa_graph.nodes.refine_clustering_node import refine_clustering_node
from ai.graphs.rpa_graph.nodes.find_include_extend_node import find_include_extend_node
from ai.graphs.rpa_graph.nodes.merge_relationships_node import merge_relationships_node
from ai.graphs.rpa_graph.nodes.name_usecases_node import name_usecases_node
from ai.graphs.rpa_graph.nodes.review_actors_node import review_actors_node
from ai.graphs.rpa_graph.nodes.review_usecases_node import review_usecases_node


def _get_model():
    load_dotenv()
    model_name = os.getenv("LLM_MODEL", "gpt-5-mini")
    if not os.getenv("OPENAI_API_KEY"):
        return None
    return init_chat_model(model_name, model_provider="openai")


def should_continue_to_clustering(state: GraphState):
    # Kiểm tra xem cả 2 nhánh đã đổ data về chưa
    if state.get("grouping_done"):
        return "continue"
    return "stop"


def build_rpa_graph():
    """
    RPA Graph (Requirement Processing Agent) - SEQUENCE:

    - Actor pipeline trước (có interrupt tại review_actors; sau khi user resume mới tiếp).
    - Goals pipeline sau: find_goals → refine_goals.
    - grouping: Merge actor+usecase (luôn dùng actor_results đã qua review).
    - refine_clustering → name_usecases → find_include_extend → merge_relationships → review_usecases.
    """

    workflow = StateGraph(GraphState)

    # Nodes
    workflow.add_node("find_actors", find_actors_node)
    workflow.add_node("synonym_check", synonym_check_node)
    workflow.add_node("find_aliases", find_aliases_node)
    workflow.add_node("review_actors", review_actors_node)

    workflow.add_node("find_goals", find_goals_node)
    workflow.add_node("refine_goals", refine_goals_node)

    workflow.add_node("grouping", grouping_node)
    workflow.add_node("refine_clustering", refine_clustering_node)
    workflow.add_node("name_usecases", name_usecases_node)
    workflow.add_node("review_usecases", review_usecases_node)
    workflow.add_node("find_include_extend", find_include_extend_node)
    workflow.add_node("merge_relationships", merge_relationships_node)

    # Sequence: Actor xong hết (kể cả resume) → Goals → grouping → ...
    workflow.add_edge(START, "find_actors")
    workflow.add_edge("find_actors", "synonym_check")
    workflow.add_edge("synonym_check", "find_aliases")
    workflow.add_edge("find_aliases", "review_actors")

    workflow.add_edge("review_actors", "find_goals")
    workflow.add_edge("find_goals", "refine_goals")
    workflow.add_edge("refine_goals", "grouping")

    workflow.add_conditional_edges(
        "grouping",
        should_continue_to_clustering,
        {
            "continue": "refine_clustering",
            "stop": END,
        },
    )

    workflow.add_edge("refine_clustering", "name_usecases")
    workflow.add_edge("name_usecases", "find_include_extend")
    workflow.add_edge("find_include_extend", "merge_relationships")
    workflow.add_edge("merge_relationships", "review_usecases")
    workflow.add_edge("review_usecases", END)

    return workflow.compile()


def run_rpa(requirement_text: str) -> RpaState:
    """Run the RPA graph and return results.

    Returns:
        dict containing:
        - requirement_text: Original requirement text
        - actors: List of canonical actors
        - actor_aliases: List of actor results with aliases
        - use_cases: List of UseCase objects with relationships (include/extend)
    """
    # sentences = requirement_text.split("\n")
    sentences = requirement_text
    requirement_text = "\n".join(requirement_text)
    llm = _get_model()
    app = build_rpa_graph()
    out = app.invoke(
        {"llm": llm, "requirement_text": requirement_text, "sentences": sentences}
    )
    return {
        "requirement_text": out.get("requirement_text", requirement_text),
        "actors": out.get("actors", []),
        "actor_aliases": out.get("actor_results", []),
        "use_cases": out.get("use_cases", []),
    }
