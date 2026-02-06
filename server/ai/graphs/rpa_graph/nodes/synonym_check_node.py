from typing import List
from ai.graphs.rpa_graph.state import ActorItem, CanonicalActorList, GraphState


def _synonym_actors_check(model, actors: List[ActorItem]) -> List[ActorItem]:
    """Remove synonymous actors using LLM."""
    if model is None or not actors:
        return actors

    structured_llm = model.with_structured_output(CanonicalActorList)
    actor_names = [item.actor for item in actors]

    system_prompt = """
    You are a Business Analyst AI specializing in requirement analysis.

    Your task is to analyze a list of actor names and remove synonymous or semantically equivalent actors.

    Rules:
    - Actors that represent the same logical role MUST be merged.
    - Choose ONE clear and generic canonical name for each group.
    - Prefer business-level, role-based names over wording variants.
    - ALL returned actor names MUST be lowercase.
    - IMPORTANT: The canonical actor name MUST be one of the existing actor names from the input list.
    - Do NOT invent new actors that are not implied by the list.
    - Do NOT explain your reasoning.
    - Return only structured data according to the output schema.
    """

    human_prompt = f"""
    The following is a list of actor names extracted from user stories.

    Actor names:
    {actor_names}

    Remove synonymous actors and return a list of unique canonical actor names.
    """

    response = structured_llm.invoke(
        [("system", system_prompt), ("human", human_prompt)]
    )

    # Lookup sentence_idx from original actors and merge if needed
    actor_lookup = {item.actor: item.sentence_idx for item in actors}

    result = []
    for canonical_name in response.actors:
        if canonical_name in actor_lookup:
            result.append(
                ActorItem(
                    actor=canonical_name, sentence_idx=actor_lookup[canonical_name]
                )
            )

    return result


def synonym_check_node(state: GraphState):
    """Remove synonymous actors using LLM."""
    model = state.get("llm")
    raw_actors = state.get("raw_actors") or []
    actors = _synonym_actors_check(model, raw_actors)
    return {"actors": actors}
