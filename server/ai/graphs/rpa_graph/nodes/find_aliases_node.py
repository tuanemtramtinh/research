from typing import List
from ai.graphs.rpa_graph.state import (
    ActorAliasList,
    ActorItem,
    ActorResult,
    GraphState,
)


def _find_actors_alias(
    model, sentences: List[str], actors: List[ActorItem]
) -> List[ActorResult]:
    """Find aliases for each canonical actor from sentences."""
    if model is None or not actors:
        # Fallback: no aliases
        return [
            ActorResult(actor=a.actor, aliases=[], sentence_idx=a.sentence_idx)
            for a in actors
        ]

    structured_llm = model.with_structured_output(ActorAliasList)
    indexed_sents = "\n".join(f"{i}: {sent}" for i, sent in enumerate(sentences))
    actor_names = [item.actor for item in actors]

    system_prompt = """
    You are a Business Analyst AI specializing in requirement analysis.

    Your task is to identify aliases (alternative names or references) for each canonical actor
    based on a list of user story sentences.

    CRITICAL RULES:
    - An alias is a different term that refers to the SAME logical actor.
    - IMPORTANT: Aliases MUST ONLY be found in the "As a [actor]" or "As an [actor]" position at the START of sentences.
    - DO NOT extract words that appear elsewhere in the sentence (e.g., "user" in "log user activities" is NOT an alias).
    - Canonical actor names MUST NOT be listed as aliases of themselves.
    - Each alias MUST map to exactly one canonical actor.
    - Sentence indices are ZERO-BASED.
    - If an actor has no aliases, return an empty alias list for that actor.
    - ALL actor and alias names MUST be lowercase.
    - Do NOT invent aliases.
    - Do NOT explain your reasoning.
    - Return only structured data according to the output schema.
    
    Example:
    - "As a customer, I want to view user profiles" → actor is "customer", NOT "user"
    - "As a system, I want to log user activities" → actor is "system", NOT "user"
    """

    human_prompt = f"""
    Canonical actor names:
    {actor_names}

    User story sentences (with indices):
    {indexed_sents}

    For each canonical actor, find all aliases that appear in the "As a [X]" position at the START of sentences.
    DO NOT include words that appear elsewhere in the sentence body.
    """

    response: ActorAliasList = structured_llm.invoke(
        [("system", system_prompt), ("human", human_prompt)]
    )

    lookup = {actor.actor: actor.sentence_idx for actor in actors}
    result = []
    for mapping in response.mappings:
        if mapping.actor in lookup:
            result.append(
                ActorResult(
                    actor=mapping.actor,
                    aliases=mapping.aliases,
                    sentence_idx=lookup[mapping.actor],
                )
            )

    return result


def find_aliases_node(state: GraphState):
    """Find aliases for each canonical actor."""

    model = state.get("llm")
    sentences = state.get("sentences") or []
    actors = state.get("actors") or []
    actor_results = _find_actors_alias(model, sentences, actors)

    # DEBUG: in canonical actors và alias
    print("\n==== find_aliases_node ====")
    for ar in actor_results:
        print(f"canonical={ar.actor}, sentence_idx={ar.sentence_idx}")
        if ar.aliases:
            for al in ar.aliases:
                print(f"  alias={al.alias}, sentences={al.sentences}")

    return {"actor_results": actor_results}
