from __future__ import annotations

from typing import List, TypedDict

import json
import os
import re

import spacy
from dotenv import load_dotenv
from langchain.chat_models import BaseChatModel, init_chat_model
from langgraph.graph import END, START, StateGraph
from spacy.language import Language
from spacy.tokens import Token

from ..state import (
    ActorAliasList,
    ActorItem,
    ActorResult,
    CanonicalActorList,
    RpaState,
    UseCase,
    UsecaseRefinement,
    UsecaseRefinementResponse,
)

# =============================================================================
# GRAPH STATE
# =============================================================================


class GraphState(TypedDict, total=False):
    llm: BaseChatModel
    requirement_text: str
    sentences: List[str]

    # Actor pipeline
    raw_actors: List[ActorItem]  # After regex extraction
    actors: List[ActorItem]  # After synonym check
    actor_results: List[ActorResult]  # After alias detection

    # UseCase pipeline
    raw_usecases: dict  # {sentence_idx: [use_case_names]}
    refined_usecases: List[UsecaseRefinement]

    # Final outputs (for compatibility with main_graph)
    use_cases: List[UseCase]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _get_model():
    load_dotenv()
    model_name = os.getenv("LLM_MODEL", "gpt-4o-mini")
    if not os.getenv("OPENAI_API_KEY"):
        return None
    return init_chat_model(model_name, model_provider="openai")


def _get_nlp():
    return spacy.load("en_core_web_lg")


# =============================================================================
# USECASE FINDER FUNCTIONS
# =============================================================================


def _find_main_verb(xcomp: Token) -> Token:
    """Find the main verb from an xcomp token."""
    # Handle case: "be Adjective to [VERB]"
    for child in xcomp.children:
        if child.dep_ == "acomp":
            for subchild in child.children:
                if subchild.dep_ == "xcomp" and subchild.pos_ == "VERB":
                    return subchild
            for subchild in child.children:
                if subchild.dep_ == "prep" and subchild.text.lower() == "of":
                    for grandchild in subchild.children:
                        if (
                            grandchild.dep_ in {"pcomp", "pobj"}
                            and grandchild.pos_ == "VERB"
                        ):
                            return grandchild

    # Handle case: "be V_3 to [VERB]"
    has_auxpass = any(
        child.dep_ == "auxpass" and child.text == "be" for child in xcomp.children
    )
    if has_auxpass:
        for child in xcomp.children:
            if child.dep_ == "xcomp" and child.pos_ == "VERB":
                return child

    return xcomp


def _get_verb_phrase(verb: Token) -> str:
    """Extract verb phrase from a verb token."""
    exclude_tokens = set()
    has_dobj = any(child.dep_ == "dobj" for child in verb.children)

    for child in verb.children:
        if child.dep_ == "conj":
            if not has_dobj:
                exclude_list = [
                    subchild for subchild in child.subtree if subchild.dep_ != "dobj"
                ]
            else:
                exclude_list = list(child.subtree)
            exclude_tokens.update(exclude_list)
        if child.dep_ == "cc":
            exclude_tokens.add(child)

    tokens = [t for t in verb.subtree if t not in exclude_tokens]
    tokens = tokens[tokens.index(verb) :]
    tokens = sorted(tokens, key=lambda t: t.i)

    # Remove "so that" clause if present
    cut_index = -1
    for i, token in enumerate(tokens):
        if (
            token.text.lower() == "so"
            and i + 1 < len(tokens)
            and tokens[i + 1].text.lower() == "that"
        ):
            cut_index = i
            break

    if cut_index != -1:
        tokens = tokens[:cut_index]

    EXCLUDE_DEPS = {"poss", "det", "nummod", "quantmod"}
    relevant_tokens = []

    for token in tokens[1:]:
        if token.dep_ == "dobj" or token.head.dep_ == "dobj":
            if token.dep_ not in EXCLUDE_DEPS:
                relevant_tokens.append(token)
        elif token.dep_ in {"prep", "pobj"} or token.head.dep_ in {"prep", "pobj"}:
            if token.dep_ not in EXCLUDE_DEPS:
                relevant_tokens.append(token)
        elif token.dep_ == "prt":
            relevant_tokens.append(token)
        elif token.dep_ in {"acomp", "advmod"}:
            relevant_tokens.append(token)
        elif token.dep_ in {"compound", "amod"}:
            relevant_tokens.append(token)

    tokens = [tokens[0]] + sorted(relevant_tokens, key=lambda t: t.i)
    result = [token.text for token in tokens]

    return " ".join(result)


def _get_all_conj(verb: Token) -> List[Token]:
    """Find all conjunctions of the root verb."""
    result = []
    for child in verb.children:
        if child.dep_ == "conj" and child.pos_ == "VERB":
            result.append(child)
            result.extend(_get_all_conj(child))
    return result


def _find_usecases_nlp(nlp: Language, sentences: List[str]) -> dict:
    """Extract usecases from all sentences using NLP pattern 'want to [verb]'."""
    res = {}

    for i, sent in enumerate(sentences):
        doc = nlp(sent)
        for token in doc:
            if token.lemma_ == "want":
                for children in token.children:
                    if children.dep_ == "xcomp" and children.pos_ in {"VERB", "AUX"}:
                        # Exclude V-ing case
                        if children.tag_ == "VBG":
                            continue

                        main_verb = _find_main_verb(children)
                        verb_phrase = _get_verb_phrase(main_verb)

                        if str(i) not in res:
                            res[str(i)] = []
                        res[str(i)].append(verb_phrase)

                        # Find ALL conj verbs (recursive)
                        all_conj_verbs = _get_all_conj(main_verb)
                        for conj in all_conj_verbs:
                            conj_verb_phrase = _get_verb_phrase(conj)
                            res[str(i)].append(conj_verb_phrase)

    return res


# =============================================================================
# GRAPH NODES
# =============================================================================


# def pre_process_node(state: GraphState):
#     """Preprocess input text and split into sentences."""
#     # text = state.get("requirement_text") or state.get("input_text") or ""
#     # text = re.sub(r"\s+", " ", text).strip()

#     sentences = state.get("requirement_text").split("\n")

#     # Create tasks (for compatibility)
#     # tasks = [TaskItem(id=i, text=sent) for i, sent in enumerate(sentences)]

#     return {
#         "sentences": sentences,
#         # "tasks": tasks,
#     }


def find_actors_node(state: GraphState):
    """Extract actors using regex pattern from user stories."""

    def _find_actors_regex(sentences: List[str]) -> List[ActorItem]:
        """Extract actors from user stories using regex pattern 'As a/an/the [actor]'."""
        pattern = r"As\s+(?:a|an|the)\s+([^,]+)"
        actor_occurrences = {}

        for i, sent in enumerate(sentences):
            match = re.search(pattern, sent, re.IGNORECASE)
            if match:
                actor = match.group(1).strip().lower()
                if actor not in actor_occurrences:
                    actor_occurrences[actor] = []
                actor_occurrences[actor].append(i)

        return [
            ActorItem(actor=actor, sentence_idx=sent_indices)
            for actor, sent_indices in actor_occurrences.items()
        ]

    sentences = state.get("sentences") or []
    raw_actors = _find_actors_regex(sentences)
    return {"raw_actors": raw_actors}


def synonym_check_node(state: GraphState):
    """Remove synonymous actors using LLM."""

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

    model = state.get("llm")
    raw_actors = state.get("raw_actors") or []
    actors = _synonym_actors_check(model, raw_actors)
    return {"actors": actors}


def find_aliases_node(state: GraphState):
    """Find aliases for each canonical actor."""

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

    Rules:
    - An alias is a different term that refers to the SAME logical actor.
    - Canonical actor names MUST NOT be listed as aliases of themselves.
    - Each alias MUST map to exactly one canonical actor.
    - Aliases must be explicitly present in the provided sentences.
    - Sentence indices are ZERO-BASED.
    - If an actor has no aliases, return an empty alias list for that actor.
    - ALL actor and alias names MUST be lowercase.
    - Do NOT invent aliases.
    - Do NOT explain your reasoning.
    - Return only structured data according to the output schema.
    """

        human_prompt = f"""
    Canonical actor names:
    {actor_names}

    User story sentences (with indices):
    {indexed_sents}

    For each canonical actor, find all aliases used in the sentences above and list the sentence indices where each alias appears.
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

    model = state.get("llm")
    sentences = state.get("sentences") or []
    actors = state.get("actors") or []
    actor_results = _find_actors_alias(model, sentences, actors)
    return {"actor_results": actor_results}


def find_usecases_node(state: GraphState):
    """Extract use cases using NLP pattern."""
    nlp = _get_nlp()
    sentences = state.get("sentences") or []
    raw_usecases = _find_usecases_nlp(nlp, sentences)
    return {"raw_usecases": raw_usecases}


def refine_usecases_node(state: GraphState):
    """Refine extracted use cases using LLM."""

    def _refine_usecases(
        model: BaseChatModel, sentences: List[str], usecases: dict
    ) -> List[UsecaseRefinement]:
        """Use LLM to refine and complete extracted use cases."""
        if model is None or not usecases:
            # Fallback: return as-is without refinement
            return [
                UsecaseRefinement(
                    sentence_idx=int(idx),
                    original=ucs,
                    refined=ucs,
                    added=[],
                    reasoning=None,
                )
                for idx, ucs in usecases.items()
            ]

        sents_text = "\n".join([f'{i}: "{sent}"' for i, sent in enumerate(sentences)])
        usecase_text = json.dumps(usecases, indent=2, ensure_ascii=False)

        system_prompt = """You are an expert in UML use case modeling and software requirements analysis.

    Your task is to refine extracted use cases so that they can be used DIRECTLY
    as use case names in a UML Use Case Diagram.

    OBJECTIVES:
    1. REVIEW
    - Verify whether each extracted use case represents a true user goal
        suitable for a UML use case.

    2. REFINE
    - Rewrite each use case into a concise UML-style use case name.
    - Use clear verb–object structure (verb + noun phrase).
    - Remove all implementation details, conditions, and variations.

    3. COMPLETE
    - Add missing use cases ONLY if they are explicitly stated in the sentence.
    - Do NOT infer system behavior or business rules not written in the text.

    UML USE CASE NAMING RULES:
    - Start with an action verb in base form (e.g., Browse, View, Create, Update, Manage).
    - Represent exactly ONE goal per use case.
    - Be short and abstract (typically 2–5 words).
    - Use business-level actions, not technical steps.
    - Use lowercase for all use case names.
    - Do NOT include:
    - "so that", purposes, or outcomes
    - constraints or options (e.g., payment methods, device types)
    - UI actions (click, tap, select) unless explicitly stated

    GOOD UML USE CASE NAMES:
    - browse products
    - view order history
    - checkout order
    - manage account
    - update profile

    BAD USE CASE NAMES:
    - browse products by category for easier searching
    - checkout using multiple payment methods
    - view order history to track purchases

    IMPORTANT CONSTRAINTS:
    - Do NOT merge multiple user goals into one use case.
    - Do NOT invent use cases not explicitly present in the sentence.
    - If a sentence does not contain a valid UML use case, return empty lists.

    OUTPUT REQUIREMENTS:
    - Follow the provided structured output schema exactly.
    - Populate:
    - original: extracted use cases
    - refined: UML-ready use case names
    - added: only missing but explicitly stated use cases
    - Provide brief reasoning for any change or addition.
    """

        human_prompt = f"""## User Stories:
    {sents_text}

    ## Extracted Use Cases (by sentence index):
    {usecase_text}

    TASK:
    - Refine each extracted use case into a UML-ready use case name.
    - Ensure each use case can be placed directly inside a Use Case Diagram.
    - Add missing use cases ONLY if they are explicitly stated in the sentence.
    - If no valid UML use case exists, return empty refined and added lists.

    Return the result strictly in the structured output format.
    """

        structured_llm = model.with_structured_output(UsecaseRefinementResponse)
        response: UsecaseRefinementResponse = structured_llm.invoke(
            [("system", system_prompt), ("human", human_prompt)]
        )

        return response.refinements

    model: BaseChatModel = state.get("llm")
    sentences = state.get("sentences") or []
    raw_usecases = state.get("raw_usecases") or {}
    refined_usecases = _refine_usecases(model, sentences, raw_usecases)
    return {"refined_usecases": refined_usecases}


def finalize_node(state: GraphState):
    """Format final output for compatibility with main_graph."""

    def _format_usecase_output(
        refined_usecases: List[UsecaseRefinement],
        actor_results: List[ActorResult],
        sentences: List[str],
    ) -> List[UseCase]:
        """Format refined use cases into UseCase objects with participating actors."""
        formatted_usecases = []

        for usecase in refined_usecases:
            usecase_list = usecase.refined + usecase.added
            actors_filter = set()

            for actor in actor_results:
                # Check if actor appears in this sentence
                if usecase.sentence_idx in actor.sentence_idx:
                    actors_filter.add(actor.actor)
                # Check if any alias appears in this sentence
                for actor_alias in actor.aliases:
                    if usecase.sentence_idx in actor_alias.sentences:
                        actors_filter.add(actor.actor)
                        break

            sentence_text = (
                sentences[usecase.sentence_idx]
                if usecase.sentence_idx < len(sentences)
                else ""
            )

            for item in usecase_list:
                formatted_usecases.append(
                    UseCase(
                        name=item,
                        participating_actors=list(actors_filter),
                        sentence_id=usecase.sentence_idx,
                        sentence=sentence_text,
                        relationships=[],
                    )
                )

        return formatted_usecases

    sentences = state.get("sentences") or []
    actor_results = state.get("actor_results") or []
    refined_usecases = state.get("refined_usecases") or []

    # Format use cases with participating actors
    use_cases = _format_usecase_output(refined_usecases, actor_results, sentences)
    return {
        "use_cases": use_cases,
    }


# =============================================================================
# GRAPH BUILDER
# =============================================================================


def build_rpa_graph():
    """
    RPA Graph (Requirement Processing Agent):
    - Extract actors using regex pattern "As a [actor]"
    - Remove synonymous actors with LLM
    - Find actor aliases with LLM
    - Extract use cases using NLP "want to [verb]" pattern
    - Refine use cases with LLM
    """

    workflow = StateGraph(GraphState)

    # Add nodes
    # workflow.add_node("pre_process", pre_process_node)
    workflow.add_node("find_actors", find_actors_node)
    workflow.add_node("synonym_check", synonym_check_node)
    workflow.add_node("find_aliases", find_aliases_node)
    workflow.add_node("find_usecases", find_usecases_node)
    workflow.add_node("refine_usecases", refine_usecases_node)
    workflow.add_node("finalize", finalize_node)

    # Define edges
    # workflow.add_edge(START, "pre_process")
    # workflow.add_edge("pre_process", "find_actors")
    workflow.add_edge(START, "find_actors")
    workflow.add_edge("find_actors", "synonym_check")
    workflow.add_edge("synonym_check", "find_aliases")
    workflow.add_edge("find_aliases", "find_usecases")
    workflow.add_edge("find_usecases", "refine_usecases")
    workflow.add_edge("refine_usecases", "finalize")
    workflow.add_edge("finalize", END)

    return workflow.compile()


def run_rpa(requirement_text: str) -> RpaState:
    """Run the RPA graph and return results."""
    sentences = requirement_text.split("\n")
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
