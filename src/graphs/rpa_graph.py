import math
from typing import List, TypedDict

import json
import os
import re
from pathlib import Path

import spacy
from dotenv import load_dotenv
from langchain.chat_models import BaseChatModel, init_chat_model
from langgraph.graph import END, START, StateGraph
from langchain_openai import OpenAIEmbeddings
from spacy.language import Language
from spacy.tokens import Token
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from ..state import (
    ActorAliasList,
    ActorItem,
    ActorResult,
    CanonicalActorList,
    NormalizedUserStoriesResponse,
    RefineClusteringResponse,
    RpaState,
    UseCase,
    UseCaseNamingResponse,
    UseCaseRelationship,
    UseCaseDomainGroupingResponse,
    UseCaseRelationshipResponse,
    UsecaseRefinement,
    UsecaseRefinementResponse,
    UserStoryItem,
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

    # Goals pipeline (sau gom cụm thành Usecase)
    raw_goals: dict  # {sentence_idx: [goal_names]}
    refined_goals: List[UsecaseRefinement]

    # Domain grouping & relationships (3-step pipeline - commented out)
    # domain_groupings: dict  # {domain_name: [use_case_names]}
    # within_domain_relationships: List[dict]  # Relationships found within domains
    # cross_domain_relationships: List[dict]  # Relationships found across domains
    # One-step include/extend (from find_include_extend_node)
    include_extend_relationships: List[
        dict
    ]  # [{source_use_case, type, target_use_case}]

    # Clustering results (đặt tên Usecase ở name_usecases_node)
    user_story_clusters: List[
        dict
    ]  # [{cluster_id, user_stories: [{actor, goal, sentence_idx, original_sentence}]}]

    # Final outputs
    use_cases: List[UseCase]  # Generated from clustering + LLM naming

    # Control flags
    grouping_done: bool


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def should_continue_to_clustering(state: GraphState):
    # Kiểm tra xem cả 2 nhánh đã đổ data về chưa
    if state.get("grouping_done"):
        return "continue"
    return "stop"


def _get_model():
    load_dotenv()
    model_name = os.getenv("LLM_MODEL", "gpt-5-mini")
    if not os.getenv("OPENAI_API_KEY"):
        return None
    return init_chat_model(model_name, model_provider="openai")


def _get_nlp():
    return spacy.load("en_core_web_lg")


# =============================================================================
# NORMALIZE SENTENCES (chuẩn hóa thành As a/an/the <actor>, I want to <goal> so that)
# =============================================================================


# def normalize_sentences_node(state: GraphState):
#     """Chuẩn hóa mỗi câu user story thành dạng chuẩn:
#     'As a/an/the <actor>, I want to <goal> so that <benefit>'.

#     Đặc biệt: câu dạng "As a X, I want NOUN to <goal> so that ..." được chuyển thành
#     "As a NOUN, I want to <goal> so that ..." (actor thực hiện goal là NOUN).
#     """
#     raw = state.get("sentences") or []
#     if not raw and state.get("requirement_text"):
#         raw = state["requirement_text"].strip().split("\n")

#     # Light cleanup: strip, remove empty, collapse spaces
#     cleaned: List[str] = []
#     for line in raw:
#         s = line.strip()
#         if not s:
#             continue
#         s = re.sub(r"\s+", " ", s)
#         cleaned.append(s)

#     if not cleaned:
#         return {"sentences": []}

#     llm = state.get("llm")
#     if llm is None:
#         return {"sentences": cleaned}

#     # Only normalize sentences that are NOT already in standard form.
#     # Các câu đã đúng dạng "As a/an/the <actor>, I want to <goal> [so that <benefit>]" sẽ được giữ nguyên.

#     standard_pattern = re.compile(
#         r"^As\s+(?:a|an|the)\s+.+?,\s*I\s+want\s+to\s+.+?(?:\s+so\s+that\s+.+)?$",
#         re.IGNORECASE,
#     )

#     def _is_standard_form(sentence: str) -> bool:
#         return bool(standard_pattern.match(sentence))

#     standard_indices = {}
#     nonstandard_sentences: List[str] = []
#     nonstandard_map: List[int] = []  # map local index -> original index

#     for idx, sent in enumerate(cleaned):
#         if _is_standard_form(sent):
#             standard_indices[idx] = sent
#         else:
#             nonstandard_map.append(idx)
#             nonstandard_sentences.append(sent)

#     # Nếu tất cả câu đều đã đúng form thì trả về nguyên trạng
#     if not nonstandard_sentences:
#         return {"sentences": cleaned}

#     structured_llm = llm.with_structured_output(NormalizedUserStoriesResponse)
#     indexed = "\n".join(f"{i}: {s}" for i, s in enumerate(nonstandard_sentences))

#     system_prompt = """**System Prompt:**
# You are a Business Analyst AI. Your task is to normalize user story sentences into the STANDARD form:

# **Standard form:** "As a/an/the <actor>, I want to <goal> [so that <benefit>]"

# RULES:
# 1. **Actor Redirection**: If the input is "As a X, I want [Actor Y] to <goal>", the true performer is Y. Output: "As a/an/the [actor y], I want to <goal>...".
#    - Example: "As an ifa, I want team managers to add players" → "As a team manager, I want to add players".

# 2. **Grammar & Case**: Use lowercase for actor names. Ensure "a/an/the" is grammatically correct. Keep the goal and benefit in natural language.

# 3. **Handle Missing Benefits**: If the input does not provide a "so that" or "in order to" clause, DO NOT invent one. End the sentence after the <goal>.
#    - *RATIONALE*: This prevents False Positives in requirement analysis.

# 4. **Relationship Awareness**: If the input mentions an "include" or "extend" relationship from a Use Case diagram, treat the primary action as the goal.
#    - Example: "As a user, I want to login (including password encryption)" → "As a user, I want to login with password encryption".

# 5. **Strict Output Mapping**:
#    - Return exactly one normalized sentence per input.
#    - Maintain the EXACT original order.
#    - Output format: A simple list of strings. Do not add introductory or concluding remarks."""

#     human_prompt = f"""Normalize each of the following user story sentences into the standard form "As a/an/the <actor>, I want to <goal> so that <benefit>".

# Input sentences (index: sentence):
# {indexed}

# Return a list of normalized sentences: one string per input, in the same order. Output list length must be {len(nonstandard_sentences)}.
# """

#     response: NormalizedUserStoriesResponse = structured_llm.invoke(
#         [("system", system_prompt), ("human", human_prompt)]
#     )

#     normalized_partial = list(response.sentences)

#     # Đảm bảo độ dài khớp số câu non-standard; nếu thiếu/thừa thì fallback một phần
#     if len(normalized_partial) < len(nonstandard_sentences):
#         normalized_partial += nonstandard_sentences[len(normalized_partial) :]
#     elif len(normalized_partial) > len(nonstandard_sentences):
#         normalized_partial = normalized_partial[: len(nonstandard_sentences)]

#     # Ghép lại theo thứ tự gốc: câu standard giữ nguyên, câu non-standard dùng bản normalized
#     final_sentences: List[str] = []
#     ns_idx = 0
#     ns_set = set(nonstandard_map)

#     for i in range(len(cleaned)):
#         if i in standard_indices:
#             final_sentences.append(standard_indices[i])
#         elif i in ns_set:
#             final_sentences.append(normalized_partial[ns_idx])
#             ns_idx += 1
#         else:
#             # Fallback an toàn (không nên xảy ra): dùng câu gốc
#             final_sentences.append(cleaned[i])

#     # DEBUG: normalized sentence list (kept only high-level marker)
#     print("\n==== normalize_sentences_node ====")
#     # If you need full details for debugging, uncomment the loop below.
#     # for i, s in enumerate(final_sentences):
#     #     print(f"{i}: {s}")

#     return {"sentences": final_sentences}


# =============================================================================
# GOALS FINDER FUNCTIONS (extract user goals; Usecase đặt tên sau khi gom cụm)
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


def _find_goals_nlp(nlp: Language, sentences: List[str]) -> dict:
    """Extract user goals from all sentences using NLP pattern 'want to [verb]'."""
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

# Keywords that identify system/infrastructure actors to filter out
SYSTEM_ACTOR_KEYWORDS = (
    "system",
    "software",
    "application",
    "platform",
    "backend",
    "server",
)

SAFE_SYSTEM_ACTORS = {
    "system administrator",
    "system admin",
    "systems analyst",
    "system analyst",
}


def _is_system_actor(actor_name: str) -> bool:
    """Return True if the actor is system-related and should be filtered out."""
    name_lower = actor_name.lower().strip()
    if name_lower in SAFE_SYSTEM_ACTORS:
        return False
    return any(kw in name_lower for kw in SYSTEM_ACTOR_KEYWORDS)


def find_actors_node(state: GraphState):
    """Extract actors using regex pattern from user stories.
    System-related actors (e.g. 'As a system', 'system operator') are filtered out.
    """

    def _find_actors_regex(sentences: List[str]) -> List[ActorItem]:
        """Extract actors from user stories using regex pattern 'As a/an/the [actor]'."""
        pattern = r"As\s+(?:a|an|the)\s+([^,]+)"
        actor_occurrences = {}

        for i, sent in enumerate(sentences):
            match = re.search(pattern, sent, re.IGNORECASE)
            if match:
                actor = match.group(1).strip().lower()
                if _is_system_actor(actor):
                    continue
                if actor not in actor_occurrences:
                    actor_occurrences[actor] = []
                actor_occurrences[actor].append(i)

        return [
            ActorItem(actor=actor, sentence_idx=sent_indices)
            for actor, sent_indices in actor_occurrences.items()
        ]

    sentences = state.get("sentences") or []
    raw_actors = _find_actors_regex(sentences)

    # DEBUG: raw actors (kept only high-level marker)
    print("\n==== find_actors_node ====")
    # If you need full details for debugging, uncomment the loop below.
    # for a in raw_actors:
    #     print(f"actor={a.actor}, sentence_idx={a.sentence_idx}")

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

    model = state.get("llm")
    sentences = state.get("sentences") or []
    actors = state.get("actors") or []
    actor_results = _find_actors_alias(model, sentences, actors)

    # DEBUG: canonical actors and aliases (kept only high-level marker)
    print("\n==== find_aliases_node ====")
    # If you need full details for debugging, uncomment the loop below.
    # for ar in actor_results:
    #     print(f"canonical={ar.actor}, sentence_idx={ar.sentence_idx}")
    #     if ar.aliases:
    #         for al in ar.aliases:
    #             print(f"  alias={al.alias}, sentences={al.sentences}")

    return {"actor_results": actor_results}


def find_goals_node(state: GraphState):
    """Extract user goals using NLP pattern (want to [verb])."""
    nlp = _get_nlp()
    sentences = state.get("sentences") or []
    raw_goals = _find_goals_nlp(nlp, sentences)
    return {"raw_goals": raw_goals}


def refine_goals_node(state: GraphState):
    """Refine extracted goals using LLM (chưa phải Usecase; đặt tên Usecase sau khi gom cụm)."""

    def _refine_goals(
        model: BaseChatModel, sentences: List[str], goals: dict
    ) -> List[UsecaseRefinement]:
        """Use LLM to refine and complete extracted goals."""
        if model is None or not goals:
            # Fallback: return as-is without refinement
            return [
                UsecaseRefinement(
                    sentence_idx=int(idx),
                    original=ucs,
                    refined=ucs,
                    added=[],
                    reasoning=None,
                )
                for idx, ucs in goals.items()
            ]

        # Chỉ dùng phần trước "so that" khi gửi cho LLM, tránh để benefit ảnh hưởng việc refine goal
        def _strip_benefit_clause(sentence: str) -> str:
            if not sentence:
                return sentence
            lower = sentence.lower()
            idx = lower.find("so that")
            if idx == -1:
                return sentence
            return sentence[:idx].strip()

        sentences_no_benefit = [_strip_benefit_clause(sent) for sent in sentences]

        sents_text = "\n".join(
            [f'{i}: "{sent}"' for i, sent in enumerate(sentences_no_benefit)]
        )
        goal_text = json.dumps(goals, indent=2, ensure_ascii=False)

        system_prompt = """You are an expert in UML use case modeling and software requirements analysis.

    Your task is to refine extracted user GOALS (from "I want to ...") so that they are
    clear, verb–object phrases. These will later be clustered and named as Use Cases.

    OBJECTIVES:
    1. REVIEW
    - Verify whether each extracted item represents a true user goal.

    2. REFINE
    - Rewrite each goal into a concise verb–object phrase (verb + noun phrase).
    - Remove all implementation details, conditions, and variations.

    3. COMPLETE
    - Add missing goals ONLY if they are explicitly stated in the sentence.
    - Do NOT infer system behavior or business rules not written in the text.

    GOAL PHRASE RULES (sau sẽ gom cụm để đặt tên Usecase):
    - Start with an action verb in base form (e.g., Browse, View, Create, Update, Manage).
    - Represent exactly ONE goal per phrase.
    - Be short and abstract (typically 2–5 words).
    - Use business-level actions, not technical steps.
    - Use lowercase for all goal phrases.
    - Do NOT include:
    - "so that", purposes, or outcomes
    - constraints or options (e.g., payment methods, device types)
    - UI actions (click, tap, select) unless explicitly stated

    GOOD GOAL PHRASES:
    - browse products
    - view order history
    - checkout order
    - manage account
    - update profile

    BAD PHRASES:
    - browse products by category for easier searching
    - checkout using multiple payment methods
    - view order history to track purchases

    IMPORTANT CONSTRAINTS:
    - Do NOT merge multiple user goals into one phrase.
    - Do NOT invent goals not explicitly present in the sentence.
    - If a sentence does not contain a valid goal, return empty lists.

    OUTPUT REQUIREMENTS:
    - Follow the provided structured output schema exactly.
    - Populate:
    - original: extracted goals
    - refined: refined goal phrases
    - added: only missing but explicitly stated goals
    - Provide brief reasoning for any change or addition.
    """

        human_prompt = f"""## User Stories:
    {sents_text}

    ## Extracted Goals (by sentence index):
    {goal_text}

    TASK:
    - Refine each extracted goal into a clear verb–object phrase.
    - Add missing goals ONLY if they are explicitly stated in the sentence.
    - If no valid goal exists, return empty refined and added lists.

    Return the result strictly in the structured output format.
    """

        structured_llm = model.with_structured_output(UsecaseRefinementResponse)
        response: UsecaseRefinementResponse = structured_llm.invoke(
            [("system", system_prompt), ("human", human_prompt)]
        )

        return response.refinements

    model: BaseChatModel = state.get("llm")
    sentences = state.get("sentences") or []
    raw_goals = state.get("raw_goals") or {}
    refined_goals = _refine_goals(model, sentences, raw_goals)

    # DEBUG: in goals trước và sau refine
    print("\n==== refine_goals_node ====")
    # print("raw_goals:", json.dumps(raw_goals, indent=2, ensure_ascii=False))
    # for item in refined_goals:
    #     print(f"- sentence_idx={item.sentence_idx}")
    #     print(f"  original={item.original}")
    #     print(f"  refined={item.refined}")
    #     print(f"  added={item.added}")

    return {"refined_goals": refined_goals}


def _extract_benefit(sentence: str) -> str:
    """Lấy phần benefit (sau 'so that') của user story. Nếu không có thì trả về ''."""
    if not sentence or "so that" not in sentence:
        return ""
    parts = sentence.split("so that", 1)
    return parts[-1].strip() if len(parts) > 1 else ""


def grouping_node(state: GraphState):
    # Skip if already processed (prevent duplicate execution from fan-in)
    if state.get("grouping_done"):
        return {}

    actor_results = state.get("actor_results")
    refined_goals = state.get("refined_goals")

    # Skip if either branch hasn't completed yet
    if not actor_results or not refined_goals:
        return {}

    sentences = state.get("sentences") or []

    input_pairs = []  # For embedding: "actor goal [benefit]"
    input_details = []  # Detailed info for each pair (actor, goal, benefit, ...)
    actors_lookup = {}

    # Build lookup: sentence_idx -> list of actors
    for actor in actor_results:
        for idx in actor.sentence_idx:  # Indices where canonical name appears
            if idx not in actors_lookup:
                actors_lookup[idx] = []
            if actor not in actors_lookup[idx]:  # Avoid duplicates
                actors_lookup[idx].append(actor)

        for alias in actor.aliases:
            for sentence in alias.sentences:  # Indices where alias appears
                if sentence not in actors_lookup:
                    actors_lookup[sentence] = []
                if actor not in actors_lookup[sentence]:  # Avoid duplicates
                    actors_lookup[sentence].append(actor)

    for item in refined_goals:
        goals = item.refined + item.added
        orig_sentence = (
            sentences[item.sentence_idx] if item.sentence_idx < len(sentences) else ""
        )
        benefit = _extract_benefit(orig_sentence)

        for goal in goals:
            # Use item.sentence_idx (int) instead of actor.sentence_idx
            if item.sentence_idx in actors_lookup:
                for actor in actors_lookup[item.sentence_idx]:
                    # String for embedding: actor + goal + benefit (benefit giúp gom cụm theo ngữ nghĩa)
                    embed_text = f"{actor.actor} {goal}"
                    if benefit:
                        embed_text += f" {benefit}"
                    input_pairs.append(embed_text)
                    # Detailed info (gắn benefit cùng actor, goal)
                    input_details.append(
                        {
                            "actor": actor.actor,
                            "goal": goal,
                            "benefit": benefit,
                            "sentence_idx": item.sentence_idx,
                            "original_sentence": orig_sentence,
                        }
                    )

    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large")

    silhouette_scores = []
    # K = range(2, len(input_pairs))
    # K_min = max(3, math.floor(len(input_pairs) / 10))
    # K_max = math.ceil(len(input_pairs) / 3)

    # K = range(K_min, K_max + 1)

    print("\n==== grouping_node ====")

    K = range(3, 11)
    X = np.array(embeddings_model.embed_documents(input_pairs))

    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        score = silhouette_score(X, labels)
        silhouette_scores.append(score)
        print(f"Số cụm k={k}, silhouette={score:.4f}")

    best_k = K[np.argmax(silhouette_scores)]
    print(f"\n=> Số cụm hợp lý nhất dựa trên ngữ nghĩa là: {best_k}")

    kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    final_labels = kmeans_final.fit_predict(X)
    clusters = {}
    for i, label in enumerate(final_labels):
        if label not in clusters:
            clusters[label] = []
        # Store detailed info instead of just string
        clusters[label].append(input_details[i])

    # print(f"\n--- CHI TIẾT PHÂN CỤM (K={best_k}) ---")
    # for label, items in sorted(clusters.items()):
    #     print(f"\nCụm {label + 1}:")
    #     for item in items:
    #         print(f"  - [{item['actor']}] {item['goal']}")
    #         if item.get("benefit"):
    #             print(f"    Benefit: {item['benefit']}")
    #         print(f"    Original: {item['original_sentence']}")

    # Convert clusters dict to list format for state
    clusters_list = [
        {"cluster_id": label, "user_stories": items}
        for label, items in sorted(clusters.items())
    ]

    # DEBUG: in input_details và clusters cuối cùng
    # print("\n==== grouping_node ====")
    # print(">>> input_details:")
    # for d in input_details:
    #     print(f"  - actor={d['actor']}, goal={d['goal']}, sent={d['sentence_idx']}")
    #     print(f"    original={d['original_sentence']}")
    # print("\n>>> clusters_list:")
    # for c in clusters_list:
    #     print(f"Cluster {c['cluster_id']}:")
    #     for s in c["user_stories"]:
    #         print(f"  - [{s['actor']}] {s['goal']} (sent={s['sentence_idx']})")
    #         print(f'    "{s["original_sentence"]}"')

    return {"grouping_done": True, "user_story_clusters": clusters_list}


def refine_clustering_node(state: GraphState):
    """
    Refine K-Means clusters using LLM: split mixed clusters, merge similar ones,
    move misplaced items. Output format is unchanged for name_usecases_node.
    """
    llm = state.get("llm")
    clusters = state.get("user_story_clusters")

    if not clusters or not llm:
        return {}

    # Flatten to list of items with (cluster_id, story_dict) for prompt
    flat_items: List[dict] = []
    for c in clusters:
        for story in c["user_stories"]:
            flat_items.append(
                {
                    "cluster_id": c["cluster_id"],
                    "sentence_idx": story["sentence_idx"],
                    "actor": story["actor"],
                    "goal": story["goal"],
                    "original_sentence": story["original_sentence"],
                }
            )

    clusters_text = ""
    for c in clusters:
        cid = c["cluster_id"]
        stories = c["user_stories"]
        clusters_text += f"\n### Cluster {cid}:\n"
        for s in stories:
            clusters_text += f"  - [{s['actor']}] {s['goal']}\n"
            clusters_text += f'    "{s["original_sentence"]}"\n'

    item_lines = "\n".join(
        f"  - sentence_idx={x['sentence_idx']} | actor={x['actor']} | goal={x['goal']}"
        for x in flat_items
    )

    n_clusters = len(clusters)
    n_items = len(flat_items)
    # Allow at most +20% more clusters than original (ceil). Prefer merging.
    cluster_budget = math.ceil(n_clusters * 1.2) if n_clusters > 0 else 0

    refine_clustering_system_prompt = """You are a senior Software Architect and UML specialist. 
    Your task is to refine clusters of user stories into distinct, valid UML Use Cases.

    CRITICAL PRINCIPLE:
    A Use Case represents a single, discrete goal for an actor. 
    - IDENTITY vs. ADMINISTRATION: 'Logging in' (Identity) is NOT the same goal as 'Managing User Accounts' (Administration).
    - CRUD vs. TRANSACTION: 'Managing Products' (Inventory) is NOT the same as 'Browsing Products' (Shopping).
    
    You must separate actions that belong to different architectural layers or business contexts, even if they involve the same data entity (like 'User')."""

    human_prompt = f"""Current clusters (count={n_clusters}, items={n_items}):
    {clusters_text}

    All items to re-cluster:
    {item_lines}

    STRICT RULES FOR CLUSTERING:
    1. **Goal-Oriented Clustering:** Group by the primary objective. 
       - AUTHENTICATION: Login, Logout, Register, Reset Password belong together.
       - ADMIN/CRUD: Create/Edit/Delete/List [Entity] (e.g., Manage Accounts, Manage Products) MUST be in their own separate clusters.
    2. **Entity vs. Action:** Do NOT group just because they share the word 'User'. 
       - [User] 'Log in' is a Security action.
       - [Admin] 'Delete User' is a Management action. 
       - These MUST be in DIFFERENT clusters.
    3. **Actor Context:** While different actors can share a Use Case (e.g., both Teacher and Student 'View Schedule'), if the action changed context (e.g., Student 'View Course' vs Admin 'Create Course'), they must be split.
    4. **Granularity:** If a cluster has more than 5-6 items that seem slightly different, split them into more specific Use Cases (e.g., 'View Reports' vs 'Export Reports').
    5. **Cluster Budget (important):** Total clusters after refinement MUST NOT exceed {cluster_budget} (≈ +20% over the current {n_clusters}). Prefer MERGING similar clusters; only SPLIT when goals are clearly unrelated.
    6. **Avoid tiny splits:** Do not create a new cluster with a single item unless it is semantically incompatible with every other item.

    Output exactly one entry per line, preserving the original strings and providing a logical 'target_cluster_id'."""

    structured_llm = llm.with_structured_output(RefineClusteringResponse)
    response = structured_llm.invoke(
        [("system", refine_clustering_system_prompt), ("human", human_prompt)]
    )

    # Build lookup (sentence_idx, actor, goal) -> target_cluster_id (RefinedClusterItem.usecase = goal text)
    def _item_key(i):
        return (i.sentence_idx, i.actor, i.usecase)

    assign = {_item_key(i): i.target_cluster_id for i in response.items}

    # Regroup by target_cluster_id, keeping full story dicts; fallback to original cluster_id if missing
    new_clusters: dict = {}
    for c in clusters:
        for story in c["user_stories"]:
            k = (story["sentence_idx"], story["actor"], story["goal"])
            tid = assign.get(k, c["cluster_id"])
            if tid not in new_clusters:
                new_clusters[tid] = []
            new_clusters[tid].append(story)

    clusters_list = [
        {"cluster_id": cid, "user_stories": items}
        for cid, items in sorted(new_clusters.items())
    ]

    print("\n==== refined clusters (after LLM) ====")
    # for c in clusters_list:
    #     print(f"\nCluster {c['cluster_id']}:")
    #     for s in c["user_stories"]:
    #         print(f"  - [{s['actor']}] {s['goal']}")

    return {"user_story_clusters": clusters_list}


def name_usecases_node(state: GraphState):
    """
    Takes clustered user stories and asks LLM to generate a UseCase name for each cluster.
    Returns a list of UseCase objects.
    """

    llm = state.get("llm")
    clusters = state.get("user_story_clusters")

    if not clusters:
        return {}

    # Build prompt with cluster information (using detailed structure)
    clusters_text = ""
    for cluster in clusters:
        cluster_id = cluster["cluster_id"]
        stories = cluster["user_stories"]  # Contains detailed dicts
        clusters_text += f"\n### Cluster {cluster_id}:\n"

        # Extract unique actors
        actors = list(set(story["actor"] for story in stories))

        clusters_text += f"  Actors: {', '.join(actors)}\n"
        clusters_text += "  Actions:\n"
        for story in stories:
            clusters_text += f"    - {story['goal']}\n"
        clusters_text += "  Sample sentences:\n"
        for story in stories[:3]:  # Show max 3 examples
            clusters_text += f'    - "{story["original_sentence"]}"\n'

    prompt = f"""You are a senior software analyst specializing in Use Case modeling for UML diagrams.

Given the following clusters of related user story actions, generate a precise UseCase name for each cluster.

**Naming Guidelines:**
1. Use verb-noun format (1-3 words preferred):
   - For login/register/password actions → "Authenticate" or "Authentication"
   - For CRUD operations on a resource → "Manage [Resource]" (e.g., "Manage Products")
   - For viewing/reading data → "View [Data]" or "Browse [Data]"
   - For creating reports → "Generate [Report Type]"
   
2. **Common patterns:**
   - login, register, logout, reset password → "Authenticate" 
   - add, edit, delete, update items → "Manage [Items]"
   - search, filter, browse → "Search [Items]" or "Browse [Items]"
   - view reports, download data → "View Reports" or "Export Data"
   - configure settings → "Configure System"

3. **Avoid:**
   - Long names (more than 4 words)
   - Generic names like "Handle User Actions"
   - Names that describe WHO does it instead of WHAT is done

4. If actions are mixed, identify the DOMINANT theme and name accordingly

{clusters_text}

For each cluster, provide:
- cluster_id: The cluster number (as shown above)
- usecase_name: A concise name following the patterns above (1-4 words)
- description: One sentence describing the use case scope
"""

    structured_llm = llm.with_structured_output(UseCaseNamingResponse)
    response = structured_llm.invoke(prompt)

    # Convert to UseCase objects
    cluster_lookup = {c["cluster_id"]: c["user_stories"] for c in clusters}
    use_cases: List[UseCase] = []

    for idx, naming in enumerate(response.usecases):
        stories = cluster_lookup.get(naming.cluster_id, [])

        # Extract unique actors
        unique_actors = list(set(story["actor"] for story in stories))

        # Convert story dicts to UserStoryItem objects (action = goal phrase)
        # Đồng thời loại bỏ duplicate theo (actor, câu gốc) trong cùng Use Case.
        # Nghĩa là nếu cùng actor và cùng original_sentence nhưng có nhiều goal khác nhau,
        # ta chỉ giữ lại một bản ghi để tránh lặp dòng trong output.
        unique_story_items: List[UserStoryItem] = []
        seen_story_keys = set()
        for story in stories:
            key = (story["actor"], story["original_sentence"].strip())
            if key in seen_story_keys:
                continue
            seen_story_keys.add(key)
            unique_story_items.append(
                UserStoryItem(
                    actor=story["actor"],
                    action=story["goal"],
                    original_sentence=story["original_sentence"],
                    sentence_idx=story["sentence_idx"],
                )
            )

        user_story_items = unique_story_items

        # Create UseCase object
        use_case = UseCase(
            id=idx + 1,
            name=naming.usecase_name,
            description=naming.description,
            participating_actors=unique_actors,
            user_stories=user_story_items,
            relationships=[],  # Will be filled later if needed
        )
        use_cases.append(use_case)

    # DEBUG: in toàn bộ use_cases trước khi in summary cuối
    print("\n==== name_usecases_node (use_cases) ====")
    # for uc in use_cases:
    #     print(f"UC-{uc.id}: {uc.name}")
    #     print(f"  Actors: {uc.participating_actors}")
    #     for s in uc.user_stories:
    #         print(f"  - [{s.actor}] {s.action} (sent={s.sentence_idx})")
    #         print(f'    "{s.original_sentence}"')

    return {"use_cases": use_cases}


# =============================================================================
# ONE-STEP INCLUDE/EXTEND RELATIONSHIP DETECTION
# =============================================================================


def find_include_extend_node(state: GraphState):
    """
    Tìm tất cả quan hệ «include» và «extend» giữa các Use Case (một bước LLM).
    Input: use_cases từ name_usecases_node (đã có name, description, user_stories).
    Output: include_extend_relationships (list dict) cho merge_relationships_node.
    """
    llm = state.get("llm")
    use_cases = state.get("use_cases") or []

    if not use_cases or not llm:
        return {}

    # Build context: name, description, goals/actions từ user_stories
    lines = []
    uc_names = []
    for uc in use_cases:
        uc_names.append(uc.name)
        parts = [f"- {uc.name}"]
        if uc.description:
            parts.append(f"  Description: {uc.description}")
        if uc.user_stories:
            goals = [s.action for s in uc.user_stories[:3]]
            parts.append(f"  Goals/actions: {', '.join(goals)}")
        lines.append("\n".join(parts))

    usecases_text = "\n\n".join(lines)
    uc_names_lower = [n.lower() for n in uc_names]

    system_prompt = """You are a UML Use Case expert. Identify «include» and «extend» relationships between the given use cases.

DEFINITIONS:
1. «include» (mandatory): Use case A ALWAYS requires use case B to complete.
   - Example: "checkout order" includes "validate cart"
   - Indicators: "must", "requires", "needs to", "first", "then"

2. «extend» (optional): Use case A MAY add behavior to use case B under some conditions.
   - Example: "apply discount" extends "checkout order"
   - Indicators: "optionally", "can also", "if", "when", "may"

RULES:
- Only output relationships clearly implied by descriptions or goals/actions. Do NOT invent.
- source_use_case and target_use_case MUST be exactly one of the use case names from the list (same spelling, use lowercase).
- A use case cannot include/extend itself.
- Return an empty list if there are no clear relationships.
- Output use case names in lowercase."""

    human_prompt = f"""Use cases (from clustered goals, already named):

{usecases_text}

Use case names (use EXACTLY as source/target, lowercase): {uc_names_lower}

Identify all «include» and «extend» relationships. Return source_use_case, relationship_type ("include" or "extend"), target_use_case, and brief reasoning."""

    structured_llm = llm.with_structured_output(UseCaseRelationshipResponse)
    response = structured_llm.invoke(
        [("system", system_prompt), ("human", human_prompt)]
    )

    # Output list of dicts (format cho merge_relationships_node)
    include_extend_relationships = []
    for rel in response.relationships:
        source = rel.source_use_case.lower().strip()
        target = rel.target_use_case.lower().strip()
        if source in uc_names_lower and target in uc_names_lower and source != target:
            include_extend_relationships.append(
                {
                    "source_use_case": source,
                    "type": rel.relationship_type,
                    "target_use_case": target,
                }
            )

    print("\n==== find include/extend ====")
    for rel in include_extend_relationships:
        print(f"  {rel['source_use_case']} --{rel['type']}--> {rel['target_use_case']}")

    return {"include_extend_relationships": include_extend_relationships}


# =============================================================================
# RELATIONSHIP DETECTION (3-STEP PIPELINE - COMMENTED OUT; dùng find_include_extend)
# =============================================================================


def group_usecases_by_domain_node(state: GraphState):
    """
    STEP 1: Group use cases by domain using LLM. (COMMENTED OUT - dùng find_include_extend)
    """
    return {}  # bypass: pipeline dùng find_include_extend thay vì 3-step

    def _group_by_domain(model: BaseChatModel, use_cases: List[UseCase]) -> dict:
        """Use LLM to group use cases into domains."""
        if model is None or not use_cases:
            # Fallback: put all in single domain
            print(
                "⚠️  Fallback: No LLM model or empty use cases, grouping all into 'general' domain"
            )
            return {"general": [uc.name for uc in use_cases]}

        structured_llm = model.with_structured_output(UseCaseDomainGroupingResponse)

        uc_list = "\n".join([f"- {uc.name}" for uc in use_cases])

        system_prompt = """You are a Business Analyst AI specializing in software requirements analysis.

Your task is to group use cases into logical business domains/categories.

RULES:
- Each use case MUST be assigned to exactly ONE domain.
- Domain names should be clear, concise business categories (e.g., "Authentication", "Shopping", "Order Management", "User Profile", "Payment").
- Use cases with similar functionality or business context should be in the same domain.
- Domain names MUST be in lowercase.
- Create between 2-7 domains depending on the variety of use cases.
- Do NOT create a domain for just 1 use case unless it's truly unique.
- Common/shared functionality (login, validate, authenticate) should be in their own domain like "authentication" or "security".

OUTPUT:
Return each use case with its assigned domain."""

        human_prompt = f"""Group the following use cases into logical business domains:

USE CASES:
{uc_list}

Assign each use case to exactly one domain."""

        print(f"\n🔄 Calling LLM to group {len(use_cases)} use cases into domains...")
        response: UseCaseDomainGroupingResponse = structured_llm.invoke(
            [("system", system_prompt), ("human", human_prompt)]
        )
        print("✅ LLM grouping completed")

        # Convert to dict: {domain: [use_case_names]}
        domain_groups = {}
        for item in response.groupings:
            domain = item.domain.lower()
            if domain not in domain_groups:
                domain_groups[domain] = []
            domain_groups[domain].append(item.use_case_name.lower())

        return domain_groups

    print("\n" + "=" * 60)
    print("STEP 1: GROUPING USE CASES BY DOMAIN")
    print("=" * 60)

    model = state.get("llm")
    use_cases = state.get("use_cases") or []

    print(f"\n📋 Input: {len(use_cases)} use cases to group")
    for i, uc in enumerate(use_cases, 1):
        print(f"  {i}. {uc.name}")

    domain_groupings = _group_by_domain(model, use_cases)

    print(f"\n✅ Grouped into {len(domain_groupings)} domain(s):")
    for domain, uc_names in sorted(domain_groupings.items()):
        print(f"\n  📁 Domain: {domain.upper()}")
        for name in uc_names:
            print(f"     - {name}")

    print("\n" + "-" * 60)

    return {"domain_groupings": domain_groupings}


def find_within_domain_relationships_node(state: GraphState):
    """
    STEP 2: Find include/extend WITHIN each domain. (COMMENTED OUT - dùng find_include_extend)
    """
    return {}  # bypass: pipeline dùng find_include_extend thay vì 3-step

    def _find_within_domain_relationships(
        model: BaseChatModel,
        domain_groupings: dict,
        use_cases: List[UseCase],
    ) -> List[dict]:
        """Find relationships between use cases in the same domain."""
        if model is None or not domain_groupings:
            return []

        structured_llm = model.with_structured_output(UseCaseRelationshipResponse)
        all_relationships = []

        # Build use case lookup for sentence context
        uc_lookup = {uc.name.lower(): uc for uc in use_cases}

        for domain, uc_names in domain_groupings.items():
            if len(uc_names) < 2:
                # Need at least 2 use cases to have relationships
                print(
                    f"  ⏭️  Skipping domain '{domain}': only {len(uc_names)} use case(s) (need at least 2)"
                )
                continue

            print(f"\n  🔍 Analyzing domain '{domain}' ({len(uc_names)} use cases)...")

            # Get use case details for this domain
            uc_details = []
            for name in uc_names:
                uc = uc_lookup.get(name.lower())
                if uc:
                    # Use description and sample user stories for context
                    context_parts = [f"- {uc.name}"]
                    if uc.description:
                        context_parts.append(f"  Description: {uc.description}")
                    if uc.user_stories:
                        samples = [s.original_sentence for s in uc.user_stories[:2]]
                        context_parts.append(
                            f"  Examples: {' | '.join(f'{s}' for s in samples)}"
                        )
                    uc_details.append("\n".join(context_parts))
                else:
                    uc_details.append(f"- {name}")

            uc_text = "\n".join(uc_details)

            system_prompt = """You are a UML Use Case expert specializing in identifying relationships between use cases.

Your task is to identify «include» and «extend» relationships between use cases WITHIN THE SAME DOMAIN.

RELATIONSHIP DEFINITIONS:
1. «include» (mandatory): Use case A ALWAYS requires use case B to complete.
   - The included use case is essential for the base use case
   - Example: "checkout order" includes "validate cart"
   - Indicators: "must first", "requires", "needs to", "depends on"

2. «extend» (optional): Use case A MAY extend use case B under certain conditions.
   - The extending use case is optional behavior
   - Example: "apply discount" extends "checkout order"
   - Indicators: "optionally", "can also", "if", "when", "may"

RULES:
- Only identify relationships explicitly supported by the use case descriptions
- Do NOT invent relationships not implied by the context
- A use case cannot include/extend itself
- Be conservative - when uncertain, don't add the relationship
- Focus on functional dependencies within this domain
- Provide brief reasoning for each relationship"""

            human_prompt = f"""Analyze the following use cases from the "{domain}" domain and identify include/extend relationships:

DOMAIN: {domain}
USE CASES:
{uc_text}

Identify any «include» or «extend» relationships between these use cases.
If no relationships exist, return an empty list."""

            response: UseCaseRelationshipResponse = structured_llm.invoke(
                [("system", system_prompt), ("human", human_prompt)]
            )

            print(
                f"    ✅ Found {len(response.relationships)} relationship(s) in '{domain}'"
            )

            for rel in response.relationships:
                all_relationships.append(
                    {
                        "source_use_case": rel.source_use_case.lower(),
                        "type": rel.relationship_type,
                        "target_use_case": rel.target_use_case.lower(),
                        "reasoning": rel.reasoning,
                        "domain": domain,
                        "relationship_scope": "within_domain",
                    }
                )

        return all_relationships

    print("\n" + "=" * 60)
    print("STEP 2: FINDING WITHIN-DOMAIN RELATIONSHIPS")
    print("=" * 60)

    model = state.get("llm")
    domain_groupings = state.get("domain_groupings") or {}
    use_cases = state.get("use_cases") or []

    print(f"\n📋 Input: {len(domain_groupings)} domain(s) to analyze")
    for domain, uc_names in sorted(domain_groupings.items()):
        print(f"  📁 {domain}: {len(uc_names)} use case(s)")

    within_relationships = _find_within_domain_relationships(
        model, domain_groupings, use_cases
    )

    print(f"\n✅ Found {len(within_relationships)} within-domain relationship(s):")
    if within_relationships:
        for rel in within_relationships:
            print(
                f"  • {rel['source_use_case']} --{rel['type']}--> {rel['target_use_case']} [{rel['domain']}]"
            )
    else:
        print("  (No relationships found)")

    print("-" * 60)

    return {"within_domain_relationships": within_relationships}


def find_cross_domain_relationships_node(state: GraphState):
    """
    STEP 3: Find include/extend ACROSS domains. (COMMENTED OUT - dùng find_include_extend)
    """
    return {}  # bypass: pipeline dùng find_include_extend thay vì 3-step

    def _find_cross_domain_relationships(
        model: BaseChatModel,
        domain_groupings: dict,
        use_cases: List[UseCase],
        existing_relationships: List[dict],
    ) -> List[dict]:
        """Find relationships between use cases from different domains."""
        if model is None or len(domain_groupings) < 2:
            return []

        structured_llm = model.with_structured_output(UseCaseRelationshipResponse)

        # Prepare domain overview
        domain_overview = []
        for domain, uc_names in domain_groupings.items():
            uc_list = ", ".join(uc_names)
            domain_overview.append(f"- {domain}: [{uc_list}]")
        domain_text = "\n".join(domain_overview)

        # Prepare existing relationships summary
        existing_text = "None found yet."
        if existing_relationships:
            existing_lines = [
                f"- {r['source_use_case']} --{r['type']}--> {r['target_use_case']}"
                for r in existing_relationships
            ]
            existing_text = "\n".join(existing_lines)

        # Build detailed use case info
        uc_details = []
        for uc in use_cases:
            # Use description and sample user stories for context
            context_parts = [f"- {uc.name}"]
            if uc.description:
                context_parts.append(f"  Description: {uc.description}")
            if uc.user_stories:
                samples = [s.original_sentence for s in uc.user_stories[:2]]
                context_parts.append(
                    f"  Examples: {' | '.join(f'{s}' for s in samples)}"
                )
            uc_details.append("\n".join(context_parts))
        uc_details_text = "\n".join(uc_details)

        system_prompt = """You are a UML Use Case expert specializing in identifying CROSS-DOMAIN relationships between use cases.

Your task is to identify «include» and «extend» relationships between use cases from DIFFERENT domains.

RELATIONSHIP DEFINITIONS:
1. «include» (mandatory): Use case A ALWAYS requires use case B to complete.
   - Common patterns: authentication required, validation needed, logging mandatory
   - Example: "checkout order" (Shopping) includes "login" (Authentication)

2. «extend» (optional): Use case A MAY extend use case B under certain conditions.
   - Common patterns: optional features, conditional behavior
   - Example: "apply coupon" (Promotions) extends "checkout order" (Shopping)

FOCUS ON CROSS-DOMAIN PATTERNS:
- Authentication/Security: Which use cases require login/authentication?
- Validation: Which use cases need input validation from another domain?
- Logging/Audit: Which use cases need activity logging?
- Notifications: Which use cases trigger notifications?
- Payment: Which use cases require payment processing?

RULES:
- ONLY identify relationships between use cases from DIFFERENT domains
- Do NOT repeat relationships already identified (see existing relationships)
- Do NOT invent relationships not implied by the use case descriptions
- A use case cannot include/extend itself
- Be thorough but conservative
- Provide brief reasoning for each relationship"""

        human_prompt = f"""Analyze use cases across different domains and identify CROSS-DOMAIN relationships:

DOMAINS AND USE CASES:
{domain_text}

USE CASE DETAILS:
{uc_details_text}

EXISTING RELATIONSHIPS (already found within domains):
{existing_text}

Identify any «include» or «extend» relationships between use cases from DIFFERENT domains.
Focus especially on:
1. Which use cases need authentication/login?
2. Which use cases share common validation?
3. Which use cases trigger or depend on use cases in other domains?

If no cross-domain relationships exist, return an empty list."""

        response: UseCaseRelationshipResponse = structured_llm.invoke(
            [("system", system_prompt), ("human", human_prompt)]
        )

        print(
            f"✅ LLM returned {len(response.relationships)} potential relationship(s)"
        )

        # Get domain for each use case
        uc_to_domain = {}
        for domain, uc_names in domain_groupings.items():
            for name in uc_names:
                uc_to_domain[name.lower()] = domain

        cross_relationships = []
        filtered_out = 0
        for rel in response.relationships:
            source_domain = uc_to_domain.get(rel.source_use_case.lower(), "unknown")
            target_domain = uc_to_domain.get(rel.target_use_case.lower(), "unknown")

            # Only keep if truly cross-domain
            if source_domain != target_domain:
                cross_relationships.append(
                    {
                        "source_use_case": rel.source_use_case.lower(),
                        "type": rel.relationship_type,
                        "target_use_case": rel.target_use_case.lower(),
                        "reasoning": rel.reasoning,
                        "source_domain": source_domain,
                        "target_domain": target_domain,
                        "relationship_scope": "cross_domain",
                    }
                )
            else:
                filtered_out += 1

        if filtered_out > 0:
            print(
                f"  ⚠️  Filtered out {filtered_out} relationship(s) (same domain, already in within-domain)"
            )

        return cross_relationships

    print("\n" + "=" * 60)
    print("STEP 3: FINDING CROSS-DOMAIN RELATIONSHIPS")
    print("=" * 60)

    model = state.get("llm")
    domain_groupings = state.get("domain_groupings") or {}
    use_cases = state.get("use_cases") or []
    within_relationships = state.get("within_domain_relationships") or []

    print(f"\n📋 Input:")
    print(f"  • {len(domain_groupings)} domain(s)")
    print(f"  • {len(use_cases)} total use case(s)")
    print(f"  • {len(within_relationships)} existing within-domain relationship(s)")

    if len(domain_groupings) < 2:
        print("\n⚠️  Skipping: Need at least 2 domains for cross-domain analysis")
        return {"cross_domain_relationships": []}

    print(f"\n🔄 Calling LLM to find cross-domain relationships...")

    cross_relationships = _find_cross_domain_relationships(
        model, domain_groupings, use_cases, within_relationships
    )

    print(f"\n✅ Found {len(cross_relationships)} cross-domain relationship(s):")
    if cross_relationships:
        for rel in cross_relationships:
            print(
                f"  • {rel['source_use_case']} ({rel['source_domain']}) --{rel['type']}--> {rel['target_use_case']} ({rel['target_domain']})"
            )
    else:
        print("  (No relationships found)")

    print("-" * 60)

    return {"cross_domain_relationships": cross_relationships}


def merge_relationships_node(state: GraphState):
    """
    Gắn relationships vào UseCase. Nhận từ find_include_extend (include_extend_relationships)
    hoặc legacy 3-step (within_domain_relationships + cross_domain_relationships).
    """
    print("\n" + "=" * 60)
    print("MERGE RELATIONSHIPS")
    print("=" * 60)

    use_cases = state.get("use_cases") or []
    include_extend_rels = state.get("include_extend_relationships") or []
    within_rels = state.get("within_domain_relationships") or []
    cross_rels = state.get("cross_domain_relationships") or []

    if include_extend_rels:
        all_relationships = include_extend_rels
        print(
            f"\n📋 Input: {len(use_cases)} use case(s), {len(all_relationships)} relationship(s) (from find_include_extend)"
        )
    else:
        all_relationships = within_rels + cross_rels
        print(
            f"\n📋 Input: {len(use_cases)} use case(s), {len(within_rels)} within + {len(cross_rels)} cross relationship(s)"
        )

    print(f"\n🔄 Merging {len(all_relationships)} total relationship(s)...")

    # Build lookup: source_use_case -> list of relationships
    rel_lookup = {}
    for rel in all_relationships:
        source = rel["source_use_case"].lower()
        if source not in rel_lookup:
            rel_lookup[source] = []
        rel_lookup[source].append(
            UseCaseRelationship(
                type=rel["type"],
                target_use_case=rel["target_use_case"],
            )
        )

    # Update use cases with their relationships (preserve new schema)
    updated_use_cases = []
    use_cases_with_rels = 0
    for uc in use_cases:
        uc_name = uc.name.lower()
        relationships = rel_lookup.get(uc_name, [])
        if relationships:
            use_cases_with_rels += 1
        updated_use_cases.append(
            UseCase(
                id=uc.id,
                name=uc.name,
                description=uc.description,
                participating_actors=uc.participating_actors,
                user_stories=uc.user_stories,
                relationships=relationships,
            )
        )

    # Print relationships summary
    print(f"\n✅ Merged relationships into {use_cases_with_rels} use case(s):")
    print("\n--- FINAL INCLUDE/EXTEND RELATIONSHIPS ---")
    for uc in updated_use_cases:
        if uc.relationships:
            print(f"\n  📌 {uc.name}:")
            for r in uc.relationships:
                print(f"     --{r.type}--> {r.target_use_case}")

    print("\n" + "=" * 60)
    print("RELATIONSHIP DETECTION COMPLETED")
    print("=" * 60)

    return {"use_cases": updated_use_cases}


# =============================================================================
# GRAPH BUILDER
# =============================================================================


def build_rpa_graph():
    """
    RPA Graph (Requirement Processing Agent):

    - Branch 1: Extract actors, remove synonyms, find aliases
    - Branch 2: Extract usecases with NLP, refine with LLM
    - grouping: Merge actor+usecase, cluster by semantic similarity (K-Means)
    - refine_clustering: LLM refines clusters (split/merge/move items)
    - name_usecases: LLM generates UseCase names for each cluster
    - find_include_extend: One-step find all include/extend relationships (LLM)
    - merge_relationships: Attach relationships to use_cases
    """

    workflow = StateGraph(GraphState)

    # Add nodes
    # Chuẩn hóa câu thành "As a/an/the <actor>, I want to <goal> so that <benefit>"
    # workflow.add_node("normalize_sentences", normalize_sentences_node)

    # Branch 1: Actor pipeline
    workflow.add_node("find_actors", find_actors_node)
    workflow.add_node("synonym_check", synonym_check_node)
    workflow.add_node("find_aliases", find_aliases_node)

    # Branch 2: UseCase pipeline
    workflow.add_node("find_goals", find_goals_node)
    workflow.add_node("refine_goals", refine_goals_node)

    # Convergence point (grouping handles both actor and usecase data)
    # NOTE: finalize_node removed - grouping_node now serves as convergence point
    workflow.add_node("grouping", grouping_node)
    workflow.add_node("refine_clustering", refine_clustering_node)
    workflow.add_node("name_usecases", name_usecases_node)
    workflow.add_node("find_include_extend", find_include_extend_node)
    workflow.add_node("merge_relationships", merge_relationships_node)

    # Define edges: chuẩn hóa câu trước, rồi hai nhánh song song
    workflow.add_edge(START, "find_actors")
    workflow.add_edge(START, "find_goals")

    # Branch 1: normalize_sentences -> find_actors -> synonym_check -> find_aliases -> grouping
    # workflow.add_edge("normalize_sentences", "find_actors")
    workflow.add_edge("find_actors", "synonym_check")
    workflow.add_edge("synonym_check", "find_aliases")
    workflow.add_edge("find_aliases", "grouping")

    # Branch 2: normalize_sentences -> find_goals -> refine_goals -> grouping
    # workflow.add_edge("normalize_sentences", "find_goals")
    workflow.add_edge("find_goals", "refine_goals")
    workflow.add_edge("refine_goals", "grouping")

    # # After grouping -> 3-Step Relationship Detection (disabled)
    # workflow.add_edge("grouping", "group_by_domain")
    # workflow.add_edge("group_by_domain", "find_within_domain_rels")
    # workflow.add_edge("find_within_domain_rels", "find_cross_domain_rels")
    # workflow.add_edge("find_cross_domain_rels", "merge_relationships")
    # grouping -> refine_clustering -> name_usecases -> END
    # workflow.add_edge("grouping", "refine_clustering")
    workflow.add_conditional_edges(
        "grouping",
        should_continue_to_clustering,
        {
            "continue": "refine_clustering",
            "stop": END,  # Kết thúc luồng rác tại đây
        },
    )
    workflow.add_edge("refine_clustering", "name_usecases")

    # After naming usecases -> find include/extend -> merge relationships
    workflow.add_edge("name_usecases", "find_include_extend")
    workflow.add_edge("find_include_extend", "merge_relationships")
    workflow.add_edge("merge_relationships", END)

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


# =============================================================================
# BATCH RUN: multiple inputs -> outputs_rpa
# =============================================================================


def _serialize_rpa_result(result: dict) -> dict:
    """Chuyển kết quả RPA sang dict có thể JSON serialize."""
    actors = result.get("actors", [])
    actor_aliases = result.get("actor_aliases", [])
    use_cases = result.get("use_cases", [])
    return {
        "requirement_text": result.get("requirement_text", ""),
        "actors": [a.model_dump() if hasattr(a, "model_dump") else a for a in actors],
        "actor_aliases": [
            a.model_dump() if hasattr(a, "model_dump") else a for a in actor_aliases
        ],
        "use_cases": [
            uc.model_dump() if hasattr(uc, "model_dump") else uc for uc in use_cases
        ],
    }


def run_rpa_batch(
    input_dir: str | Path | None = None,
    output_dir: str | Path = "outputs_rpa",
    input_files: List[str | Path] | None = None,
    write_txt: bool = True,
) -> List[Path]:
    """Chạy RPA cho nhiều file input và ghi kết quả ra folder outputs_rpa.

    Args:
        input_dir: Thư mục chứa file .txt (mặc định: project_root/inputs).
        output_dir: Thư mục ghi output (mặc định: outputs_rpa).
        input_files: Danh sách đường dẫn file cụ thể; nếu set thì bỏ qua input_dir.
        write_txt: Nếu True, ghi thêm file .txt tóm tắt (actors, use cases) bên cạnh .json.

    Returns:
        Danh sách đường dẫn file output đã ghi.
    """
    project_root = Path(__file__).resolve().parent.parent.parent
    if input_files is not None:
        paths = [Path(p) for p in input_files]
    else:
        base = Path(input_dir) if input_dir else project_root / "inputs"
        paths = sorted(base.glob("*.txt"))
    out_base = Path(output_dir)
    if not out_base.is_absolute():
        out_base = project_root / out_base
    out_base.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    for inp in paths:
        if not inp.is_absolute():
            inp = project_root / inp
        if not inp.exists():
            continue
        raw = inp.read_text(encoding="utf-8").strip()
        result = run_rpa(raw)
        data = _serialize_rpa_result(result)
        stem = inp.stem
        json_path = out_base / f"{stem}.json"
        json_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        written.append(json_path)
        if write_txt:
            lines = [
                f"=== INPUT: {inp.name} ===",
                f"User stories: {len(raw.split(chr(10)))}",
                "",
                "=== ACTORS ===",
            ]
            for a in result.get("actor_aliases", []):
                name = getattr(a, "actor", a) if not isinstance(a, str) else a
                lines.append(f"  - {name}")
            lines.append("")
            lines.append("=== USE CASES ===")
            for uc in result.get("use_cases", []):
                uid = getattr(uc, "id", 0)
                uc_name = getattr(uc, "name", "")
                actors = ", ".join(getattr(uc, "participating_actors", []) or [])
                lines.append(f"[{uid}] {uc_name} (actors: {actors})")
                # User stories thuộc use case này
                user_stories = getattr(uc, "user_stories", []) or []
                if user_stories:
                    lines.append("  User stories:")
                    for us in user_stories:
                        orig = getattr(us, "original_sentence", "") or getattr(
                            us, "action", ""
                        )
                        if orig:
                            lines.append(f"    - {orig}")
                # Relationships của use case này
                rels = getattr(uc, "relationships", []) or []
                if rels:
                    lines.append("  Relationships:")
                    for rel in rels:
                        rtype = getattr(rel, "type", "include")
                        target = getattr(rel, "target_use_case", "")
                        arrow = "──include──>" if rtype == "include" else "··extend··>"
                        lines.append(f"    {uc_name} {arrow} {target}")
                lines.append("")
            # Tổng hợp relationships giữa các use case (một lần)
            lines.append("=== USE CASE RELATIONSHIPS (summary) ===")
            any_rel = False
            for uc in result.get("use_cases", []):
                uc_name = getattr(uc, "name", "")
                for rel in getattr(uc, "relationships", []) or []:
                    any_rel = True
                    rtype = getattr(rel, "type", "include")
                    target = getattr(rel, "target_use_case", "")
                    arrow = "──include──>" if rtype == "include" else "··extend··>"
                    lines.append(f"  {uc_name} {arrow} {target}")
            if not any_rel:
                lines.append("  (none)")
            txt_path = out_base / f"{stem}.txt"
            txt_path.write_text("\n".join(lines), encoding="utf-8")
            written.append(txt_path)
    return written


# =============================================================================
# TEST FUNCTION
# =============================================================================


def test_rpa_graph(requirements_file: str | Path | None = None):
    """Test the RPA graph with user stories read from a file."""
    # Default: input_user_stories.txt in project root (research/)
    if requirements_file is None:
        project_root = Path(__file__).resolve().parent.parent.parent
        requirements_file = project_root / "inputs" / "input_5.txt"
    else:
        requirements_file = Path(requirements_file)

    if not requirements_file.exists():
        raise FileNotFoundError(
            f"Requirements file not found: {requirements_file}. "
            "Create the file or pass a valid path to test_rpa_graph(requirements_file=...)."
        )

    with open(requirements_file, encoding="utf-8") as f:
        sample_requirements = f.read()

    print("=" * 60)
    print("TESTING RPA GRAPH")
    print("=" * 60)
    print("\n📝 INPUT: User Stories")
    print("-" * 40)
    for i, line in enumerate(sample_requirements.strip().split("\n")):
        print(f"  {i}: {line}")

    print("\n⏳ Running RPA Graph...")
    result = run_rpa(sample_requirements)

    # Print actors
    # print("\n👤 ACTORS:")
    # print("-" * 40)
    # for actor in result.get("actors", []):
    #     print(f"  - {actor.actor}")

    # # Print use cases with relationships
    # print("\n📋 USE CASES WITH RELATIONSHIPS:")
    # print("-" * 40)
    # for uc in result.get("use_cases", []):
    #     print(f"\n  📌 {uc.name}")
    #     print(f"     Actors: {', '.join(uc.participating_actors)}")
    #     if uc.relationships:
    #         print("     Relationships:")
    #         for rel in uc.relationships:
    #             arrow = "──include──>" if rel.type == "include" else "··extend··>"
    #             print(f"       {arrow} {rel.target_use_case}")
    #     else:
    #         print("     Relationships: (none)")

    # # Print JSON format of use_cases
    # print("\n📄 USE CASES (JSON FORMAT):")
    # print("-" * 40)
    # use_cases_json = [uc.model_dump() for uc in result.get("use_cases", [])]
    # print(json.dumps(use_cases_json, indent=2, ensure_ascii=False))

    # # Save output to file
    # output_data = {
    #     "actors": [
    #         actor.model_dump() if hasattr(actor, "model_dump") else actor
    #         for actor in result.get("actors", [])
    #     ],
    #     "actor_aliases": [
    #         alias.model_dump() if hasattr(alias, "model_dump") else alias
    #         for alias in result.get("actor_aliases", [])
    #     ],
    #     "use_cases": use_cases_json,
    # }

    # output_file = "rpa_output.json"
    # with open(output_file, "w", encoding="utf-8") as f:
    #     json.dump(output_data, f, indent=2, ensure_ascii=False)

    # print("\n" + "=" * 60)
    # print("TEST COMPLETED")
    # print(f"✅ Output saved to: {output_file}")
    # print("=" * 60)

    # return result


if __name__ == "__main__":
    # paths = run_rpa_batch(input_files=["inputs/input_5.txt"])
    test_rpa_graph()
