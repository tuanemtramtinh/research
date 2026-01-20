"""
Usecase Finder Module
Handles usecase extraction from user stories using NLP.
"""

import json
from typing import List, Optional
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from spacy.language import Language
from spacy.tokens import Token

from actor_finder import ActorAliasMapping, ActorResult


# =============================================================================
# STRUCTURED OUTPUT MODELS
# =============================================================================
class UsecaseRefinement(BaseModel):
    """Refinement result for a single sentence."""

    sentence_idx: int = Field(description="Index of the original sentence")
    original: List[str] = Field(description="Original extracted use cases")
    refined: List[str] = Field(description="Refined/improved use cases")
    added: List[str] = Field(
        default_factory=list, description="Missing use cases that should be added"
    )
    reasoning: Optional[str] = Field(
        default=None, description="Brief explanation of changes made"
    )


class UsecaseRefinementResponse(BaseModel):
    """Complete response containing all refined use cases."""

    refinements: List[UsecaseRefinement] = Field(
        description="List of refinements for each sentence"
    )

    def to_dict(self) -> dict:
        """Convert to simple dict format: {sentence_idx: [refined_usecases]}"""
        result = {}
        for r in self.refinements:
            result[str(r.sentence_idx)] = r.refined + r.added
        return result


# =============================================================================
# USECASE FINDER CLASS
# =============================================================================
class UsecaseFinder:
    """Handles usecase extraction from user stories using NLP."""

    def __init__(self, nlp_model: Language, sents: List[str], llm: ChatOpenAI):
        self.nlp = nlp_model
        self.sents = sents
        self.llm = llm

    def _find_main_verb(self, xcomp: Token) -> Token:
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

    def _get_verb_phrase(self, verb: Token) -> str:
        """Extract verb phrase from a verb token."""
        exclude_tokens = set()
        has_dobj = any(child.dep_ == "dobj" for child in verb.children)
        for child in verb.children:
            if child.dep_ == "conj":  # login, swim, etc.
                exclude_list = None
                if not has_dobj:
                    exclude_list = [
                        subchild
                        for subchild in child.subtree
                        if subchild.dep_ != "dobj"
                    ]
                else:
                    exclude_list = child.subtree
                exclude_tokens.update(exclude_list)
            if child.dep_ == "cc":  # and, or, but
                exclude_tokens.add(child)

        tokens = [t for t in verb.subtree if t not in exclude_tokens]
        tokens = tokens[tokens.index(verb) :]
        tokens = sorted(tokens, key=lambda t: t.i)

        cut_index = -1

        # Remove "so that" clause if present
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

            # Keep prepositional phrases
            elif token.dep_ in {"prep", "pobj"} or token.head.dep_ in {"prep", "pobj"}:
                if token.dep_ not in EXCLUDE_DEPS:
                    relevant_tokens.append(token)

            # Keep particle (phrasal verbs)
            elif token.dep_ == "prt":
                relevant_tokens.append(token)

            # Keep adjective/adverb complements
            elif token.dep_ in {"acomp", "advmod"}:
                relevant_tokens.append(token)

            # Keep compound nouns and adjective modifiers
            elif token.dep_ in {"compound", "amod"}:
                relevant_tokens.append(token)

        tokens = [tokens[0]] + sorted(relevant_tokens, key=lambda t: t.i)
        result = [token.text for token in tokens]

        return " ".join(result)

    def _get_all_conj(self, verb: Token) -> List[Token]:
        """
        Find all conjunctions of the root verb (swim and sleep and eat).
        For root verb 'swim', conj verbs are 'sleep' and 'eat'.
        """
        result = []
        for child in verb.children:
            if child.dep_ == "conj" and child.pos_ == "VERB":
                result.append(child)
                result.extend(self._get_all_conj(child))  # Recursive search
        return result

    def find_usecases(self) -> dict:
        """Extract usecases from all sentences."""
        res = {}

        for i, sent in enumerate(self.sents):
            doc = self.nlp(sent)
            for token in doc:
                if token.lemma_ == "want":
                    for children in token.children:
                        if children.dep_ == "xcomp" and children.pos_ in {
                            "VERB",
                            "AUX",
                        }:
                            # Exclude V-ing case that becomes xcomp of 'want'
                            # (checkout using multiple payment methods)
                            if children.tag_ == "VBG":
                                continue

                            main_verb = self._find_main_verb(children)
                            verb_phrase = self._get_verb_phrase(main_verb)

                            if str(i) not in res:
                                # res[str(i)] = {}
                                res[str(i)] = []
                            # res[str(i)][main_verb.text] = verb_phrase
                            res[str(i)] += [verb_phrase]

                            # Find ALL conj verbs (recursive)
                            all_conj_verbs = self._get_all_conj(main_verb)
                            for conj in all_conj_verbs:
                                conj_verb_phrase = self._get_verb_phrase(conj)
                                # res[str(i)][conj.text] = conj_verb_phrase
                                res[str(i)] += [conj_verb_phrase]

        return res

    def refine_usecases(self, usecases: dict) -> UsecaseRefinementResponse:
        """
        Use LLM to refine and complete extracted use cases.

        Args:
            llm: LangChain ChatOpenAI instance
            usecases: Dict of {sentence_idx: [use_cases]}

        Returns:
            UsecaseRefinementResponse with refined use cases
        """
        sents_text = "\n".join([f'{i}: "{sent}"' for i, sent in enumerate(self.sents)])

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
  - “so that”, purposes, or outcomes
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

        # Use structured output
        structured_llm = self.llm.with_structured_output(UsecaseRefinementResponse)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": human_prompt},
        ]

        response: UsecaseRefinementResponse = structured_llm.invoke(messages)

        return response.refinements

    def format_usecase_output(
        self, usecases: List[UsecaseRefinement], actors: List[ActorResult]
    ):
        formatted_usecases = []

        for usecase in usecases:
            usecase_list = usecase.refined + usecase.added
            actors_filter = set()
            for actor in actors:
                if usecase.sentence_idx in actor.sentence_idx:
                    actors_filter.add(actor.actor)
                for actor_alias in actor.aliases:
                    if usecase.sentence_idx in actor_alias.sentences:
                        actors_filter.add(actor.actor)
                        break
            for item in usecase_list:
                new_usecase_format = {
                    "name": item,
                    "participating_actors": list(actors_filter),
                    "relationship": [],
                }
                formatted_usecases = formatted_usecases + [new_usecase_format]

        print(json.dumps(formatted_usecases, indent=2, ensure_ascii=False))

        return
