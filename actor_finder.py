"""
Actor Finder Module
Handles all actor-related operations including finding and alias detection.
"""

import re
from typing import List
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI


# =============================================================================
# DTO CLASSES (Data Transfer Objects for Actor)
# =============================================================================


class ActorItem(BaseModel):
    actor: str = Field()
    sentence_idx: List[int] = Field()

    def __str__(self):
        return f"{self.actor} (sentences: {self.sentence_idx})"

    def __repr__(self):
        return self.__str__()


class CanonicalActorList(BaseModel):
    """List of canonical actor names (used for synonym checking)."""

    actors: List[str] = Field(
        description="A list of canonical actor names after removing synonyms."
    )


class AliasItem(BaseModel):
    """Represents an alias for an actor with its sentence occurrences."""

    alias: str = Field(description="Name of the alias")
    sentences: List[int] = Field(
        description="List of sentence indices where THIS specific alias appears (starting from 0)"
    )

    def __str__(self):
        return f"'{self.alias}' -> sentences: {self.sentences}"


class ActorAliasMapping(BaseModel):
    """Maps an actor name to their aliases."""

    actor: str = Field(description="The canonical actor's name")
    aliases: List[AliasItem] = Field(
        description="List of alternative names/references for this actor"
    )

    def __str__(self):
        if not self.aliases:
            return f"Actor: '{self.actor}' (no aliases)"
        aliases_str = ", ".join(str(alias) for alias in self.aliases)
        return f"Actor: '{self.actor}' | Aliases: [{aliases_str}]"


class ActorAliasList(BaseModel):
    """Collection of actor-alias mappings."""
    mappings: List[ActorAliasMapping] = Field(description="List of actor-alias mappings")


class ActorResult(ActorAliasMapping):
    """Final result combining actor, aliases, and sentence indices."""

    sentence_idx: List[int] = Field(
        description="List of sentence indices where the canonical actor appears"
    )

    def __str__(self):
        aliases_str = ", ".join(str(alias) for alias in self.aliases) if self.aliases else "none"
        return f"Actor: '{self.actor}' | Sentences: {self.sentence_idx} | Aliases: [{aliases_str}]"

# =============================================================================
# ACTOR FINDER CLASS
# =============================================================================
class ActorFinder:
    """Handles all actor-related operations including finding and alias detection."""

    def __init__(self, llm: ChatOpenAI, sents: List[str]):
        self.llm = llm
        self.sents = sents

    def find_actors(self, input_text: str) -> List[ActorItem]:
        """Extract actors from user stories using regex pattern."""
        pattern = r"As\s+(?:a|an|the)\s+([^,]+)"

        # Dictionary to group occurrences by actor
        actor_occurrences = {}

        for i, sent in enumerate(self.sents):
            match = re.search(pattern, sent)
            if match:
                actor = match.group(1).strip()

                if actor not in actor_occurrences:
                    actor_occurrences[actor] = []

                actor_occurrences[actor].append(i)

        # Convert to list format
        actors = [
            ActorItem(actor=actor, sentence_idx=sent_indices)
            for actor, sent_indices in actor_occurrences.items()
        ]

        return actors

    def synonym_actors_check(self, actors: List[ActorItem]) -> List[ActorItem]:
        """Remove synonymous actors using LLM."""
        structured_llm = self.llm.with_structured_output(CanonicalActorList)

        # Only send actor names to LLM (without sentence_idx)
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

        messages = [("system", system_prompt), ("human", human_prompt)]

        response = structured_llm.invoke(messages)

        # Lookup sentence_idx from original actors
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

    def find_actors_alias(self, actors: List[ActorItem]) -> List[ActorResult]:
        """Find aliases for each canonical actor from sentences."""
        structured_llm = self.llm.with_structured_output(ActorAliasList)
        indexed_sents = "\n".join(f"{i}: {sent}" for i, sent in enumerate(self.sents))

        # Only send actor names to LLM (without sentence_idx)
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

        messages = [("system", system_prompt), ("human", human_prompt)]

        response: ActorAliasList = structured_llm.invoke(messages)
        lookup = {actor.actor: actor.sentence_idx for actor in actors}
        result = [ActorResult(actor=actor.actor, aliases=actor.aliases, sentence_idx=lookup[actor.actor]) for actor in response.mappings if actor.actor in lookup]
                        
        return result
