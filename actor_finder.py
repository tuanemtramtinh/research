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
class ActorList(BaseModel):
    """List of actors extracted from user stories."""
    actors: List[str] = Field(
        description="A list of actors who perform actions in the user stories."
    )


class AliasItem(BaseModel):
    """Represents an alias for an actor with its sentence occurrences."""
    alias: str = Field(description="Name of the alias")
    sentences: List[int] = Field(
        description="List of sentence indices where THIS specific alias appears (starting from 0)"
    )
    
    def __str__(self):
        return f"'{self.alias}' -> sentences: {self.sentences}"


class ActorAlias(BaseModel):
    """Maps an actor to their aliases."""
    actor: str = Field(description="The original actor's name")
    aliases: List[AliasItem] = Field(
        description="List of alternative names/references for this actor"
    )
    
    def __str__(self):
        if not self.aliases:
            return f"Actor: {self.actor} (no aliases)"
        aliases_str = ", ".join(str(alias) for alias in self.aliases)
        return f"Actor: {self.actor} | Aliases: [{aliases_str}]"


class ActorAliasList(BaseModel):
    """Collection of actor-alias mappings."""
    mappings: List[ActorAlias] = Field(
        description="List of actor-alias mappings"
    )


# =============================================================================
# ACTOR FINDER CLASS
# =============================================================================
class ActorFinder:
    """Handles all actor-related operations including finding and alias detection."""
    
    def __init__(self, llm: ChatOpenAI, sents: List[str]):
        self.llm = llm
        self.sents = sents
    
    def find_actors(self, input_text: str) -> List[str]:
        """Extract actors from user stories using regex pattern."""
        pattern = r"As\s+(?:a|an|the)\s+([^,]+)"
        actors = set(
            map(lambda sent: re.search(pattern, sent).group(1).strip(), self.sents)
        )
        return list(actors)
    
    def synonym_actors_check(self, actors: List[str]) -> List[str]:
        """Remove synonymous actors using LLM."""
        structured_llm = self.llm.with_structured_output(ActorList)
        
        system_prompt = """
        You are a Business Analyst AI specializing in requirement analysis.

        Your task is to analyze a list of actors and remove synonymous or semantically equivalent actors.

        Rules:
        - Actors that represent the same logical role MUST be merged.
        - Choose ONE clear and generic canonical name for each group.
        - Prefer business-level, role-based names over wording variants.
        - ALL returned actor names MUST be lowercase.
        - Do NOT invent new actors that are not implied by the list.
        - Do NOT explain your reasoning.
        - Return only structured data according to the output schema.
        """
        
        human_prompt = f"""
        The following is a list of actors extracted from user stories.

        Actors:
        {actors}

        Remove synonymous actors and return a list of unique canonical actors.
        """
        
        messages = [
            ("system", system_prompt),
            ("human", human_prompt)
        ]
        
        responses = structured_llm.invoke(messages)
        return responses.actors
    
    def find_actors_alias(self, actors: List[str]) -> List[ActorAlias]:
        """Find aliases for each canonical actor from sentences."""
        structured_llm = self.llm.with_structured_output(ActorAliasList)
        indexed_sents = "\n".join(
            f"{i}: {sent}" for i, sent in enumerate(self.sents)
        )
        
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
        Canonical actors:
        {actors}

        User story sentences (with indices):
        {indexed_sents}

        For each canonical actor, find all aliases used in the sentences above and list the sentence indices where each alias appears.
        """
        
        messages = [
            ("system", system_prompt),
            ("human", human_prompt)
        ]
        
        response = structured_llm.invoke(messages)
        return response.mappings
