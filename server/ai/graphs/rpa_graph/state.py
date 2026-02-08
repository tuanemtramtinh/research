# Define State
from typing import List, Literal, Optional, TypedDict
from langchain.chat_models import BaseChatModel
from pydantic import BaseModel, Field


class AliasItem(BaseModel):
    alias: str = Field(description="Name of the alias")
    sentences: List[int] = Field(
        description="List of sentence indices where THIS specific alias appears (starting from 1)"
    )


class ActorItem(BaseModel):
    """Actor with sentence indices where it appears."""

    actor: str = Field(description="Actor name")
    sentence_idx: List[int] = Field(description="Sentence indices (0-based)")


class CanonicalActorList(BaseModel):
    """List of canonical actor names after removing synonyms."""

    actors: List[str] = Field(
        description="A list of canonical actor names after removing synonyms."
    )


class ActorAliasItem(BaseModel):
    """Represents an alias for an actor with its sentence occurrences."""

    alias: str = Field(description="Name of the alias")
    sentences: List[int] = Field(
        description="List of sentence indices where THIS specific alias appears (0-based)"
    )


class ActorAliasMapping(BaseModel):
    """Maps an actor name to their aliases."""

    actor: str = Field(description="The canonical actor's name")
    aliases: List[ActorAliasItem] = Field(
        description="List of alternative names/references for this actor"
    )


class ActorAliasList(BaseModel):
    """Collection of actor-alias mappings."""

    mappings: List[ActorAliasMapping] = Field(
        description="List of actor-alias mappings"
    )


class ActorResult(BaseModel):
    """Final result combining actor, aliases, and sentence indices."""

    actor: str = Field(description="The canonical actor's name")
    aliases: List[ActorAliasItem] = Field(description="List of aliases")
    sentence_idx: List[int] = Field(
        description="Sentence indices where the actor appears"
    )


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


class UseCaseRelationship(BaseModel):
    type: str = Field(description="Relationship type, e.g. 'include' or 'extend'")
    target_use_case: str = Field(description="Target use case name")


class UserStoryItem(BaseModel):
    """Represents a single user story within a use case cluster."""

    actor: str = Field(description="Actor performing the action")
    action: str = Field(description="The action/usecase extracted from user story")
    original_sentence: str = Field(description="Original user story sentence")
    sentence_idx: int = Field(description="Index of the original sentence")


class UseCase(BaseModel):
    """Use case generated from clustered user stories."""

    id: int = Field(description="The id of the Usecase")

    name: str = Field(
        description="Use case name in verb-noun format, e.g. 'Browse Products', 'Manage Inventory'"
    )
    description: str = Field(
        default="", description="Brief description of what this use case does"
    )
    participating_actors: List[str] = Field(
        default_factory=list,
        description="Unique actors involved in this use case",
    )
    user_stories: List[UserStoryItem] = Field(
        default_factory=list,
        description="List of user stories that belong to this use case",
    )
    relationships: List[UseCaseRelationship] = Field(
        default_factory=list, description="Include/extend relationships"
    )


class RefinedClusterItem(BaseModel):
    """One user story item with its assigned cluster after refinement."""

    sentence_idx: int = Field(description="Index of the original sentence")
    actor: str = Field(description="Actor name")
    usecase: str = Field(description="The usecase/action text")
    target_cluster_id: int = Field(
        description="ID of the cluster this item should belong to (0-based)"
    )


class RefineClusteringResponse(BaseModel):
    """LLM output for refining K-Means clusters."""

    items: List[RefinedClusterItem] = Field(
        description="Each user story item with its target cluster id"
    )


class UseCaseNaming(BaseModel):
    """Single usecase naming result."""

    cluster_id: int = Field(description="The cluster ID")
    usecase_name: str = Field(
        description="A concise, verb-noun format usecase name (e.g., 'Manage Products', 'Process Orders')"
    )
    description: str = Field(description="Brief description of what this usecase does")


class UseCaseNamingResponse(BaseModel):
    """Response containing all usecase names."""

    usecases: List[UseCaseNaming] = Field(description="List of usecase naming results")


class UseCaseRelationshipItem(BaseModel):
    """A single relationship between two use cases."""

    source_use_case: str = Field(description="Use case that initiates the relationship")
    relationship_type: Literal["include", "extend"] = Field(
        description="Relationship type: 'include' (mandatory) or 'extend' (optional)"
    )
    target_use_case: str = Field(description="Target use case of the relationship")
    reasoning: str = Field(
        default="", description="Brief explanation why this relationship exists"
    )


class UseCaseRelationshipResponse(BaseModel):
    """Response containing all identified relationships."""

    relationships: List[UseCaseRelationshipItem] = Field(
        default_factory=list, description="List of identified relationships"
    )


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
