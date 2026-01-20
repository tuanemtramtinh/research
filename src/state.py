from __future__ import annotations

from typing import Annotated, Dict, List, Literal, Optional, TypedDict

import operator

from pydantic import BaseModel, Field


# class TaskItem(BaseModel):
#     id: int = Field(description="Task id starting from 1")
#     text: str = Field(description="Task text (a chunk of the requirement)")


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


class UseCase(BaseModel):
    name: str = Field(description="Use case name (verb phrase), e.g. 'Borrow Book'")
    participating_actors: List[str] = Field(
        default_factory=list, description="Actors participating in this use case"
    )
    sentence_id: int = Field(
        default=0, description="Which task/sentence produced this use case"
    )
    sentence: str = Field(default="", description="Original sentence text")
    relationships: List[UseCaseRelationship] = Field(default_factory=list)


class Scenario(BaseModel):
    use_case_name: str = Field(
        description="Use case name this scenario is derived from"
    )
    actors: List[str] = Field(description="Actors involved in the scenario")
    preconditions: List[str] = Field(default_factory=list, description="Preconditions")
    trigger: str = Field(default="", description="Trigger that starts the scenario")
    main_flow: List[str] = Field(
        default_factory=list, description="Main success flow steps"
    )
    alternate_flows: List[str] = Field(
        default_factory=list, description="Alternate/exception flows"
    )
    postconditions: List[str] = Field(
        default_factory=list, description="Postconditions"
    )


class ScenarioFieldCheck(BaseModel):
    """Per-field check results from a single detector.

    Each detector must score the SAME 3 criteria for each Scenario field.
    The combiner then applies a 2/3 rule across detectors per (field, criterion).
    """

    detector: str = Field(description="Detector name")
    field: str = Field(description="Scenario field name")
    scenario_version: int = Field(
        default=0,
        description="Scenario version/id these checks refer to (used to support regeneration loops)",
    )
    criteria: List["ScenarioCriterionCheck"] = Field(
        default_factory=list,
        description="List of 3 criteria checks (score/pass/rationale) for this field by this detector",
    )


CriterionName = Literal["c1", "c2", "c3"]


class ScenarioCriterionCheck(BaseModel):
    criterion: CriterionName = Field(
        description="Criterion id/name shared across all detectors"
    )
    score: float = Field(description="Criterion score (scale decided by detectors)")
    passed: bool = Field(description="Pass/fail for this criterion")
    rationale: str = Field(
        default="", description="Rationale for this criterion result"
    )


class ScenarioValidation(BaseModel):
    passed: bool = Field(description="True if the scenario passed all checks")
    # field -> criterion -> combined rationale across detectors
    failed_fields: Dict[str, Dict[str, str]] = Field(
        default_factory=dict,
        description="Map: field -> (criterion -> combined rationale across detectors)",
    )
    regen_rationale: str = Field(
        default="",
        description="Combined rationale/instructions for regenerating the scenario",
    )


class ScenarioResult(BaseModel):
    use_case: UseCase
    # Back-compat: older pipeline returned a structured Scenario.
    # New SCA pipeline returns a fully-dressed Use Case Specification string.
    scenario: Scenario | None = None
    use_case_spec: str = Field(
        default="", description="Fully-dressed use case specification (template text)"
    )
    evaluation: UseCaseEvaluation | None = None
    validation: UseCaseSpecValidation | ScenarioValidation


class CompletenessEvaluation(BaseModel):
    score: int = Field(description="0-100")
    result: Literal["PASS", "FAIL"]
    rationale: str
    missing_or_weak_fields: List[str] = Field(default_factory=list)


class SimpleCriterionEvaluation(BaseModel):
    score: int = Field(description="0-100")
    result: Literal["PASS", "FAIL"]
    rationale: str


class UseCaseEvaluation(BaseModel):
    Completeness: CompletenessEvaluation
    Coherence: SimpleCriterionEvaluation
    Relevance: SimpleCriterionEvaluation


class UseCaseSpecJudgeResult(BaseModel):
    detector: str
    spec_version: int = Field(default=0)
    evaluation: UseCaseEvaluation


class UseCaseSpecValidation(BaseModel):
    passed: bool
    failed_criteria: Dict[str, str] = Field(default_factory=dict)
    regen_rationale: str = Field(default="")


# class MainState(TypedDict, total=False):
#     requirement_text: str
#     # tasks: List[TaskItem]

#     # Map-reduce: worker results are accumulated here
#     results: List[RpaResult]

#     # Agent-2 output
#     use_cases: List[UseCase]

#     # Agent-3 output (scenario generation)
#     scenario_results: List[ScenarioResult]

#     # Convenience: reduced/merged view
#     merged_actors: List[str]
#     merged_actor_aliases: List[ActorAlias]


class ScaState(TypedDict, total=False):
    requirement_text: str
    actors: List[str]
    use_case: UseCase
    # SCA output (new): fully-dressed use case specification text
    use_case_spec: str
    spec_version: int

    # Back-compat (old): structured Scenario
    scenario: Scenario
    scenario_version: int

    # New judging: 3 judge results then combiner
    judge_results: Annotated[List[UseCaseSpecJudgeResult], operator.add]
    validation: UseCaseSpecValidation


class RpaState(TypedDict, total=False):
    requirement_text: str
    # tasks: List[TaskItem]
    actors: List[str]
    actor_aliases: List[ActorResult]
    use_cases: List[UseCase]


# For LangGraph reducers
ADD_RPA_RESULTS = operator.add


ADD_USECASES = operator.add


ADD_FIELD_CHECKS = operator.add


ADD_SCENARIO_RESULTS = operator.add


ADD_JUDGE_RESULTS = operator.add
