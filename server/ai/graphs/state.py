import operator
from typing import Annotated, Dict, List, Literal, TypedDict

from pydantic import BaseModel, Field

from ai.graphs.rpa_graph.state import ActorResult, UseCase


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


class CompletenessEvaluation(BaseModel):
    score: int = Field(description="0-100")
    result: Literal["PASS", "FAIL"]
    rationale: str
    missing_or_weak_fields: List[str] = Field(default_factory=list)


class SimpleCriterionEvaluation(BaseModel):
    score: int = Field(description="0-100")
    result: Literal["PASS", "FAIL"]
    rationale: str


class UseCaseSpecValidation(BaseModel):
    passed: bool
    failed_criteria: Dict[str, str] = Field(default_factory=dict)
    regen_rationale: str = Field(default="")


class UseCaseEvaluation(BaseModel):
    Completeness: CompletenessEvaluation
    Coherence: SimpleCriterionEvaluation
    Relevance: SimpleCriterionEvaluation


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


class RpaState(TypedDict, total=False):
    requirement_text: str
    actors: List[str]
    actor_aliases: List[ActorResult]
    use_cases: List[UseCase]


class OrchestratorState(TypedDict, total=False):
    requirement_text: str
    # tasks: List[TaskItem]

    actors: List[str]
    actor_aliases: List[ActorResult]

    # RPA output
    use_cases: List[UseCase]

    # Map-reduce accumulator (Agent2 output)
    scenario_results_acc: Annotated[List[ScenarioResult], operator.add]

    # Final reduced view
    scenario_results: List[ScenarioResult]

    # Reduced view
    # merged_actors: List[str]
