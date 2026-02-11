"""SCA (Scenario-Completion Agent) state definitions for the server."""

from __future__ import annotations

import operator
from typing import Annotated, Any, Dict, List, Literal, Optional, TypedDict

from pydantic import BaseModel, Field

from ai.graphs.rpa_graph.state import UseCase


# ---------------------------------------------------------------------------
# Evaluation models
# ---------------------------------------------------------------------------


class CompletenessEvaluation(BaseModel):
    score: int = Field(description="0-100")
    result: Literal["PASS", "FAIL"]
    rationale: str
    sub_scores: Dict[str, int] = Field(
        default_factory=dict,
        description="Per-sub-criterion scores (e.g., 'Primary Actor': 15).",
    )
    missing_or_weak_fields: List[str] = Field(default_factory=list)


class CorrectnessEvaluation(BaseModel):
    """Optional correctness check against a reference scenario/spec.

    When no reference is provided, result should be 'N/A'.
    """

    score: Optional[int] = Field(default=None, description="0-100, or null when N/A")
    result: Literal["PASS", "FAIL", "N/A"]
    rationale: str
    reference_path: Optional[str] = None
    sub_scores: Dict[str, int] = Field(
        default_factory=dict,
        description="Per-sub-criterion correctness scores; empty when result is N/A.",
    )


class SimpleCriterionEvaluation(BaseModel):
    score: int = Field(description="0-100")
    result: Literal["PASS", "FAIL"]
    rationale: str
    sub_scores: Dict[str, int] = Field(
        default_factory=dict,
        description="Per-sub-criterion scores.",
    )


class UseCaseEvaluation(BaseModel):
    Completeness: CompletenessEvaluation
    Correctness: CorrectnessEvaluation
    Relevance: SimpleCriterionEvaluation


class UseCaseSpecJudgeResult(BaseModel):
    detector: str
    spec_version: int = Field(default=0)
    evaluation: UseCaseEvaluation


class UseCaseSpecValidation(BaseModel):
    passed: bool
    failed_criteria: Dict[str, str] = Field(default_factory=dict)
    regen_rationale: str = Field(default="")


# ---------------------------------------------------------------------------
# Scenario result
# ---------------------------------------------------------------------------


class ScenarioResult(BaseModel):
    use_case: UseCase
    use_case_spec_json: Dict[str, Any] = Field(
        default_factory=dict,
        description="Structured Use Case Specification as a JSON object.",
    )
    evaluation: Optional[UseCaseEvaluation] = None
    comparison_spec_path: Optional[str] = Field(
        default=None,
        description="Optional path to an external use case spec.",
    )
    comparison_evaluation: Optional[UseCaseEvaluation] = Field(
        default=None,
        description="Scores for comparison_spec_path.",
    )
    validation: UseCaseSpecValidation


# ---------------------------------------------------------------------------
# LangGraph state
# ---------------------------------------------------------------------------


class ScaState(TypedDict, total=False):
    requirement_text: List[str]
    actors: List[str]
    use_case: UseCase

    # SCA output: structured JSON object for the spec
    use_case_spec_json: Dict[str, Any]
    spec_version: int

    # Optional: per-node model configuration (writer/judges/combiner)
    model_configs: Dict[str, Dict[str, Any]]

    # Optional: per-use-case reference file for Correctness evaluation
    reference_spec_path: Optional[str]

    # Optional: per-use-case comparison spec path
    comparison_spec_path: Optional[str]

    # 3 judge results then combiner
    judge_results: Annotated[List[UseCaseSpecJudgeResult], operator.add]
    validation: UseCaseSpecValidation
