from typing import Any, Dict, List, Optional

from ai.graphs.sca_graph.state import ScenarioResult


def _evaluation_to_dict(ev) -> Optional[Dict[str, Any]]:
    """Convert a UseCaseEvaluation to a JSON-friendly dict with scores & sub-scores."""
    if ev is None:
        return None
    return {
        "completeness": {
            "score": ev.Completeness.score,
            "result": ev.Completeness.result,
            "rationale": ev.Completeness.rationale,
            "sub_scores": ev.Completeness.sub_scores,
            "missing_or_weak_fields": ev.Completeness.missing_or_weak_fields,
        },
        "correctness": {
            "score": ev.Correctness.score,
            "result": ev.Correctness.result,
            "rationale": ev.Correctness.rationale,
            "reference_path": ev.Correctness.reference_path,
            "sub_scores": ev.Correctness.sub_scores,
        },
        "relevance": {
            "score": ev.Relevance.score,
            "result": ev.Relevance.result,
            "rationale": ev.Relevance.rationale,
            "sub_scores": ev.Relevance.sub_scores,
        },
    }


def _scenario_result_to_response(sr: ScenarioResult) -> Dict[str, Any]:
    """Flatten a ScenarioResult into the API response shape."""
    return {
        "use_case": sr.use_case.model_dump(),
        "use_case_spec_json": sr.use_case_spec_json,
        "evaluation": _evaluation_to_dict(sr.evaluation),
        "comparison_spec_path": sr.comparison_spec_path,
        "comparison_evaluation": _evaluation_to_dict(sr.comparison_evaluation),
        "validation": {
            "passed": sr.validation.passed,
            "failed_criteria": sr.validation.failed_criteria,
            "regen_rationale": sr.validation.regen_rationale,
        },
    }
