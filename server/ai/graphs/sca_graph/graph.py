"""SCA graph: generate a fully-dressed Use Case Specification, evaluate with 3 judges, and optionally regenerate."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError as PydanticValidationError

from ai.graphs.rpa_graph.state import UseCase
from ai.graphs.sca_graph.state import (
    CompletenessEvaluation,
    CorrectnessEvaluation,
    ScaState,
    ScenarioResult,
    SimpleCriterionEvaluation,
    UseCaseEvaluation,
    UseCaseSpecJudgeResult,
    UseCaseSpecValidation,
)


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

_DEFAULT_MODEL_CONFIG: Dict[str, Any] = {
    "provider": "openai",
    "name": "gpt-5-mini",
    "temperature": 0,
}

_MODEL_CONFIG_1: Dict[str, Any] = {
    "provider": "openai",
    "name": "gpt-5-mini",
    "temperature": 0,
}

_MODEL_CONFIG_2: Dict[str, Any] = {
    "provider": "openai",
    "name": "gpt-5-mini",
    "temperature": 0,
}

_MODEL_CONFIG_3: Dict[str, Any] = {
    "provider": "openai",
    "name": "gpt-5-mini",
    "temperature": 0,
}

_MODEL_CACHE: Dict[Tuple[str, str, float], Any] = {}


def _get_model(model_config: Optional[dict] = None):
    load_dotenv()

    cfg: Dict[str, Any] = dict(_DEFAULT_MODEL_CONFIG)
    if isinstance(model_config, dict):
        cfg.update({k: v for k, v in model_config.items() if v is not None})

    provider = str(cfg.get("provider") or "openai")
    model_name = str(cfg.get("name") or "gpt-4.1")
    temperature = float(
        cfg.get("temperature") if cfg.get("temperature") is not None else 0
    )

    if provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        return None

    cache_key = (provider, model_name, temperature)
    cached = _MODEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    try:
        model = init_chat_model(model_name, model_provider=provider, temperature=temperature)
    except TypeError:
        model = init_chat_model(model_name, model_provider=provider)

    _MODEL_CACHE[cache_key] = model
    return model


def _get_model_for(state: ScaState, key: str):
    model_configs = state.get("model_configs")
    cfg = None
    if isinstance(model_configs, dict):
        cfg = model_configs.get(key)
    return _get_model(cfg)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _use_case_payload(use_case: UseCase) -> dict:
    try:
        return use_case.model_dump()
    except Exception:
        return {
            "id": int(getattr(use_case, "id", 0) or 0),
            "name": getattr(use_case, "name", ""),
            "description": getattr(use_case, "description", "") or "",
            "participating_actors": getattr(use_case, "participating_actors", []) or [],
            "user_stories": [
                getattr(s, "model_dump", lambda: s)()
                for s in (getattr(use_case, "user_stories", []) or [])
            ],
            "relationships": [
                getattr(r, "model_dump", lambda: r)()
                for r in (getattr(use_case, "relationships", []) or [])
            ],
        }


def _heuristic_use_case_spec(
    *, requirement_text: str, actors: List[str], use_case: UseCase
) -> dict:
    uc = _use_case_payload(use_case)
    uid = f"UC-{int(uc.get('id') or 0)}"
    primary_actors = [
        str(a)
        for a in (getattr(use_case, "participating_actors", None) or [])
        if str(a).strip()
    ]
    supporting_actors = [
        str(a)
        for a in (actors or [])
        if str(a).strip() and str(a).strip() not in set(primary_actors)
    ]

    trig_raw = (getattr(use_case, "description", "") or "").strip()
    if not trig_raw and getattr(use_case, "user_stories", None):
        parts = [
            getattr(s, "original_sentence", "") or getattr(s, "action", "")
            for s in (use_case.user_stories or [])[:3]
        ]
        trig_raw = " ".join(p for p in parts if p).strip()
    trig = trig_raw or f"A primary actor initiates '{use_case.name}'."

    include_rels = [
        r for r in (uc.get("relationships") or [])
        if str(r.get("type", "")).lower() == "include"
    ]
    extend_rels = [
        r for r in (uc.get("relationships") or [])
        if str(r.get("type", "")).lower() == "extend"
    ]

    main_steps: List[str] = []
    step_no = 1
    for r in include_rels:
        target = str(r.get("target_use_case", "")).strip()
        if not target:
            continue
        main_steps.append(
            f"{step_no}. Primary Actor requests to perform mandatory sub-flow «include» {target} → System completes «include» {target}."
        )
        step_no += 1

    main_steps.append(
        f"{step_no}. Primary Actor initiates '{use_case.name}' → System begins processing the request."
    )
    step_no += 1
    main_steps.append(
        f"{step_no}. Primary Actor provides required input → System validates the input and proceeds."
    )
    step_no += 1
    main_steps.append(
        f"{step_no}. Primary Actor confirms completion → System completes '{use_case.name}' and records the outcome."
    )

    af_lines: List[str] = []
    for i, r in enumerate(extend_rels, 1):
        target = str(r.get("target_use_case", "")).strip()
        if not target:
            continue
        af_lines.append(
            f"AF-{i} (from Step 2): <condition to trigger «extend» {target}>\n"
            f"1. Primary Actor requests optional sub-flow «extend» {target} → System performs «extend» {target}.\n"
            "Return to Step 3."
        )

    ef_lines = [
        "EF-1 (from Step 2): <system cannot process the request>\n"
        "1. Primary Actor submits the request → System displays an error and ends the use case."
    ]

    spec_obj = {
        "use_case_name": str(getattr(use_case, "name", "") or "").strip(),
        "unique_id": uid,
        "area": "Requirements Domain",
        "context_of_use": "The primary actor performs the use case within the normal operation of the system based on the provided requirement text.",
        "scope": "The target system described by the requirement text.",
        "level": "User-goal",
        "primary_actors": primary_actors or ["Primary Actor"],
        "supporting_actors": supporting_actors or ["System"],
        "stakeholders_and_interests": [
            "Primary Actor: Achieve the goal expressed by the use case with a verifiable outcome.",
            "System Owner: Ensure the use case is executed consistently and traceably.",
        ],
        "description": (
            str(getattr(use_case, "description", "") or "").strip()
            or f"Enable the primary actor to complete '{use_case.name}' as described in the requirement text."
        ),
        "triggering_event": trig,
        "trigger_type": "External",
        "preconditions": [
            "The system is available and able to accept requests.",
            "The primary actor has access to the system interface required to initiate the use case.",
        ],
        "postconditions": [
            f"The system records a completed outcome for '{use_case.name}'.",
            "Any relevant system state changes are persisted.",
        ],
        "assumptions": [
            "The requirement text provides sufficient context to identify actors and intended outcomes.",
        ],
        "requirements_met": [
            f"The system shall support the use case '{use_case.name}'.",
        ],
        "priority": "Medium",
        "risk": "Medium",
        "outstanding_issues": [
            "Confirm domain/area classification and any business policy constraints implied by the requirement text.",
        ],
        "main_flow": main_steps,
        "alternative_flows": af_lines or ["AF-1 (from Step 2): <valid variation>"],
        "exception_flows": ef_lines,
        "information_for_steps": [
            "1. Request",
            "2. Input Data",
            "3. Confirmation",
        ],
        "input": {
            "requirement_text": requirement_text,
            "actors": [str(a) for a in (actors or [])],
            "target_use_case": _use_case_payload(use_case),
        },
    }
    return spec_obj


def _extract_json_object(text: str) -> dict:
    raw = (text or "").strip()
    if not raw:
        raise ValueError("Empty model output")
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        candidate = raw[start : end + 1]
        obj = json.loads(candidate)
        if isinstance(obj, dict):
            return obj
    raise ValueError("Could not parse JSON object from model output")


def _resolve_reference_path(path_str: str) -> Path:
    p = Path(path_str).expanduser()
    if not p.is_absolute():
        repo_root = Path(__file__).resolve().parents[3]
        p = (repo_root / p).resolve()
    return p


def _read_reference_text(path_str: str) -> str:
    p = _resolve_reference_path(path_str)
    return p.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SCA_UC_PROMPT_PATH = Path(__file__).resolve().parents[3] / "docs" / "sca_use_case_spec_json_prompt.md"
_SCA_UC_PROMPT_CACHE: str | None = None


def _load_sca_uc_prompt() -> str:
    global _SCA_UC_PROMPT_CACHE
    if _SCA_UC_PROMPT_CACHE is not None:
        return _SCA_UC_PROMPT_CACHE
    try:
        _SCA_UC_PROMPT_CACHE = _SCA_UC_PROMPT_PATH.read_text(encoding="utf-8").strip()
    except Exception:
        _SCA_UC_PROMPT_CACHE = ""
    return _SCA_UC_PROMPT_CACHE


_WRITER_SYSTEM_PROMPT = (
    "You are a professional, strict, and detail-oriented Use Case Specification Writer. "
    "You output ONLY a single JSON object and nothing else. "
    "You never leave any required field empty. "
    "You only infer information that is logically supported by the given Requirement Text. "
    "If information is implicit, you must state it explicitly using formal requirement language."
)

_WRITER_HUMAN_PROMPT_TEMPLATE = r"""
Your task is to generate EXACTLY ONE complete and fully-dressed Use Case Specification
based ONLY on the given input.

The output must be suitable for formal Software Requirement Specification (SRS) documents.

========================
CRITICAL ENFORCEMENT RULES
========================
1. You MUST output ONLY a single JSON object (no prose, no markdown, no code fences).
2. EVERY field in the JSON schema below MUST be present and MUST be non-empty.
3. Do NOT output placeholder values such as "N/A", "None", "Not specified", "-" or "".
4. If a field is not explicitly stated in the Requirement Text:
   - You MUST infer it conservatively and logically.
   - The inference MUST NOT introduce new system functionality.
5. If inference is required, ensure consistency by:
   - Reflecting it in Assumptions
   - Keeping it aligned with the Requirement Text
6. Do NOT invent:
   - New actors
   - New system capabilities
   - New business rules
7. Use only formal, neutral, system-oriented language.
8. Generate ONLY ONE use case.
9. If the Target Use Case contains UML-like relationships (see `relationships` in the Target Use Case input object):
    - For each relationship with type "include": you MUST represent it as a mandatory sub-flow in `main_flow`.
    - For each relationship with type "extend": you MUST represent it as an optional variation in `alternative_flows`.
    - You MUST NOT invent new relationships that are not present in the Target Use Case.

========================
OUTPUT JSON SCHEMA (MUST FOLLOW EXACTLY)
========================

{
    "use_case_name": "<non-empty string>",
    "unique_id": "<non-empty string>",
    "area": "<non-empty string>",
    "context_of_use": "<non-empty string>",
    "scope": "<non-empty string>",
    "level": "Summary | User-goal | Sub-function",
    "primary_actors": ["<non-empty string>", "..."],
    "supporting_actors": ["<non-empty string>", "..."],
    "stakeholders_and_interests": ["<non-empty string>", "..."],
    "description": "<non-empty string>",
    "triggering_event": "<non-empty string>",
    "trigger_type": "External | Temporal",
    "preconditions": ["<non-empty string>", "..."],
    "postconditions": ["<non-empty string>", "..."],
    "assumptions": ["<non-empty string>", "..."],
    "requirements_met": ["<non-empty string>", "..."],
    "priority": "Low | Medium | High",
    "risk": "Low | Medium | High",
    "outstanding_issues": ["<non-empty string>", "..."],
    "main_flow": ["1. <Actor action> → <System response>", "2. ..."],
    "alternative_flows": ["AF-1 (from Step X): ...", "AF-2 (from Step Y): ..."],
    "exception_flows": ["EF-1 (from Step X): ..."],
    "information_for_steps": ["1. <Data>", "2. <Data>", "3. <Data>"]
}

========================
INPUT (DO NOT MODIFY)
========================
Requirement Text:
{{requirement_text}}

Actors:
{{actors}}

Target Use Case:
{{single_use_case_object}}

========================
OUTPUT RULE (ABSOLUTE)
========================
- Output ONLY the JSON object.
"""

_JUDGE_SYSTEM_PROMPT = (
    "You are acting as a STRICT, RULE-BASED Use Case Specification JUDGE."
)

_JUDGE_HUMAN_PROMPT_TEMPLATE = r"""
Your role is ONLY to EVALUATE, not to improve, rewrite, or suggest changes.
You must assign scores objectively and consistently so that different evaluators would reach the same result.

You must assess a COMPLETE USE CASE SPECIFICATION exactly as provided,
without assuming missing information or filling any gaps.

All judgments MUST strictly follow the evaluation criteria and scoring rules defined below.
If any required field is missing, unclear, or weakly defined, you MUST treat it as a deficiency.

You will evaluate the use case using EXACTLY THREE criteria:
1. Completeness
2. Correctness
3. Relevance

You MUST output results in the specified JSON format ONLY.

--------------------------------
INPUT
--------------------------------
<REFERENCE_SCENARIO>
{{reference_scenario}}
</REFERENCE_SCENARIO>

<USE_CASE_SPEC>
{{use_case_spec}}
</USE_CASE_SPEC>

--------------------------------
EVALUATION CRITERIA
--------------------------------

================================
1. COMPLETENESS (0-100)
================================
Sub-scores:
- Primary Actor (0-15): 15=exactly one correct external actor, 10=vague, 0=system/internal
- Use Case Name (0-10): 10=verb+object user goal, 5=vague, 0=technical
- Preconditions (0-10): 10=explicit verifiable states, 5=vague, 0=missing/actions
- Postconditions (0-10): 10=measurable states, 5=vague, 0=missing
- Stakeholders & Interests (0-5): 5=all listed with interests, 2=vague, 0=missing
- Main Flow (0-25): 25=complete actor-system steps, 15=gaps, 0=not success path
- Alternative Flows (0-15): 15=all variations with step refs, 10=incomplete, 0=missing
- Exception Flows (0-10): 10=clear with step refs, 5=generic, 0=missing
PASS >= 70, FAIL < 70

================================
2. CORRECTNESS (0-100)
================================
Compare against REFERENCE_SCENARIO. If reference is empty → result="N/A", score=null.
Sub-scores:
- Primary Actor (0-20)
- Use Case Name (0-20)
- Main Success Scenario (0-25)
- Alternative Flows (0-15)
- Exception Flows (0-10)
- Preconditions (0-5)
- Postconditions (0-5)
PASS >= 70, FAIL < 70

================================
3. RELEVANCE (0-100)
================================
Sub-scores:
- Primary Actor ↔ Use Case Name (0-15)
- Use Case Name ↔ Main Flow (0-25)
- Main Flow ↔ Alternative Flows (0-20)
- Preconditions & Trigger ↔ Main Flow (0-10)
- (Main Flow & Alternative Flows) ↔ Postconditions (0-15)
- Stakeholders & Interests ↔ Postconditions (0-15)
PASS >= 70, FAIL < 70

--------------------------------
OUTPUT FORMAT (STRICT JSON)
--------------------------------
{
  "Completeness": {
    "score": <0-100>, "result": "PASS|FAIL", "rationale": "...",
    "sub_scores": {"Primary Actor":<0-15>,"Use Case Name":<0-10>,"Preconditions":<0-10>,"Postconditions":<0-10>,"Stakeholders & Interests":<0-5>,"Main Flow":<0-25>,"Alternative Flows":<0-15>,"Exception Flows":<0-10>},
    "missing_or_weak_fields": ["..."]
  },
  "Correctness": {
    "score": <0-100 or null>, "result": "PASS|FAIL|N/A", "rationale": "...",
    "reference_path": "<string or null>",
    "sub_scores": {"Primary Actor":<0-20>,"Use Case Name":<0-20>,"Main Success Scenario (MSS)":<0-25>,"Alternative Flows":<0-15>,"Exception Flows":<0-10>,"Preconditions":<0-5>,"Postconditions":<0-5>}
  },
  "Relevance": {
    "score": <0-100>, "result": "PASS|FAIL", "rationale": "...",
    "sub_scores": {"Primary Actor ↔ Use Case Name":<0-15>,"Use Case Name ↔ Main Flow":<0-25>,"Main Flow ↔ Alternative Flows":<0-20>,"Preconditions & Trigger ↔ Main Flow":<0-10>,"(Main Flow & Alternative Flows) ↔ Postconditions":<0-15>,"Stakeholders & Interests ↔ Postconditions":<0-15>}
  }
}

If <REFERENCE_SCENARIO> is empty set Correctness.result="N/A", score=null, sub_scores={}.
"""


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------


def generate_use_case_spec_node(state: ScaState):
    model = _get_model_for(state, "writer")
    use_case: UseCase = state.get("use_case")  # type: ignore[assignment]
    requirement_text = str(state.get("requirement_text") or "")
    actors = state.get("actors") or (use_case.participating_actors or [])
    spec_version = int(state.get("spec_version") or 0) + 1

    if model is None:
        spec_obj = _heuristic_use_case_spec(
            requirement_text=requirement_text,
            actors=[str(a) for a in (actors or [])],
            use_case=use_case,
        )
        return {"use_case_spec_json": spec_obj, "spec_version": spec_version}

    use_case_json = json.dumps(_use_case_payload(use_case), ensure_ascii=False, indent=2)
    prompt = (
        _WRITER_HUMAN_PROMPT_TEMPLATE
        .replace("{{requirement_text}}", requirement_text)
        .replace("{{actors}}", json.dumps(list(actors or []), ensure_ascii=False))
        .replace("{{single_use_case_object}}", use_case_json)
    )

    sca_uc_prompt = _load_sca_uc_prompt()
    messages = [("system", _WRITER_SYSTEM_PROMPT)]
    if sca_uc_prompt:
        messages.append(("system", sca_uc_prompt))
    messages.append(("human", prompt))

    resp = model.invoke(messages)
    content = str(getattr(resp, "content", "") or "").strip()
    spec_obj = _extract_json_object(content)
    return {"use_case_spec_json": spec_obj, "spec_version": spec_version}


def regenerate_use_case_spec_node(state: ScaState):
    model = _get_model_for(state, "writer")
    use_case: UseCase = state.get("use_case")  # type: ignore[assignment]
    requirement_text = str(state.get("requirement_text") or "")
    actors = state.get("actors") or (use_case.participating_actors or [])
    spec_version = int(state.get("spec_version") or 0) + 1

    current_spec_obj = state.get("use_case_spec_json") or {}
    validation: UseCaseSpecValidation | None = state.get("validation")  # type: ignore[assignment]
    issues = (
        (getattr(validation, "regen_rationale", "") or "").strip() if validation else ""
    )

    if model is None:
        return {
            "use_case_spec_json": dict(current_spec_obj) if isinstance(current_spec_obj, dict) else {},
            "spec_version": spec_version,
        }

    use_case_json = json.dumps(_use_case_payload(use_case), ensure_ascii=False, indent=2)
    prompt = (
        _WRITER_HUMAN_PROMPT_TEMPLATE
        .replace("{{requirement_text}}", requirement_text)
        .replace("{{actors}}", json.dumps(list(actors or []), ensure_ascii=False))
        .replace("{{single_use_case_object}}", use_case_json)
    )

    system_prompt = (
        "You are a professional, strict Use Case Specification Writer. "
        "You output ONLY a single JSON object and nothing else. "
        "Regenerate the use case specification to address the judge-identified deficiencies. "
        "Only change what is necessary and keep all content consistent with the Requirement Text."
    )
    human_prompt = f"""{prompt}

---
CURRENT USE CASE SPECIFICATION:
{json.dumps(current_spec_obj, ensure_ascii=False, indent=2)}

---
DEFICIENCIES TO FIX (DO NOT IGNORE):
{issues}

Output ONLY the corrected JSON object using the exact schema.
"""

    sca_uc_prompt = _load_sca_uc_prompt()
    messages = [("system", system_prompt)]
    if sca_uc_prompt:
        messages.append(("system", sca_uc_prompt))
    messages.append(("human", human_prompt))

    resp = model.invoke(messages)
    content = str(getattr(resp, "content", "") or "").strip()
    spec_obj = _extract_json_object(content)
    return {"use_case_spec_json": spec_obj, "spec_version": spec_version}


def _build_evaluation_from_partial(
    pve: PydanticValidationError,
    reference_spec_path: str | None = None,
) -> UseCaseEvaluation:
    raw: dict = {}
    for err in pve.errors():
        iv = err.get("input")
        if isinstance(iv, dict):
            raw = iv
            break
    if not raw:
        raise ValueError("Could not extract partial data from ValidationError")

    comp_raw = raw.get("Completeness")
    comp = (
        CompletenessEvaluation(**comp_raw)
        if isinstance(comp_raw, dict)
        else CompletenessEvaluation(
            score=0, result="FAIL", rationale="Not returned by LLM.",
            sub_scores={}, missing_or_weak_fields=[],
        )
    )

    corr_raw = raw.get("Correctness")
    corr = (
        CorrectnessEvaluation(**corr_raw)
        if isinstance(corr_raw, dict)
        else CorrectnessEvaluation(
            score=None, result="N/A",
            rationale="Not returned by LLM; defaulting to N/A.",
            reference_path=(reference_spec_path.strip() if isinstance(reference_spec_path, str) else None),
            sub_scores={},
        )
    )

    rel_raw = raw.get("Relevance")
    rel = (
        SimpleCriterionEvaluation(**rel_raw)
        if isinstance(rel_raw, dict)
        else SimpleCriterionEvaluation(
            score=0, result="FAIL", rationale="Not returned by LLM.", sub_scores={},
        )
    )

    return UseCaseEvaluation(Completeness=comp, Correctness=corr, Relevance=rel)


def _judge_node(detector_name: str, *, model_key: str | None = None):
    def _fn(state: ScaState):
        model = _get_model_for(state, model_key or detector_name)
        spec_obj = state.get("use_case_spec_json") or {}
        spec = json.dumps(spec_obj, ensure_ascii=False, indent=2)
        spec_version = int(state.get("spec_version") or 0)
        reference_spec_path = state.get("reference_spec_path")

        reference_text = ""
        if isinstance(reference_spec_path, str) and reference_spec_path.strip():
            try:
                reference_text = _read_reference_text(reference_spec_path).strip()
            except Exception:
                reference_text = ""

        def _heuristic_eval() -> dict:
            missing: List[str] = []
            required_keys = [
                "use_case_name", "primary_actors", "stakeholders_and_interests",
                "preconditions", "postconditions", "main_flow",
                "alternative_flows", "exception_flows",
            ]
            presence: Dict[str, bool] = {}
            spec_obj_local: dict = {}
            try:
                spec_obj_local = _extract_json_object(spec)
                for key in required_keys:
                    val = spec_obj_local.get(key)
                    if val is None:
                        missing.append(key); presence[key] = False; continue
                    if isinstance(val, str) and not val.strip():
                        missing.append(key); presence[key] = False; continue
                    if isinstance(val, list) and len([x for x in val if str(x).strip()]) == 0:
                        missing.append(key); presence[key] = False; continue
                    presence[key] = True
            except Exception:
                missing = required_keys[:]
                for key in required_keys:
                    presence[key] = False

            comp_sub_scores: Dict[str, int] = {
                "Primary Actor": 15
                if presence.get("primary_actors")
                and isinstance(spec_obj_local.get("primary_actors"), list)
                and len([x for x in (spec_obj_local.get("primary_actors") or []) if str(x).strip()]) == 1
                else 0,
                "Use Case Name": 10 if presence.get("use_case_name") else 0,
                "Preconditions": 10 if presence.get("preconditions") else 0,
                "Postconditions": 10 if presence.get("postconditions") else 0,
                "Stakeholders & Interests": 5 if presence.get("stakeholders_and_interests") else 0,
                "Main Flow": 25 if presence.get("main_flow") else 0,
                "Alternative Flows": 15 if presence.get("alternative_flows") else 0,
                "Exception Flows": 10 if presence.get("exception_flows") else 0,
            }
            comp_score = int(sum(comp_sub_scores.values()))
            completeness_pass = comp_score >= 70 and len(missing) == 0

            correctness_obj = {
                "score": None, "result": "N/A",
                "rationale": "No reference scenario was provided; correctness evaluation was skipped.",
                "reference_path": (reference_spec_path.strip() if isinstance(reference_spec_path, str) else None),
                "sub_scores": {},
            }

            rel_sub_scores: Dict[str, int] = {
                "Primary Actor ↔ Use Case Name": 15 if presence.get("primary_actors") and presence.get("use_case_name") else 0,
                "Use Case Name ↔ Main Flow": 25 if presence.get("use_case_name") and presence.get("main_flow") else 0,
                "Main Flow ↔ Alternative Flows": 20
                if presence.get("main_flow") and presence.get("alternative_flows")
                and any("from step" in str(x).lower() for x in (spec_obj_local.get("alternative_flows") or []))
                else (10 if presence.get("alternative_flows") else 0),
                "Preconditions & Trigger ↔ Main Flow": 10
                if presence.get("preconditions")
                and bool(str(spec_obj_local.get("triggering_event", "") or "").strip())
                and presence.get("main_flow")
                else 0,
                "(Main Flow & Alternative Flows) ↔ Postconditions": 15 if presence.get("postconditions") and presence.get("main_flow") else 0,
                "Stakeholders & Interests ↔ Postconditions": 15 if presence.get("stakeholders_and_interests") and presence.get("postconditions") else 0,
            }
            rel_score = int(sum(rel_sub_scores.values()))

            evaluation = UseCaseEvaluation(
                Completeness={
                    "score": comp_score,
                    "result": "PASS" if completeness_pass else "FAIL",
                    "rationale": "All required JSON fields are present." if completeness_pass else "Missing required JSON fields.",
                    "sub_scores": comp_sub_scores,
                    "missing_or_weak_fields": missing,
                },
                Correctness=correctness_obj,
                Relevance={
                    "score": rel_score,
                    "result": "PASS" if rel_score >= 70 else "FAIL",
                    "rationale": "Heuristic relevance scoring (no LLM configured).",
                    "sub_scores": rel_sub_scores,
                },
            )
            jr = UseCaseSpecJudgeResult(detector=detector_name, spec_version=spec_version, evaluation=evaluation)
            return {"judge_results": [jr]}

        if model is None:
            return _heuristic_eval()

        structured_llm = model.with_structured_output(UseCaseEvaluation, method="function_calling")
        prompt = (
            _JUDGE_HUMAN_PROMPT_TEMPLATE
            .replace("{{use_case_spec}}", spec)
            .replace("{{reference_scenario}}", reference_text)
        )
        try:
            evaluation = structured_llm.invoke(
                [("system", _JUDGE_SYSTEM_PROMPT), ("human", prompt)]
            )
        except PydanticValidationError as pve:
            try:
                evaluation = _build_evaluation_from_partial(pve, reference_spec_path)
            except Exception:
                return _heuristic_eval()
        except Exception:
            return _heuristic_eval()

        # Hard guard: if no reference, correctness must be N/A
        if not (reference_text or "").strip():
            try:
                if getattr(evaluation, "Correctness", None) is not None:
                    evaluation.Correctness.result = "N/A"
                    evaluation.Correctness.score = None
                    evaluation.Correctness.rationale = "No reference scenario was provided; correctness evaluation was skipped."
                    evaluation.Correctness.reference_path = None
                    evaluation.Correctness.sub_scores = {}
            except Exception:
                pass

        try:
            if getattr(evaluation, "Correctness", None) is not None:
                if getattr(evaluation.Correctness, "reference_path", None) in (None, ""):
                    evaluation.Correctness.reference_path = (
                        reference_spec_path.strip() if isinstance(reference_spec_path, str) else None
                    )
        except Exception:
            pass

        jr = UseCaseSpecJudgeResult(detector=detector_name, spec_version=spec_version, evaluation=evaluation)
        return {"judge_results": [jr]}

    return _fn


def combiner_node(state: ScaState):
    model = _get_model_for(state, "summarizer")
    spec_version = int(state.get("spec_version") or 0)
    judge_results = [
        r for r in (state.get("judge_results") or [])
        if int(getattr(r, "spec_version", 0)) == spec_version
    ]

    def _threshold(n: int) -> int:
        return max(1, int(math.ceil((2.0 * n) / 3.0)))

    criteria_names = ["Completeness", "Correctness", "Relevance"]
    failed: Dict[str, str] = {}
    missing_fields: List[str] = []

    for crit in criteria_names:
        rows = []
        for r in judge_results:
            ev = getattr(r, "evaluation", None)
            if ev is None:
                continue
            crit_ev = getattr(ev, crit, None)
            if crit_ev is None:
                continue
            rows.append((
                r.detector,
                str(getattr(crit_ev, "result", "")),
                str(getattr(crit_ev, "rationale", "") or ""),
            ))
            if crit == "Completeness":
                for f in (getattr(crit_ev, "missing_or_weak_fields", []) or []):
                    if f and f not in missing_fields:
                        missing_fields.append(f)

        n = len(rows)
        if n == 0:
            failed[crit] = "No judge results produced."
            continue

        fails = [(d, rat) for (d, res, rat) in rows if res not in ("PASS", "N/A")]
        if len(fails) >= _threshold(n):
            msgs = []
            for d, rat in fails:
                m = rat.strip() or f"Failed {crit}."
                if m not in msgs:
                    msgs.append(m)
            failed[crit] = " | ".join(msgs)

    passed = len(failed) == 0

    regen_rationale = ""
    if not passed:
        raw_lines = []
        for crit in criteria_names:
            if crit in failed:
                raw_lines.append(f"- {crit}: {failed[crit]}")
        if missing_fields:
            raw_lines.append("- Missing/weak fields: " + ", ".join(missing_fields))
        raw = "\n".join(raw_lines)

        if model is None:
            regen_rationale = raw
        else:
            system_prompt = "You summarize judge feedback into actionable regeneration instructions."
            human_prompt = (
                "Summarize the following evaluation failures into concise, actionable instructions.\n"
                f"Focus on what to change in the use case specification to pass.\n\nFAILURES:\n{raw}"
            )
            try:
                regen_rationale = str(
                    model.invoke([("system", system_prompt), ("human", human_prompt)]).content or ""
                ).strip()
            except Exception:
                regen_rationale = raw

    return {
        "validation": UseCaseSpecValidation(
            passed=passed, failed_criteria=failed, regen_rationale=regen_rationale,
        )
    }


# ---------------------------------------------------------------------------
# Graph builders
# ---------------------------------------------------------------------------


def build_sca_graph():
    """Generate ONE fully-dressed Use Case Spec, evaluate with 3 judges, optionally regenerate."""

    workflow = StateGraph(ScaState)
    workflow.add_node("generate_spec", generate_use_case_spec_node)
    workflow.add_node("judge_1", _judge_node("judge_1", model_key="judge_1"))
    workflow.add_node("judge_2", _judge_node("judge_2", model_key="judge_2"))
    workflow.add_node("judge_3", _judge_node("judge_3", model_key="judge_3"))
    workflow.add_node("combine", combiner_node)
    workflow.add_node("regen_spec", regenerate_use_case_spec_node)

    workflow.add_edge(START, "generate_spec")
    # Fan-out: all 3 judges run in parallel after spec generation
    workflow.add_edge("generate_spec", "judge_1")
    workflow.add_edge("generate_spec", "judge_2")
    workflow.add_edge("generate_spec", "judge_3")
    # Fan-in: combine waits for all 3 judges to finish
    workflow.add_edge("judge_1", "combine")
    workflow.add_edge("judge_2", "combine")
    workflow.add_edge("judge_3", "combine")

    def _route_after_combine(state: ScaState):
        v = state.get("validation")
        if v and getattr(v, "passed", False):
            return "end"
        return "regen"

    workflow.add_conditional_edges("combine", _route_after_combine, {"end": END, "regen": "regen_spec"})
    # After regen, fan-out to all 3 judges again in parallel
    workflow.add_edge("regen_spec", "judge_1")
    workflow.add_edge("regen_spec", "judge_2")
    workflow.add_edge("regen_spec", "judge_3")

    return workflow.compile()


def build_sca_eval_graph():
    """Evaluate an existing Use Case Specification JSON with the same 3 judges + combiner."""

    workflow = StateGraph(ScaState)
    workflow.add_node("judge_1", _judge_node("judge_1", model_key="judge_1"))
    workflow.add_node("judge_2", _judge_node("judge_2", model_key="judge_2"))
    workflow.add_node("judge_3", _judge_node("judge_3", model_key="judge_3"))
    workflow.add_node("combine", combiner_node)

    # Fan-out: all 3 judges run in parallel
    workflow.add_edge(START, "judge_1")
    workflow.add_edge(START, "judge_2")
    workflow.add_edge(START, "judge_3")
    # Fan-in: combine waits for all 3 judges to finish
    workflow.add_edge("judge_1", "combine")
    workflow.add_edge("judge_2", "combine")
    workflow.add_edge("judge_3", "combine")
    workflow.add_edge("combine", END)

    return workflow.compile()


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------


def _avg_int(values: List[int]) -> int:
    if not values:
        return 0
    return int(round(sum(values) / float(len(values))))


def _avg_sub_scores(items: List[Dict[str, int]]) -> Dict[str, int]:
    if not items:
        return {}
    sums: Dict[str, int] = {}
    counts: Dict[str, int] = {}
    for d in items:
        if not isinstance(d, dict):
            continue
        for k, v in d.items():
            key = str(k).strip()
            if not key:
                continue
            try:
                iv = int(v)
            except Exception:
                continue
            sums[key] = sums.get(key, 0) + iv
            counts[key] = counts.get(key, 0) + 1
    return {
        k: int(round(sums[k] / float(counts[k])))
        for k in sums
        if counts.get(k, 0) > 0
    }


def aggregate_evaluation_from_out(out: dict) -> UseCaseEvaluation | None:
    """Aggregate the per-judge evaluations for the latest spec_version."""
    spec_version = int(out.get("spec_version") or 0)
    matching: List[UseCaseSpecJudgeResult] = [
        r for r in (out.get("judge_results") or [])
        if int(getattr(r, "spec_version", 0)) == spec_version
    ]
    if not matching:
        return None

    comp_scores: List[int] = []
    corr_scores: List[int] = []
    rel_scores: List[int] = []
    comp_subs: List[Dict[str, int]] = []
    corr_subs: List[Dict[str, int]] = []
    rel_subs: List[Dict[str, int]] = []
    missing_fields: List[str] = []
    corr_reference_path: str | None = None
    corr_has_any: bool = False

    for r in matching:
        ev = getattr(r, "evaluation", None)
        if ev is None:
            continue

        comp = getattr(ev, "Completeness", None)
        if comp is not None:
            comp_scores.append(int(getattr(comp, "score", 0) or 0))
            ss = getattr(comp, "sub_scores", None)
            if isinstance(ss, dict) and ss:
                comp_subs.append(ss)
            for f in (getattr(comp, "missing_or_weak_fields", None) or []):
                fs = str(f).strip()
                if fs and fs not in missing_fields:
                    missing_fields.append(fs)

        corr = getattr(ev, "Correctness", None)
        if corr is not None:
            corr_has_any = True
            s = getattr(corr, "score", None)
            if isinstance(s, int):
                corr_scores.append(s)
            ss = getattr(corr, "sub_scores", None)
            if isinstance(ss, dict) and ss:
                corr_subs.append(ss)
            rp = getattr(corr, "reference_path", None)
            if corr_reference_path is None and isinstance(rp, str) and rp.strip():
                corr_reference_path = rp.strip()

        rel = getattr(ev, "Relevance", None)
        if rel is not None:
            rel_scores.append(int(getattr(rel, "score", 0) or 0))
            ss = getattr(rel, "sub_scores", None)
            if isinstance(ss, dict) and ss:
                rel_subs.append(ss)

    n = len(matching)
    avg_comp = _avg_int(comp_scores)
    avg_corr = _avg_int(corr_scores)
    avg_rel = _avg_int(rel_scores)

    correctness_obj: dict
    if not corr_has_any or not corr_scores:
        correctness_obj = {
            "score": None, "result": "N/A",
            "rationale": "No reference scenario was provided; correctness evaluation was skipped.",
            "reference_path": corr_reference_path, "sub_scores": {},
        }
    else:
        correctness_obj = {
            "score": avg_corr,
            "result": "PASS" if avg_corr >= 70 else "FAIL",
            "rationale": f"Average across {n} judge(s).",
            "reference_path": corr_reference_path,
            "sub_scores": _avg_sub_scores(corr_subs),
        }

    return UseCaseEvaluation(
        Completeness={
            "score": avg_comp,
            "result": "PASS" if avg_comp >= 70 else "FAIL",
            "rationale": f"Average across {n} judge(s).",
            "sub_scores": _avg_sub_scores(comp_subs),
            "missing_or_weak_fields": missing_fields,
        },
        Correctness=correctness_obj,
        Relevance={
            "score": avg_rel,
            "result": "PASS" if avg_rel >= 70 else "FAIL",
            "rationale": f"Average across {n} judge(s).",
            "sub_scores": _avg_sub_scores(rel_subs),
        },
    )


# ---------------------------------------------------------------------------
# Public runners
# ---------------------------------------------------------------------------


def run_sca_use_case(
    *,
    use_case: UseCase,
    requirement_text: List[str],
    actors: List[str] | None = None,
    reference_spec_path: str | None = None,
    comparison_spec_path: str | None = None,
    model_configs: dict | None = None,
) -> ScenarioResult:
    app = build_sca_graph()
    out = app.invoke({
        "requirement_text": requirement_text,
        "actors": actors or (use_case.participating_actors or []),
        "use_case": use_case,
        "use_case_spec_json": {},
        "spec_version": 0,
        "judge_results": [],
        "validation": None,
        "reference_spec_path": reference_spec_path,
        "model_configs": model_configs or {
            "writer": dict(_DEFAULT_MODEL_CONFIG),
            "judge_1": dict(_MODEL_CONFIG_1),
            "judge_2": dict(_MODEL_CONFIG_2),
            "judge_3": dict(_MODEL_CONFIG_3),
            "summarizer": dict(_DEFAULT_MODEL_CONFIG),
        },
    })
    spec_json = out.get("use_case_spec_json") or {}
    validation = out.get("validation") or UseCaseSpecValidation(
        passed=True, failed_criteria={}, regen_rationale=""
    )

    aggregated_evaluation = aggregate_evaluation_from_out(out)

    comparison_evaluation: UseCaseEvaluation | None = None
    if isinstance(comparison_spec_path, str) and comparison_spec_path.strip():
        try:
            comparison_spec_obj = _read_json_object_file(comparison_spec_path)
            eval_app = build_sca_eval_graph()
            eval_out = eval_app.invoke({
                "requirement_text": requirement_text,
                "actors": actors or (use_case.participating_actors or []),
                "use_case": use_case,
                "use_case_spec_json": comparison_spec_obj,
                "spec_version": 1,
                "judge_results": [],
                "validation": None,
                "reference_spec_path": reference_spec_path,
                "comparison_spec_path": None,
                "model_configs": model_configs or {
                    "writer": dict(_DEFAULT_MODEL_CONFIG),
                    "judge_1": dict(_DEFAULT_MODEL_CONFIG),
                    "judge_2": dict(_DEFAULT_MODEL_CONFIG),
                    "judge_3": dict(_DEFAULT_MODEL_CONFIG),
                    "summarizer": dict(_DEFAULT_MODEL_CONFIG),
                },
            })
            comparison_evaluation = aggregate_evaluation_from_out(eval_out)
        except Exception:
            comparison_evaluation = None

    return ScenarioResult(
        use_case=use_case,
        use_case_spec_json=spec_json,
        evaluation=aggregated_evaluation,
        comparison_spec_path=(comparison_spec_path.strip() if isinstance(comparison_spec_path, str) else None),
        comparison_evaluation=comparison_evaluation,
        validation=validation,
    )


def _read_json_object_file(path_str: str) -> dict:
    p = _resolve_reference_path(path_str)
    raw = p.read_text(encoding="utf-8")
    obj = json.loads(raw)
    if not isinstance(obj, dict):
        raise ValueError("JSON root must be an object")
    return obj


def run_sca(input_data: dict) -> List[ScenarioResult]:
    """Accept RPA-like JSON input and generate scenarios for all provided use cases."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    requirement_text = input_data.get("requirement_text", "")
    raw_use_cases = input_data.get("use_cases") or []
    actors = input_data.get("actors")
    actors_list: List[str] = []
    if isinstance(actors, list):
        for a in actors:
            if isinstance(a, str):
                actors_list.append(a)
            elif isinstance(a, dict) and a.get("name"):
                actors_list.append(str(a.get("name")))

    model_configs = input_data.get("model_configs")
    model_configs_dict = model_configs if isinstance(model_configs, dict) else None

        # Filter valid use cases and maintain order
    valid_items: List[tuple] = []
    for i, raw in enumerate(raw_use_cases):
            if not isinstance(raw, dict):
                continue
            valid_items.append((i, UseCase.model_validate(raw)))


    results: List[ScenarioResult] = [None] * len(valid_items)  # type: ignore[list-item]

    def _run_one(index: int, uc: UseCase) -> tuple:
        sr = run_sca_use_case(
            use_case=uc,
            requirement_text=requirement_text,
            actors=actors_list or None,
            model_configs=model_configs_dict,
        )
        return index, sr

    with ThreadPoolExecutor(max_workers=min(len(valid_items) or 1, 8)) as pool:
        futures = {
            pool.submit(_run_one, idx, uc): idx
            for idx, (_, uc) in enumerate(valid_items)
        }
        for future in as_completed(futures):
            idx, sr = future.result()
            results[idx] = sr

    return [r for r in results if r is not None]
