from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from pathlib import Path

import json
import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.graph import END, START, StateGraph

from ..state import (
    ScaState,
    ScenarioResult,
    UseCase,
    UseCaseEvaluation,
    UseCaseSpecJudgeResult,
    UseCaseSpecValidation,
)


_DEFAULT_MODEL_CONFIG: Dict[str, Any] = {
    "provider": "openai",
    "name": "gpt-4o-mini",
    "temperature": 0,
}

# Cache initialized models so multiple nodes can reuse them.
_MODEL_CACHE: Dict[Tuple[str, str, float], Any] = {}


def _get_model(model_config: Optional[dict] = None):
    """Return an initialized chat model, or None if not configured.

    This is intentionally generic so different nodes (writer/judges/combiner)
    can use different model configurations later.
    """

    load_dotenv()

    cfg: Dict[str, Any] = dict(_DEFAULT_MODEL_CONFIG)
    if isinstance(model_config, dict):
        # Allow per-node overrides like {"name": "gpt-4o", "temperature": 0.2}
        cfg.update({k: v for k, v in model_config.items() if v is not None})

    provider = str(cfg.get("provider") or "openai")
    model_name = str(cfg.get("name") or "gpt-4o-mini")
    temperature = float(
        cfg.get("temperature") if cfg.get("temperature") is not None else 0
    )

    # Gate on required credentials for known providers.
    if provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        return None

    cache_key = (provider, model_name, temperature)
    cached = _MODEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    # Bias toward determinism.
    try:
        model = init_chat_model(model_name, model_provider=provider, temperature=temperature)
    except TypeError:
        model = init_chat_model(model_name, model_provider=provider)

    _MODEL_CACHE[cache_key] = model
    return model


def _get_model_for(state: ScaState, key: str):
    """Get model config from state['model_configs'][key], otherwise default."""

    model_configs = state.get("model_configs")
    cfg = None
    if isinstance(model_configs, dict):
        cfg = model_configs.get(key)
    return _get_model(cfg)


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
    # Minimal deterministic JSON object when no LLM is configured.
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
    
    # trig = (
    #     use_case.sentence or ""
    # ).strip() or f"A primary actor initiates '{use_case.name}'."
    
    trig_raw = (getattr(use_case, "description", "") or "").strip()
    if not trig_raw and getattr(use_case, "user_stories", None):
        parts = [
            getattr(s, "original_sentence", "") or getattr(s, "action", "")
            for s in (use_case.user_stories or [])[:3]
        ]
        trig_raw = " ".join(p for p in parts if p).strip()
    trig = trig_raw or f"A primary actor initiates '{use_case.name}'."

    include_rels = [
        r
        for r in (uc.get("relationships") or [])
        if str(r.get("type", "")).lower() == "include"
    ]
    extend_rels = [
        r
        for r in (uc.get("relationships") or [])
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
            f"AF-{i} (from Step 2): <condition to trigger «extend» {target}>\n1. Primary Actor requests optional sub-flow «extend» {target} → System performs «extend» {target}.\nReturn to Step 3."
        )

    ef_lines = [
        "EF-1 (from Step 2): <system cannot process the request>\n1. Primary Actor submits the request → System displays an error and ends the use case."
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
        "description": (str(getattr(use_case, "description", "") or "").strip() or f"Enable the primary actor to complete '{use_case.name}' as described in the requirement text."),
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
    """Best-effort extraction of a JSON object from model output."""
    raw = (text or "").strip()
    if not raw:
        raise ValueError("Empty model output")

    # Common case: already pure JSON
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Best-effort: find first JSON object block
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        candidate = raw[start : end + 1]
        obj = json.loads(candidate)
        if isinstance(obj, dict):
            return obj

    raise ValueError("Could not parse JSON object from model output")


def _save_use_case_spec_json(*, use_case: UseCase, spec_version: int, spec_obj: dict) -> str:
    repo_root = Path(__file__).resolve().parents[2]
    out_dir = repo_root / "sca_specs"
    out_dir.mkdir(parents=True, exist_ok=True)

    uc_id = int(getattr(use_case, "id", 0) or 0)
    raw_name = str(getattr(use_case, "name", "") or "").strip()
    safe_name = "".join(
        ch for ch in raw_name if ch.isalnum() or ch in ("-", "_", " ")
    ).strip()
    safe_name = "_".join(safe_name.split())[:60] or f"usecase_{uc_id}"

    out_path = out_dir / f"usecase_{uc_id}_{safe_name}_v{spec_version}.json"
    out_path.write_text(
        json.dumps(spec_obj, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return str(out_path)


def _resolve_reference_path(path_str: str) -> Path:
    p = Path(path_str).expanduser()
    if not p.is_absolute():
        repo_root = Path(__file__).resolve().parents[2]
        p = (repo_root / p).resolve()
    return p


def _read_reference_text(path_str: str) -> str:
    p = _resolve_reference_path(path_str)
    return p.read_text(encoding="utf-8")


_WRITER_SYSTEM_PROMPT = "You are a professional, strict, and detail-oriented Use Case Specification Writer. You output ONLY a single JSON object and nothing else. You never leave any required field empty. You only infer information that is logically supported by the given Requirement Text. If information is implicit, you must state it explicitly using formal requirement language."

# Prompt reference (documentation): docs/sca_use_case_spec_json_prompt.md

_SCA_UC_PROMPT_PATH = Path(__file__).resolve().parents[2] / "docs" / "sca_use_case_spec_json_prompt.md"
_SCA_UC_PROMPT_CACHE: str | None = None


def _load_sca_uc_prompt() -> str:
    """Load the field definitions + writing rules prompt block.

    This content is sent to the LLM as additional system guidance so it
    understands what each JSON field means and the rules for writing them.
    """

    global _SCA_UC_PROMPT_CACHE
    if _SCA_UC_PROMPT_CACHE is not None:
        return _SCA_UC_PROMPT_CACHE
    try:
        _SCA_UC_PROMPT_CACHE = _SCA_UC_PROMPT_PATH.read_text(encoding="utf-8").strip()
    except Exception:
        _SCA_UC_PROMPT_CACHE = ""
    return _SCA_UC_PROMPT_CACHE

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
      Add one or more explicit Actor→System steps that perform the included use case, before or during the primary goal execution.
    - For each relationship with type "extend": you MUST represent it as an optional variation in `alternative_flows`.
      Each alternative flow MUST branch from a specific `main_flow` step (e.g., "AF-1 (from Step 3): ...") and MUST return to a later main step or end successfully.
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
        _save_use_case_spec_json(
            use_case=use_case, spec_version=spec_version, spec_obj=spec_obj
        )
        return {
            "use_case_spec_json": spec_obj,
            "spec_version": spec_version,
        }

    use_case_json = json.dumps(
        _use_case_payload(use_case), ensure_ascii=False, indent=2
    )
    prompt = (
        _WRITER_HUMAN_PROMPT_TEMPLATE.replace("{{requirement_text}}", requirement_text)
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
    _save_use_case_spec_json(use_case=use_case, spec_version=spec_version, spec_obj=spec_obj)
    return {
        "use_case_spec_json": spec_obj,
        "spec_version": spec_version,
    }


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

    use_case_json = json.dumps(
        _use_case_payload(use_case), ensure_ascii=False, indent=2
    )
    prompt = (
        _WRITER_HUMAN_PROMPT_TEMPLATE.replace("{{requirement_text}}", requirement_text)
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
    _save_use_case_spec_json(use_case=use_case, spec_version=spec_version, spec_obj=spec_obj)
    return {
        "use_case_spec_json": spec_obj,
        "spec_version": spec_version,
    }


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
1. COMPLETENESS
================================

Evaluate whether the use case specification contains all REQUIRED fields
and whether each field is sufficiently defined as a testable system artifact.

Evaluation consists of TWO dimensions:
- STRUCTURAL COMPLETENESS
- BEHAVIORAL COMPLETENESS

Each field has explicit scoring rules.
Partial credit is allowed ONLY where explicitly defined.

--------------------------------
STRUCTURAL COMPLETENESS
--------------------------------

1. PRIMARY ACTOR (15 points)

Evaluate whether the use case clearly identifies exactly ONE correct primary actor.

- 15 points:
  Exactly ONE primary actor is defined.
  The actor is external to the system (human or external system).
  The actor name is consistent throughout the use case.

- 10 points:
  An actor exists but is vague, generic, or partially mixed with system responsibility.

- 0 points:
  The actor is the system itself, an internal component (UI, module, database),
  or multiple conflicting primary actors exist.

--------------------------------

2. USE CASE NAME (10 points)

Evaluate whether the use case name expresses a single, observable user goal.

- 10 points:
  The name follows the pattern <Verb + Object>.
  It represents a concrete user goal with an observable outcome.

- 5 points:
  The name relates to a user goal but is vague, overly broad,
  or combines multiple goals into one.

- 0 points:
  The name describes a technical action, UI flow, or system function
  (e.g., "Validate Input", "Open Screen").

--------------------------------

3. PRECONDITIONS (10 points)

Evaluate whether preconditions define necessary system states before execution.

- 10 points:
  Preconditions are explicit system states.
  They are necessary and verifiable.

- 5 points:
  Preconditions exist but are vague or redundant.

- 0 points:
  Preconditions are missing or written as actions.

--------------------------------

4. POSTCONDITIONS (10 points)

Evaluate whether postconditions describe resulting system states.

- 10 points:
  Postconditions clearly describe measurable and verifiable system states
  after successful completion.

- 5 points:
  Postconditions exist but are vague or describe actions instead of system states.

- 0 points:
  Postconditions are missing or incorrect.

--------------------------------

5. STAKEHOLDERS & INTERESTS (5 points)

Evaluate whether stakeholders and their interests are explicitly defined.

- 5 points:
  All relevant stakeholders are listed.
  Each stakeholder has a specific, concrete, and testable interest.

- 2 points:
  Stakeholders are listed but interests are vague, generic, or overlapping.

- 0 points:
  Stakeholders are missing or only names are listed without interests.

--------------------------------
BEHAVIORAL COMPLETENESS
--------------------------------

6. MAIN FLOW (25 points)

Evaluate the Main Success Scenario (MSS).

- 25 points:
  The Main Flow is a complete step-by-step Actor–System interaction.
  No internal logic or UI details are included.
  The flow starts from preconditions and ends with goal achievement.

- 15 points:
  The Main Flow exists but misses steps, system responses,
  or contains minor logical gaps.

- 0 points:
  The Main Flow is not a success path or is written as a technical workflow.

--------------------------------

7. ALTERNATIVE FLOWS (15 points)

Evaluate valid variations of the Main Flow.

- 15 points:
  All meaningful variations are covered.
  Each alternative flow explicitly references a specific Main Flow step
  and still leads to successful goal completion.

- 10 points:
  Alternative flows exist but lack Main Flow step references
  or contain unclear or incomplete logic.

- 0 points:
  Alternative flows are missing or written as independent scenarios.

--------------------------------

8. EXCEPTION FLOWS (10 points)

Evaluate error and failure handling.

- 10 points:
  Exception situations are clearly defined.
  Each exception references a specific Main Flow step
  and explicitly states the system's handling behavior.

- 5 points:
  Exception flows exist but are generic,
  lack clear system responses, or are partially defined.

- 0 points:
  Exception flows are missing or incorrectly mixed with alternative flows.

--------------------------------
SCORING & DECISION
--------------------------------

Total Score = Sum of all field scores (0-100).

PASS if Total Score ≥ 70.
FAIL if Total Score < 70.

Provide rationale:
- Which required fields are weak, missing, or incorrect

================================
2. CORRECTNESS
================================

Your task is to evaluate the CORRECTNESS of a use case specification generated by an LLM (Scenario B) by comparing it against a human-written reference use case (Scenario A).

Correctness is defined as the degree to which Scenario B preserves the semantic intent, actor responsibility, causal structure, and outcome semantics of Scenario A — even if wording, step granularity, ordering, or structure differs.

You must perform a semantic and comparative validation, not a textual or stylistic comparison.

Scenario B may contain:

- merged or split steps

- different phrasing

- additional or omitted details

These differences are acceptable only if they do not change the underlying intent, responsibility, causal logic, or outcome defined in Scenario A.

No-reference rule:
If Scenario A (human reference) is missing or incomplete, you MUST NOT penalize Scenario B.
In that case, set all scores to N/A and return total_score = null.

--------------------------------
EVALUATION CRITERIA & SCORING
--------------------------------

Evaluate Scenario B against Scenario A using the following fields.

1. Primary Actor (0-20 points)

Evaluate whether the primary actor in Scenario B matches the real-world external actor in Scenario A.

20 points: Scenario B identifies the same real-world external role as Scenario A. Actor intent and responsibility are semantically equivalent, regardless of wording or step structure.

10 points: Actor is conceptually the same but described vaguely, ambiguously, or partially mixed with system behavior.

0 points: Scenario B uses a different actor, or treats the system, UI, or internal component as the primary actor.

--------------------------------

2. Use Case Name (0-20 points)

Evaluate whether the use case name in Scenario B represents the same high-level actor goal as Scenario A.

20 points: Name is semantically equivalent and represents the same single actor goal, even if phrased differently.

10 points: Name is related but noticeably broader, narrower, or more function-oriented than the human version.

0 points: Name describes a technical task, UI flow, internal function, or a different goal.

--------------------------------

3. Main Success Scenario (MSS) (0-25 points)

Evaluate whether the Main Success Scenario in Scenario B preserves the same success intent and causal structure as Scenario A.

Step count, granularity, and wording may differ. Steps may be merged, split, added, or omitted as long as: 

- actor-system interaction remains equivalent 
- logical prerequisites are preserved
- the final successful outcome is the same

25 points: MSS is semantically equivalent in intent, interaction flow, and outcome.

15 points: MSS is mostly equivalent but includes minor logical gaps, redundant steps, or slight internal logic leakage. Final outcome remains equivalent.

0 points: MSS cannot be semantically mapped to the human success path, or is written as a technical / implementation workflow.

--------------------------------

4. Alternative Flows (0-15 points)

Evaluate whether alternative flows in Scenario B represent valid success variations of the MSS.

15 points: Alternative flows are causally linked to the MSS and still lead to the same successful outcome. Human alternatives may be merged or split without semantic loss.

8 points: Alternative flows are partially correct but weakly connected to the MSS or have unclear success outcomes.

0 points: Alternative flows are missing, irrelevant, or actually describe failure scenarios.

--------------------------------

5. Exception Flows (0-10 points)

Evaluate whether exception flows in Scenario B match the failure intent and outcomes of Scenario A.

10 points: Exception flows cover semantically equivalent failure conditions and system results. Step structure and placement may differ.

5 points: Exception flows exist but triggering conditions or resulting system states are vague or incomplete.

0 points: Exception flows are missing, incorrect, or confused with alternative (success) flows.

--------------------------------

6. Preconditions (0-5 points)

Evaluate whether preconditions in Scenario B describe the same required system states as Scenario A.

5 points: Preconditions are semantically equivalent and necessary for the MSS to occur.

2 points: Preconditions are related but incomplete, overly vague, or include unnecessary conditions.

0 points: Preconditions are missing or written as actions, UI steps, or technical setup.

--------------------------------

7. Postconditions (0-5 points)

Evaluate whether postconditions describe the same system-observable end states as Scenario A.

5 points: Postconditions describe equivalent final system states. Wording and number of states may differ.

2 points: Postconditions partially describe outcomes but mix actions with states or omit key results.

0 points: Postconditions are missing or written as internal processing steps.

--------------------------------
SCORING & DECISION
--------------------------------

Total Correctness Score = Sum of all field scores = 0-100

PASS: Total Score ≥ 70
FAIL: Total Score < 70

Your rationale MUST explicitly identify:

- Which fields in Scenario B diverge semantically from Scenario A

- Where actor responsibility or intent is weakened or altered

- Any causal or logical inconsistencies in the MSS

- Whether alternative or exception flows are misclassified or misplaced

- Whether missing or extra elements affect semantic equivalence

DO NOT

- Compare wording, grammar, or writing style

- Require step-by-step textual identity

- Penalize acceptable step merging or splitting

- Infer behavior not explicitly stated

- Evaluate completeness, design quality, or implementation details

================================
3. RELEVANCE
================================

Your task is to evaluate the RELEVANCE of a given Use Case by examining the explicit traceability relationships between its fields (e.g., Actor, Use Case Name, Main Flow, Alternative Flows, Preconditions, Postconditions, Stakeholders).

Relevance is defined as the degree to which these fields are logically and explicitly connected to each other to support a single, coherent use case goal.
A Use Case is relevant only if its fields mutually reinforce the same intent and execution outcome, without introducing unrelated behavior or scope drift.

You must evaluate field-to-field consistency, not writing quality or functional correctness.


1. Primary Actor ↔ Use Case Name (15 points)

Evaluate whether the Use Case name represents a goal that directly benefits the primary actor.

15: The primary actor is an external entity and directly receives the main benefit when the Use Case goal is achieved

8: The actor benefits indirectly, or the Use Case goal is partially system-centric

0: The actor is unrelated to the goal, or is an internal system component

--------------------------------

2. Use Case Name ↔ Main Flow (25 points)

Evaluate whether the Main Flow clearly achieves the stated Use Case goal.

25: The final step of the MSS establishes a clear, observable system state that fulfills the Use Case goal

15: The MSS is mostly complete but lacks an explicit final state proving goal completion

0: The MSS does not demonstrate that the goal has been achieved

--------------------------------

3. Main Flow ↔ Alternative Flows (20 points)

Evaluate traceability between the Main Flow and all Alternative / Exception Flows.

20: Every Alternative Flow explicitly references the exact MSS step where it branches (e.g., “At step 4…”)

10: Alternative Flows exist, but step linkage is unclear or only implicit

0: Alternative Flows cannot be traced to the MSS

--------------------------------

4. Preconditions & Trigger ↔ Main Flow (10 points)

Evaluate whether Preconditions and Trigger are necessary for initiating the Main Flow.

10: Preconditions and Trigger are necessary conditions for starting the Main Flow

5: Preconditions exist but are not meaningfully used by the Main Flow

0: Preconditions or Trigger are irrelevant, contradictory, or disconnected from the Main Flow

--------------------------------

5. Main Flow & Alternative Flows ↔ Postconditions (15 points)

Evaluate whether all execution paths terminate in a defined system state.

15: The Main Flow and every Alternative Flow explicitly terminate in a defined Postcondition ensuring a consistent and acceptable system state

8: A Postcondition exists, but some Alternative Flows terminate ambiguously or converge only implicitly

0: The Main Flow and/or Alternative Flows do not converge to any defined Postcondition

--------------------------------

6. Stakeholders & Interests ↔ Postconditions (15 points)

Evaluate whether stakeholder interests are satisfied by the resulting system state.

15: Every stated stakeholder interest is explicitly mapped to one or more Postconditions

8: Stakeholder interests are only partially covered by Postconditions

0: Stakeholder interests cannot be traced to any Postcondition

--------------------------------
SCORING & DECISION
--------------------------------

Total Relevance Score = Sum of all field-to-field traceability scores (0-100).

PASS if Total Relevance Score ≥ 70

FAIL if Total Relevance Score < 70


Your rationale must explicitly identify:

- Which fields are weakly linked to other fields

- Which fields are isolated or cannot be traced to the Use Case goal

- Which flows or conditions introduce off-scope behavior

- Where traceability breaks (e.g., missing linkage between Main Flow and Postcondition, Alternative Flow not linked to Main Flow step)

Do not:

- Speculate about intended behavior

- Infer missing relationships

- Evaluate correctness or system design quality

--------------------------------
OUTPUT FORMAT (STRICT JSON)
--------------------------------

{
  "Completeness": {
    "score": <0-100>,
    "result": "PASS | FAIL",
    "rationale": "...",
        "sub_scores": {
            "Primary Actor": <0-15>,
            "Use Case Name": <0-10>,
            "Preconditions": <0-10>,
            "Postconditions": <0-10>,
            "Stakeholders & Interests": <0-5>,
            "Main Flow": <0-25>,
            "Alternative Flows": <0-15>,
            "Exception Flows": <0-10>
        },
    "missing_or_weak_fields": [
      "field name",
      "field name"
    ]
  },
    "Correctness": {
        "score": <0-100 or null>,
        "result": "PASS | FAIL | N/A",
        "rationale": "...",
                "reference_path": "<string or null>",
                "sub_scores": {
                    "Primary Actor": <0-20>,
                    "Use Case Name": <0-20>,
                    "Main Success Scenario (MSS)": <0-25>,
                    "Alternative Flows": <0-15>,
                    "Exception Flows": <0-10>,
                    "Preconditions": <0-5>,
                    "Postconditions": <0-5>
                }
    },
  "Relevance": {
    "score": <0-100>,
    "result": "PASS | FAIL",
        "rationale": "...",
        "sub_scores": {
            "Primary Actor ↔ Use Case Name": <0-15>,
            "Use Case Name ↔ Main Flow": <0-25>,
            "Main Flow ↔ Alternative Flows": <0-20>,
            "Preconditions & Trigger ↔ Main Flow": <0-10>,
            "(Main Flow & Alternative Flows) ↔ Postconditions": <0-15>,
            "Stakeholders & Interests ↔ Postconditions": <0-15>
        }
  }
}

Rules:
- Do NOT assume missing fields
- Judge only based on provided content
- Be strict and consistent

Correctness rule:
- If <REFERENCE_SCENARIO> is empty or only whitespace, you MUST set:
    - Correctness.result = "N/A"
    - Correctness.score = null
    - Correctness.rationale = a short explanation that correctness was skipped due to missing reference
    - Correctness.reference_path = null
    - Correctness.sub_scores = {}
"""


def _judge_node(detector_name: str, *, model_key: str | None = None):
    def _fn(state: ScaState):
        model = _get_model_for(state, model_key or detector_name)
        spec_obj = state.get("use_case_spec_json") or {}
        spec = json.dumps(spec_obj, ensure_ascii=False, indent=2)
        spec_version = int(state.get("spec_version") or 0)
        reference_spec_path = state.get("reference_spec_path")

        # Load reference scenario text (if any) for correctness evaluation.
        reference_text = ""
        if isinstance(reference_spec_path, str) and reference_spec_path.strip():
            try:
                reference_text = _read_reference_text(reference_spec_path).strip()
            except Exception as e:
                # Keep as empty; judges will return N/A.
                reference_text = ""

        if model is None:
            # Conservative fallback: fail if any required JSON fields are missing.
            missing: List[str] = []
            required_keys = [
                "use_case_name",
                "primary_actors",
                "stakeholders_and_interests",
                "preconditions",
                "postconditions",
                "main_flow",
                "alternative_flows",
                "exception_flows",
            ]
            presence: Dict[str, bool] = {}
            spec_obj: dict = {}
            try:
                spec_obj = _extract_json_object(spec)
                for key in required_keys:
                    val = spec_obj.get(key)
                    if val is None:
                        missing.append(key)
                        presence[key] = False
                        continue
                    if isinstance(val, str) and not val.strip():
                        missing.append(key)
                        presence[key] = False
                        continue
                    if isinstance(val, list) and len([x for x in val if str(x).strip()]) == 0:
                        missing.append(key)
                        presence[key] = False
                        continue
                    presence[key] = True
            except Exception:
                missing = required_keys[:]
                for key in required_keys:
                    presence[key] = False
                spec_obj = {}

            # Sub-scores (deterministic) aligned to the rubric point allocations.
            comp_sub_scores: Dict[str, int] = {
                "Primary Actor": 15
                if presence.get("primary_actors")
                and isinstance(spec_obj.get("primary_actors"), list)
                and len([x for x in (spec_obj.get("primary_actors") or []) if str(x).strip()]) == 1
                else 0,
                "Use Case Name": 10 if presence.get("use_case_name") else 0,
                "Preconditions": 10 if presence.get("preconditions") else 0,
                "Postconditions": 10 if presence.get("postconditions") else 0,
                "Stakeholders & Interests": 5
                if presence.get("stakeholders_and_interests")
                else 0,
                "Main Flow": 25 if presence.get("main_flow") else 0,
                "Alternative Flows": 15 if presence.get("alternative_flows") else 0,
                "Exception Flows": 10 if presence.get("exception_flows") else 0,
            }
            comp_score = int(sum(comp_sub_scores.values()))
            completeness_pass = comp_score >= 70 and len(missing) == 0

            correctness_obj = {
                "score": None,
                "result": "N/A",
                "rationale": "No reference scenario was provided; correctness evaluation was skipped.",
                "reference_path": (reference_spec_path.strip() if isinstance(reference_spec_path, str) else None),
                "sub_scores": {},
            }

            rel_sub_scores: Dict[str, int] = {
                "Primary Actor ↔ Use Case Name": 15
                if presence.get("primary_actors") and presence.get("use_case_name")
                else 0,
                "Use Case Name ↔ Main Flow": 25
                if presence.get("use_case_name") and presence.get("main_flow")
                else 0,
                "Main Flow ↔ Alternative Flows": 20
                if presence.get("main_flow")
                and presence.get("alternative_flows")
                and any(
                    "from step" in str(x).lower()
                    for x in (spec_obj.get("alternative_flows") or [])
                )
                else (10 if presence.get("alternative_flows") else 0),
                "Preconditions & Trigger ↔ Main Flow": 10
                if presence.get("preconditions")
                and bool(str(spec_obj.get("triggering_event", "") or "").strip())
                and presence.get("main_flow")
                else 0,
                "(Main Flow & Alternative Flows) ↔ Postconditions": 15
                if presence.get("postconditions") and presence.get("main_flow")
                else 0,
                "Stakeholders & Interests ↔ Postconditions": 15
                if presence.get("stakeholders_and_interests")
                and presence.get("postconditions")
                else 0,
            }
            rel_score = int(sum(rel_sub_scores.values()))

            evaluation = UseCaseEvaluation(
                Completeness={
                    "score": comp_score,
                    "result": "PASS" if completeness_pass else "FAIL",
                    "rationale": "All required JSON fields are present."
                    if completeness_pass
                    else "Missing required JSON fields.",
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
            jr = UseCaseSpecJudgeResult(
                detector=detector_name, spec_version=spec_version, evaluation=evaluation
            )
            return {"judge_results": [jr]}

        # NOTE: OpenAI JSON-schema structured outputs can be brittle with complex schemas.
        # Use function-calling mode to avoid response_format schema rejections.
        structured_llm = model.with_structured_output(
            UseCaseEvaluation, method="function_calling"
        )
        prompt = (
            _JUDGE_HUMAN_PROMPT_TEMPLATE.replace("{{use_case_spec}}", spec)
            .replace("{{reference_scenario}}", reference_text)
        )
        evaluation = structured_llm.invoke(
            [("system", _JUDGE_SYSTEM_PROMPT), ("human", prompt)]
        )

        # Hard guard: if no reference scenario was provided, correctness must be N/A.
        if not (reference_text or "").strip():
            try:
                if getattr(evaluation, "Correctness", None) is not None:
                    evaluation.Correctness.result = "N/A"
                    evaluation.Correctness.score = None
                    evaluation.Correctness.rationale = (
                        "No reference scenario was provided; correctness evaluation was skipped."
                    )
                    evaluation.Correctness.reference_path = None
                    evaluation.Correctness.sub_scores = {}
            except Exception:
                pass
        # Ensure reference path is always captured.
        try:
            if getattr(evaluation, "Correctness", None) is not None:
                if getattr(evaluation.Correctness, "reference_path", None) in (None, ""):
                    evaluation.Correctness.reference_path = (
                        reference_spec_path.strip()
                        if isinstance(reference_spec_path, str)
                        else None
                    )
        except Exception:
            pass
        jr = UseCaseSpecJudgeResult(
            detector=detector_name, spec_version=spec_version, evaluation=evaluation
        )
        return {"judge_results": [jr]}

    return _fn


def combiner_node(state: ScaState):
    model = _get_model_for(state, "summarizer")
    spec_version = int(state.get("spec_version") or 0)
    judge_results = [
        r
        for r in (state.get("judge_results") or [])
        if int(getattr(r, "spec_version", 0)) == spec_version
    ]

    def _threshold(n: int) -> int:
        import math

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
            rows.append(
                (
                    r.detector,
                    str(getattr(crit_ev, "result", "")),
                    str(getattr(crit_ev, "rationale", "") or ""),
                )
            )
            if crit == "Completeness":
                mf = list(getattr(crit_ev, "missing_or_weak_fields", []) or [])
                for f in mf:
                    if f and f not in missing_fields:
                        missing_fields.append(f)

        n = len(rows)
        if n == 0:
            failed[crit] = "No judge results produced."
            continue

        fails = [(d, rat) for (d, res, rat) in rows if res not in ("PASS", "N/A")]
        if len(fails) >= _threshold(n):
            # Majority FAIL
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
            human_prompt = f"""Summarize the following evaluation failures into concise, actionable instructions.
Focus on what to change in the use case specification to pass.

FAILURES:
{raw}
"""
            try:
                regen_rationale = str(
                    model.invoke(
                        [("system", system_prompt), ("human", human_prompt)]
                    ).content
                    or ""
                ).strip()
            except Exception:
                regen_rationale = raw

    return {
        "validation": UseCaseSpecValidation(
            passed=passed,
            failed_criteria=failed,
            regen_rationale=regen_rationale,
        )
    }


def build_sca_graph():
    """Agent 2 (Worker): generate ONE fully-dressed Use Case Specification, then evaluate with 3 judges and a combiner."""

    workflow = StateGraph(ScaState)
    workflow.add_node("generate_spec", generate_use_case_spec_node)
    workflow.add_node("judge_1", _judge_node("judge_1", model_key="judge_1"))
    workflow.add_node("judge_2", _judge_node("judge_2", model_key="judge_2"))
    workflow.add_node("judge_3", _judge_node("judge_3", model_key="judge_3"))
    workflow.add_node("combine", combiner_node)
    workflow.add_node("regen_spec", regenerate_use_case_spec_node)

    workflow.add_edge(START, "generate_spec")
    workflow.add_edge("generate_spec", "judge_1")
    workflow.add_edge("judge_1", "judge_2")
    workflow.add_edge("judge_2", "judge_3")
    workflow.add_edge("judge_3", "combine")

    def _route_after_combine(state: ScaState):
        v = state.get("validation")
        if v and getattr(v, "passed", False):
            return "end"
        return "regen"

    workflow.add_conditional_edges(
        "combine",
        _route_after_combine,
        {"end": END, "regen": "regen_spec"},
    )

    # One regeneration pass, then re-run judges and combine again.
    workflow.add_edge("regen_spec", "judge_1")

    return workflow.compile()


def run_sca_use_case(
    *,
    use_case: UseCase,
    requirement_text: List[str],
    actors: List[str] | None = None,
    reference_spec_path: str | None = None,
    model_configs: dict | None = None,
) -> ScenarioResult:
    app = build_sca_graph()
    out = app.invoke(
        {
            "requirement_text": requirement_text,
            "actors": actors or (use_case.participating_actors or []),
            "use_case": use_case,
            "use_case_spec_json": {},
            "spec_version": 0,
            "judge_results": [],
            "validation": None,
            "reference_spec_path": reference_spec_path,
            "model_configs": model_configs
            or {
                "writer": dict(_DEFAULT_MODEL_CONFIG),
                "judge_1": dict(_DEFAULT_MODEL_CONFIG),
                "judge_2": dict(_DEFAULT_MODEL_CONFIG),
                "judge_3": dict(_DEFAULT_MODEL_CONFIG),
                "summarizer": dict(_DEFAULT_MODEL_CONFIG),
            },
        }
    )
    spec_json = out.get("use_case_spec_json") or {}
    validation = out.get("validation") or UseCaseSpecValidation(
        passed=True, failed_criteria={}, regen_rationale=""
    )

    def _avg_int(values: List[int]) -> int:
        if not values:
            return 0
        return int(round(sum(values) / float(len(values))))

    def _aggregate_evaluation() -> UseCaseEvaluation | None:
        spec_version = int(out.get("spec_version") or 0)
        matching: List[UseCaseSpecJudgeResult] = [
            r
            for r in (out.get("judge_results") or [])
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
                for k in sums.keys()
                if counts.get(k, 0) > 0
            }

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
                "score": None,
                "result": "N/A",
                "rationale": "No reference scenario was provided; correctness evaluation was skipped.",
                "reference_path": corr_reference_path,
                "sub_scores": {},
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

    aggregated_evaluation = _aggregate_evaluation()

    return ScenarioResult(
        use_case=use_case,
        use_case_spec_json=spec_json,
        evaluation=aggregated_evaluation,
        validation=validation,
    )


def run_sca(input_data: dict) -> List[ScenarioResult]:
    """Convenience wrapper: accept the RPA-like JSON input and generate scenarios for all provided use cases.

    Expected shape:
    {
      "requirement_text": List[str],
      "use_cases": [{"name": str, "participating_actors": [...], "relationships": [...]}, ...]
    }
    Other keys (e.g., actors/aliases) are ignored.
    """

    requirement_text = str(input_data.get("requirement_text", ""))
    raw_use_cases = input_data.get("use_cases") or []
    actors = input_data.get("actors")
    actors_list: List[str] = []
    if isinstance(actors, list):
        # Support both [{name, aliases}, ...] and ["Actor", ...]
        for a in actors:
            if isinstance(a, str):
                actors_list.append(a)
            elif isinstance(a, dict) and a.get("name"):
                actors_list.append(str(a.get("name")))

    results: List[ScenarioResult] = []
    model_configs = input_data.get("model_configs")
    model_configs_dict = model_configs if isinstance(model_configs, dict) else None
    for raw in raw_use_cases:
        if not isinstance(raw, dict):
            continue

        # Allow partial use case objects (id/description/user_stories may be missing).
        uc = UseCase.model_validate(raw)
        results.append(
            run_sca_use_case(
                use_case=uc,
                requirement_text=requirement_text,
                actors=actors_list or None,
                model_configs=model_configs_dict,
            )
        )

    return results
