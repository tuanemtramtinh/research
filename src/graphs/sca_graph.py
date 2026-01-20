from __future__ import annotations

from typing import Dict, List

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


def _get_model():
    load_dotenv()
    model_name = "gpt-5-mini"
    if not os.getenv("OPENAI_API_KEY"):
        return None
    # Bias toward determinism.
    try:
        return init_chat_model(model_name, model_provider="openai", temperature=0)
    except TypeError:
        return init_chat_model(model_name, model_provider="openai")


def _use_case_payload(use_case: UseCase) -> dict:
    try:
        return use_case.model_dump()
    except Exception:
        return {
            "name": getattr(use_case, "name", ""),
            "participating_actors": getattr(use_case, "participating_actors", []) or [],
            "sentence_id": int(getattr(use_case, "sentence_id", 0) or 0),
            "sentence": getattr(use_case, "sentence", ""),
            "relationships": [getattr(r, "model_dump", lambda: r)() for r in (getattr(use_case, "relationships", []) or [])],
        }


def _heuristic_use_case_spec(*, requirement_text: str, actors: List[str], use_case: UseCase) -> str:
    # Minimal deterministic template output when no LLM is configured.
    uc = _use_case_payload(use_case)
    uid = f"UC-{int(uc.get('sentence_id') or 0)}"
    primary = ", ".join(use_case.participating_actors or [])
    all_actors = ", ".join(actors or (use_case.participating_actors or []))
    trig = (use_case.sentence or "").strip() or f"A primary actor initiates '{use_case.name}'."

    include_rels = [r for r in (uc.get("relationships") or []) if str(r.get("type", "")).lower() == "include"]
    extend_rels = [r for r in (uc.get("relationships") or []) if str(r.get("type", "")).lower() == "extend"]

    main_steps: List[str] = []
    step_no = 1
    for r in include_rels:
        target = str(r.get("target_use_case", "")).strip()
        if not target:
            continue
        main_steps.append(f"{step_no}. Primary Actor requests to perform mandatory sub-flow «include» {target} → System completes «include» {target}.")
        step_no += 1

    main_steps.append(f"{step_no}. Primary Actor initiates '{use_case.name}' → System begins processing the request.")
    step_no += 1
    main_steps.append(f"{step_no}. Primary Actor provides required input → System validates the input and proceeds.")
    step_no += 1
    main_steps.append(f"{step_no}. Primary Actor confirms completion → System completes '{use_case.name}' and records the outcome.")

    af_lines: List[str] = []
    for i, r in enumerate(extend_rels, 1):
        target = str(r.get("target_use_case", "")).strip()
        if not target:
            continue
        af_lines.append(f"AF-{i} (from Step 2): <condition to trigger «extend» {target}>\n1. Primary Actor requests optional sub-flow «extend» {target} → System performs «extend» {target}.\nReturn to Step 3.")

    ef_lines = [
        "EF-1 (from Step 2): <system cannot process the request>\n1. Primary Actor submits the request → System displays an error and ends the use case."
    ]

    return "\n".join(
        [
            "Use case name:",
            f"{use_case.name}",
            "Area:",
            "",
            "UniqueID:",
            uid,
            "Primary Actor(s):",
            primary,
            "Supporting Actor(s):",
            "",
            "Description:",
            "",
            "Triggering Event:",
            trig,
            "Trigger type: (External | Temporal)",
            "External",
            "",
            "Preconditions:",
            "",
            "Postconditions:",
            "",
            "Assumptions:",
            "",
            "Requirements Met:",
            "",
            "Priority:",
            "",
            "Risk:",
            "",
            "Outstanding Issues:",
            "",
            "",
            "-------------------------------------------------",
            "MAIN FLOW (SUCCESS SCENARIO)",
            "-------------------------------------------------",
            *main_steps,
            "",
            "-------------------------------------------------",
            "ALTERNATIVE FLOWS",
            "-------------------------------------------------",
            *(af_lines or [""]),
            "",
            "-------------------------------------------------",
            "EXCEPTION FLOWS",
            "-------------------------------------------------",
            *ef_lines,
            "",
            "-------------------------------------------------",
            "INFORMATION FOR STEPS (OPTIONAL BUT RECOMMENDED)",
            "-------------------------------------------------",
            "1. Request\n2. Input Data\n3. Confirmation",
            "",
            "---",
            "INPUT (DO NOT MODIFY)",
            "---",
            "Requirement Text:",
            requirement_text,
            "",
            "Actors:",
            all_actors,
            "",
            "Target Use Case:",
            json.dumps(_use_case_payload(use_case), ensure_ascii=False, indent=2),
        ]
    ).strip() + "\n"


# _WRITER_SYSTEM_PROMPT = "You are a professional and strict Use Case Specification Writer."

# _WRITER_HUMAN_PROMPT_TEMPLATE = r"""
# Your task is to generate ONE complete Use Case Scenario based on the given input.
# You must strictly follow the Use Case Specification template defined below.
# The output must be deterministic, unambiguous, and suitable for software requirement documentation.

# IMPORTANT RULES:
# - Generate ONLY ONE use case per prompt.
# - Do NOT invent actors, steps, or system behavior not supported by the input.
# - Use formal, neutral, and system-oriented language.
# - Use numbered steps.
# - Clearly separate Main Path, Alternative Flows, and Exception Flows.
# - Respect «include» and «extend» relationships semantically:
#   - «include» = mandatory sub-flow
#   - «extend» = optional or conditional flow
# - Assume this is a fully-dressed use case.

# ------------------------------------------------------------------
# USE CASE SPECIFICATION TEMPLATE (YOU MUST FOLLOW EXACTLY)
# ------------------------------------------------------------------

# Use case name:
# Area:
# UniqueID:
# Primary Actor(s):
# Supporting Actor(s):
# Description:
# Triggering Event:
# Trigger type: (External | Temporal)

# Preconditions:
# Postconditions:
# Assumptions:
# Requirements Met:
# Priority:
# Risk:
# Outstanding Issues:

# -------------------------------------------------
# MAIN FLOW (SUCCESS SCENARIO)
# -------------------------------------------------
# Steps must:
# - Be written as Actor action → System response
# - Represent the happy path only
# - Include all mandatory «include» use cases inline as steps

# Format:
# 1. ...
# 2. ...

# -------------------------------------------------
# ALTERNATIVE FLOWS
# -------------------------------------------------
# - Each alternative flow must:
#   - Reference the step number it branches from
#   - Clearly state the condition
#   - Return to a specific step OR end the use case

# Format:
# AF-1 (from Step X): <condition>
# AF-2 (from Step Y): <condition>

# -------------------------------------------------
# EXCEPTION FLOWS
# -------------------------------------------------
# - Exception flows represent errors or failures
# - Must describe system handling behavior

# Format:
# EF-1 (from Step X): <failure condition>

# -------------------------------------------------
# INFORMATION FOR STEPS (OPTIONAL BUT RECOMMENDED)
# -------------------------------------------------
# - Map each main step to the data involved
# - Use concise data names (e.g., Credentials, Cart, Payment Info)

# ------------------------------------------------------------------
# INPUT (DO NOT MODIFY)
# ------------------------------------------------------------------
# Requirement Text:
# {{requirement_text}}

# Actors:
# {{actors}}

# Target Use Case:
# {{single_use_case_object}}

# ------------------------------------------------------------------
# OUTPUT RULE (STRICT)
# ------------------------------------------------------------------
# - Output ONLY the completed Use Case Specification using the template above.
# - Do NOT output JSON.
# """

_WRITER_SYSTEM_PROMPT = "You are a professional, strict, and detail-oriented Use Case Specification Writer. You never leave any required field empty. You only infer information that is logically supported by the given Requirement Text. If information is implicit, you must state it explicitly using formal requirement language."

_WRITER_HUMAN_PROMPT_TEMPLATE = r"""
Your task is to generate EXACTLY ONE complete and fully-dressed Use Case Specification
based ONLY on the given input.

The output must be suitable for formal Software Requirement Specification (SRS) documents.

========================
CRITICAL ENFORCEMENT RULES
========================
1. EVERY field in the Use Case Specification template MUST be filled.
2. NO field may be empty, omitted, or marked as:
   - "N/A"
   - "None"
   - "Not specified"
   - "-"
3. If a field is not explicitly stated in the Requirement Text:
   - You MUST infer it conservatively and logically.
   - The inference MUST NOT introduce new system functionality.
4. If inference is required, ensure consistency by:
   - Reflecting it in Assumptions
   - Keeping it aligned with the Requirement Text
5. Do NOT invent:
   - New actors
   - New system capabilities
   - New business rules
6. Use only formal, neutral, system-oriented language.
7. Generate ONLY ONE use case.

========================
USE CASE SPECIFICATION TEMPLATE
(MUST FOLLOW EXACTLY)
========================

Use case name:
Area:
UniqueID:
Primary Actor(s):
Supporting Actor(s):
Description:
Triggering Event:
Trigger type: (External | Temporal)

Preconditions:
(Postconditions must be system-observable states)
Postconditions:
Assumptions:
(Explicit assumptions derived from missing but necessary information)
Requirements Met:
(List concrete system requirements satisfied by this use case)
Priority:
(Low | Medium | High – must be justified implicitly by requirement context)
Risk:
(Low | Medium | High – based on system dependency or failure impact)
Outstanding Issues:
(Open questions or policy decisions implied but not resolved)

-------------------------------------------------
MAIN FLOW (SUCCESS SCENARIO)
-------------------------------------------------
Rules:
- Each step MUST follow: Actor action → System response
- Steps represent ONLY the successful (happy) path
- System actions must be observable or verifiable
- Do NOT mix errors or conditions here

Format:
1. <Actor> performs action → <System> responds
2. ...

-------------------------------------------------
ALTERNATIVE FLOWS
-------------------------------------------------
Rules:
- Alternative flows are valid, non-error variations
- Each flow MUST:
  - Reference a step number
  - State a clear condition
  - Explicitly state where it returns or that it ends

Format:
AF-1 (from Step X): <condition>
AF-2 (from Step Y): <condition>

-------------------------------------------------
EXCEPTION FLOWS
-------------------------------------------------
Rules:
- Exception flows represent failures or abnormal conditions
- Each flow MUST describe system handling behavior
- Each flow MUST clearly end the use case or stop progression

Format:
EF-1 (from Step X): <failure condition>

-------------------------------------------------
INFORMATION FOR STEPS
-------------------------------------------------
Rules:
- Map EACH main step to the data involved
- Use concise and consistent data names
- Do NOT invent new data entities

Format:
1. <Data>
2. <Data>
3. <Data>

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
- Output ONLY the completed Use Case Specification.
- Do NOT output explanations, notes, JSON, or markdown.
- The output MUST strictly follow the template structure above.
"""

def generate_use_case_spec_node(state: ScaState):
    model = _get_model()
    use_case: UseCase = state.get("use_case")  # type: ignore[assignment]
    requirement_text = str(state.get("requirement_text") or "")
    actors = state.get("actors") or (use_case.participating_actors or [])
    spec_version = int(state.get("spec_version") or 0) + 1

    if model is None:
        return {
            "use_case_spec": _heuristic_use_case_spec(
                requirement_text=requirement_text,
                actors=[str(a) for a in (actors or [])],
                use_case=use_case,
            ),
            "spec_version": spec_version,
        }

    use_case_json = json.dumps(_use_case_payload(use_case), ensure_ascii=False, indent=2)
    prompt = (
        _WRITER_HUMAN_PROMPT_TEMPLATE
        .replace("{{requirement_text}}", requirement_text)
        .replace("{{actors}}", json.dumps(list(actors or []), ensure_ascii=False))
        .replace("{{single_use_case_object}}", use_case_json)
    )

    resp = model.invoke([("system", _WRITER_SYSTEM_PROMPT), ("human", prompt)])
    content = str(getattr(resp, "content", "") or "").strip()
    return {"use_case_spec": content, "spec_version": spec_version}


def regenerate_use_case_spec_node(state: ScaState):
    model = _get_model()
    use_case: UseCase = state.get("use_case")  # type: ignore[assignment]
    requirement_text = str(state.get("requirement_text") or "")
    actors = state.get("actors") or (use_case.participating_actors or [])
    spec_version = int(state.get("spec_version") or 0) + 1

    current_spec = str(state.get("use_case_spec") or "")
    validation: UseCaseSpecValidation | None = state.get("validation")  # type: ignore[assignment]
    issues = (getattr(validation, "regen_rationale", "") or "").strip() if validation else ""

    if model is None:
        return {"use_case_spec": current_spec, "spec_version": spec_version}

    use_case_json = json.dumps(_use_case_payload(use_case), ensure_ascii=False, indent=2)
    prompt = (
        _WRITER_HUMAN_PROMPT_TEMPLATE
        .replace("{{requirement_text}}", requirement_text)
        .replace("{{actors}}", json.dumps(list(actors or []), ensure_ascii=False))
        .replace("{{single_use_case_object}}", use_case_json)
    )

    system_prompt = (
        "You are a professional and strict Use Case Specification Writer. "
        "Regenerate the use case specification to address the judge-identified deficiencies. "
        "Only change what is necessary."
    )
    human_prompt = f"""{prompt}

---
CURRENT USE CASE SPECIFICATION:
{current_spec}

---
DEFICIENCIES TO FIX (DO NOT IGNORE):
{issues}

Output ONLY the corrected use case specification in the exact template format.
"""

    resp = model.invoke([("system", system_prompt), ("human", human_prompt)])
    content = str(getattr(resp, "content", "") or "").strip()
    return {"use_case_spec": content, "spec_version": spec_version}


_JUDGE_SYSTEM_PROMPT = "You are acting as a STRICT, RULE-BASED Use Case Specification JUDGE."

_JUDGE_HUMAN_PROMPT_TEMPLATE = r"""
Your role is ONLY to EVALUATE, not to improve, rewrite, or suggest changes.

You must assess a COMPLETE USE CASE SPECIFICATION exactly as provided,
without assuming missing information or filling any gaps.

All judgments MUST strictly follow the evaluation criteria and scoring rules defined below.
If any required field is missing, unclear, or weakly defined, you MUST treat it as a deficiency.

You will evaluate the use case using EXACTLY THREE criteria:
1. Completeness (Đầy đủ)
2. Coherence (Mạch lạc)
3. Relevance (Liên quan)

You MUST output results in the specified JSON format ONLY.

--------------------------------
INPUT
--------------------------------
<USE_CASE_SPEC>
{{use_case_spec}}
</USE_CASE_SPEC>

--------------------------------
EVALUATION CRITERIA
--------------------------------

================================
1. COMPLETENESS (Đầy đủ)
================================

Evaluate whether the use case specification contains all REQUIRED fields
and whether each field is sufficiently defined.

STRUCTURE COMPLETENESS:
- Use Case Name is present and clearly describes a single user goal
- Actor(s) are clearly identified and appropriate
- Description explains WHAT the user wants to achieve and WHY
- Triggering Event is defined and realistic
- Preconditions are stated and necessary
- Postconditions describe the system state after completion

BEHAVIOR COMPLETENESS:
- Main Flow exists and covers the happy path end-to-end
- Alternative Flows exist for valid variations
- Exception Flows exist for error or failure cases
- Flows align with preconditions and postconditions

Scoring (0–100):
- 90–100: All required fields present and well-defined
- 70–89: Minor missing or weak fields
- 50–69: Important fields missing or incomplete
- <50: Many required fields missing

PASS if score ≥ 70, otherwise FAIL.

Provide rationale:
- Which fields are correct
- Which fields are missing or insufficient

================================
2. COHERENCE (Mạch lạc)
================================

Evaluate logical consistency ACROSS ALL FIELDS.

Check:
- Use Case Name matches Description and flows
- Actor actions are consistent across all flows
- Triggering Event logically leads to Main Flow
- Preconditions are required by the flows
- Postconditions are achieved by the flows
- Alternative and Exception Flows clearly branch from Main Flow
- No contradictions between fields

Scoring (0–100):
- 90–100: Fully consistent and logical
- 70–89: Minor inconsistencies
- 50–69: Several logical mismatches
- <50: Conflicting or illogical structure

PASS if score ≥ 70.

================================
3. RELEVANCE (Liên quan)
================================

Evaluate whether all fields and flows focus on ONE use case goal.

Check:
- Use Case Name reflects the actual goal
- Description and flows serve the same goal
- No steps unrelated to the stated goal
- No scope creep into other use cases
- Alternative/Exception flows are still relevant

Scoring (0–100):
- 90–100: Highly focused
- 70–89: Mostly relevant
- 50–69: Several off-scope elements
- <50: Unfocused or mixed goals

PASS if score ≥ 70.

--------------------------------
OUTPUT FORMAT (STRICT JSON)
--------------------------------

{
  "Completeness": {
    "score": <0-100>,
    "result": "PASS | FAIL",
    "rationale": "...",
    "missing_or_weak_fields": [
      "field name",
      "field name"
    ]
  },
  "Coherence": {
    "score": <0-100>,
    "result": "PASS | FAIL",
    "rationale": "..."
  },
  "Relevance": {
    "score": <0-100>,
    "result": "PASS | FAIL",
    "rationale": "..."
  }
}

Rules:
- Do NOT assume missing fields
- Judge only based on provided content
- Be strict and consistent
"""


def _judge_node(detector_name: str):
    def _fn(state: ScaState):
        model = _get_model()
        spec = str(state.get("use_case_spec") or "")
        spec_version = int(state.get("spec_version") or 0)

        if model is None:
            # Conservative fallback: fail if any required text is missing.
            missing: List[str] = []
            for required in [
                "Use case name:",
                "Primary Actor(s):",
                "Description:",
                "Triggering Event:",
                "Preconditions:",
                "Postconditions:",
                "MAIN FLOW (SUCCESS SCENARIO)",
                "ALTERNATIVE FLOWS",
                "EXCEPTION FLOWS",
            ]:
                if required not in spec:
                    missing.append(required)

            completeness_pass = len(missing) == 0
            evaluation = UseCaseEvaluation(
                Completeness={
                    "score": 80 if completeness_pass else 40,
                    "result": "PASS" if completeness_pass else "FAIL",
                    "rationale": "All required structural sections are present." if completeness_pass else "Missing required sections.",
                    "missing_or_weak_fields": missing,
                },
                Coherence={
                    "score": 70 if completeness_pass else 40,
                    "result": "PASS" if completeness_pass else "FAIL",
                    "rationale": "Cannot verify coherence without full content." if completeness_pass else "Insufficient content to verify coherence.",
                },
                Relevance={
                    "score": 70 if completeness_pass else 40,
                    "result": "PASS" if completeness_pass else "FAIL",
                    "rationale": "Assumed focused if structure exists." if completeness_pass else "Unclear goal due to missing content.",
                },
            )
            jr = UseCaseSpecJudgeResult(detector=detector_name, spec_version=spec_version, evaluation=evaluation)
            return {"judge_results": [jr]}

        structured_llm = model.with_structured_output(UseCaseEvaluation)
        prompt = _JUDGE_HUMAN_PROMPT_TEMPLATE.replace("{{use_case_spec}}", spec)
        evaluation: UseCaseEvaluation = structured_llm.invoke([("system", _JUDGE_SYSTEM_PROMPT), ("human", prompt)])
        jr = UseCaseSpecJudgeResult(detector=detector_name, spec_version=spec_version, evaluation=evaluation)
        return {"judge_results": [jr]}

    return _fn


def combiner_node(state: ScaState):
    model = _get_model()
    spec_version = int(state.get("spec_version") or 0)
    judge_results = [
        r for r in (state.get("judge_results") or [])
        if int(getattr(r, "spec_version", 0)) == spec_version
    ]

    def _threshold(n: int) -> int:
        import math

        return max(1, int(math.ceil((2.0 * n) / 3.0)))

    criteria_names = ["Completeness", "Coherence", "Relevance"]
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
            rows.append((r.detector, str(getattr(crit_ev, "result", "")), str(getattr(crit_ev, "rationale", "") or "")))
            if crit == "Completeness":
                mf = list(getattr(crit_ev, "missing_or_weak_fields", []) or [])
                for f in mf:
                    if f and f not in missing_fields:
                        missing_fields.append(f)

        n = len(rows)
        if n == 0:
            failed[crit] = "No judge results produced."
            continue

        fails = [(d, rat) for (d, res, rat) in rows if res != "PASS"]
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
                regen_rationale = str(model.invoke([("system", system_prompt), ("human", human_prompt)]).content or "").strip()
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
    workflow.add_node("judge_1", _judge_node("judge_1"))
    workflow.add_node("judge_2", _judge_node("judge_2"))
    workflow.add_node("judge_3", _judge_node("judge_3"))
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


def run_sca_use_case(*, use_case: UseCase, requirement_text: str, actors: List[str] | None = None) -> ScenarioResult:
    app = build_sca_graph()
    out = app.invoke(
        {
            "requirement_text": requirement_text,
            "actors": actors or (use_case.participating_actors or []),
            "use_case": use_case,
            "use_case_spec": "",
            "spec_version": 0,
            "judge_results": [],
            "validation": None,
        }
    )

    spec = str(out.get("use_case_spec") or "")
    validation = out.get("validation") or UseCaseSpecValidation(passed=True, failed_criteria={}, regen_rationale="")

    # If judge nodes ran with structured output, the combined evaluation is embedded in judge_results;
    # store the first judge's evaluation as a representative snapshot.
    jr = None
    for r in out.get("judge_results") or []:
        if int(getattr(r, "spec_version", 0)) == int(out.get("spec_version") or 0):
            jr = r
            break

    return ScenarioResult(
        use_case=use_case,
        scenario=None,
        use_case_spec=spec,
        evaluation=getattr(jr, "evaluation", None),
        validation=validation,
    )


def run_sca(input_data: dict) -> List[ScenarioResult]:
    """Convenience wrapper: accept the RPA-like JSON input and generate scenarios for all provided use cases.

    Expected shape:
    {
      "requirement_text": str,
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
    for raw in raw_use_cases:
        if not isinstance(raw, dict):
            continue

        # Allow partial use case objects (sentence_id/sentence may be missing).
        uc = UseCase.model_validate(raw)
        results.append(run_sca_use_case(use_case=uc, requirement_text=requirement_text, actors=actors_list or None))

    return results
