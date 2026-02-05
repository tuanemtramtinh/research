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
    model_name = "gpt-4o-mini"
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
) -> str:
    # Minimal deterministic template output when no LLM is configured.
    uc = _use_case_payload(use_case)
    uid = f"UC-{int(uc.get('id') or 0)}"
    primary = ", ".join(use_case.participating_actors or [])
    all_actors = ", ".join(actors or (use_case.participating_actors or []))
    
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

    return (
        "\n".join(
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
        ).strip()
        + "\n"
    )


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

Use Case Name:
(The name of the use case, expressed as a short active verb phrase that represents the goal of the primary actor and leads to an observable result when completed)
Unique ID:
(A unique identifier assigned to the use case for reference, traceability, and linkage with requirements, test cases, or other artifacts)
Area:
(The business domain or functional area to which the use case belongs, used for classification and organization)
Context of Use:
(A longer statement describing the goal of the use case within its normal operating context, including assumptions about the environment and typical conditions)
Scope:
(The design scope of the use case, specifying which system is considered as a black box and is responsible for the described behavior)
Level:
(The abstraction level of the use case, one of: Summary, User-goal, or Sub-function, indicating its role within the overall system behavior)
Primary Actor(s):
(The main external role that initiates the use case and whose goal is directly fulfilled by the successful execution of the use case)
Supporting Actor(s):
(Secondary actors that assist in the execution of the use case, such as external systems or services, but do not pursue the primary goal)
Stakeholders and Interests:
(A list of stakeholders involved in or affected by the use case, along with their key interests, expectations, or concerns regarding its outcome)
Description:
(A brief summary explaining the purpose of the use case and the value it provides to the primary actor)
Triggering Event:
(The event that initiates the use case, such as an action performed by an actor or a condition that requires the system to respond)
Trigger Type (External | Temporal):
(The classification of the trigger: External if initiated by an actor or external system; Temporal if initiated by time or a scheduled event)

Preconditions:
(Conditions that are assumed to be true before the use case begins; the use case does not establish these conditions itself)
Postconditions:
(Postconditions must be system-observable states)
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

    use_case_json = json.dumps(
        _use_case_payload(use_case), ensure_ascii=False, indent=2
    )
    prompt = (
        _WRITER_HUMAN_PROMPT_TEMPLATE.replace("{{requirement_text}}", requirement_text)
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
    issues = (
        (getattr(validation, "regen_rationale", "") or "").strip() if validation else ""
    )

    if model is None:
        return {"use_case_spec": current_spec, "spec_version": spec_version}

    use_case_json = json.dumps(
        _use_case_payload(use_case), ensure_ascii=False, indent=2
    )
    prompt = (
        _WRITER_HUMAN_PROMPT_TEMPLATE.replace("{{requirement_text}}", requirement_text)
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
2. Coherence
3. Relevance

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

Total Score = Sum of all field scores (0–100).

PASS if Total Score ≥ 70.
FAIL if Total Score < 70.

Provide rationale:
- Which fields are weak, missing, or incorrect

================================
2. COHERENCE
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
                    "rationale": "All required structural sections are present."
                    if completeness_pass
                    else "Missing required sections.",
                    "missing_or_weak_fields": missing,
                },
                Coherence={
                    "score": 70 if completeness_pass else 40,
                    "result": "PASS" if completeness_pass else "FAIL",
                    "rationale": "Cannot verify coherence without full content."
                    if completeness_pass
                    else "Insufficient content to verify coherence.",
                },
                Relevance={
                    "score": 70 if completeness_pass else 40,
                    "result": "PASS" if completeness_pass else "FAIL",
                    "rationale": "Assumed focused if structure exists."
                    if completeness_pass
                    else "Unclear goal due to missing content.",
                },
            )
            jr = UseCaseSpecJudgeResult(
                detector=detector_name, spec_version=spec_version, evaluation=evaluation
            )
            return {"judge_results": [jr]}

        structured_llm = model.with_structured_output(UseCaseEvaluation)
        prompt = _JUDGE_HUMAN_PROMPT_TEMPLATE.replace("{{use_case_spec}}", spec)
        evaluation: UseCaseEvaluation = structured_llm.invoke(
            [("system", _JUDGE_SYSTEM_PROMPT), ("human", prompt)]
        )
        jr = UseCaseSpecJudgeResult(
            detector=detector_name, spec_version=spec_version, evaluation=evaluation
        )
        return {"judge_results": [jr]}

    return _fn


def combiner_node(state: ScaState):
    model = _get_model()
    spec_version = int(state.get("spec_version") or 0)
    judge_results = [
        r
        for r in (state.get("judge_results") or [])
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


def run_sca_use_case(
    *, use_case: UseCase, requirement_text: List[str], actors: List[str] | None = None
) -> ScenarioResult:
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
    validation = out.get("validation") or UseCaseSpecValidation(
        passed=True, failed_criteria={}, regen_rationale=""
    )

    # If judge nodes ran with structured output, the combined evaluation is embedded in judge_results;
    # store the first judge's evaluation as a representative snapshot.
    jr = None
    for r in out.get("judge_results") or []:
        if int(getattr(r, "spec_version", 0)) == int(out.get("spec_version") or 0):
            jr = r
            break

    return ScenarioResult(
        use_case=use_case,
        use_case_spec=spec,
        evaluation=getattr(jr, "evaluation", None),
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
            )
        )

    return results
