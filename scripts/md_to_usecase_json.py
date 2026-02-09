from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

SECTION_RE = re.compile(r"^##\s+\d+\.\s+(.+?)\s*$")
BOLD_LINE_RE = re.compile(r"^\*\*(.+?)\*\*\s*$")
MAIN_STEP_RE = re.compile(r"^\s*(\d+)\.\s+(.*)$")
ALT_FLOW_START_RE = re.compile(r"^\s*-\s*\*\*(AF\d+)\s*–\s*(.+?)\*\*\s*$")
EXC_FLOW_START_RE = re.compile(r"^\s*(E\d+)\s*–\s*(.+?)\s*(?:\(Step\s*(.+?)\))?\s*$")
TRIGGER_RE = re.compile(r"^\s*\*Trigger:\*\s*(.+?)\s*$", re.IGNORECASE)


def _split_sections(text: str) -> Dict[str, List[str]]:
    sections: Dict[str, List[str]] = {}
    current = None
    for line in text.splitlines():
        m = SECTION_RE.match(line.strip())
        if m:
            current = m.group(1).strip().lower()
            sections[current] = []
            continue
        if current is not None:
            sections[current].append(line.rstrip())
    return sections


def _strip_empty(lines: List[str]) -> List[str]:
    out = []
    for ln in lines:
        if ln.strip() == "" and (not out or out[-1] == ""):
            continue
        out.append(ln)
    while out and out[0].strip() == "":
        out.pop(0)
    while out and out[-1].strip() == "":
        out.pop()
    return out


def _parse_single_value(lines: List[str]) -> str:
    lines = _strip_empty(lines)
    if not lines:
        return ""
    if len(lines) == 1:
        m = BOLD_LINE_RE.match(lines[0].strip())
        if m:
            return m.group(1).strip()
        return lines[0].strip()
    # If the first non-empty line is bold, take it.
    for ln in lines:
        m = BOLD_LINE_RE.match(ln.strip())
        if m:
            return m.group(1).strip()
    return " ".join(ln.strip() for ln in lines if ln.strip())


def _parse_paragraph(lines: List[str]) -> str:
    return " ".join(ln.strip() for ln in _strip_empty(lines) if ln.strip())


def _parse_main_flow(lines: List[str]) -> List[str]:
    steps: List[str] = []
    for ln in lines:
        m = MAIN_STEP_RE.match(ln.strip())
        if m:
            steps.append(f"{m.group(1)}. {m.group(2).strip()}")
    return steps


def _parse_alternative_flows(lines: List[str]) -> List[str]:
    flows: List[str] = []
    current_id = None
    current_title = None
    current_trigger = None
    current_body: List[str] = []

    def _flush():
        nonlocal current_id, current_title, current_trigger, current_body
        if not current_id:
            return
        parts = []
        head = f"{current_id}"
        if current_title:
            head += f" ({current_title})"
        if current_trigger:
            head += f" [Trigger: {current_trigger}]"
        parts.append(head + ":")
        if current_body:
            parts.append(" ".join(s.strip() for s in current_body if s.strip()))
        flows.append(" ".join(parts).strip())
        current_id = None
        current_title = None
        current_trigger = None
        current_body = []

    for ln in lines:
        m = ALT_FLOW_START_RE.match(ln.strip())
        if m:
            _flush()
            current_id = m.group(1).strip()
            current_title = m.group(2).strip()
            continue
        t = TRIGGER_RE.match(ln.strip())
        if t:
            current_trigger = t.group(1).strip()
            continue
        if current_id:
            if ln.strip():
                current_body.append(ln.strip())
    _flush()
    return flows


def _parse_exception_flows(lines: List[str]) -> List[str]:
    flows: List[str] = []
    current_id = None
    current_title = None
    current_step = None
    current_body: List[str] = []

    def _flush():
        nonlocal current_id, current_title, current_step, current_body
        if not current_id:
            return
        head = f"{current_id}"
        if current_title:
            head += f" ({current_title})"
        if current_step:
            head += f" [Step: {current_step}]"
        parts = [head + ":"]
        if current_body:
            parts.append(" ".join(s.strip() for s in current_body if s.strip()))
        flows.append(" ".join(parts).strip())
        current_id = None
        current_title = None
        current_step = None
        current_body = []

    for ln in lines:
        m = MAIN_STEP_RE.match(ln.strip())
        if m and m.group(1).startswith("E"):
            # Not expected; keep for safety
            pass
        m2 = re.match(r"^\s*\d+\.\s*\*\*(E\d+)\s*–\s*(.+?)\*\*\s*$", ln.strip())
        if m2:
            _flush()
            current_id = m2.group(1).strip()
            current_title = m2.group(2).strip()
            current_step = None
            continue
        m3 = re.match(r"^\s*\d+\.\s*\*\*(E\d+)\s*–\s*(.+?)\*\*\s*\(Step\s*(.+?)\)\s*$", ln.strip())
        if m3:
            _flush()
            current_id = m3.group(1).strip()
            current_title = m3.group(2).strip()
            current_step = m3.group(3).strip()
            continue
        # Fallback: plain “E1 – …” lines
        m4 = EXC_FLOW_START_RE.match(ln.strip())
        if m4:
            _flush()
            current_id = m4.group(1).strip()
            current_title = m4.group(2).strip()
            current_step = (m4.group(3) or "").strip() or None
            continue
        if current_id:
            if ln.strip():
                current_body.append(ln.strip())
    _flush()
    return flows


def _info_from_steps(steps: List[str]) -> List[str]:
    info: List[str] = []
    for s in steps:
        m = MAIN_STEP_RE.match(s)
        if m:
            idx = m.group(1)
            body = m.group(2)
            info.append(f"{idx}. {body}")
    return info or ["1. Actor input", "2. System response", "3. Confirmation"]


def convert_md_to_json(md_path: Path, uid: str) -> dict:
    text = md_path.read_text(encoding="utf-8")
    sections = _split_sections(text)

    use_case_name = _parse_single_value(sections.get("use case name", []))
    description = _parse_paragraph(sections.get("description", []))
    primary_actor = _parse_single_value(sections.get("primary actor", []))
    context = _parse_paragraph(sections.get("problem domain context", []))
    preconditions_raw = _parse_paragraph(sections.get("preconditions", []))
    postconditions_raw = _parse_paragraph(sections.get("postconditions", []))

    main_flow = _parse_main_flow(sections.get("main flow", []))
    alternative_flows = _parse_alternative_flows(sections.get("alternative flows", []))
    exception_flows = _parse_exception_flows(sections.get("exceptions", []))

    preconditions = [preconditions_raw] if preconditions_raw and preconditions_raw.lower() != "none." else ["None"]
    postconditions = [postconditions_raw] if postconditions_raw else ["Outcome recorded by system"]

    primary_actors = [primary_actor] if primary_actor else ["Primary Actor"]

    spec = {
        "use_case_name": use_case_name or md_path.stem.replace("_", " ").replace("-", " ").strip(),
        "unique_id": uid,
        "area": "Urban Mobility" if "mobility" in context.lower() else "Requirements Domain",
        "context_of_use": context or "The primary actor performs the use case within the system's operational context.",
        "scope": "Target System",
        "level": "User-goal",
        "primary_actors": primary_actors,
        "supporting_actors": ["System"],
        "stakeholders_and_interests": [
            f"{primary_actors[0]}: Achieve the goal described in the use case."
        ],
        "description": description or f"Enable {primary_actors[0]} to perform {use_case_name}.",
        "triggering_event": f"{primary_actors[0]} initiates '{use_case_name}'.",
        "trigger_type": "External",
        "preconditions": preconditions,
        "postconditions": postconditions,
        "assumptions": ["The actor has appropriate access to the system."],
        "requirements_met": [description] if description else [f"The system shall support '{use_case_name}'."],
        "priority": "Medium",
        "risk": "Medium",
        "outstanding_issues": ["None identified."],
        "main_flow": main_flow,
        "alternative_flows": alternative_flows,
        "exception_flows": exception_flows,
        "information_for_steps": _info_from_steps(main_flow),
    }
    return spec


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert paradigm_scenario markdown files to use case JSON.")
    parser.add_argument(
        "--input-dir",
        "-i",
        type=str,
        default=r"C:\Users\kyluo\research\paradigm_scenario\1",
        help="Directory containing markdown files",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="",
        help="Directory for JSON output (default: same as input)",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    md_files = sorted(input_dir.glob("*.md"))
    if not md_files:
        print(f"No .md files found in {input_dir}")
        return 1

    for idx, md_path in enumerate(md_files, 1):
        uid = f"UC-{idx:03d}"
        spec = convert_md_to_json(md_path, uid)
        out_path = output_dir / f"{md_path.stem}.json"
        out_path.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
