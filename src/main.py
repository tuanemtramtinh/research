from __future__ import annotations

from pathlib import Path
import sys

from dotenv import load_dotenv

from .main_graph import build_main_graph


def main(input_file: str = "input_user_stories.txt", output_file: str | None = None):
    # Load environment variables from .env if present (OPENAI_API_KEY, etc.)
    load_dotenv()

    # Windows terminals can default to cp1252 and crash when printing Unicode (e.g., '→').
    # Force UTF-8 when possible; otherwise replace unencodable characters.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    root = Path(__file__).resolve().parents[1]
    input_path = root / input_file

    # Read input file - each line is a user story
    raw_text = input_path.read_text(encoding="utf-8")

    # Join lines with newline to preserve user story format
    # (RPA graph will split by sentences/lines internally)
    requirement_text = raw_text.strip()

    # Helper function to write to both stdout and file
    output_lines: list[str] = []

    def log(text: str = ""):
        print(text)
        output_lines.append(text)

    log(f"=== INPUT FILE: {input_file} ===")
    log(f"Loaded {len(requirement_text.split(chr(10)))} user stories\n")

    app = build_main_graph()
    result = app.invoke(
        {
            "requirement_text": requirement_text,
            "use_cases": [],
            "scenario_results_acc": [],
        }
    )

    # log("\n=== TASKS ===")
    # for t in result.get("tasks", []):
    #     log(f"{t.id}. {t.text}")

    log("\n=== ACTORS ALIASES ===")
    log(str(result.get("actor_aliases", [])))

    log("\n=== USE CASES ===")
    for uc in result.get("use_cases", []):
        actors = ", ".join(getattr(uc, "participating_actors", []) or [])
        uc_id = getattr(uc, "id", 0)
        log(f"[{uc_id}] {uc.name} (actors: {actors})")
        desc = getattr(uc, "description", "") or ""
        if desc:
            log(f"  description: {desc}")
        us = getattr(uc, "user_stories", None) or []
        if us:
            for us_item in us[:3]:
                orig = getattr(us_item, "original_sentence", "") or getattr(
                    us_item, "action", ""
                )
                if orig:
                    log(f"  - {orig}")

    log("\n=== SCENARIOS ===")
    for idx, sr in enumerate(result.get("scenario_results", []), 1):
        v = sr.validation
        log(f"\n--- SCENARIO {idx} ---")
        log(f"Use case: [{getattr(sr.use_case, 'id', 0)}] {sr.use_case.name}")
        log("\n--- CONTENT ---")
        spec = (sr.use_case_spec or "").strip()
        if spec:
            log(spec)
        elif sr.scenario is not None:
            sc = sr.scenario
            log(f"trigger: {sc.trigger}")
            for i, step in enumerate(sc.main_flow, 1):
                log(f"{i}. {step}")
        else:
            log("<EMPTY SCENARIO>")

        passed = bool(getattr(v, "passed", False))
        if not passed:
            log("\n--- VALIDATION: FAILED ---")
            failed_criteria = getattr(v, "failed_criteria", None)
            if isinstance(failed_criteria, dict):
                for crit, rationale in failed_criteria.items():
                    log(f"- {crit}: {rationale}")
            else:
                failed_fields = getattr(v, "failed_fields", None)
                if isinstance(failed_fields, dict):
                    for field, crits in (failed_fields or {}).items():
                        for criterion, rationale in (crits or {}).items():
                            log(f"- {field} ({criterion}): {rationale}")

            regen = str(getattr(v, "regen_rationale", "") or "").strip()
            if regen:
                log("\n--- REGEN RATIONALE ---")
                log(regen)
        else:
            log("\n--- VALIDATION: PASSED ---")

    # Write output to file if specified
    if output_file:
        output_path = root / output_file
        output_path.write_text("\n".join(output_lines), encoding="utf-8")
        print(f"\n=== OUTPUT SAVED TO: {output_path} ===")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run RPA + SCA pipeline on user stories"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="input_user_stories.txt",
        help="Input file containing user stories (default: input_user_stories.txt)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="output.txt",
        help="Output file to save results (default: output.txt)",
    )
    args = parser.parse_args()

    main(input_file=args.input, output_file=args.output)
