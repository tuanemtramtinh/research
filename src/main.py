from __future__ import annotations

from pathlib import Path
import sys

from dotenv import load_dotenv

from .main_graph import build_main_graph


def main():
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
    input_path = root / "input.txt"

    requirement_text = input_path.read_text(encoding="utf-8")

    app = build_main_graph()
    result = app.invoke({"requirement_text": requirement_text, "use_cases": [], "scenario_results_acc": []})

    print("\n=== TASKS ===")
    for t in result.get("tasks", []):
        print(f"{t.id}. {t.text}")

    print("\n=== MERGED ACTORS ===")
    print(result.get("merged_actors", []))

    print("\n=== USE CASES ===")
    for uc in result.get("use_cases", []):
        actors = ", ".join(uc.participating_actors)
        print(f"[{uc.sentence_id}] {uc.name} (actors: {actors})")
        print(f"  sentence: {uc.sentence}")

    print("\n=== SCENARIOS ===")
    for idx, sr in enumerate(result.get("scenario_results", []), 1):
        v = sr.validation
        print(f"\n--- SCENARIO {idx} ---")
        print(f"Use case: [{sr.use_case.sentence_id}] {sr.use_case.name}")
        print("\n--- CONTENT ---")
        spec = (sr.use_case_spec or "").strip()
        if spec:
            print(spec)
        elif sr.scenario is not None:
            sc = sr.scenario
            print(f"trigger: {sc.trigger}")
            for i, step in enumerate(sc.main_flow, 1):
                print(f"{i}. {step}")
        else:
            print("<EMPTY SCENARIO>")

        passed = bool(getattr(v, "passed", False))
        if not passed:
            print("\n--- VALIDATION: FAILED ---")
            failed_criteria = getattr(v, "failed_criteria", None)
            if isinstance(failed_criteria, dict):
                for crit, rationale in failed_criteria.items():
                    print(f"- {crit}: {rationale}")
            else:
                failed_fields = getattr(v, "failed_fields", None)
                if isinstance(failed_fields, dict):
                    for field, crits in (failed_fields or {}).items():
                        for criterion, rationale in (crits or {}).items():
                            print(f"- {field} ({criterion}): {rationale}")

            regen = str(getattr(v, "regen_rationale", "") or "").strip()
            if regen:
                print("\n--- REGEN RATIONALE ---")
                print(regen)
        else:
            print("\n--- VALIDATION: PASSED ---")


if __name__ == "__main__":
    main()
