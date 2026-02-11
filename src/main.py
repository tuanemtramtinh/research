from __future__ import annotations

from pathlib import Path
import sys

from dotenv import load_dotenv

from .main_graph import build_main_graph


def main(
    input_file: str = "input_user_stories.txt",
    output_file: str | None = None,
    reference_dir: str | None = None,
    comparison_dir: str | None = None,
):
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
            "reference_dir": reference_dir,
            "comparison_dir": comparison_dir,
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

        ev = getattr(sr, "evaluation", None)
        if ev is not None:
            comp = getattr(ev, "Completeness", None)
            corr = getattr(ev, "Correctness", None)
            rel = getattr(ev, "Relevance", None)

            def _avg_overall_score(*criteria) -> str:
                scores: list[float] = []
                for c in criteria:
                    if c is None:
                        continue
                    try:
                        if str(getattr(c, "result", "") or "").strip().upper() == "N/A":
                            continue
                    except Exception:
                        pass
                    s = getattr(c, "score", None)
                    if isinstance(s, (int, float)):
                        scores.append(float(s))
                if not scores:
                    return "N/A"
                return f"{int(round(sum(scores) / float(len(scores))))}/100"

            def _fmt_crit(name: str, crit) -> str:
                if crit is None:
                    return f"- {name}: <no score>"
                score = getattr(crit, "score", None)
                result_txt = getattr(crit, "result", None)
                if score is None and result_txt is None:
                    return f"- {name}: <no score>"
                if score is None:
                    return f"- {name}: {result_txt}"
                if result_txt is None:
                    return f"- {name}: {score}/100"
                return f"- {name}: {score}/100 ({result_txt})"

            def _fmt_correctness(crit) -> str:
                if crit is None:
                    return "- Correctness: <no score>"
                res = getattr(crit, "result", None)
                score = getattr(crit, "score", None)
                if res == "N/A":
                    return "- Correctness: N/A"
                if score is None and res is not None:
                    return f"- Correctness: {res}"
                if score is not None and res is not None:
                    return f"- Correctness: {score}/100 ({res})"
                if score is not None:
                    return f"- Correctness: {score}/100"
                return "- Correctness: <no score>"

            log("\n--- SCORES ---")
            log(_fmt_crit("Completeness", comp))
            log(_fmt_correctness(corr))
            log(_fmt_crit("Relevance", rel))
            log(f"- Overall (avg): {_avg_overall_score(comp, corr, rel)}")

            cmp_path = getattr(sr, "comparison_spec_path", None)
            cmp_ev = getattr(sr, "comparison_evaluation", None)
            if cmp_path:
                log("\n--- COMPARISON SCENARIO SCORES (FILE) ---")
                log(str(cmp_path))
                if cmp_ev is None:
                    log("<NO COMPARISON EVALUATION>")
                else:
                    cmp_comp = getattr(cmp_ev, "Completeness", None)
                    cmp_corr = getattr(cmp_ev, "Correctness", None)
                    cmp_rel = getattr(cmp_ev, "Relevance", None)
                    log(_fmt_crit("Completeness", cmp_comp))
                    log(_fmt_correctness(cmp_corr))
                    log(_fmt_crit("Relevance", cmp_rel))
                    log(f"- Overall (avg): {_avg_overall_score(cmp_comp, cmp_corr, cmp_rel)}")
            else:
                log("\n--- COMPARISON SCENARIO SCORES ---")
                log("N/A")

            def _log_sub_scores(label: str, crit) -> None:
                if crit is None:
                    return
                sub = None
                try:
                    sub = crit.get("sub_scores")
                except Exception:
                    sub = getattr(crit, "sub_scores", None)

                if not isinstance(sub, dict) or not sub:
                    return

                log(f"\n--- {label.upper()} SUB-SCORES ---")
                for k, val in sub.items():
                    ks = str(k).strip()
                    if not ks:
                        continue
                    log(f"- {ks}: {val}")

            _log_sub_scores("Completeness", comp)
            _log_sub_scores("Correctness", corr)
            _log_sub_scores("Relevance", rel)

            missing = list(getattr(comp, "missing_or_weak_fields", []) or []) if comp else []
            if missing:
                log("- Missing/weak fields: " + ", ".join(str(x) for x in missing if str(x).strip()))
        else:
            log("\n--- SCORES ---")
            log("<NO EVALUATION>")

        log("\n--- CONTENT ---")
        spec_obj = getattr(sr, "use_case_spec_json", None) or {}
        if isinstance(spec_obj, dict) and spec_obj:
            import json

            log(json.dumps(spec_obj, ensure_ascii=False, indent=2))
        else:
            log("<EMPTY USE CASE SPEC JSON>")

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

        # Full raw scenario result JSON (includes evaluation + sub_scores + validation)
        try:
            payload = sr.model_dump()  # pydantic v2
        except Exception:
            try:
                payload = sr.dict()  # pydantic v1 fallback
            except Exception:
                payload = None

        if payload is not None:
            import json

            log("\n--- RAW SCENARIO RESULT JSON ---")
            log(json.dumps(payload, ensure_ascii=False, indent=2))

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
    parser.add_argument(
        "--reference-dir",
        type=str,
        default=None,
        help="Directory containing reference JSON files for correctness evaluation (auto-matched to use cases by name)",
    )
    parser.add_argument(
        "--comparison-dir",
        type=str,
        default=None,
        help="Directory containing comparison JSON files to score alongside generated specs (auto-matched to use cases by name)",
    )
    args = parser.parse_args()

    main(
        input_file=args.input,
        output_file=args.output,
        reference_dir=args.reference_dir,
        comparison_dir=args.comparison_dir,
    )
