from pathlib import Path
import json
from drift import compute_drift


def write_report(drift_rows, out_path: Path, drift_trigger_count: int) -> str:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("# Drift Report (Checkpoint 5)")
    lines.append("")
    lines.append("| feature | drift_score | flag |")
    lines.append("|---|---:|---|")

    drift_count = 0
    for r in drift_rows:
        lines.append(f"| {r['feature']} | {r['score']:.3f} | {r['flag']} |")
        if r["flag"] == "DRIFT":
            drift_count += 1

    lines.append("")
    recommendation = "INVESTIGATE/ROLLBACK" if drift_count >= drift_trigger_count else "OK_TO_PROCEED"
    lines.append(f"Recommendation: {recommendation}")

    out_path.write_text("\n".join(lines))
    return recommendation


def main() -> None:
    cfg = json.loads(Path("configs/baseline.json").read_text())
    threshold = float(cfg["monitor"]["threshold"])
    drift_trigger_count = int(cfg["monitor"]["drift_feature_count_trigger"])

    drift_rows = compute_drift("data/transactions_train.csv", "data/transactions_today.csv", threshold=threshold)
    rec = write_report(drift_rows, Path("reports/drift_report.md"), drift_trigger_count=drift_trigger_count)
    print("Wrote reports/drift_report.md")
    print(f"Recommendation: {rec}")


if __name__ == "__main__":
    main()
