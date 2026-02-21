from pathlib import Path
import json

def main() -> None:
    runs = Path("runs")
    if not runs.exists():
        print("No runs/ folder found.")
        return
    run_dirs = sorted([p for p in runs.iterdir() if p.is_dir()])
    if not run_dirs:
        print("No run subfolders under runs/.")
        return
    latest = run_dirs[-1]
    required = ["params.json","metrics.json","model.pkl","metadata.json"]
    missing = [f for f in required if not (latest/f).exists()]
    print(f"Latest run: {latest}")
    print("Missing:", missing if missing else "None")
    if not missing:
        meta = json.loads((latest/"metadata.json").read_text())
        print("metadata.json:", meta)

    report = Path("reports/drift_report.md")
    print("Drift report exists:", report.exists())
    if report.exists():
        print("Report last line:", report.read_text().splitlines()[-1])

if __name__ == "__main__":
    main()
