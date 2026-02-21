import json
import pickle
import subprocess
import hashlib
import datetime
from pathlib import Path
from typing import Any, Dict


def sha256_file(path: str) -> str:
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def get_git_commit() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"])
        return out.decode().strip()
    except Exception:
        return "UNKNOWN"


def log_run(run_id: str, params: Dict[str, Any], metrics: Dict[str, Any], model: Any, train_csv_path: str) -> str:
    run_dir = Path("runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "params.json").write_text(json.dumps(params, indent=2))
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    with open(run_dir / "model.pkl", "wb") as f:
        pickle.dump(model, f)

    # FILL_THIS (1/3): timestamp string, use datetime.datetime.now().isoformat(timespec="seconds")
    timestamp = "FILL_THIS"

    # FILL_THIS (2/3): git commit hash, call get_git_commit()
    git_commit = "FILL_THIS"

    # FILL_THIS (3/3): dataset fingerprint, call sha256_file(train_csv_path)
    dataset_sha256 = "FILL_THIS"

    metadata = {"timestamp": timestamp, "git_commit": git_commit, "dataset_sha256": dataset_sha256}
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    return str(run_dir)
