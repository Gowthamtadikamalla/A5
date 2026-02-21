import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple

from mlops_utils import log_run


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def load_csv(path: Path) -> List[Dict[str, str]]:
    import csv
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def train_valid_split(rows: List[Dict[str, str]], valid_frac: float, seed: int) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    rng = random.Random(seed)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    cut = int(len(rows) * (1.0 - valid_frac))
    train = [rows[i] for i in idx[:cut]]
    valid = [rows[i] for i in idx[cut:]]
    return train, valid


def predict_prob(model: Dict, x: Dict[str, float], features: List[str]) -> float:
    score = float(model["bias"])
    weights = model["weights"]
    for f in features:
        score += float(weights[f]) * float(x[f])
    return sigmoid(score)


def eval_metrics(model: Dict, rows: List[Dict[str, str]], features: List[str], label_col: str) -> Dict[str, float]:
    tp = fp = tn = fn = 0
    thr = float(model["threshold"])
    for r in rows:
        x = {f: float(r[f]) for f in features}
        y = int(float(r[label_col]))
        p = predict_prob(model, x, features)
        yhat = 1 if p >= thr else 0
        if yhat == 1 and y == 1:
            tp += 1
        elif yhat == 1 and y == 0:
            fp += 1
        elif yhat == 0 and y == 0:
            tn += 1
        else:
            fn += 1

    total = tp + fp + tn + fn
    acc = (tp + tn) / total if total else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) else 0.0
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "n_valid": total}


def generate_run_id() -> str:
    import datetime
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"{random.randint(0, 9999):04d}"
    return f"{ts}_{suffix}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text())

    feats = cfg["features"]
    label_col = cfg["label_col"]
    seed = int(cfg.get("seed", 123))
    valid_frac = float(cfg.get("valid_frac", 0.2))

    train_csv = Path(cfg["train_csv"])
    rows = load_csv(train_csv)
    _, valid_rows = train_valid_split(rows, valid_frac=valid_frac, seed=seed)

    model = cfg["model"]
    metrics = eval_metrics(model, valid_rows, feats, label_col)

    print("Validation metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    run_id = generate_run_id()
    run_dir = log_run(run_id, cfg, metrics, model, str(train_csv))
    print(f"Saved run to {run_dir}")


if __name__ == "__main__":
    main()
