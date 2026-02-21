import csv
from pathlib import Path
from typing import Dict, List, Tuple


def _numeric_columns(header: List[str]) -> List[str]:
    return [c for c in header if c != "label"]


def _mean_std(values: List[float]) -> Tuple[float, float]:
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    mean = sum(values) / n
    var = sum((x - mean) ** 2 for x in values) / max(1, (n - 1))
    std = var ** 0.5
    return mean, std


def compute_drift(train_csv: str, today_csv: str, threshold: float = 0.75) -> List[Dict]:
    train_rows = list(csv.DictReader(Path(train_csv).open("r", newline="")))
    today_rows = list(csv.DictReader(Path(today_csv).open("r", newline="")))

    feats = _numeric_columns(list(train_rows[0].keys()))

    out: List[Dict] = []
    for feat in feats:
        train_vals = [float(r[feat]) for r in train_rows]
        today_vals = [float(r[feat]) for r in today_rows]
        mean_train, std_train = _mean_std(train_vals)
        mean_today, std_today = _mean_std(today_vals)

        # FILL_THIS (1/2): score = abs(mean_today - mean_train) / (std_train + 1e-6)
        score = 0.0

        # FILL_THIS (2/2): flag = "DRIFT" if score > threshold else "OK"
        flag = "OK"

        out.append({
            "feature": feat,
            "score": float(score),
            "flag": flag,
            "mean_train": mean_train,
            "std_train": std_train,
            "mean_today": mean_today,
            "std_today": std_today,
        })

    out.sort(key=lambda d: d["score"], reverse=True)
    return out
