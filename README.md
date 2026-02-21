# Checkpoint 5: Minimal MLOps Loop (40 minutes)

## Goal
Practice a minimal MLOps loop using only Python and Git:

- Train a baseline model
- Log a run record with lineage metadata (Git commit + dataset fingerprint)
- Generate a drift report on today's data
- Make an operational decision (`OK_TO_PROCEED` vs `INVESTIGATE/ROLLBACK`)

---

## Setup
1) Clone this repo and enter it.

2) Create a branch:
```bash
git checkout -b cp5-<yourUID>
```

---

## Part A: Run logging + lineage (about 20 minutes)

### 1) Run baseline training
```bash
python train.py --config configs/baseline.json
```

### 2) Fill the lineage fields
Open `mlops_utils.py` and fill the lines marked `FILL_THIS`.

### 3) Commit your changes
```bash
git add mlops_utils.py
git commit -m "CP5: run logging lineage complete"
```

### 4) Re-run training (captures your new commit hash)
```bash
python train.py --config configs/baseline.json
```

### 5) Verify lineage and save proof (required)
1) Open: `runs/<run_id>/metadata.json`

2) Run:
```bash
git log -1 --oneline
```

3) Confirm the commit hash in `metadata.json` matches the hash shown by `git log -1 --oneline`.

4) Take a screenshot showing both:
- `runs/<run_id>/metadata.json` open, and
- the `git log -1 --oneline` output in your terminal.

5) Save the screenshot inside your run folder:
- `runs/<run_id>/`
Example filename: `runs/<run_id>/git_commit_proof.png`

---

## Part B: Drift report + decision (about 15 minutes)

### 1) Run monitoring once
```bash
python monitor.py
```

### 2) Fill the drift logic
Open `drift.py` and fill the lines marked `FILL_THIS`.

### 3) Re-run monitoring
```bash
python monitor.py
```

### 4) Confirm the drift report is complete
Open: `reports/drift_report.md`

It must include:
- Per-feature drift scores
- A final `Recommendation:` line

### 5) Create `DECISION.txt`
Create `DECISION.txt` with exactly one line matching the recommendation:
- `OK_TO_PROCEED` or
- `INVESTIGATE/ROLLBACK`

---

## Submit (Canvas zip)
Zip and upload the following:

- `runs/<run_id>/`
  - Must include: `params.json`, `metrics.json`, `model.pkl`, `metadata.json`, and your screenshot proof file
- `reports/drift_report.md`
- `DECISION.txt`
