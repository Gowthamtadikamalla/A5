# Checkpoint 5: Minimal MLOps Loop (40 minutes)

## Goal
Practice a minimal MLOps loop using only Python and Git:
- Train a baseline model
- Log a run record with lineage metadata (git commit + dataset fingerprint)
- Generate a drift report on today's data
- Make an operational decision (OK_TO_PROCEED vs INVESTIGATE/ROLLBACK)

## Setup
1) Clone this repo and enter it
2) Create a branch:
   git checkout -b cp5-<yourUID>

## Part A: Run logging + lineage (about 20 minutes)
1) Run baseline:
   python train.py --config configs/baseline.json

2) In `mlops_utils.py`, fill the 3 lines marked `FILL_THIS`.
   Re-run training and confirm metadata.json has real values.

3) Commit:
   git add .
   git commit -m "CP5: run logging lineage complete"

## Part B: Drift report + decision
4) Run once:
   python monitor.py

5) In `drift.py`, fill the lines marked `FILL_THIS`.
   Re-run:
   python monitor.py

## Optional: Release tag
git tag model-v0.1

## Submit (Canvas zip)
Zip and upload:
- runs/<run_id>/
- reports/drift_report.md
- DECISION.txt

Optional: include `GIT.txt` with:
- git log -1 --oneline
- git branch --show-current
