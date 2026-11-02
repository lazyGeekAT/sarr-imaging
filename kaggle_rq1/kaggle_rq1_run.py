#!/usr/bin/env python
"""
RQ1 Baseline — Kaggle GPU Kernel Runner.

Runs EXP-001 (YOLO11s baseline) under leave-one-source-out (LOSO) scenery folds
(Inshore held-out vs. Offshore held-out) against the in-domain baseline on SSDD.

RQ1 question:
  Under a guarded leave-one-source-out split, does a stock detector's mAP drop
  measurably relative to in-domain, and does the drop concentrate in particular sources?
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

# Force unbuffered output so stdout streams live to the Kaggle web log
os.environ["PYTHONUNBUFFERED"] = "1"

print("=" * 70)
print("SARVO / SAR-YOLO: RQ1 Baseline LOSO Experiment")
print("=" * 70)
print(f"Python: {sys.version}")
print(f"Working Directory: {Path.cwd()}")

# ── Step 1: Ensure repository and package are installed ────────────────────
repo_dir = Path("/kaggle/working/sarr-imaging")
if not repo_dir.exists():
    local_repo = Path(__file__).resolve().parents[1]
    if (local_repo / "saryolo").is_dir():
        repo_dir = local_repo
        print(f"📦 Using local repository at: {repo_dir}")
    else:
        print("📥 Cloning sarr-imaging repository from GitHub...")
        subprocess.run(
            ["git", "clone", "https://github.com/lazyGeekAT/sarr-imaging.git", str(repo_dir)],
            check=True,
        )

# Install ultralytics and dependencies
print("📦 Installing ultralytics and dependencies...")
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "ultralytics>=8.3.0", "pycocotools>=2.0.7"],
    check=True,
)

# Ensure repo root is on sys.path and install package in editable mode
if str(repo_dir) not in sys.path:
    sys.path.insert(0, str(repo_dir))

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", str(repo_dir)], check=True)

import saryolo  # noqa: E402

print(f"✅ saryolo package imported successfully from {saryolo.__file__}")

from saryolo.training.runner import run_experiment  # noqa: E402

# ── Step 2: Locate SSDD Dataset ────────────────────────────────────────────
possible_roots = [
    Path("/kaggle/input/ssdd-sarvo"),
    Path("/kaggle/input/datasets/arindamtripathi/ssdd-sarvo"),
    repo_dir / "datasets/staging/ssdd-sarvo",
    repo_dir / "datasets/processed/ssdd",
]

dataset_root = None
for candidate in possible_roots:
    if candidate.is_dir() and (candidate / "images").is_dir():
        dataset_root = candidate
        break

if dataset_root is None:
    print("❌ Could not find SSDD dataset directory. Checked:")
    for c in possible_roots:
        print(f"   - {c}")
    sys.exit(1)

print(f"📁 Dataset root: {dataset_root}")

# Verify data YAML files
data_configs = {
    "indomain": dataset_root / "ssdd_official.yaml",
    "inshore_heldout": dataset_root / "ssdd_inshore_heldout.yaml",
    "offshore_heldout": dataset_root / "ssdd_offshore_heldout.yaml",
}

for name, path in data_configs.items():
    if not path.is_file():
        print(f"❌ Missing dataset YAML for {name}: {path}")
        sys.exit(1)
    print(f"   Found config for {name}: {path}")

# ── Step 3: Run Configuration ──────────────────────────────────────────────
EPOCHS = int(os.environ.get("EPOCHS", "50"))
BATCH = int(os.environ.get("BATCH", "16"))
IMGSZ = int(os.environ.get("IMGSZ", "512"))

if os.environ.get("CUDA_VISIBLE_DEVICES"):
    DEVICE = os.environ.get("CUDA_VISIBLE_DEVICES")
elif torch.cuda.is_available():
    DEVICE = "0"
else:
    DEVICE = "cpu"

EXP_CONFIG = repo_dir / "configs/exp/EXP-001_baseline.yaml"

print(f"⚙️  Settings: epochs={EPOCHS}, batch={BATCH}, imgsz={IMGSZ}, device={DEVICE}")

runs_out = Path("/kaggle/working/results") if Path("/kaggle/working").is_dir() else Path("results")
runs_out.mkdir(parents=True, exist_ok=True)

experiments_to_run = [
    {
        "id": "EXP-001-indomain",
        "description": "SSDD official in-domain baseline (713 train / 178 val / 232 test)",
        "data_yaml": str(data_configs["indomain"]),
        "held_out_source": "none",
    },
    {
        "id": "EXP-001-inshore-heldout",
        "description": "SSDD scenery Fold 1: inshore held out (720 train / 180 val / 223 test)",
        "data_yaml": str(data_configs["inshore_heldout"]),
        "held_out_source": "inshore",
    },
    {
        "id": "EXP-001-offshore-heldout",
        "description": "SSDD scenery Fold 2: offshore held out (178 train / 45 val / 900 test)",
        "data_yaml": str(data_configs["offshore_heldout"]),
        "held_out_source": "offshore",
    },
]

# ── Step 4: Execute Experiments ────────────────────────────────────────────
results = []
for i, exp in enumerate(experiments_to_run):
    exp_id = exp["id"]
    desc = exp["description"]
    data_path = exp["data_yaml"]
    print("\n" + "=" * 70)
    print(f"RUNNING ({i+1}/{len(experiments_to_run)}): {exp_id}")
    print(f"Description: {desc}")
    print(f"Data config: {data_path}")
    print("=" * 70)

    overrides = {
        "data": exp["data_yaml"],
        "epochs": EPOCHS,
        "batch": BATCH,
        "imgsz": IMGSZ,
        "name": exp_id,
        "device": DEVICE,
        "project": str(runs_out / "runs"),
        "seed": 0,
        "deterministic": True,
    }

    t0 = time.time()
    try:
        record = run_experiment(
            EXP_CONFIG,
            ledger_root=runs_out,
            project=runs_out / "runs",
            device=DEVICE,
            extra_overrides=overrides,
            validate_after=True,
            efficiency=True,
            skip_ledger=False,
        )
        elapsed = time.time() - t0
        print(f"\n✅ {exp_id} finished in {elapsed:.1f}s with status: {record.status}")
        metrics = record.metrics or {}
        map50_str = str(metrics.get("mAP50", "N/A"))
        map_str = str(metrics.get("mAP50_95", "N/A"))
        prec_str = str(metrics.get("precision", "N/A"))
        rec_str = str(metrics.get("recall", "N/A"))
        print(f"   mAP50:    {map50_str}")
        print(f"   mAP50-95: {map_str}")
        print(f"   Precision:{prec_str}")
        print(f"   Recall:   {rec_str}")
        results.append({
            "id": exp_id,
            "held_out_source": exp["held_out_source"],
            "status": record.status,
            "elapsed_seconds": round(elapsed, 2),
            "metrics": metrics,
        })
    except Exception as exc:
        elapsed = time.time() - t0
        print(f"\n❌ {exp_id} failed after {elapsed:.1f}s: {exc}")
        import traceback
        traceback.print_exc()
        results.append({
            "id": exp_id,
            "held_out_source": exp["held_out_source"],
            "status": "failed",
            "error": str(exc),
            "elapsed_seconds": round(elapsed, 2),
            "metrics": {},
        })

# ── Step 5: RQ1 Synthesis & Aggregate Output ───────────────────────────────
print("\n" + "=" * 70)
print("RQ1 ANALYSIS AND SYNTHESIS")
print("=" * 70)

indomain_res = next((r for r in results if r["held_out_source"] == "none"), None)
inshore_res = next((r for r in results if r["held_out_source"] == "inshore"), None)
offshore_res = next((r for r in results if r["held_out_source"] == "offshore"), None)

map50_indomain = (indomain_res.get("metrics") or {}).get("mAP50") if indomain_res else None
map50_inshore = (inshore_res.get("metrics") or {}).get("mAP50") if inshore_res else None
map50_offshore = (offshore_res.get("metrics") or {}).get("mAP50") if offshore_res else None

drop_inshore = (map50_indomain - map50_inshore) if (map50_indomain is not None and map50_inshore is not None) else None
drop_offshore = (map50_indomain - map50_offshore) if (map50_indomain is not None and map50_offshore is not None) else None

degrades = None
if drop_inshore is not None and drop_offshore is not None:
    max_drop = max(drop_inshore, drop_offshore)
    degrades = max_drop > 0.01

rq1_answer = {
    "question": "Does the baseline actually degrade across acquisition sources?",
    "dataset": "SSDD",
    "architecture": "YOLO11s (EXP-001 baseline)",
    "epochs": EPOCHS,
    "imgsz": IMGSZ,
    "in_domain_mAP50": map50_indomain,
    "inshore_heldout_mAP50": map50_inshore,
    "offshore_heldout_mAP50": map50_offshore,
    "drop_inshore": drop_inshore,
    "drop_offshore": drop_offshore,
    "baseline_degrades_measurably": degrades,
    "all_completed": all(r["status"] == "completed" for r in results),
    "results": results,
}

print(f"In-domain mAP50:           {map50_indomain}")
print(f"Inshore held-out mAP50:    {map50_inshore} (drop = {drop_inshore})")
print(f"Offshore held-out mAP50:   {map50_offshore} (drop = {drop_offshore})")
print(f"Baseline degrades across sources: {degrades}")

output_json = Path("rq1_aggregate.json")
with open(output_json, "w") as f:
    json.dump(rq1_answer, f, indent=2)

print(f"\n💾 Aggregate written to: {output_json.resolve()}")
print("=" * 70)
print("RQ1 KERNEL EXECUTION COMPLETE")
print("=" * 70)
