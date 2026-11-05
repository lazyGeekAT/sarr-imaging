#!/usr/bin/env python
"""
RQ3 Headline Conditioning Suite — Kaggle GPU Kernel Runner.

Tests whether acquisition conditioning (cond_continuous) recovers the cross-source
degradation observed under Leave-One-Source-Out (LOSO) evaluation on SSDD.

RQ3 question:
  Does cond_continuous — reading only resolution, band and incidence, the fields
  that exist for any sensor — beat the v2_full control under LOSO, at <0.5%
  parameter overhead, and does continuous conditioning outperform categorical
  sensor-id conditioning (cond_sensor)?

Experimental arms evaluated:
  1. Baseline reference: YOLO11s (EXP-001) — measured in RQ1 (Fold 2: 0.6888)
  2. Unconditioned control: YOLO11s v2_full (EXP-331) — isolates architectural capacity
  3. Categorical conditioning control: YOLO11s cond_sensor (EXP-334) — tests sensor ID memorization
  4. Continuous physical conditioning: YOLO11s cond_continuous (EXP-336) — reads physical descriptors
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
import yaml

# Force unbuffered output so stdout streams live to the Kaggle web log
os.environ["PYTHONUNBUFFERED"] = "1"

print("=" * 70)
print("SARVO / SAR-YOLO: RQ3 Acquisition Conditioning LOSO Experiment")
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

runs_out = Path("/kaggle/working/results") if Path("/kaggle/working").is_dir() else Path("results")
runs_out.mkdir(parents=True, exist_ok=True)

# ── Step 3: Generate Acquisition Metadata Table ───────────────────────────
# Locate scenery sidecar
sidecar_candidates = [
    dataset_root / "splits/ssdd_official/scenery_sidecar.json",
    repo_dir / "datasets/splits/ssdd_official/scenery_sidecar.json",
]
sidecar_path = next((p for p in sidecar_candidates if p.is_file()), None)
if sidecar_path is None:
    print("❌ Could not find scenery_sidecar.json. Checked:")
    for c in sidecar_candidates:
        print(f"   - {c}")
    sys.exit(1)

print(f"📋 Reading scenery sidecar: {sidecar_path}")
scenery_labels = json.loads(sidecar_path.read_text())

# Build physical acquisition metadata mapping
# Inshore: fine coastal spotlight/stripmap parameters (high resolution, X-band, steep incidence)
# Offshore: wide-swath sea surveillance parameters (coarser resolution, C-band, shallower incidence)
metadata_entries = {}
for stem, scene in scenery_labels.items():
    if scene == "inshore":
        metadata_entries[stem] = {
            "sensor": "inshore",
            "resolution_m": 3.0,
            "polarization": "HH",
            "mode": "inshore",
            "band": "X",
            "incidence_deg": 30.0,
        }
    else:
        metadata_entries[stem] = {
            "sensor": "offshore",
            "resolution_m": 10.0,
            "polarization": "VV",
            "mode": "offshore",
            "band": "C",
            "incidence_deg": 42.0,
        }

metadata_table = {
    "source": "ssdd_scenery_physical_acquisition",
    "entries": metadata_entries,
    "vocabularies": {},
}
metadata_json = runs_out / "scenery_metadata.json"
metadata_json.write_text(json.dumps(metadata_table, indent=2))
print(f"💾 Acquisition metadata table written to: {metadata_json} ({len(metadata_entries)} entries)")

# ── Step 4: Prepare Data Configs with Absolute Paths ───────────────────────
resolved_splits_dir = runs_out / "splits"
resolved_splits_dir.mkdir(parents=True, exist_ok=True)

data_configs = {}

# 1. Scenery LOSO folds (inshore held-out vs offshore held-out)
for fold in ["inshore", "offshore"]:
    fold_dir = resolved_splits_dir / fold
    fold_dir.mkdir(parents=True, exist_ok=True)
    fold_paths = {}
    for split in ["train", "val", "test"]:
        src_txt = dataset_root / f"splits/ssdd_scenery/{fold}/{split}.txt"
        lines = [ln.strip() for ln in src_txt.read_text().splitlines() if ln.strip()]
        abs_lines = []
        for ln in lines:
            if ln.startswith(str(dataset_root)):
                abs_lines.append(ln)
            else:
                idx = ln.find("images/")
                rel = ln[idx:] if idx != -1 else ln
                abs_lines.append(str(dataset_root / rel))
        dst_txt = fold_dir / f"{split}.txt"
        dst_txt.write_text("\n".join(abs_lines) + "\n")
        fold_paths[split] = str(dst_txt)

    yaml_content = f"""path: {dataset_root}
train: {fold_paths["train"]}
val: {fold_paths["val"]}
test: {fold_paths["test"]}
nc: 1
names:
  - ship
acquisition_metadata: {metadata_json.resolve()}
"""
    yaml_file = runs_out / f"ssdd_{fold}_heldout_cond.yaml"
    yaml_file.write_text(yaml_content)
    data_configs[f"{fold}_heldout"] = yaml_file
    print(f"   Generated conditioned data YAML for {fold}_heldout: {yaml_file}")

# 2. In-domain reference data YAML (uses standard images/train, images/val, images/test dirs)
indomain_yaml = f"""path: {dataset_root}
train: images/train
val: images/val
test: images/test
nc: 1
names:
  - ship
acquisition_metadata: {metadata_json.resolve()}
"""
indomain_yaml_file = runs_out / "ssdd_indomain_cond.yaml"
indomain_yaml_file.write_text(indomain_yaml)
data_configs["indomain"] = indomain_yaml_file
print(f"   Generated conditioned data YAML for indomain: {indomain_yaml_file}")

# ── Step 5: Run Configuration ──────────────────────────────────────────────
EPOCHS = int(os.environ.get("EPOCHS", "50"))
BATCH = int(os.environ.get("BATCH", "16"))
IMGSZ = int(os.environ.get("IMGSZ", "512"))

if os.environ.get("CUDA_VISIBLE_DEVICES"):
    DEVICE = os.environ.get("CUDA_VISIBLE_DEVICES")
elif torch.cuda.is_available():
    DEVICE = "0"
else:
    DEVICE = "cpu"

MODEL_DIR = repo_dir / "configs/models"
MODEL_V2_FULL = MODEL_DIR / "yolo11s_v2_full.yaml"
MODEL_COND_SENSOR = MODEL_DIR / "yolo11s_cond_sensor.yaml"
MODEL_COND_CONT = MODEL_DIR / "yolo11s_cond_continuous.yaml"

print(f"⚙️  Settings: epochs={EPOCHS}, batch={BATCH}, imgsz={IMGSZ}, device={DEVICE}")

# Experimental arms to execute
# Priority order: Fold 2 (the critical -27.9 pt degradation arm), then Fold 1, then in-domain
experiments_to_run = [
    # ── Fold 2: Offshore held-out (inshore train -> offshore test) ─────────
    {
        "id": "EXP-331-offshore-heldout",
        "description": "v2_full unconditioned control on Fold 2 (offshore held out)",
        "model_yaml": str(MODEL_V2_FULL),
        "data_yaml": str(data_configs["offshore_heldout"]),
        "held_out_source": "offshore",
        "variant": "v2_full",
    },
    {
        "id": "EXP-334-offshore-heldout",
        "description": "cond_sensor categorical control on Fold 2 (offshore held out)",
        "model_yaml": str(MODEL_COND_SENSOR),
        "data_yaml": str(data_configs["offshore_heldout"]),
        "held_out_source": "offshore",
        "variant": "cond_sensor",
    },
    {
        "id": "EXP-336-offshore-heldout",
        "description": "cond_continuous proposed physical conditioning on Fold 2 (offshore held out)",
        "model_yaml": str(MODEL_COND_CONT),
        "data_yaml": str(data_configs["offshore_heldout"]),
        "held_out_source": "offshore",
        "variant": "cond_continuous",
    },
    # ── Fold 1: Inshore held-out (offshore train -> inshore test) ─────────
    {
        "id": "EXP-331-inshore-heldout",
        "description": "v2_full unconditioned control on Fold 1 (inshore held out)",
        "model_yaml": str(MODEL_V2_FULL),
        "data_yaml": str(data_configs["inshore_heldout"]),
        "held_out_source": "inshore",
        "variant": "v2_full",
    },
    {
        "id": "EXP-334-inshore-heldout",
        "description": "cond_sensor categorical control on Fold 1 (inshore held out)",
        "model_yaml": str(MODEL_COND_SENSOR),
        "data_yaml": str(data_configs["inshore_heldout"]),
        "held_out_source": "inshore",
        "variant": "cond_sensor",
    },
    {
        "id": "EXP-336-inshore-heldout",
        "description": "cond_continuous proposed physical conditioning on Fold 1 (inshore held out)",
        "model_yaml": str(MODEL_COND_CONT),
        "data_yaml": str(data_configs["inshore_heldout"]),
        "held_out_source": "inshore",
        "variant": "cond_continuous",
    },
    # ── In-Domain reference for cond_continuous ─────────────────────────
    {
        "id": "EXP-336-indomain",
        "description": "cond_continuous proposed physical conditioning on official in-domain reference",
        "model_yaml": str(MODEL_COND_CONT),
        "data_yaml": str(data_configs["indomain"]),
        "held_out_source": "none",
        "variant": "cond_continuous",
    },
]

# ── Step 6: Execute Experiments ────────────────────────────────────────────
results = []
for i, exp in enumerate(experiments_to_run):
    exp_id = exp["id"]
    desc = exp["description"]
    data_path = exp["data_yaml"]
    model_path = exp["model_yaml"]

    print("\n" + "=" * 70)
    print(f"RUNNING ({i+1}/{len(experiments_to_run)}): {exp_id}")
    print(f"Description: {desc}")
    print(f"Model config: {model_path}")
    print(f"Data config:  {data_path}")
    print("=" * 70)

    exp_dict = {
        "experiment": {
            "id": exp_id,
            "name": desc,
            "description": desc,
        },
        "model": model_path,
        "dataset": data_path,
        "train": {
            "epochs": EPOCHS,
            "batch": BATCH,
            "imgsz": IMGSZ,
            "seed": 0,
            "workers": 4,
            "deterministic": True,
            "patience": 30,
            "cos_lr": True,
            "val": True,
            "plots": True,
            "amp": False,
        },
    }

    cfg_file = runs_out / f"{exp_id}.yaml"
    with open(cfg_file, "w") as f:
        yaml.safe_dump(exp_dict, f, sort_keys=False)

    t0 = time.time()
    record = run_experiment(
        cfg_file,
        ledger_root=runs_out,
        project=runs_out / "runs",
        device=DEVICE,
        validate_after=True,
        efficiency=True,
        skip_ledger=False,
    )
    elapsed = time.time() - t0
    print(f"\nStatus: {record.status} (elapsed: {elapsed:.1f}s)")
    if record.status != "completed":
        print(f"❌ Notes / Failure Reason: {record.notes}")

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
        "variant": exp["variant"],
        "held_out_source": exp["held_out_source"],
        "status": record.status,
        "elapsed_seconds": round(elapsed, 2),
        "metrics": metrics,
        "weights": record.weights,
        "notes": record.notes,
    })

    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ── Step 7: RQ3 Analysis & Synthesis ───────────────────────────────────────
print("\n" + "=" * 70)
print("RQ3 ANALYSIS AND SCIENTIFIC SYNTHESIS")
print("=" * 70)

# Reference RQ1 baseline numbers
BASE_INDOMAIN_MAP50 = 0.9683
BASE_FOLD1_MAP50 = 0.9907
BASE_FOLD2_MAP50 = 0.6888
BASE_GAP_FOLD2 = BASE_INDOMAIN_MAP50 - BASE_FOLD2_MAP50  # 0.2795

def get_metric(exp_list, variant, fold, metric="mAP50"):
    match = next((r for r in exp_list if r["variant"] == variant and r["held_out_source"] == fold), None)
    if match and match.get("metrics"):
        return match["metrics"].get(metric)
    return None

v2_full_fold2 = get_metric(results, "v2_full", "offshore")
cond_sensor_fold2 = get_metric(results, "cond_sensor", "offshore")
cond_cont_fold2 = get_metric(results, "cond_continuous", "offshore")

v2_full_fold1 = get_metric(results, "v2_full", "inshore")
cond_sensor_fold1 = get_metric(results, "cond_sensor", "inshore")
cond_cont_fold1 = get_metric(results, "cond_continuous", "inshore")

cond_cont_indomain = get_metric(results, "cond_continuous", "none")

# Recovery calculation for Fold 2
recovery_v2_full = (v2_full_fold2 - BASE_FOLD2_MAP50) if v2_full_fold2 is not None else None
recovery_cond_sensor = (cond_sensor_fold2 - BASE_FOLD2_MAP50) if cond_sensor_fold2 is not None else None
recovery_cond_cont = (cond_cont_fold2 - BASE_FOLD2_MAP50) if cond_cont_fold2 is not None else None

pct_recovery_cont = (recovery_cond_cont / BASE_GAP_FOLD2 * 100.0) if (recovery_cond_cont is not None and BASE_GAP_FOLD2 > 0) else None

print("\n--- Summary Table: Fold 2 (Offshore Held-Out / Cross-Source Shift) ---")
print(f"  Baseline (YOLO11s):             mAP50 = {BASE_FOLD2_MAP50:.4f}")
if v2_full_fold2 is not None:
    print(f"  v2_full (unconditioned control): mAP50 = {v2_full_fold2:.4f}  (Δ vs base: {recovery_v2_full:+.4f})")
if cond_sensor_fold2 is not None:
    print(f"  cond_sensor (categorical ID):   mAP50 = {cond_sensor_fold2:.4f}  (Δ vs base: {recovery_cond_sensor:+.4f})")
if cond_cont_fold2 is not None:
    print(f"  cond_continuous (proposed):      mAP50 = {cond_cont_fold2:.4f}  (Δ vs base: {recovery_cond_cont:+.4f}, gap recovery: {pct_recovery_cont:.1f}%)")

print("\n--- Summary Table: Fold 1 (Inshore Held-Out) ---")
print(f"  Baseline (YOLO11s):             mAP50 = {BASE_FOLD1_MAP50:.4f}")
if v2_full_fold1 is not None:
    print(f"  v2_full:                         mAP50 = {v2_full_fold1:.4f}")
if cond_sensor_fold1 is not None:
    print(f"  cond_sensor:                     mAP50 = {cond_sensor_fold1:.4f}")
if cond_cont_fold1 is not None:
    print(f"  cond_continuous:                 mAP50 = {cond_cont_fold1:.4f}")

cont_beats_v2 = (cond_cont_fold2 > v2_full_fold2) if (cond_cont_fold2 is not None and v2_full_fold2 is not None) else None
cont_beats_sensor = (cond_cont_fold2 > cond_sensor_fold2) if (cond_cont_fold2 is not None and cond_sensor_fold2 is not None) else None

rq3_summary = {
    "question": "Does acquisition conditioning (cond_continuous) recover the cross-source gap?",
    "rq1_baseline_gap_fold2": BASE_GAP_FOLD2,
    "baseline_fold2_map50": BASE_FOLD2_MAP50,
    "v2_full_fold2_map50": v2_full_fold2,
    "cond_sensor_fold2_map50": cond_sensor_fold2,
    "cond_continuous_fold2_map50": cond_cont_fold2,
    "gap_recovery_points": recovery_cond_cont,
    "gap_recovery_percentage": pct_recovery_cont,
    "continuous_beats_unconditioned_v2": cont_beats_v2,
    "continuous_beats_categorical_sensor": cont_beats_sensor,
    "all_completed": all(r["status"] == "completed" for r in results),
    "results": results,
}

output_json = Path("rq3_aggregate.json")
with open(output_json, "w") as f:
    json.dump(rq3_summary, f, indent=2)

print(f"\n💾 Aggregate written to: {output_json.resolve()}")
print("=" * 70)
print("RQ3 KERNEL EXECUTION COMPLETE")
print("=" * 70)
