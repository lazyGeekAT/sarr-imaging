# SAR-YOLO: Project Overview

## Project Purpose
SAR-YOLO is a research pipeline for SAR (Synthetic Aperture Radar) object detection, built on the YOLO11 architecture. Unlike RGB object detection, SAR imagery is corrupted by multiplicative speckle, targets are a handful of pixels across, and the clutter background competes for the same gradient signal as the object. SAR-YOLO encodes SAR formation properties directly into its architecture through a series of physically-informed modules.

## Problem Being Solved
SAR object detection differs fundamentally from RGB detection:
- **Multiplicative speckle** noise rather than additive Gaussian noise
- **Small targets** (a few pixels) against cluttered backgrounds
- **Coherent measurement physics** that violate RGB-derived assumptions
- Targets compete with clutter for the same gradient signal

## Main Features
- **110 architecture variants** wired and tested, every one builds and runs a forward pass
- **134 configured experiments** in the master workflow
- **984 passing tests** — no dataset download and no GPU needed
- **Efficiency frontier (SARVO-Lite)** that removes the two dominant compute slots while retaining all physical priors
- **Exact identity at initialisation**: every custom module is an exact identity at init, so a freshly built SAR-YOLO predicts identically to its stock YOLO11 counterpart
- **No fabricated numbers**: the ledger is append-only; unmeasured cells render as TBD
- **Controls are structural, not rhetorical**: every claim has a capacity-matched control built from the same graph

## Technology Stack
- **Primary language**: Python 3.10+
- **Deep learning framework**: PyTorch >= 2.2.0
- **Model backbone**: Ultralytics YOLO11
- **Key dependencies**: torch, numpy, opencv-python, PyYAML, pandas, matplotlib, tqdm, thop, seaborn, pycocotools
- **CLI framework**: argparse-based saryolo command with 19 subcommands
- **Demo**: Streamlit web application (app.py)
- **Testing**: pytest >= 8.0.0 with 984 checks

## High-Level Architecture
The project follows a **layered architecture** with these major subsystems:

1. **Input Adapter (Module A)**: SAR Input Adapter operating on the image before the first convolution
2. **Feature Enhancement (Component 1)**: SAR Feature Enhancement (SFE) — local contrast normalisation
3. **Speckle Module (Component 2)**: Speckle-Aware Feature Module (SFM) — separates target from speckle/clutter
4. **Attention (Component 3)**: SAR-Adaptive Attention (SAA) — channel + spatial + local-contrast fusion
5. **Multi-Scale Fusion (Component 4)**: Adaptive Multi-Scale Fusion (AMF) — per-scale adaptive reweighting
6. **P2 Detection Level**: Small-object detection head
7. **Target Prior (Component 8)**: Target Prior Modulation (TPM) — signed content-adaptive prior
8. **Spatial-Frequency (Component 9)**: Spatial-Frequency Representation (SFR) — radial FFT branch
9. **Context Aggregation (Component 10)**: Context Aggregation (CAG) — multi-extent + regional context
10. **Refinement (Component 11)**: Target-Aware Deformable Refinement (TADR) — per-sample offset sampling
11. **SSAC (Scatter-Selective Adaptive Computation)**: Per-region allocation of expensive computation
12. **SAR-Aware Loss**: Custom loss function with separation, small-target, and smooth terms

**Key design guarantees**:
- Every custom module is an exact identity at initialisation (alpha = 0)
- Channel preservation: modules preserve channel count so parse_model can wire them
- Structural controls isolate mechanism from capacity: every claim has a parameter-identical control

## Major Subsystems
- **Architecture builder** (`saryolo/nn/arch.py`): Symbolic YAML generator for 110+ variants
- **Neural modules** (`saryolo/nn/modules/`): 12+ SAR-specific modules with mode ablations
- **Training pipeline** (`saryolo/training/`): SAR-aware loss, conditioning, runner
- **Evaluation** (`saryolo/evaluation/`): metrics, efficiency, robustness, gain diagnostics
- **CLI** (`saryolo/cli.py`): 19 subcommands (train, eval, efficiency, robustness, etc.)
- **Streamlit demo** (`app.py`): Web interface for inference
- **Paper/results** (`paper/`, `docs/`): Generated tables, claim-to-evidence audit

## Important Entry Points
- `python -m saryolo train --exp <config>` — train a model
- `python -m saryolo eval --weights <ckpt> --data <data.yaml>` — evaluate mAP
- `python -m saryolo efficiency --weights <ckpt> --data <data.yaml>` — profile FLOPs/params
- `python -m saryolo efficiency --ssac-execution sparse --ssac-keep ...` — SSAC sparse/sweep
- `python -m saryolo predict --weights <ckpt> --source <image>` — single-image inference
- `streamlit run app.py` — web demo
- `python -m saryolo arch --variant <name> --out <dir>` — generate YAML variant
- `python -m saryolo bench --variants baseline_s full_s --models <dir>` — architecture benchmark

## Key Architectural Observations
- **110 wired architectures**: every variant builds and runs a forward pass; baseline equals stock YOLO11 exactly
- **Parameter counts**: YOLO11n = 2,624,080; YOLO11s = 9,458,752
- **Efficiency frontier**: SARVO-Lite (s) = 11.016 M params, 32.55 GFLOPs at scale s (42% reduction from full)
- **Modules are zero-initialised**: every residual gate starts at 0, so a freshly built model is numerically identical to baseline
- **SAR-native statistics**: CFAR log-ratio + local coefficient of variation, SFE local contrast norm, SFM speckle/clutter separation
- **Acquisition conditioning**: Per-sample metadata (sensor, resolution, gain, mode) modulates features
- **P2 level**: Optional small-object head; when enabled must include P3, P4, P5

## Development and Execution Commands
```bash
# Install
python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

# Run tests
pytest tests/ -q  # 984 tests

# Generate YAML variant
python -m saryolo arch --variant full_s --nc 1 --out configs/models/

# Train
python -m saryolo train --exp configs/exp/REAL-001_hrsid_baseline.yaml

# Evaluate
python -m saryolo eval --weights results/runs/REAL-001/weights/best.pt --data configs/datasets/hrsid_real.yaml --imgsz 320

# Efficiency profiling
python -m saryolo efficiency --weights results/runs/REAL-004/weights/best.pt --imgsz 320 --real --data configs/datasets/hrsid_real.yaml --batch 8 --runs 3

# SSAC sparse execution sweep
python -m saryolo efficiency --weights results/runs/SSAC-001/weights/best.pt --imgsz 320 --real --data configs/datasets/hrsid_real.yaml --batch 4 --runs 5 --ssac-execution sparse --ssac-keep 1.0 --ssac-keep 0.5 --ssac-keep 0.25 --ssac-keep 0.1

# Predict
python -m saryolo predict --weights weights/sarvo_ssac001.pt --source test.jpg --imgsz 640

# Streamlit demo
streamlit run app.py

# Generate paper tables
python -m saryolo assets --require-complete  # fails if any cell is TBD
```

## Known Limitations
- **One dataset (HRSID), one subset (200/60/60), one machine, no GPU**
- **Primary comparison is three seeds** — a noise check, not validation
- **No cross-sensor or cross-resolution result** (LOSO machinery untested on real data)
- **Ablation ladder, tuned LoRA sweep and full-release numbers remain TBD**
- **Front end costs ~3× CPU latency at this resolution**
- **Two synthetic acquisition-shift probes are null results** (kept for completeness)
- **No arm was trained at 640px for the efficiency claim** — scale-bound result only

## Citations
```bibtex
@software{bose_sarvo,
  title  = {SARVO: SAR Acquisition-Robust Visual Optimization},
  author = {Bose, Arghya},
  year   = {2026},
  url    = {https://github.com/officialarghya29/sarr-imaging}
}
```