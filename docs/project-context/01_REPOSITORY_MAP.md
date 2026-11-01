# SAR-YOLO: Repository Map

## Directory Tree
```saryolo/
  __init__.py          # Package init, VERSION, module exports
  __main__.py          # Entry point for `python -m saryolo`
  cli.py               # 19 subcommands behind one entry point
  inference.py         # Inference pipeline (load, predict, annotate)
  augmentation/        # SAR-augmentation model (shape-preserving degradation)
  data/                # Dataset registry, YOLO/VOC conversion, splits, metadata
  evaluation/          # metrics, efficiency, robustness, gain, cross-dataset
  nn/                  # Neural network modules
    __init__.py
    arch.py            # Symbolic architecture builder for 110+ variants
    losses.py          # SAR-aware criterion build + consistency term
    model.py           # SARYOLODetectionModel, SARYOLORTDetectionModel
    modules/           # 12+ SAR-specific neural modules
      __init__.py
      input_adapter.py # Module A — SAR Input Adapter (SIA)
      speckle.py       # Component 2 — Speckle-Aware Feature Module (SFM)
      enhancement.py   # Component 1 — SAR Feature Enhancement (SFE)
      attention.py     # Component 3 — SAR-Adaptive Attention (SAA)
      fusion.py        # Component 4 — Adaptive Multi-Scale Fusion (AMF)
      cfar.py          # Ratio-space CFAR front end (master Phase 5)
      ssac.py          # Scatter-Selective Adaptive Computation (SSAC)
      frequency.py     # Component 9 — Spatial-Frequency Representation (SFR)
      context.py       # Component 10 — Context Aggregation (CAG)
      target_prior.py  # Component 8 — Target Prior Modulation (TPM)
      refinement.py    # Component 11 — Target-Aware Deformable Refinement (TADR)
    _common.py         # Shared blocks: ConvBNAct, DWConvBNAct, LocalStats, ZeroGate, etc.
  training/            # Trainer, runner, LoRA, hard-example mining, config
  data/                # dataset registry, YOLO/VOC conversion, splits, metadata
  paper/               # table generators (unmeasured cell = TBD)
  tracking/            # append-only experiment ledger
  configs/             # datasets/, models/ (110 generated), exp/ (EXP-001..019 + ablations + frontier)
  docs/                # physics-to-architecture proposals, method drafts, claim audit
  weights/             # the one committed checkpoint (sarvo_ssac001.pt) with provenance/checksum
  tests/               # 984 checks; the instrument is tested as hard as the model
  results/             # experiment results (out of git)
  reports/             # reproduction status, defect log
  app.py               # Streamlit demo — draws the pipeline's detections
  pyproject.toml       # Project config, dependencies, script saryolo = saryolo.cli:main
  requirements.txt     # Pinned dependencies
  README.md            # Project overview, status badges, quickstart
  .gitignore           # Git ignore rules
  runtime.txt          # Runtime specification
```
## Directory Responsibilities

### saryolo/
- **Package root**: CLI entry point, inference pipeline, demo
- **nn/**: Neural network architecture, modules, loss functions
- **training/**: Training pipeline, SAR-aware loss, conditioning, runner
- **data/**: Dataset handling, metadata, splits, conversion, validation
- **evaluation/**: Metrics, efficiency profiling, robustness sweeps, gain diagnostics
- **augmentation/**: SAR-augmentation model (shape-preserving degradation)
- **tracking/**: Experiment ledger management

### configs/
- **models/**: 110 generated model YAML variants (VARIANTS dict)
- **exp/**: Experiment configs (EXP-001..019 + ablations + frontier)
- **datasets/**: Dataset YAML configs

### paper/
- **Table generators**: build_ablation, build_efficiency_frontier, build_real_data_pilot, etc.
- **Results and claims**: RESULTS.md, claim_evidence_audit.md

### tests/
- **984 checks** across multiple categories:
  - Baseline matching (baseline = stock YOLO11)
  - Every variant builds and forwards
  - Module identity at init
  - No module frozen at init
  - All gates leave zero during real training
  - Conditioning per-sample not per-batch
  - VOCAB mismatch raises instead of clamping
  - AP implementation reproduces pycocotools
  - Hard image from another split is error not silent no-op
  - Augmentation cannot invalidate labels
  - No table emits unmeasured number
  - README cost table matches measured models
- **Test categories**: unit, integration, smoke, slow, gpu

### weights/
- **Committed checkpoint**: sarvo_ssac001.pt (the only weight in git)
- **Provenance and checksum**: weights/README.md
- **Resolution order**: docs/DEPLOYMENT.md

### docs/
- **Architecture proposals**: docs/architecture_proposals.md
- **Method brief**: docs/METHOD.md
- **Claim-to-evidence audit**: docs/claim_evidence_audit.md
- **DEPLOYMENT docs**: docs/DEPLOYMENT.md
- **Generated figures and tables**: docs/assets/
- **Test composition**: docs/assets/tests.svg

### reports/
- **Release readiness**: reports/release_readiness.md
- **Defect log**: reports/defect_log.md

### app.py
- **Streamlit web demo**: SAR ship detection interface
- **Loads one committed checkpoint**: weights/sarvo_ssac001.pt
- **Never substitutes a model**: if checkpoint missing, says so and disables detection

### cli.py
- **19 subcommands**: datasets, synth-data, check-data, stats, metadata, probe, arch, train, eval, predict, robustness, gain, efficiency, cross-dataset, mine-hard, augment, loso, bench, assets, ledger
- **Single entry point** for the whole pipeline

## File Inventory Summary

| Category | Count |
|---|---|
| First-party source files | ~80 |
| Configuration files | ~20 |
| Test files | 984+ (pytest checks, many are structural/import checks) |
| Documentation files | ~15+ |
| Asset/data files | ~10+ |
| Generated/vendor files | results/, runs/, wandb/, mlruns/ |

## Module Boundaries

### Neural Module Interface Contract
Every SAR-YOLO neck module must satisfy three constraints to be wireable through `ultralytics.nn.tasks.parse_model`:

1. **Single input, single output**: `parse_model` resolves output channels with `c2 = ch[f]` for unknown modules, which breaks if `f` is a list.
2. **Channel preserving**: Must return the same number of channels it received. Projection back to a new width is left to the stock `Conv`/`C3k2` that follows.
3. **Identity at initialisation**: Every residual gate starts at `alpha = 0`, so a freshly built SAR-YOLO is numerically identical to its YOLO baseline.

### Channel Arguments
`parse_model` substitutes any string argument that names a local variable, and `ch` (the running channel list) is one of them. Passing `"ch"` therefore hands the module the channel list as it stands at that layer, letting us resolve width-scaled channels exactly at build time for any `scale`/`width_multiple`.

### Module Registration
`saryolo.nn.register` makes modules resolvable from any Ultralytics YAML, including RT-DETR's, by registering custom layer names.

## Important File Relationships

### YAML Generation Pipeline
1. `arch.py` `VARIANTS` dict defines 110+ ModelSpec objects
2. `variant_filename(spec)` generates canonical filename: `yolo11<scale>_<name>.yaml`
3. `build_yaml_dict(spec)` generates the dict (backbone + head rows + optional sar_loss)
4. `build_yaml_text(spec)` serialises to YAML with provenance header
5. CLI `saryolo arch --variant <name>` writes YAML files to `configs/models/`

### Model → Forward Pass
1. Model YAML is loaded by Ultralytics `parse_model`
2. Backbone rows build feature maps from input
3. Head rows apply top-down pathway, per-level research slots (prior, attention, context, refinement, SSAC)
4. Detect head reads the finest-level output
5. SAR-aware criterion computes loss with separation, small-target, smooth terms

### Experiment Ledger
- `tracking/ledger.py` stores every run: experiment_id, model, dataset, status, metrics, seeds, notes
- `configs/exp/` YAML configs reference VARIANT names (EXP-001 = baseline, EXP-007 = full, etc.)
- `python -m saryolo ledger` displays all records

## Generated/Vendored Exclusions
- `results/` — experiment runs, out of git
- `runs/` — training runs, out of git
- `wandb/` — experiment tracking, out of git
- `mlruns/` — MLOps tracking, out of git
- `dataset_statistics/` — generated stats, out of git
- `validation_reports/` — generated reports, out of git
- `*.pt`, `*.pth` — model weights, out of git (except sarvo_ssac001.pt)
- `configs/datasets/generated/` — machine-specific paths, excluded
- `.venv/`, `env/`, `env/` — virtual environments, excluded
- `__pycache__/`, `*.pyc` — Python cache, excluded

## Cross-References

### Module ↔ Module Dependencies
- **CFAR front end** (cfar.py) → provides statistic stack used by SFE, SFM, SSAC assessment
- **SFE** (enhancement.py) → uses LocalStats, produces enhanced features consumed by downstream modules
- **SFM** (speckle.py) → uses LocalStats, produces noise/clutter maps consumed by downstream
- **SAA** (attention.py) → uses LocalStats, produces attention map fused by AMF
- **AMF** (fusion.py) → takes concatenated multi-scale features, produces reweighted output
- **TPM** (target_prior.py) → produces signed prior map, can condition spectral filter (Module G)
- **SFR** (frequency.py) → radial FFT branch, can be conditioned by TPM
- **CAG** (context.py) → gathers multi-extent + regional context, added after attention/refinement
- **TADR** (refinement.py) → per-sample offset sampling, inserted per detection level pre-head
- **SSAC** (ssac.py) → per-region allocation, inserted per detection level pre-head (after prior/attention/context)
- **SIA** (input_adapter.py) → operates on image before first convolution, output feeds backbone

### Key Cross-Cutting Concerns
- **Zero-initialisation**: every module's gate starts at 0 → exact identity at init
- **Channel preservation**: every module preserves channel count
- **Acquisition metadata**: per-sample conditioning (sensor, resolution, gain, mode)
- **SAR-native statistics**: CFAR log-ratio, local coefficient of variation, local contrast
- **SAR-aware loss**: separation, small-target, smooth terms

## Module Count
- **12+ neural modules** in `saryolo/nn/modules/`, each with multiple mode ablations
- **110 architecture variants** in `VARIANTS` dict
- **8 experiment levels** (EXP-001 through EXP-019 + ablations + frontier arms)
- **Efficiency frontier arms**: SARVO-Lite (s), SARVO-Lite (n), SARVO-Lite (m)