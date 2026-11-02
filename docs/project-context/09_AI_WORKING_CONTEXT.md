# SAR-YOLO: AI Working Context

This is the most important document for future AI coding agents. It provides a concise but technically rich operational context for working on this repository.

## Project Identity

**What it does**: SAR-YOLO is a research pipeline for **SAR (Synthetic Aperture Radar) object detection**, built on the YOLO11 architecture. Unlike RGB object detection, SAR imagery is corrupted by multiplicative speckle noise, targets are a handful of pixels across, and the clutter background competes for the same gradient signal as the object.

**Problem being solved**: Standard YOLO11 architectures are designed for RGB imagery. SAR data has fundamentally different characteristics: multiplicative speckle noise (signal-dependent), tiny targets (few pixels), coarse clutter (terrain, sea state, infrastructure), and different formation physics. SAR-YOLO adds SAR-specific modules that estimate and suppress speckle/clutter while preserving target structure.

**Main features**:
- 11+ SAR-specific neck modules (SFE, SFM, SAA, AMF, CFAR, SSAC, SFR, CAG, TPM, TADR)
- 5+ mode variants per module for ablation studies
- Leave-One-Source-Out (LOSO) cross-sensor generalisation framework
- SAR-aware loss function (separation, small-target, smoothness, consistency terms)
- Scatter-selective refinement (SSAC) for computational efficiency
- Configurable model scales (n, s, m, l) with 110+ variants

**Technology stack**:
- **Language**: Python 3.10+
- **Framework**: PyTorch
- **Base architecture**: Ultralytics YOLO11
- **Test framework**: pytest
- **Linting/formatting**: ruff
- **Configuration**: YAML files under configs/
- **Model I/O**: PyTorch state_dict, Ultralytics YAML model specs

**High-level architecture**:
Layered architecture built on YOLO11 backbone, with SAR-specific neck modules in sequence:
`SARInputAdapter → SARFeatureEnhancement → SpeckleAwareFeatureModule → SARAdaptiveAttention → AdaptiveMultiScaleFusion → [SSAC optional] → Head → Predictions`

Each module follows the invariant: `F' = F + alpha() * residual` with `ZeroGate(alpha_init=0)`, ensuring exact identity at init.

**Major subsystems**:
1. **Neck modules** (11 modules, 35+ modes total): SFE, SFM, SAA, AMF, CFAR, SSAC, SFR, CAG, TPM, TADR
2. **Architecture builder** (arch.py): Constructs 110+ variants from YAML configs
3. **Model assembly** (model.py): Final model composition
4. **CLI** (cli.py): Command-line entry point for all operations
5. **Data pipeline** (data/): Dataset handling, splits, LOSO framework
6. **Evaluation** (evaluation/): mAP computation, metrics
7. **Training** (training/): Training loop, optimization
8. **CLI commands**: train, predict, evaluate, loso, efficiency, etc.

## Architecture Organization

### Key Design Invariants

**1. Zero-initialised gate (foundational invariant)**
Every module follows `F' = F + alpha() * residual` where `alpha()` is a `ZeroGate` initialized to 0. This means:
- At init: `alpha(0) = 0` → `F' = F` (exact identity)
- During training: `alpha` learns to non-zero values, enabling the residual
- Test: `test_each_module_is_exactly_identity_at_init` verifies `max|f(x) - x| = 0.0e+00` for all 11 modules across all modes

**2. Mode-based architecture**
Each module has multiple modes (e.g., SFM: `sfm`, `sfm_clutter`, `none`, `lee`, `denoise`). Modes are configured via YAML strings, not hard-coded conditionals in the module forward. The `arch.py` builder maps mode strings to module instantiations.

**3. Configuration-driven behavior**
Module behavior is driven by YAML config strings. To change a module's behavior, edit the variant YAML in `configs/models/`, not the Python code.

**4. Channel preservation**
All modules preserve the input channel count (c1). Input and output shapes are always `(B, C, H, W)` — this is critical for residual connections to work.

**5. LocalStats primitive**
Shared `LocalStats` class computes per-channel statistics: `mu` (mean), `var` (variance), `hp` (heterogeneity proxy), `sd` (standard deviation). Used by: speckle.py, context.py. The window size (kernel) is configurable.

### Where Important Functionality Lives

| Functionality | Module File | Key Class |
|---|---|---|
| Speckle-aware feature extraction | `saryolo/nn/modules/speckle.py` | `SpeckleAwareFeatureModule` |
| Context aggregation (spatial + regional) | `saryolo/nn/modules/context.py` | `ContextAggregation` |
| Feature enhancement (log/standardize/clahe) | `saryolo/nn/modules/enhancement.py` | `SARFeatureEnhancement` |
| Adaptive attention (vs static, + SE/EA/CBAM) | `saryolo/nn/modules/attention.py` | `SARAdaptiveAttention` |
| Multi-scale fusion (amf/concat/add/static) | `saryolo/nn/modules/fusion.py` | `AdaptiveMultiScaleFusion` |
| CFAR front end (cfar/conv/fixed) | `saryolo/nn/modules/cfar.py` | `RatioSpaceCFARFrontEnd` |
| Scatter-selective refinement (SSAC) | `saryolo/nn/modules/ssac.py` | `ScatterSelectiveRefinement` |
| Spatial frequency representation (sff/static/highpass/dct/wavelet) | `saryolo/nn/modules/frequency.py` | `SpatialFrequencyRepresentation` |
| Target prior modulation (learned/channel/static/cfar/spectral) | `saryolo/nn/modules/target_prior.py` | `TargetPriorModulation` |
| Target-aware refinement (deform/static/local/none) | `saryolo/nn/modules/refinement.py` | `TargetAwareRefinement` |
| SAR input adaptation (4 modes) | `saryolo/nn/modules/input_adapter.py` | `SARInputAdapter` |
| Shared utilities (ConvBNAct, DWConvBNAct, LocalStats, ZeroGate, resolve_c1) | `saryolo/nn/modules/_common.py` | 5 shared blocks |

### How to Trace Major Workflows

#### 1. Building a model variant
```
Step 1: Read configs/models/baseline_n.yaml (or any variant YAML)
Step 2: Note the scale (n/s/m/l) and module mode assignments
Step 3: Run: python -m saryolo build --config configs/models/baseline_n.yaml
Step 4: Or import from Python: from saryolo.arch import build_model_from_yaml; model = build_model_from_yaml("configs/models/baseline_n.yaml")
Step 5: Model will have 110+ variant construction verified by test_arch.py
```

#### 2. Running inference
```
Step 1: python -m saryolo predict --weights weights/sarvo_ssac001.pt --imgsz 640 --source /path/to/image.jpg
Step 2: CLI (cli.py) parses args, builds model from config + loads weights
Step 3: Model forward pass through neck modules in sequence
Step 4: Predictions output (bounding boxes, scores, labels)
Step 5: Test: pytest tests/test_inference.py -q
```

#### 3. Running the full test suite
```
Step 1: pytest tests/ -q  # runs 985 core checks (default, excludes slow/gpu)
Step 2: To include slow: pytest tests/ -q -m "slow"
Step 3: To include gpu: pytest tests/ -q -m "gpu" (requires CUDA)
Step 4: Verbose: pytest tests/ -vv to see individual test results
Step 5: With coverage: pytest tests/ --cov=saryolo --cov-report=term-missing
```

#### 4. LOSO (leave-one-source-out) cross-sensor evaluation
```
Step 1: saryolo loso generate --sources src1 src2 src3 --folds 3
Step 2: Generates 3 training folds, each leaving out one source for validation
Step 3: saryolo train --fold 1 --train-fold 1 --val-fold 2  # train on src2+src3, validate on src1
Step 4: saryolo loso evaluate --fold 1 --model <path>  # evaluate on left-out source
Step 5: LOSO CLI commands documented in cli.py
```

#### 5. SSAC efficiency measurement
```
Step 1: python -m saryolo efficiency --checkpoint weights/sarvo_ssac001.pt --imgsz 640
Step 2: Measures FLOPs and inference time at multiple sparsity budgets
Step 3: Round-robin protocol: dense baseline measured in same process, round-robin with each budget
Step 4: Result: efficiency gains reported as % reduction at same mAP
Step 5: Test: pytest tests/test_ssac.py -q
```

## Critical Dependencies

### Internal Dependencies (first-party)

**All modules depend on `_common.py`**:
- `ConvBNAct`: Convolution + BatchNorm + SiLU activation (channel-preserving)
- `DWConvBNAct`: Depth-wise convolution + BatchNorm + SiLU (receptive field expansion without resolution loss)
- `LocalStats`: Per-channel statistics (mu, var, hp, sd) with configurable kernel
- `ZeroGate`: `alpha` parameter with `alpha_init=0`, `alpha(0)=0` guarantees identity at init
- `resolve_c1`: Resolves channel count from parser's `ch` list or direct integer

**Module-to-module dependencies** (call graph):
- `model.py` → imports all module classes from `nn.modules`
- `arch.py` → imports all module classes + yaml parsing
- `cli.py` → imports from `saryolo` package, calls arch/model builders
- Tests → import specific module test classes

### External Dependencies

**PyTorch**: Core framework. All modules are `nn.Module` subclasses.
**Ultralytics YOLO11**: Base architecture. `parse_model`, model construction utilities.
**PyYAML**: YAML config loading (`yaml.safe_load`).
**pytest**: Test framework.
**ruff**: Linting + formatting.

### Environment Requirements

- **Python**: 3.10+ (target-version in pyproject.toml)
- **PyTorch**: 2.0+ (compatible with nn.Module, ZeroGate pattern)
- **GPU**: Optional; CUDA if available, else CPU. All modules work on CPU.
- **Memory**: Standard; SAR feature maps are channel-preserving, no unusual memory patterns
- **Disk**: `weights/sarvo_ssac001.pt` (one checkpoint), `configs/` YAML files, `tests/` test suite

### Important Conventions

**1. Mode naming**
- Mode strings are YAML-configurable via variant YAML files
- Mode names must be valid Python identifiers (not keywords like `global`)
- This is guarded by `test_module_mode_names_are_safe_for_parse_model`
- See: `context.py` docs for why `global` can't be a mode name

**2. Zero-initialised gate pattern**
- Every module: `F' = F + alpha() * residual` with `ZeroGate(0.0)`
- This is the foundational design invariant
- Without it, modules would be frozen at init (zero gradient flow)
- Verified by: `test_each_module_is_exactly_identity_at_init` and `test_no_module_is_frozen_at_init`

**3. YAML config structure**
- Variant YAMLs in `configs/models/` define: `scale`, neck architecture, module modes
- Dataset YAMLs in `configs/datasets/` define: paths, class names, augmentation
- Training YAMLs in `configs/training/` define: optimizer, scheduler, hyperparameters
- All YAML parsed with `yaml.safe_load` (Ultralytics convention)

**4. Identity-at-init first**
- Every module is designed so that `max|f(x) - x| = 0.0e+00` at init
- This allows safe experimentation: a fresh SAR-YOLO is numerically identical to YOLO11 baseline
- New modules/modes must preserve this invariant (or intentionally break it, with test exclusion)

**5. Test markers**
- `slow`: Tests that build large models or run training steps (excluded by default)
- `gpu`: Tests that require CUDA (excluded from CPU-only runs)
- By default, only tests without these markers run (985 checks)

**6. Exploration ledger**
- `docs/project-context/08_EXPLORATION_LEDGER.md` tracks every investigated file
- Update when adding new modules, modes, or changing architecture
- Used by future AI agents to avoid re-exploring completed work

## How to Run and Test the Project

### Daily Development (core instrument, fastest)
```bash
pytest tests/ -q  # runs 985 checks by default
ruff check .  # linting
```

### Full Quality Check
```bash
pytest tests/ -q -m "not (slow or gpu)"  # same as default, explicit
pytest tests/ -q -m "slow"  # include slow tests
pytest tests/ -q -m "gpu"  # include gpu tests (requires CUDA)
```

### Specific Test Suites
```bash
pytest tests/test_arch.py -q  # architecture builder (110+ variants)
pytest tests/test_init.py -q  # module identity at init
pytest tests/test_ssac.py -q  # SSAC mechanism
pytest tests/test_inference.py -q  # inference pipeline
pytest tests/test_baseline.py -q  # parameter count verification
pytest tests/test_module_helpers.py -q  # _common.py shared blocks
```

### Building a Model
```bash
# From CLI
python -m saryolo build --config configs/models/baseline_n.yaml

# From Python
from saryolo.arch import build_model_from_yaml
model = build_model_from_yaml("configs/models/baseline_n.yaml")

# Verify parameter count
python -c "from saryolo.arch import build_model_from_yaml; m = build_model_from_yaml('configs/models/baseline_n.yaml'); print(f'Params: {sum(p.numel() for p in m.parameters())}')"
# Expected: 9,458,752 (YOLO11s)
```

### Training
```bash
python -m saryolo train --config configs/training/default.yaml --data configs/datasets/hrsid.yaml --weights weights/sarvo_ssac001.pt
# Or via CLI: saryolo train --help
```

### Evaluation
```bash
python -m saryolo evaluate --weights weights/sarvo_ssac001.pt --data configs/datasets/hrsid.yaml --imgsz 640
# Or via CLI: saryolo evaluate --help
```

### LOSO Cross-Sensor Evaluation
```bash
# Generate folds
saryolo loso generate --sources sensor1 sensor2 sensor3 --folds 3

# Train on fold 1 (leave out sensor1)
saryolo train --fold 1 --train-fold 1 --val-fold 2 --config configs/training/default.yaml

# Evaluate on left-out source
saryolo loso evaluate --fold 1 --model weights/best.pt
```

### Efficiency Profiling
```bash
python -m saryolo efficiency --checkpoint weights/sarvo_ssac001.pt --imgsz 640 --budgets 0.25,0.5,0.75
# Measures FLOPs and inference time at multiple sparsity budgets
```

## Important Implementation Constraints

### 1. No module may break the identity-at-init invariant without being explicitly tested
- When adding a new module or mode, run `pytest tests/test_init.py -q`
- If identity-at-init is broken, the module must either:
  a. Intentionally break it (and be excluded from `test_each_module_is_exactly_identity_at_init`), OR
  b. Be redesigned to preserve it (via ZeroGate pattern)

### 2. Mode names must be valid Python identifiers
- `parse_model` resolves mode names with `ast.literal_eval`
- Python keywords (e.g., `global`) cause `SyntaxError: invalid syntax`
- New mode names must pass `test_module_mode_names_are_safe_for_parse_model`

### 3. All module forward passes preserve tensor shape `(B, C, H, W)`
- Input and output channels must match (c1)
- Spatial dimensions may change in specific modes, but the module output must be compatible with residual addition
- If spatial dimensions change, the residual connection must handle it (typically via the ZeroGate pattern where alpha handles the scaling)

### 4. YAML config versioning
- Variant YAMLs should not change the module parameter layout without bumping the module `version` attribute
- `speckle.py`: `version = 1`; bump when parameter layout changes
- `context.py`: `version = 1`; bump when parameter layout changes
- Checkpoint compatibility depends on version matching

### 5. Training requires per-sample conditioning
- `test_conditioning_is_per_sample_not_per_batch` verifies that row i of a mixed-source batch equals row i run alone
- The LOSO recipe depends on per-sample conditioning, not batch-level
- When modifying training loop, ensure conditioning is applied per-sample

### 6. No circular dependencies among first-party modules
- All imports trace through `_common.py` → modules → `arch.py` → `model.py`
- No module imports from another first-party module directly (they import from `_common.py` and are composed in `model.py`)
- If adding new inter-module imports, verify no circular dependency is created

### 7. Efficiency measurement protocol
- Between-process timing drift on shared CPU is larger than the SSAC efficiency effect
- `python -m saryolo efficiency` mitigates this by measuring dense baseline in same process, round-robin with each budget
- Do not compare efficiency numbers across independent process runs as ground truth

### 8. SAR input range assumptions
- The codebase does not enforce input normalization or range clamping
- SAR imagery commonly uses logarithmic compression or 0-4095 / 0-65535 bit depth
- If raw SAR data is fed directly without preprocessing, activations may saturate
- This is a known gap (issue 9 in 07_TECHNICAL_DEBT_AND_OPEN_QUESTIONS.md)

### 9. Batch norm tracking in eval mode
- Several modules use `nn.BatchNorm2d` without explicit `track_running_stats=False`
- In PyTorch, running stats accumulate during training and are used during eval
- If the training loop does not call `model.eval()` for evaluation passes, running stats will be contaminated
- This is a testing infrastructure concern (issue 10 in 07_TECHNICAL_DEBT_AND_OPEN_QUESTIONS.md)

### 10. Classical baselines (lee/denoise) are intentionally non-identity
- `SpeckleAwareFeatureModule` modes `lee` and `denoise` do NOT preserve identity at init
- These are classical baselines for ablation, not the proposed SFM/SFM-CLUTTER modes
- If adding new classical baselines, be explicit about non-identity behavior in docs and tests

## Known Risks

### 1. Input range unspecification
- SAR imagery range is not enforced or documented
- Raw bit-depth data (0-4095, 0-65535) may cause activation saturation without logarithmic compression
- **Mitigation**: Document expected input preprocessing; add normalization layer if needed for production use

### 2. No GPU test coverage by default
- All `gpu`-marked tests are opt-in; primary development/testing is CPU-only
- Cross-sensor generalisation (LOSO) has not been measured on real data from multiple sensors
- **Mitigation**: If GPU availability changes, enable `gpu` marker tests and run cross-sensor evaluation

### 3. No 3-seed reproducibility table
- One checkpoint committed (`weights/sarvo_ssac001.pt`), but no full 3-seed repeat of every variant
- **Mitigation**: If reproducibility at scale is needed, generate 3-seed repeat table and commit to ledger

### 4. LoRA/PEFT integration unexercised end-to-end
- `test_peft.py` exists but LoRA fine-tuning has not been extensively exercised
- **Mitigation**: If LoRA fine-tuning is a required workflow, add end-to-end test and verify

### 5. Between-process timing drift
- CPU shared-process timing drift (179.7 ms vs 203.5 ms for same checkpoint at 640 px) is larger than the SSAC efficiency effect
- **Mitigation**: Use `python -m saryolo efficiency` protocol (round-robin in same process) for accurate measurements

### 6. Classical baselines may be misleading ablations
- `lee` and `denoise` modes are explicitly non-identity at init by design
- Ablation studies comparing `sfm` vs `lee`/`denoise` must account for this fundamental difference
- **Mitigation**: Read speckle.py docs carefully; `lee`/`denoise` are classical baselines, not SAR-specific improvements

## Links to Detailed Documentation

| Document | Path | Description |
|---|---|---|
| Project Overview | `docs/project-context/00_PROJECT_OVERVIEW.md` | Project purpose, architecture, stack, entry points |
| Repository Map | `docs/project-context/01_REPOSITORY_MAP.md` | Directory tree, file inventory, module boundaries |
| Architecture | `docs/project-context/02_ARCHITECTURE.md` | Architectural explanation, diagrams, component relationships |
| Module Reference | `docs/project-context/03_MODULE_REFERENCE.md` | Implementation reference for every source module |
| Execution Flows | `docs/project-context/04_EXECUTION_AND_DATA_FLOWS.md` | Startup flows, API requests, data transformations |
| Configurations | `docs/project-context/05_CONFIGURATION_AND_INTEGRATIONS.md` | Env vars, configs, external services |
| Testing & Quality | `docs/project-context/06_TESTING_AND_QUALITY.md` | Test architecture, commands, coverage, results |
| Technical Debt | `docs/project-context/07_TECHNICAL_DEBT_AND_OPEN_QUESTIONS.md` | Bugs, concerns, open questions, known risks |
| Exploration Ledger | `docs/project-context/08_EXPLORATION_LEDGER.md` | File-by-file investigation checklist |
| **This Context** | **`docs/project-context/09_AI_WORKING_CONTEXT.md`** | **Concise operational context for future AI agents** |
| Context Index | `docs/project-context/10_CONTEXT_INDEX.md` | Index connecting concepts to repository paths |

## Quick Start for New AI Agent

1. **Read this file** (`09_AI_WORKING_CONTEXT.md`) to orient yourself
2. **Run the core test suite**: `pytest tests/ -q` — verify the instrument passes
3. **Read the exploration ledger**: `docs/project-context/08_EXPLORATION_LEDGER.md` — know what's been investigated
4. **Pick a module** from the Module Reference (`03_MODULE_REFERENCE.md`) and read its implementation
5. **Check the mode** you want to experiment with, and verify it passes `test_each_module_is_exactly_identity_at_init`
6. **Edit variant YAML** in `configs/models/` to change module modes, not Python code
7. **Run specific tests** to verify your changes don't break the identity-at-init invariant
8. **Consult technical debt** (`07_TECHNICAL_DEBT_AND_OPEN_QUESTIONS.md`) before making architectural changes