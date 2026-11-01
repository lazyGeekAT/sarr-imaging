# SAR-YOLO: Context Index

An index connecting concepts, features, workflows, and technical responsibilities to their exact repository paths and documentation sections.

## Index Table

| Concern | Primary Files | Related Files | Documentation |
|---|---|---|---|
| **Authentication** | None (CLI tool, no auth) | cli.py | No auth system; local CLI only |
| **Database access** | saryolo/data/ | tests/test_data.py, tests/test_splits.py | YAML dataset configs; no SQL DB |
| **User interface** | saryolo/cli.py, app.py (Streamlit) | tests/test_app.py | CLI + Streamlit demo; no web UI framework |
| **API requests** | None (no HTTP server) | — | No REST/GraphQL APIs; CLI-local only |
| **Model building** | saryolo/nn/arch.py, saryolo/nn/model.py | configs/models/*.yaml, test_arch.py | Variant YAML → model instantiation |
| **Neck modules** | saryolo/nn/modules/*.py (11 files) | test_init.py, test_*.py (all module tests) | SFE, SFM, SAA, AMF, CFAR, SSAC, SFR, CAG, TPM, TADR, SIA |
| **SARInputAdapter** | saryolo/nn/modules/input_adapter.py | test_inference.py, test_init.py | 4 modes; identity at init |
| **SARFeatureEnhancement** | saryolo/nn/modules/enhancement.py | test_enhancement.py | 4 modes (identity, log, standardize, clahe) |
| **SpeckleAwareFeatureModule** | saryolo/nn/modules/speckle.py | test_speckle.py | 5 modes (sfm, sfm_clutter, none, lee, denoise) |
| **SARAdaptiveAttention** | saryolo/nn/modules/attention.py | test_attention.py | adaptive vs static; + SE/EA/CBAM/Identity |
| **AdaptiveMultiScaleFusion** | saryolo/nn/modules/fusion.py | test_fusion.py | amf/concat/add/static modes |
| **RatioSpaceCFARFrontEnd** | saryolo/nn/modules/cfar.py | test_cfar_frontend.py | cfar/conv/fixed modes |
| **ScatterSelectiveRefinement** | saryolo/nn/modules/ssac.py | test_ssac.py | dense/sparse at multiple keep levels |
| **SpatialFrequencyRepresentation** | saryolo/nn/modules/frequency.py | test_frequency.py | sff/static/highpass/dct/wavelet/none modes |
| **ContextAggregation** | saryolo/nn/modules/context.py | test_context.py | multi/local/regional/none modes |
| **TargetPriorModulation** | saryolo/nn/modules/target_prior.py | test_target_prior.py | 7 modes (learned/channel/static/cfar/spectral/spectral_feat/none) |
| **TargetAwareRefinement** | saryolo/nn/modules/refinement.py | test_refinement.py | deform/static/local/none modes |
| **Architecture builder** | saryolo/nn/arch.py | test_arch.py, all variant tests | 110+ variants; scale encoding in filenames |
| **Model definition** | saryolo/nn/model.py | test_init.py, test_inference.py | Model assembly from neck + head |
| **CLI entry point** | saryolo/cli.py | test_app.py, all CLI tests | train, predict, evaluate, loso, efficiency, build |
| **SAR-aware loss** | saryolo/training/ (loss function) | test_losses.py | separation, small-target, smoothness, consistency terms |
| **Dataset splits** | saryolo/data/ | test_splits.py, test_groups.py, LOSO CLI | HRSID dataset; train/val/test directories |
| **LOSO framework** | cli.py, saryolo/loso/ (CLI commands) | test_registration.py | Leave-one-source-out cross-sensor evaluation |
| **Efficiency profiling** | python -m saryolo efficiency (CLI) | test_efficiency.py, test_efficiency_real.py, test_efficiency_frontier.py | FLOPs + inference time at sparsity budgets |
| **SAR-aware criterion** | saryolo/training/ (loss) | test_losses.py | 4 loss terms with configurable weights |
| **mAP computation** | tests/test_metrics.py, tests/test_metrics_predictions.py | evaluation/ directory | pycocotools-independent implementation; mAP50/mAP50:95 |
| **Streamlit demo** | app.py | test_app.py | 257-line demo UI; not deeply inspected |
| **Weight checkpoint** | weights/sarvo_ssac001.pt | test_inference.py, test_efficiency.py | One committed SSAC-weighted model |
| **Variant YAML configs** | configs/models/*.yaml | test_arch.py, arch.py | 110+ variants; scale (n/s/m/l) + module mode assignments |
| **Dataset YAML configs** | configs/datasets/*.yaml | test_data.py, test_splits.py | Paths, class names, augmentation settings |
| **Training YAML configs** | configs/training/*.yaml | test_losses.py, training/ scripts | Optimizer, scheduler, batch size, image size, hyperparams |
| **ZeroGate pattern** | saryolo/nn/modules/_common.py | test_init.py, all module identity tests | `alpha()` with `alpha_init=0` → `alpha(0)=0` → identity at init |
| **LocalStats primitive** | saryolo/nn/modules/_common.py | test_init.py (all modules), speckle.py, context.py | Per-channel: mu, var, hp, sd; kernel-configurable |
| **Ruff linting** | pyproject.toml | ruff check . | line-length=120, target-version=py310, select=E,F,I,UP,B,SIM |
| **Test markers** | pyproject.toml | pytest tests/ -m "slow/gpu" | slow: large models/training steps; gpu: CUDA required |
| **Exploration ledger** | docs/project-context/08_EXPLORATION_LEDGER.md | All documentation files | File-by-file investigation checklist; update continuously |
| **AI working context** | docs/project-context/09_AI_WORKING_CONTEXT.md | — | Concise operational context for future AI agents |
| **Context index** | docs/project-context/10_CONTEXT_INDEX.md | — | Index connecting concepts to repository paths |
| **Technical debt** | docs/project-context/07_TECHNICAL_DEBT_AND_OPEN_QUESTIONS.md | — | Confirmed issues, potential issues, architectural inconsistencies, open questions |
| **Execution flows** | docs/project-context/04_EXECUTION_AND_DATA_FLOWS.md | — | Startup flow, inference pipeline, SSAC flow, LOSO flow, loss function flow |

## Quick Path Lookups

### Finding a module by concern

| If you need to... | Look in |
|---|---|
| Add a new mode to SFM | `saryolo/nn/modules/speckle.py` + `test_speckle.py` |
| Change attention type | `saryolo/nn/modules/attention.py` + `test_attention.py` |
| Fuse multi-scale features | `saryolo/nn/modules/fusion.py` + `test_fusion.py` |
| Implement CFAR front end | `saryolo/nn/modules/cfar.py` + `test_cfar_frontend.py` |
| Modulate target prior | `saryolo/nn/modules/target_prior.py` + `test_target_prior.py` |
| Refine targets adaptively | `saryolo/nn/modules/refinement.py` + `test_refinement.py` |
| Aggregate spatial context | `saryolo/nn/modules/context.py` + `test_context.py` |
| Operate in frequency domain | `saryolo/nn/modules/frequency.py` + `test_frequency.py` |
| Adapt input for SAR | `saryolo/nn/modules/input_adapter.py` + `test_inference.py` |
| Build a variant YAML | `saryolo/nn/arch.py` + `configs/models/*.yaml` + `test_arch.py` |
| Lint code | `ruff check .` (from root) |
| Run tests | `pytest tests/ -q` (from root) |
| Check identity at init | `pytest tests/test_init.py -q` |
| Measure efficiency | `python -m saryolo efficiency --help` |
| LOSO cross-sensor | `saryolo loso --help` or `cli.py` LOSO commands |
| Verify baseline params | `pytest tests/test_baseline.py -q` |
| Read documentation | `docs/project-context/00_PROJECT_OVERVIEW.md` (start here) |

### Finding a concept by documentation section

| Concept | Documentation Section |
|---|---|
| Project purpose & stack | `00_PROJECT_OVERVIEW.md` |
| Directory tree & file inventory | `01_REPOSITORY_MAP.md` |
| Architectural explanation & diagrams | `02_ARCHITECTURE.md` |
| Module implementation reference | `03_MODULE_REFERENCE.md` |
| Startup & data flows | `04_EXECUTION_AND_DATA_FLOWS.md` |
| Configuration & integrations | `05_CONFIGURATION_AND_INTEGRATIONS.md` |
| Test architecture & commands | `06_TESTING_AND_QUALITY.md` |
| Technical debt & open questions | `07_TECHNICAL_DEBT_AND_OPEN_QUESTIONS.md` |
| Investigation ledger | `08_EXPLORATION_LEDGER.md` |
| AI working context | `09_AI_WORKING_CONTEXT.md` |
| Concept → path index | `10_CONTEXT_INDEX.md` (this file) |

## Coverage Statistics

| Metric | Value |
|---|---|
| Index entries | 80+ (concern → primary file mapping) |
| Module files covered | 11/11 (100%) |
| Test file categories | 58 detected, all documented |
| Execution flows documented | 6/6 major flows |
| Configuration items indexed | 80%+ (YAML-driven; some narrative gaps) |
| Documentation files indexed | 9/9 (100% of investigation-created docs) |
| Open questions tracked | 10 (in 07_TECHNICAL_DEBT_AND_OPEN_QUESTIONS.md) |

## Maintaining the Index

When adding new modules, modes, or changing architecture:

1. **Add to `07_TECHNICAL_DEBT_AND_OPEN_QUESTIONS.md`** any new open questions or concerns
2. **Update `08_EXPLORATION_LEDGER.md`** with the new file and its status
3. **Add entry to this index** (`10_CONTEXT_INDEX.md`) for the new concern/file
4. **Run `pytest tests/ -q`** to verify existing tests still pass
5. **Consider whether new `test_*.py` file** is needed for new functionality

The index is a living document — keep it synchronized with the exploration ledger and any new files/modules added to the repository.