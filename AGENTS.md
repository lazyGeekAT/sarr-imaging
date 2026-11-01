# SAR-YOLO: AI Agent Instructions

## How to Use This Repository

### 1. Read This File First
Before making substantial changes, read `docs/project-context/09_AI_WORKING_CONTEXT.md`. This provides a concise but technically rich operational context for working on this repository.

### 2. Consult Detailed Documentation
For any subsystem modification, consult the relevant detailed documentation:
- **Module implementations**: `docs/project-context/03_MODULE_REFERENCE.md`
- **Architectural overview**: `docs/project-context/02_ARCHITECTURE.md`
- **Execution flows**: `docs/project-context/04_EXECUTION_AND_DATA_FLOWS.md`
- **Configuration**: `docs/project-context/05_CONFIGURATION_AND_INTEGRATIONS.md`
- **Test architecture**: `docs/project-context/06_TESTING_AND_QUALITY.md`
- **Technical debt & open questions**: `docs/project-context/07_TECHNICAL_DEBT_AND_OPEN_QUESTIONS.md`

### 3. Verify Important Claims Against Source Code
Never assume documentation is up-to-date. Important claims about:
- Module behavior and invariants
- Identity-at-init properties
- Mode configurations
- Parameter counts

...must be verified against actual source code in `saryolo/nn/modules/` and config YAML files in `configs/`.

### 4. Preserve Established Architecture and Conventions
The project's foundational design invariant is that **every module preserves identity at init**: `max|f(x) - x| = 0.0e+00` at initialization. This is enforced via the `ZeroGate(alpha_init=0)` pattern in every module.

When adding new modules or modes:
- Maintain the `F' = F + alpha() * residual` pattern
- Initialize `ZeroGate` with `alpha_init=0.0`
- Run `pytest tests/test_init.py -q` to verify identity-at-init
- Do not break the identity-at-init invariant without explicit testing

### 5. Update Documentation When Architectural Behavior Changes
If you change architectural behavior (e.g., module initialization, mode patterns, residual connections), update these documentation files:
- `docs/project-context/09_AI_WORKING_CONTEXT.md` (operational context)
- `docs/project-context/07_TECHNICAL_DEBT_AND_OPEN_QUESTIONS.md` (if introducing known issues)
- `docs/project-context/08_EXPLORATION_LEDGER.md` (file status updates)
- `docs/project-context/10_CONTEXT_INDEX.md` (concept-to-path mappings)

Do not assume old documentation is always current.

### 6. Important Invariants to Preserve

| Invariant | Description | Verification |
|---|---|---|
| **Identity at init** | `max|f(x) - x| = 0.0e+00` for all modules at init | `pytest tests/test_init.py -q` |
| **Zero-initialised gate** | `ZeroGate(0.0)` → `alpha(0) = 0` → `F' = F` at init | Code review of each module's `__init__` |
| **Channel preservation** | All modules preserve input channel count (c1) | Forward pass shape: `(B, C, H, W)` → `(B, C, H, W)` |
| **Mode names must be valid Python identifiers** | `parse_model` resolves with `ast.literal_eval`; keywords like `global` cause `SyntaxError` | `test_module_mode_names_are_safe_for_parse_model` |
| **No circular dependencies** | All imports trace through `_common.py` → modules → `arch.py` → `model.py` | Import graph analysis |

### 7. Test Commands (Familiarize Yourself)

**Daily development** (core instrument, fastest):
```bash
pytest tests/ -q  # runs 984 checks by default
```

**Full quality check**:
```bash
pytest tests/ -q -m "not (slow or gpu)"
```

**Specific suites**:
```bash
pytest tests/test_arch.py -q      # architecture builder (110+ variants)
pytest tests/test_init.py -q      # module identity at init
pytest tests/test_ssac.py -q      # SSAC mechanism
pytest tests/test_inference.py -q  # inference pipeline
pytest tests/test_baseline.py -q  # parameter count verification
pytest tests/test_module_helpers.py -q  # _common.py shared blocks
```

**Linting**:
```bash
ruff check .
```

### 8. Adding New Modules or Modes

When adding a new module:

1. **Implement in** `saryolo/nn/modules/new_module.py`
2. **Follow the pattern**:
   - `F' = F + alpha() * residual` with `ZeroGate(0.0)`
   - `max|f(x) - x| = 0.0e+00` at init (identity at init)
   - Mode validation in `__init__` (raise `ValueError` for invalid modes)
   - Import from `_common.py` for shared blocks
3. **Register in** `saryolo/nn/arch.py` (add to variant YAML mapping)
4. **Add test in** `tests/test_new_module.py` (at minimum: identity at init, build + forward)
5. **Run**: `pytest tests/test_new_module.py -q` to verify
6. **Run**: `pytest tests/test_init.py -q` to ensure no regression

When adding a new mode to an existing module:

1. **Add mode to** the module's `MODES` tuple
2. **Implement mode forward logic**
3. **Run** `pytest tests/test_*.py -q` for that module to verify
4. **Check** `test_module_mode_names_are_safe_for_parse_model` still passes

### 9. Verifying Changes Don't Break Identity-at-Init

After any change to a module or the architecture builder:

```bash
# Run the identity-at-init test suite
pytest tests/test_init.py -q

# If any module fails, revert or adjust the change
# The test checks: max|f(x) - x| = 0.0e+00 for all 11 modules across all modes
```

### 10. Reporting Important Discoveries

If you discover something important that is not yet documented:

1. **Add it to** `docs/project-context/07_TECHNICAL_DEBT_AND_OPEN_QUESTIONS.md` if it's a concern, gap, or open question
2. **Add it to** `docs/project-context/08_EXPLORATION_LEDGER.md` with appropriate status
3. **Add an entry to** `docs/project-context/10_CONTEXT_INDEX.md` for the new concept-to-path mapping
4. **Consider** whether `docs/project-context/09_AI_WORKING_CONTEXT.md` needs updating

### 11. Commands Summary

| Action | Command |
|---|---|
| Orient yourself | `read docs/project-context/09_AI_WORKING_CONTEXT.md` |
| Run core tests | `pytest tests/ -q` |
| Check identity at init | `pytest tests/test_init.py -q` |
| Lint code | `ruff check .` |
| Build model from variant | `python -m saryolo build --config configs/models/baseline_n.yaml` |
| Run inference | `python -m saryolo predict --weights weights/sarvo_ssac001.pt --imgsz 640 --source /path/to/image.jpg` |
| LOSO: generate folds | `saryolo loso generate --sources src1 src2 src3 --folds 3` |
| Measure efficiency | `python -m saryolo efficiency --checkpoint weights/sarvo_ssac001.pt --imgsz 640` |
| Run full architecture builder test | `pytest tests/test_arch.py -q` |
| Run all module identity tests | `pytest tests/test_init.py -q` |
| Run SSAC tests | `pytest tests/test_ssac.py -q` |
| Check ruff compliance | `ruff check .` |

### 12. What Not To Do

- ❌ Do not modify application source code to "fix" things unless the task explicitly requires it
- ❌ Do not refactor existing implementations unless the task requires it
- ❌ Do not upgrade dependencies unnecessarily
- ❌ Do not install packages beyond what `requirements.txt` / `pyproject.toml` specifies
- ❌ Do not delete or rename files
- ❌ Do not change application configurations
- ❌ Do not execute destructive commands
- ❌ Do not modify database contents
- ❌ Do not trigger production deployments
- ❌ Do not assume old documentation is always current (always verify against source)
- ❌ Do not skip tests or weaken assertions to make verification pass
- ❌ Do not add type/lint suppressions just to make verification pass
- ❌ Do not mention Codebuff, the AI agent, or any AI assistant attributions in commit messages, PR titles, or PR bodies -- treat these as internal implementation details, not part of the project's public record

---

**Codebuff attributions are internal only.** Do not include "Generated with Codebuff" or similar attribution in any commit message, PR title, or PR description. These are implementation details, not project heritage.

---