# SAR-YOLO: Module Reference

This document provides an implementation-oriented reference for every relevant source module in the SAR-YOLO codebase. Each entry documents the module's purpose, declarations, dependencies, callers, inputs/outputs, side effects, error handling, and relationships with other modules.

## Legend

| Symbol | Meaning |
|---|---|
| `📍` | File path (repository-relative) |
| `🎯` | Module responsibility/category |
| `📥` | Primary inputs |
| `📤` | Primary outputs |
| `⚙️` | Key side effects |
| `❌` | Error handling |
| `🔗` | Relationships with other modules |

---

## Module A: SAR Input Adapter (SIA)

📍 `saryolo/nn/modules/input_adapter.py`

🎯 **Responsibility**: Input-level SAR representation adapter. Operates on the image *before* the first convolution. Channel-preserving and identity at init. Four arms (identity, local, learned, hybrid) compared in the same slot under one init contract, so a difference between arms is attributable to the representation rather than where the module was inserted.

📥 **Primary inputs**:
- `x`: Input tensor (B, C, H, W), typically the SAR image
- `c1`: Input channels (resolved via `resolve_c1`, at first row this is the stem's input width, typically 3 for a stock YOLO model)

📤 **Primary outputs**:
- `z`: Adapted representation (B, C, H, W), same shape as input
- `forward(x)`: `x + alpha() * (representation(x) - x)`, residual with ZeroGate

⚙️ **Key side effects**:
- Identity at init: `alpha = 0` → `forward(x) == x` bit-for-bit before training
- Mode controls what representation is computed before gating
- Hybrid mode: identity at init comes from the gate alone (not the fusion); a zero-initialised fusion would freeze the adapter

🔗 **Callers**:
- Architecture builder (`arch.py`): Emitted as the **first** backbone row (`b.add("backbone", -1, 1, "SARInputAdapter", ["ch", mode])`)
- Downstream every module in the graph sees the adapter's output as the image representation

🔗 **Consumers**:
- Every backbone row after row 0 operates on the adapter's output
- The channel list `ch` propagated through `parse_model` includes the adapter's output width

📋 **Important declarations**:
- `MODES = ("hybrid", "local", "learned", "identity")`
- `version = 1`
- `c1 = resolve_c1(c1, source)` — resolves channel count from explicit int or parser's `ch` list
- `self.stats = LocalStats(k=self.kernel)` — when mode is `local` or `hybrid`
- `self.project = ConvBNAct(4 * self.c1, self.c1, k=1, act=True)` — projects 4-stream concatenation back to c1
- `self.learned = DWConvBNAct(self.c1, hidden, k=3)` — depth-wise then point-wise learned representation (when mode is `learned` or `hybrid`)
- `self.expand = ConvBNAct(hidden, self.c1, k=1, act=False)` — expands back to c1
- `self.fuse = ConvBNAct(2 * self.c1, self.c1, k=1, act=True)` — fuses stats + learned (hybrid only; NOT zero-initialised; identity at init comes from gate alone)
- `self.alpha = ZeroGate(alpha_init)` — tanh-bounded gate, initialised at 0
- `representation(x)`: Returns the adapted representation before gating (exposed for visualization)

### Mode Details

| Mode | Parameters | Description |
|---|---|---|
| `identity` | 0 | Raw intensity. Exactly zero parameters: the control. Measured: dropping the adapter row from `in_identity` leaves a graph layer-for-layer identical to `v2_full`, and after synchronising shared weights the two produce bit-identical outputs (max abs diff 0.0). |
| `local` | ~43 (at c1=3) | Local-statistics representation only: `psi([x ; mu ; sd ; hp])` through a 1x1 conv. 4 streams: raw, local mean, local std, high-pass residual. |
| `learned` | ~98 (at c1=3) | Learned representation only: `DWConv(x)` → `expand` → project back to c1. No explicit statistics. |
| `hybrid` | ~164 (at c1=3) | Proposed: both streams fused by a learned 1x1 projection. Stats branch (`4*c1 → c1` via `project`) + learned branch (`DWConv → expand`). Fused by `fuse` (2*c1 → c1 via 1x1 conv). Identity at init from gate alone. |

❌ **Error handling**:
- `mode not in MODES` → `ValueError`
- `c1` must be `int` or `list`/`tuple` (not `bool`, not other types)
- `source` must be `None` at the first YAML row (no source layer to index at position 0)

🔗 **Relationships**:
- **Upstream**: None (it is the first row in the YAML)
- **Downstream**: Every module in the backbone/head sees the adapter's output as the input tensor
- **Related to**: CFAR front end (`cfar.py`) — both are input-side SAR representations; CFAR replaces raw intensity with analytic statistic, SIA computes a learned/local representation. They are mutually exclusive (only one is emitted as row 0 of backbone).
- **Slot study arms** (from `VARIANTS`): `in_identity`, `in_local`, `in_learned`, `in_hybrid` — all keep every other v2 slot at its setting, so differences are attributable to the input representation alone.

---

## Component 1: SAR Feature Enhancement (SFE)

📍 `saryolo/nn/modules/enhancement.py`

🎯 **Responsiciency**: Learnable local-contrast feature enhancement with a zero-initialised residual gate. Re-expresses features through local contrast normalisation and mixes back through a gated residual, so enhancement is additive and reversible rather than destructive.

📥 **Primary inputs**:
- `x`: Input tensor (B, C, H, W)

📤 **Primary outputs**:
- `forward(x)`: `x * gamma + beta + alpha() * z`, where `z` is the learned structure mix, `gamma/beta` are per-channel affine corrections
- Identity at init: `gamma = 1`, `beta = 0`, `alpha = 0` → `forward(x) == x` bit-for-bit

⚙️ **Key side effects**:
- Mode `sfe`: `gamma` (ones), `beta` (zeros), `alpha` (ZeroGate at 0), `z` = learned structure mix
- Mode `identity`: returns `x` unchanged
- Mode `log`: log compression — `sign(x) * log1p(|x|)`
- Mode `standardize`: `(x - mu) / sd` — local contrast normalisation
- Mode `clahe`: differentiable CLAHE analogue — `refined * gamma + beta`

🔗 **Callers**:
- Architecture builder (`arch.py`): Emitted after P2/4 C3k2 row: `b.add("backbone", -1, 1, "SARFeatureEnhancement", ["ch", mode, kernel, reduction])`
- Downstream modules (SFM, SAA, AMF, etc.) see the enhanced features

🔗 **Consumers**:
- Every module after SFE in the per-level chain operates on the enhanced features
- The `LocalStats` block inside SFE produces `mu, var, sd, hp` — these are also exposed for diagnostics

📋 **Important declarations**:
- `MODES = ("sfe", "identity", "log", "standardize", "clahe")`
- `version = 1`
- `self.c1 = resolve_c1(c1)` — input channels
- `self.mode = mode` — selects the enhancement variant
- `self.stats = LocalStats(k=kernel)` — local statistics computation (3x3 avg-pool based)
- **Mode `sfe` specifics**:
  - `self.gamma = nn.Parameter(torch.ones(1, self.c1, 1, 1))` — per-channel gain, initialised to 1
  - `self.beta = nn.Parameter(torch.zeros(1, self.c1, 1, 1))` — per-channel bias, initialised to 0
  - `self.alpha = ZeroGate(alpha_init)` — residual gate, initialised at 0
  - `self.reduce = ConvBNAct(3 * self.c1, hidden, k=1)` — mixes raw + structure + sd
  - `self.expand = ConvBNAct(hidden, self.c1, k=1, act=False)` — projects hidden back to c1
  - Forward: `structure = (x - mu) / sd`; `z = self.expand(self.reduce(torch.cat((x, structure, sd), dim=1)))`; `return x * gamma + beta + alpha() * z`
- **Mode `clahe` specifics**:
  - `self.gamma = nn.Parameter(torch.ones(1, self.c1, 1, 1))`
  - `self.beta = nn.Parameter(torch.zeros(1, self.c1, 1, 1))`
  - `self.clip = ZeroGate(0.0)` — clip bounds the structure branch
  - Forward: `structure = (x - mu) / sd`; `clip = self.clip() * 2.0`; `structure = structure.clamp(-1.0 - clip, 1.0 + clip)`; `refined = (x - mu) * structure + mu`; `return refined * gamma + beta`
- **Mode `log`**: `return sign(x) * log1p(|x|)` — guarded to stay finite
- **Mode `standardize`**: `return (x - mu) / sd` — local contrast normalisation
- **Mode `identity`**: `return x`

❌ **Error handling**:
- `mode not in MODES` → `ValueError`
- `kernel` must be odd and >= 3 (enforced by... actually not in __init__, but LocalStats enforces it)

🔗 **Relationships**:
- **Upstream**: CFAR front end (if enabled, row 0 of backbone); enhancement operates on whatever the CFAR produces, or on raw intensity (if no CFAR)
- **Downstream**: SFM (Component 2), SAA (Component 3), AMF (Component 4), or any per-level slot (prior, attention, context, refinement, SSAC)
- **Related to**: SFM (speckle module) — both use `LocalStats` as the core SAR-specific primitive. SFE enhances local contrast; SFM separates target from speckle/clutter. They are different mechanisms in the same slot.
- **Ablation arms** (from `VARIANTS`): `pre_identity` (no enhancement), `pre_log`, `pre_clahe`, `pre_standardize` — all keep every other v2 slot at its setting

---

## Component 2: Speckle-Aware Feature Module (SFM)

📍 `saryolo/nn/modules/speckle.py`

🎯 **Responsibility**: Separates target structure from speckle/clutter responses and reweights them. Following the project brief's hypothesis: if a network is given an explicit notion of "where target-like structure is likely to be", and that notion is used to modulate detection features rather than merely plotted, hard targets become easier to detect.

📥 **Primary inputs**:
- `x`: Input tensor (B, C, H, W)

📤 **Primary outputs**:
- `forward(x)`: `x + alpha() * suppressed`, where `suppressed = structure - beta * noise_map * x` (or with clutter term subtracted)
- Identity at init: `alpha = 0` → `forward(x) == x` bit-for-bit

⚙️ **Key side effects**:
- Mode `sfm`: Proposed — estimates per-channel speckle likelihood `N` and learned target-response map `T`; `F' = F + alpha * (T - beta * N * F)`
- Mode `sfm_clutter`: Extended — separate estimators for speckle (near window) and clutter (wide window); `F' = F + alpha * (T - beta * N * F - gamma * C * F)`
- Mode `lee`: Classical Lee filter — `F + k * (mu - F)`, `k = var / (var + noise_var)`
- Mode `denoise`: Fixed low-pass — `0.5 * x + 0.5 * mu`
- Mode `none`: Returns `x` unchanged

🔗 **Callers**:
- Architecture builder (`arch.py`): Emitted after SFE row: `b.add("backbone", -1, 1, "SpeckleAwareFeatureModule", ["c1", mode, reduction, kernel])`
- Downstream modules see the speckle-separated features

🔗 **Consumers**:
- Every module after SFM in the per-level chain operates on the speckle-separated features
- The `noise` head produces a noise map; the `structure` branch produces structure-preserving features

📋 **Important declarations**:
- `MODES = ("sfm", "sfm_clutter", "none", "lee", "denoise")`
- `version = 1`
- `self.c1 = resolve_c1(c1)` — input channels
- `self.mode = mode` — selects the SFM variant
- `self.stats = LocalStats(k=kernel)` — local statistics (mean, var, sd, high-pass)
- **Mode `sfm` specifics**:
  - `self.noise = nn.Sequential(
      ConvBNAct(3 * self.c1, hidden, k=1),
      ConvBNAct(hidden, hidden, k=3),  # spatial context: speckle is spatially correlated
      nn.Conv2d(hidden, self.c1, 1))` — estimates noise likelihood from [x, hp, var]
  - `nn.init.zeros_(self.noise[-1].weight)`; `nn.init.constant_(self.noise[-1].bias, -2.0)` — noise head starts near "no noise"
  - `self.structure = DWConvBNAct(self.c1, self.c1, k=3)` — structure-preserving target extractor (3x3 depth-wise conv)
  - `self.beta = nn.Parameter(torch.tensor(float(beta_init)))` — initial `beta` logit; `sigmoid(0.5) ~= 0.62`
  - `self.alpha = ZeroGate(0.0)` — residual gate, initialised at 0
  - Forward: `mu, var, _, hp = self.stats(x)`; `noise_map = torch.sigmoid(self.noise(torch.cat((x, hp, var), dim=1)))`; `structure = self.structure(x)`; `beta_val = torch.sigmoid(self.beta)`; `suppressed = structure - beta_val * noise_map * x`; `return x + alpha() * suppressed`
- **Mode `sfm_clutter` specifics**:
  - Additional `self.wide = LocalStats(k=2 * kernel + 1)` — wide-window statistics for clutter
  - `self.clutter = nn.Sequential(...)` — estimates clutter likelihood from [x, hp_wide, sd_wide]
  - `self.gamma = nn.Parameter(torch.tensor(0.5))` — clutter suppression weight
  - Forward also subtracts `sigmoid(gamma) * clutter_map * x`
- **Mode `lee`**: 
  - `mu = self.stats.pool(x)`; `_, var, _, _ = self.stats(x)`; `noise_var = var.mean().clamp_min(1e-5)`; `k = (var / (var + noise_var)).clamp(0.0, 1.0)`; `return x + k * (mu - x)`
- **Mode `denoise`**: `return 0.5 * x + 0.5 * mu` — fixed low-pass blend toward local mean

❌ **Error handling**:
- `mode not in MODES` → `ValueError`
- `reduction` must be >= 1 (enforced by `max(self.c1 // max(reduction, 1), 8)`)
- `beta_init` is just a float, no range enforcement in __init__ (sigmoid bounds it to (0,1) at forward)

🔗 **Relationships**:
- **Upstream**: SFE (Component 1) — operates on enhanced features; SFM separates target from speckle in the enhanced features
- **Downstream**: SAA (Component 3), AMF (Component 4), or per-level slots (prior, attention, context, refinement, SSAC)
- **Related to**: SFE (feature enhancement) — both use `LocalStats` as the core SAR-specific primitive. SFE enhances local contrast; SFM separates target from speckle/clutter. They are different mechanisms in the same slot (SFE is Component 1, SFM is Component 2).
- **Slot study arms** (from `VARIANTS`): `speckle_none`, `speckle_lee`, `speckle_denoise` — all keep every other v2 slot at its setting

---

## Component 3: SAR-Adaptive Attention (SAA) and Baselines

📍 `saryolo/nn/modules/attention.py`

🎯 **Responsibility**: Proposed SAR-Adaptive Attention: channel + spatial + local-contrast evidence, adaptively fused with per-sample softmax weights predicted from the feature's own global statistics. All alternatives share the `F + alpha * (F * M - F)` residual form and the same `alpha = 0` initialisation, so every block is an exact identity at init and the comparison isolates the attention mechanism itself.

📥 **Primary inputs**:
- `x`: Input tensor (B, C, H, W)

📤 **Primary outputs**:
- `forward(x)`: `x + alpha() * (x * m - x)`, where `m` is the fused attention map
- Attention map `m`: `w0 * a + w1 * s + w2 * l`, weighted sum of channel, spatial, and local-contrast branches
- Identity at init: `alpha = 0` → `forward(x) == x`

⚙️ **Key side effects**:
- Three branches: channel, spatial, local-contrast
- Adaptive fusion: per-sample softmax weights over the three branches
- `static` ablation: replaces predicted weights with learned constants shared across the batch, isolating adaptivity from having three branches
- All arms are channel-preserving and identity at init

🔗 **Callers**:
- Architecture builder (`arch.py`): Emitted per detection level: `b.add("head", heads[lvl], 1, "SARAdaptiveAttention", ["ch", heads[lvl], 16, 7, gate])`
- The `gate` argument selects `adaptive` (proposed) or `static` (ablation)

🔗 **Consumers**:
- The detection head reads the attended features
- The `desc = ChannelDescriptor()` output is also used by other modules (SFR, TPM)

📋 **Important declarations**:
- `ATTENTION_BUILDERS = {"none": IdentityAttention, "se": SEAttention, "eca": ECAAttention, "cbam": CBAMAttention, "saa": SARAdaptiveAttention, "saa_static": lambda c1, source=None, **kw: SARAdaptiveAttention(c1, source, gate="static", **kw)}`
- `build_attention(name, c1, source, **kwargs)` — instantiates a variant by name
- `version = 1` (on `_ResidualAttention`, `SEAttention`, `ECAAttention`, `CBAMAttention`, `SARAdaptiveAttention`, `IdentityAttention`)

### Sub-modules

#### IdentityAttention
- `MODE`: returns `torch.ones_like(x)` — the no-attention control
- Zero parameters

#### SEAttention (Squeeze-and-Excitation)
- `hidden = max(c1 // max(reduction, 1), 4)`
- `self.mlp = nn.Sequential(nn.Linear(2 * c1, hidden), nn.SiLU(inplace=True), nn.Linear(hidden, c1))`
- `nn.init.zeros_(self.mlp[-1].weight)`; `nn.init.zeros_(self.mlp[-1].bias)` — zero-init MLP makes gate start at 0
- `self.desc = ChannelDescriptor()`
- Forward: compute descriptor (avg + max over spatial), pass through MLP, sigmoid, reshape to (B, C, 1, 1), expand_as(x)
- Channel-only baseline

#### ECAAttention (ECA-Net)
- `self.conv = nn.Conv1d(1, 1, k, padding=k // 2, bias=False)`
- `nn.init.zeros_(self.conv.weight)` — zero-init
- Forward: adaptive avg pool2d → flatten → Conv1d → sigmoid → reshape to (B, C, 1, 1), expand_as(x)
- 1D-conv channel baseline

#### CBAMAttention (Convolutional Block Attention Module)
- `self.mlp = nn.Sequential(nn.Linear(2 * c1, hidden), nn.SiLU(inplace=True), nn.Linear(hidden, c1))` — channel descriptor MLP
- `nn.init.zeros_(self.mlp[-1].weight)`; `nn.init.zeros_(self.mlp[-1].bias)`
- `self.spatial = nn.Conv2d(2, 1, kernel, padding=kernel // 2, bias=False)` — spatial attention
- `nn.init.zeros_(self.spatial.weight)`
- Forward: channel attention → multiply → spatial attention → return `ch * sp`
- Sequential channel-then-spatial baseline

#### SARAdaptiveAttention (Proposed)
- `hidden = max(c1 // max(reduction, 1), 4)`
- **Channel branch**: `self.channel_branch = nn.Sequential(nn.Linear(3 * c1, hidden), nn.SiLU(inplace=True), nn.Linear(hidden, c1))`; `nn.init.zeros_(...)`; `self.desc = ChannelDescriptor()`
- **Spatial + local-contrast branches**: `self.local = LocalStats(k=3)`; `self.spatial_branch = nn.Sequential(...)` (Conv2d → BN → SiLU → Conv2d); `self.contrast_branch = nn.Sequential(ConvBNAct(2, ...), Conv2d(..., 1, 1, bias=False))`; all with zero-init final weights
- **Adaptive fusion**: `self.gate_mlp = nn.Sequential(nn.Linear(3 * c1, hidden), nn.SiLU(inplace=True), nn.Linear(hidden, 3))`; `nn.init.zeros_(...)` — neutral start: all three branches weighted equally until training disagrees
- **Static fusion**: `self.static_logits = nn.Parameter(torch.zeros(3))` — learned constants, softmaxmed across the batch
- Forward computes all three branch maps, then fuses with either adaptive or static weights

❌ **Error handling**:
- `gate not in ("adaptive", "static")` → `ValueError` in `SARAdaptiveAttention.__init__`
- All sub-modules: zero-init of final weights ensures identity at init

🔗 **Relationships**:
- **Upstream**: AMF (Component 4), per-level slots (prior, context, refinement, SSAC), SFM (Component 2), SFE (Component 1)
- **Downstream**: Detection head
- **Related to attention baselines**: `SEAttention`, `ECAAttention`, `CBAMAttention` — same residual form (`F + alpha * (F * M - F)`), same identity-at-init contract, same channel-preserving design. The module-level ablation table (`tests/test_arch.py`) compares full model against each baseline.
- **Ablation arms** (from `VARIANTS`): `att_none`, `att_se`, `att_eca`, `att_cbam`, `att_saa_static` — all keep every other v2 slot at its setting

---

## Component 4: Adaptive Multi-Scale Fusion (AMF)

📍 `saryolo/nn/modules/fusion.py`

🎯 **Responsibility**: Input-adaptive, channel-preserving reweighting of concatenated multi-scale features. Inserted immediately after each neck `Concat` and reweights the concatenated channel groups before the (stock) projection block that follows.

📥 **Primary inputs**:
- `x`: Input tensor (B, C, H, W), the concatenated feature from FPN

📤 **Primary outputs**:
- `forward(x)`: 
  - `mode == "concat"`: returns `x` unchanged (stock behaviour)
  - `mode == "add"`: projects each branch to shared width, sums, projects back to c1
  - `mode == "amf"`: `_reweight(x)` then `gated = fused * sigmoid(channel_gate(mix(fused)))`; `return x + alpha() * (gated - x)`
  - `mode == "static"`: `_reweight(x)` with static logits; `return x + alpha() * (gated - x)`
- Identity at init: `alpha = 0` → all modes produce `forward(x) == x` (except `add` which is an operator replacement, not an identity)

⚙️ **Key side effects**:
- `mode == "amf"`: Learned per-scale weights via MLP over channel descriptors; DWConv + channel gate
- `mode == "static"`: Learnable but input-independent weights (ablation isolating adaptivity)
- `mode == "add"`: Projected additive fusion; kept close to average branch width
- `mode == "concat"`: Returns input unchanged (the block is present in graph but is a no-op)

🔗 **Callers**:
- Architecture builder (`arch.py`): Emitted after each `Concat` in the top-down pathway. The `sources` argument lists which backbone rows were concatenated; the block reads `ch[sources]` to determine group sizes.
- Example: `b.add("head", -1, 1, "AdaptiveMultiScaleFusion", ["ch", sources, reduction, mode])`

🔗 **Consumers**:
- The output feeds into the next C3k2 block in the FPN pathway
- The `channel_gate` output (1x1 conv) is a single-channel gate applied across all channels

📋 **Important declarations**:
- `MODES = ("amf", "concat", "add", "static")`
- `version = 1`
- `self.c1 = int(sum(self.groups))` — total width = sum of group sizes
- `self.groups = list(int(g) for g in groups)` — explicit group sizes from Concat inputs
- **Mode `amf` specifics**:
  - `self.desc = ChannelDescriptor()` — computes (avg, max, std) over spatial dims → (B, 3*c1)
  - `self.weight_mlp = nn.Sequential(nn.Linear(3 * self.c1, hidden), nn.SiLU(inplace=True), nn.Linear(hidden, len(self.groups)))`; `nn.init.zeros_(...)` — starts from uniform weights
  - `self.mix = DWConvBNAct(self.c1, self.c1, k=3)` — cheap depth-wise mix after reweighting
  - `self.channel_gate = nn.Sequential(nn.Conv2d(self.c1, self.c1, 1, bias=False))`; `nn.init.zeros_(...)` — channel gate, zero-init
  - `self.alpha = ZeroGate(0.0)` — residual gate
  - `_reweight(x)`: `chunks = split(x, self.groups)`; `w = softmax(weight_mlp(desc(x)))`; `return cat(chunks[g] * w[:, g].view(b, 1, 1, 1) for g)`
  - Forward: `fused = _reweight(x)`; `gated = fused * sigmoid(channel_gate(mix(fused)))`; `return x + alpha() * (gated - x)`
- **Mode `static` specifics**:
  - `self.static_logits = nn.Parameter(torch.zeros(len(self.groups)))` — learnable but input-independent weights
  - Same `mix`, `channel_gate`, `alpha` as `amf` mode
  - `_reweight(x)` uses `softmax(self.static_logits)` instead of MLP
- **Mode `add` specifics**:
  - `self.proj_in = nn.ModuleList(nn.Conv2d(g, hidden, 1, bias=False) for g in self.groups)` — projects each group to shared hidden width
  - `self.proj_out = nn.Conv2d(hidden, self.c1, 1, bias=False)` — projects back to concatenated width
  - Forward: split into chunks, `summed = sum(proj(chunk) for proj, chunk in zip(...))`; `return proj_out(summed)`
- **Mode `concat`**: `return x` — the block is present but is a no-op

❌ **Error handling**:
- `mode not in MODES` → `ValueError`
- With `sources`: `c1` must be the parser's `ch` list; groups must sum to `c1`; otherwise `ValueError`
- `groups` must be non-empty; `len(self.groups)` is used for softmax dimension

🔗 **Relationships**:
- **Upstream**: Top-down FPN pathway — each `Concat` input receives output from an `Upsample` + previous `AMF` (or stock Concat)
- **Downstream**: Next `C3k2` block in the FPN pathway; per-level slots (prior, attention, context, refinement, SSAC) operate on the output
- **Related to fusion baselines**: `concat` (stock behaviour), `add` (projected sum), `static` (learnable but input-independent weights) — all in the same slot, same init contract, so differences are attributable to the fusion operator
- **Ablation arms** (from `VARIANTS`): `fus_concat`, `fus_add`, `fus_static` — all keep every other v2 slot at its setting

---

## Ratio-Space CFAR Front End

📍 `saryolo/nn/modules/cfar.py`

🎯 **Responsibility**: Analytic multi-scale CFAR statistic as the detector's first representation. Replaces "a convolution reads the raw magnitude image" with "the radar's own statistic is computed analytically, and a learned gain decides per pixel how much of it to use."

📥 **Primary inputs**:
- `x`: Input tensor (B, C, H, W), typically the SAR image (grayscale, often replicated to 3 channels by Ultralytics)

📤 **Primary outputs**:
- `forward(x)`: `x * (1 + g)`, where `g` is the per-pixel gain in (-1, 1) via `tanh`
- Mode `cfar`: `g = tanh(gain(MFP_stack))` — learned gain over CFAR statistic
- Mode `conv`: `g = tanh(gain(raw_intensity_expanded))` — matched-cost control: same layer shapes, reads raw intensity instead of statistic
- Mode `fixed`: `g = tanh(2.0 * (mean_ratio - mean_cf ar).clamp(-1, 1))` — no learnable parameters; threshold the mean local contrast

⚙️ **Key side effects**:
- Identity at init in `cfar` and `conv` modes (zero-initialised gain network); `fixed` mode is deliberately not an identity (it perturbs the baseline)
- Three control modes answer different questions:
  - `cfar`: the proposed arm — learned, zero-initialised per-pixel gain over the radar statistic
  - `conv`: matched-cost control — identical parameter count, fed raw intensity instead of statistic. Separates "the radar statistic helps" from "217 extra parameters and a learned transform help"
  - `fixed`: fixed-threshold control — no learnable parameters. Separates "the statistics help" from "the learnable part helps"

🔗 **Callers**:
- Architecture builder (`arch.py`): Emitted as the **first** backbone row (before everything else): `b.add("backbone", -1, 1, "RatioSpaceCFARFrontEnd", ["ch", mode])`
- Downstream every module in the graph sees the CFAR statistic as the first representation

🔗 **Consumers**:
- Every backbone row after row 0 operates on the CFAR statistic output
- The statistic stack (`mean_channels, log, r_k, c_k`) is also exposed via `statistics(x)` for diagnostics

📋 **Important declarations**:
- `MODES = ("cfar", "conv", "fixed")`
- `version = 1`
- `IDENTITY_AT_INIT = {"cfar": True, "conv": True, "fixed": False}` — only `cfar` and `conv` are identities at init
- `self.c1 = resolve_c1(c1)` — input channels
- `self.mode = mode` — selects the CFAR variant
- `self.scales = tuple(int(k) for k in scales)` — local-statistics window sizes; default `(3, 7, 15)`
- `self.eps = float(eps)` — numerical floor for log and ratio denominators
- `self.pools = nn.ModuleList(AvgPool2d(k, stride=1, padding=k // 2) for k in self.scales)` — one pool per scale
- **Gain network** (shared by `cfar` and `conv` modes, same layer shapes → same parameter count):
  - `self.gain = nn.Sequential(nn.Conv2d(n_stats, hidden, k=1), nn.SiLU(inplace=True), nn.Conv2d(hidden, 1, k=1))`
  - `n_stats = 1 + 2 * len(self.scales)` — e.g., with scales=(3,7,15): n_stats = 1 + 2*3 = 7 (mean, then for each pool: log-ratio and CoV)
  - `nn.init.zeros_(self.gain[-1].weight)`; `nn.init.zeros_(self.gain[-1].bias)` — exact identity at init: g == 0, out == x
- **Mode `cfar` forward**: `stats = self.statistics(x)`; `return x * (1.0 + torch.tanh(self.gain(stats)))`
- **Mode `conv` forward**: `raw = x.mean(dim=1, keepdim=True)`; `return x * (1.0 + torch.tanh(self.gain(raw.expand(-1, self.gain[0].in_channels, -1, -1))))` — the "matched-cost control": same layer shapes, reads raw intensity
- **Mode `fixed` forward**: 
  - `stats = self.statistics(x)`
  - `contrast = stats[:, 2::2].mean(dim=1, keepdim=True)` — the c_k channels (local coefficient of variation)
  - `ratio = stats[:, 1::2].mean(dim=1, keepdim=True)` — the r_k channels (log-ratio)
  - `return x * (1.0 + torch.tanh(2.0 * (ratio - contrast).clamp(-1.0, 1.0)))` — threshold the mean local contrast at a constant, no parameters

❌ **Error handling**:
- `mode not in MODES` → `ValueError`
- `scales` must not be empty; each k must be odd and >= 3

🔗 **Relationships**:
- **Upstream**: None (it is the first row of the backbone, row 0)
- **Downstream**: Every module in the backbone/head sees the CFAR statistic as the input tensor. The statistic stack's structure (mean, log-ratio r_k, CoV c_k) is also consumed by:
  - SFE (Component 1): uses `LocalStats` which produces similar (mu, var, sd, hp) — the two mechanisms read the same kind of signal
  - SFM (Component 2): uses `LocalStats` for speckle/clutter estimation
  - SSAC (ssac.py): uses `statistics()` for per-region assessment
- **Ablation arms** (from `VARIANTS`): `pre_identity` (no CFAR), `pre_log`, `pre_clahe`, `pre_standardize`, `pre_standardize` — all keep every other v2 slot at its setting
- **Related to**: SAR Input Adapter (input_adapter.py) — both are input-side SAR representations. CFAR replaces raw intensity with analytic statistic; SIA computes a learned/local representation. They are mutually exclusive (only one is emitted as row 0 of backbone).

---

## Scatter-Selective Adaptive Computation (SSAC)

📍 `saryolo/nn/modules/ssac.py`

🎯 **Responsibility**: Scatter-Selective Adaptive Computation (SSAC): spend expensive feature processing only where the image evidence says it is useful. A cheap shared path runs everywhere; a multi-scale SAR statistic drives a per-region allocation `g ∈ (0, 1)`; the expensive refinement is added in proportion to `g`; and the whole block is gated by a zero-initialised residual, so it is an exact identity until it has learned something.

📥 **Primary inputs**:
- `x`: Input tensor (B, C, H, W), typically the feature map at a detection level

📤 **Primary outputs**:
- `forward(x)`: `x + alpha() * (mix - x)`, where `mix = g * rich + (1 - g) * cheap`, `g` is the gain map, `rich` is the expensive path, `cheap` is the cheap path
- `last_gate_mean`: recorded when `penalty > 0`; mean allocation of the last forward pass
- `gain_map(x)`: per-region allocation `g` in (0, 1), shape (B, 1, H, W), measurable rather than inferred

⚙️ **Key side effects**:
- Two execution modes: `dense` (expensive path runs everywhere) and `sparse` (expensive path runs only on selected tiles)
- Two modes: `adaptive` (proposal, SAR statistic drives allocation), `adaptive_raw` (assessment alternative, raw features drive allocation), `fixed` (matched fixed-computation control, spatially constant allocation)
- Exact identity at init in every mode (alpha = 0 → out == x)
- Sparse execution: tiles with highest-mean evidence are refined; rejected tiles come back as the cheap path's value (no mask needed)
- Halo of 2 pixels on each tile ensures sparse eval reproduces dense eval to floating-point tolerance
- Penalty term: `sparsity * mean(gate_means)` added to loss when `penalty > 0`; makes the allocation differentiable and pushes it to be small

🔗 **Callers**:
- Architecture builder (`arch.py`): Emitted per detection level: `b.add("head", heads[lvl], 1, "ScatterSelectiveRefinement", ["ch", heads[lvl], ssac, list(ssac_scales), ssac_hidden, ssac_expand, ssac_execution, ssac_tile, ssac_keep, ssac_penalty, ssac_tau, ssac_select])`
- The `mode` selects adaptive/adaptive_raw/fixed; `execution` selects dense/sparse; `keep` is the fraction of tiles refined (sparse only); `penalty` adds the sparsity term to loss

🔗 **Consumers**:
- The detection head reads the adaptively-refined representation
- `last_gate_mean` is used by the training loss (sparsity term) and by evaluation diagnostics

📋 **Important declarations**:
- `MODES = ("adaptive", "adaptive_raw", "fixed")` — three modes: proposal, assessment alternative, fixed control
- `EXECUTIONS = ("dense", "sparse")` — orthogonal to modes: mode decides how allocation is computed, execution decides how much arithmetic is spent
- `version = 1`
- `IDENTITY_AT_INIT = {"adaptive": True, "adaptive_raw": True, "fixed": True}` — every mode is an exact identity at init
- **Constructor arguments**:
  - `c1`: input channels (resolved via `resolve_c1`)
  - `mode`: "adaptive", "adaptive_raw", or "fixed"
  - `scales`: tuple of window sizes for assessment; default `(3, 7)` — geometric: only meaningful relative to a window larger than the scatterer and smaller than the scene
  - `hidden`: width of scorer MLP; default 16 — small on purpose, the assessment must be cheap
  - `expand`: bottleneck expansion of expensive path; default 2 — makes expensive path's arithmetic clearly larger than cheap path's
  - `execution`: "dense" or "sparse"; default "dense"
  - `tile`: spatial tile size for sparse execution; default 16 — smaller tiles route more finely, pay proportionally more halo
  - `keep`: fraction of tiles refined; default 0.25 — at least one tile always selected (`max(1, ceil(keep * n_tiles))`)
  - `penalty`: sparsity weight added to loss; default 0.0 — off in every arm except the deliberate sparsity arm
  - `tau`: temperature of allocation sigmoid; default 1.0
  - `select`: threshold on g for reporting selected fraction; default 0.5 — does not change forward pass
  - `alpha_init`: initial residual gate; default 0.0 — exact identity at init
  - `eps`: numerical floor for log and ratio denominators; default 1e-5
- `self.pools = nn.ModuleList(AvgPool2d(k, stride=1, padding=k // 2) for k in self.scales)` — one avg-pool per scale
- `self.head = nn.Sequential(nn.Conv2d(head_in, hidden, 1), nn.SiLU(inplace=True), nn.Conv2d(hidden, 1, 1))` — scorer MLP
  - `head_in = self.c1 if mode == "adaptive_raw" else 1 + 2 * len(self.scales)` — with adaptive: input is the statistic stack (1 mean + 2*K log-ratio/CoV); with adaptive_raw: input is the channel mean
- `self.cheap = DWConvBNAct(self.c1, self.c1, k=3)` — cheap path, always computed, shared by every location
- `self.rich = nn.Sequential(ConvBNAct(c1, c1 * expand, 1), DWConvBNAct(c1 * expand, c1 * expand, k=5), ConvBNAct(c1 * expand, c1, 1))` — expensive path, bottleneck (expand → DW 5x5 → contract)
- `self.alpha = ZeroGate(alpha_init)` — residual gate
- `self.halo = self._receptive_halo()` — receptive radius of expensive path, summed over its spatial kernels
- `statistics(x)`: The analytic SAR assessment statistic stack, shape (B, 1 + 2K, H, W). Mean channels, then for each scale: log-ratio (CFAR variable) and local CoV.
- `evidence_logit(x)`: Raw evidence logit `a` before sigmoid. In `fixed` mode: logit is averaged over spatial axes, turning per-region decision into one difficulty scalar per image while keeping parameter count equal to proposal's.
- `gain_map(x)`: Per-region allocation `g` about to be applied, shape (B, 1, H, W). Factored out of forward so it's measurable.
- `selected_fraction(x)`: Fraction of locations above the reporting threshold `select`; diagnostic, not a switch.
- `_tile_scores(g)`: Per-tile mean allocation (B, nH*nW), in row-major tile order.
- `_selected_tiles(scores)`: Indices of tiles sparse execution refines (B, k), sorted. `topk` returns k highest-evidence tiles; sort makes gather order a pure function of scores.
- `routing_stats(x)`: Diagnostics: what a sparse build would skip, and at what halo cost. `executed_rich_fraction` is the share of expensive path's arithmetic that sparse would perform, halo included: `k/nTiles * ((tile + 2*halo)/tile)^2`, capped at 1.
- `_refine_sparse(f, g)`: `rich(f)` computed only on selected tiles, halo-corrected. Padding to whole tiles, each gathered tile carries halo pixels of context. Rejected tiles come back as `f` itself (exactly what the mix at g=0 would produce).
- `forward(x)`: `g = gain_map(x)`; `cheap = self.cheap(x)`; `rich = self._refine_sparse(cheap, g) if execution == "sparse" else self.rich(cheap)`; `mix = g * rich + (1.0 - g) * cheap`; `return x + self.alpha().to(x.dtype) * (mix - x)`
- `extra_repr()`: `f"c1={self.c1}, mode={self.mode}, scales={self.scales}, expand={self.expand}, execution={self.execution}, tile={self.tile}, keep={self.keep}"`

❌ **Error handling**:
- `mode not in MODES` → `ValueError`
- `execution not in EXECUTIONS` → `ValueError`
- `tile < 1` → `ValueError`
- `not 0.0 < keep <= 1.0` → `ValueError`
- `penalty < 0.0` → `ValueError`
- `not scales` → `ValueError` ("scales must not be empty; the assessment needs at least one window")
- `any(k < 3 or k % 2 == 0 for k in scales)` → `ValueError` (scales must be odd and >= 3)
- `expand < 1` → `ValueError`
- `not 0.0 < select < 1.0` → `ValueError`

🔗 **Relationships**:
- **Upstream**: Per-level slots (prior, attention, context, refinement) — SSAC acts last in the chain, so the detection head reads the adaptively-refined representation
- **Downstream**: Detection head
- **Related to**: CFAR front end (cfar.py) — both use SAR-native statistics (log-ratio + local CoV). SSAC reuses the same assessment statistic family the CFAR front end is built on, because the repository already measured that these statistics carry usable signal on real HRSID chips (REAL-004). Whether the signal predicts *where* expensive processing helps is the separate question SSAC exists to test.
- **Related to**: SAR-Aware loss (model.py) — when `w_ssac_sparsity > 0`, the training loss adds `sparsity * mean(gate_means)` term. The means are read from modules immediately after the primary forward, before the perturbed pass can overwrite them.
- **Ablation arms** (from `VARIANTS`): `SSAC-001` through `SSAC-006` — the master workflow's key comparison arms. SSAC-001 vs SSAC-002 (adaptive vs fixed computation, parameter-identical), SSAC-003 (raw feature vs SAR statistic), SSAC-004 (sparse execution), SSAC-005 (expand=1 expensive path), SSAC-006 (sparsity penalty).
- **Execution mode test**: `test_ssac_execution_sweep` — profiles dense vs sparse at multiple `keep` levels, in one process, so comparison is valid (between-process drift is larger than the effect)

---

## Component 9: Spatial-Frequency Representation (SFR)

📍 `saryolo/nn/modules/frequency.py`

🎯 **Responsibility**: Learnable radial spectral filter, fused back into the spatial feature. Gives the network an explicit spectral branch and lets *training* decide how much of the spectrum to keep, instead of assuming a particular classical filter is the right one. Parameterised as a small number of radial frequency bands rather than one free coefficient per frequency bin.

📥 **Primary inputs**:
- `x`: Input tensor (B, C, H, W)

📤 **Primary outputs**:
- `forward(x)`: `x + alpha() * (y - x)`, where `y` is the spectral-filtered result. Exact identity at init (alpha = 0).
- Mode `sff`: Proposed — learnable radial bands modulated per sample by the feature's own global descriptor
- Mode `static`: Learnable radial bands, input-independent
- Mode `highpass`: Fixed classical response (monotonically increasing gain with radius), registered as buffer (never trained)
- Mode `dct`: 8x8 block DCT-II with learnable radial bands inside the block
- Mode `wavelet`: One-level Haar analysis/synthesis with learnable per-sub-band gains
- Mode `none`: Returns `x` unchanged

🔗 **Callers**:
- Architecture builder (`arch.py`): Emitted on the deepest backbone stage (P5/32): `b.add("backbone", -1, 1, "SpatialFrequencyRepresentation", ["ch", mode, bands, reduction])`
- The P5/32 stage is the cheapest place to attach FFT (cost grows with spatial resolution)

🔗 **Consumers**:
- The output feeds into the top-down FPN pathway and per-level slots (prior, attention, context, refinement, SSAC)
- The `sff` mode's `modulate` head takes a 3-vector (avg, max, std) from `ChannelDescriptor` and outputs per-band log-gains

📋 **Important declarations**:
- `MODES = ("sff", "static", "highpass", "dct", "wavelet", "none")`
- `version = 3` (bumped when parameter layout changes, e.g., "none" arm no longer carries unreachable gate parameter)
- `self.c1 = resolve_c1(c1)` — input channels
- `self.mode = mode` — selects the SFR variant
- `self.bands = int(bands)` — number of radial frequency bands; must be >= 2
- **Mode `highpass` specifics**:
  - `self.register_buffer("band_gain", torch.linspace(-1.0, 1.0, self.bands).view(1, -1))` — fixed buffer, never trained
  - Fixed classical response: monotonically increasing gain with radius ("attenuate slowly varying background, keep fine structure")
- **Mode `dct` specifics**:
  - `self.band_gain = nn.Parameter(torch.randn(self.c1, self.bands) * 1e-2)` — small random init
  - `self.register_buffer("dct_basis", torch.zeros(0), persistent=False)` — cached on first use, re-created if device changes
  - `_basis(device, dtype)`: Orthonormal DCT-II basis (64×64), rows are coefficient, columns are pixel
  - `_forward_dct(x)`: Overlapping-free 8x8 block DCT round trip with learnable radial gain
    - Padding: replication mode (hard edge avoided)
    - `unfold` into blocks, `basis @ coeffs`, apply `exp(gain)`, `basis.t() @ coeffs`, `fold` back
    - Replication padding avoids hard edges at block boundaries
- **Mode `wavelet` specifics**:
  - `self.band_gain = nn.Parameter(torch.randn(self.c1, 4) * 1e-2)` — four octant sub-bands (LL, LH, HL, HH)
  - `_haar_split(x)` / `_haar_merge(ll, lh, hl, hh)`: One-level orthonormal Haar analysis/synthesis
  - `_forward_wavelet(x)`: Haar analysis/synthesis with learnable per-sub-band gains
  - Gain: `exp(band_gain.clamp(-6.0, 6.0)).view(1, c1, 4, 1, 1)` — applied to each of the 4 sub-bands
- **Mode `sff` specifics**:
  - `self.desc = ChannelDescriptor()` — global channel descriptor (avg, max, std) → (B, 3*c1)
  - `self.modulate = nn.Sequential(nn.Linear(3 * c1, hidden), nn.SiLU(inplace=True), nn.Linear(hidden, self.bands))`; `nn.init.zeros_(...)` — zero-init modulation head; exact identity at init comes from the gate alone
  - `sff - static` measures the value of *input adaptivity* alone — the same decomposition the attention module uses with its own static arm
  - Per-sample spectral adaptation: `gain = gain.unsqueeze(0) + self.modulate(self.desc(x)).unsqueeze(1)` — (1, S, bands) + (B, 1, bands) → (B, C, bands)
- **Mode `static` specifics**:
  - `self.band_gain = nn.Parameter(torch.randn(self.c1, self.bands) * 1e-2)` — small random init, not zeros (zero prior would freeze the gate)
- **Mode `none`**: No gate created (mode != "none" check in __init__)
- Forward (all modes except none):
  - `compute_dtype = x.dtype if x.dtype in (float32, float64) else float32` — torch.fft has no half-precision CPU kernel
  - `xh = x.to(compute_dtype)`
  - **Mode `dct`**: `y = self._forward_dct(xh)`
  - **Mode `wavelet`**: `y = self._forward_wavelet(xh)`
  - **Other modes** (sff, static, highpass): `spectrum = torch.fft.rfft2(xh, norm="ortho")`; `filtered = torch.fft.irfft2(spectrum * self.spectral_gain(xh), s=xh.shape[-2:], norm="ortho")`; `y = filtered`
  - `y = y.to(x.dtype)`; `return x + self.alpha() * (y - x)`
- `spectral_gain(x)`: Per-bin spectral gain, shaped (1, C, H, W // 2 + 1). Interpolated over radius using `interp_weights` / `radial_band_index` (shared with TPM's spectral arm).
  - `lo, hi, weight = radial_band_index(h, w, self.bands, x.device, float32)` — two-band interpolation weights
  - `g = band_gain.index_select(-1, lo) * (1 - weight) + band_gain.index_select(-1, hi) * weight`
  - If mode `sff`: `gain = gain.unsqueeze(0) + modulate(self.desc(x)).unsqueeze(1)` — per-sample adaptation
  - `g = g.clamp(-6.0, 6.0)`; `return exp(g).view(*gain.shape[:-1], h, w // 2 + 1)`

❌ **Error handling**:
- `mode not in MODES` → `ValueError`
- `bands < 2` → `ValueError` ("bands must be >= 2 to interpolate between bands")
- `highpass`: no bands constraint (but bands set from __init__ argument)
- `dct`: basis re-creation on device change

🔗 **Relationships**:
- **Upstream**: Backbone P5/32 stage (row after SPPF); the spectral branch is placed here because FFT cost grows with spatial resolution, and P5/32 (10×10 at scale s, 5×5 at scale n) is the cheapest stage
- **Downstream**: Top-down FPN pathway, per-level slots (prior, attention, context, refinement, SSAC)
- **Shared infrastructure**: `interp_weights` and `radial_band_index` now live in `_common.py`, because both the backbone spectral filter (Component 9) and the prior-conditioned spectral selection (Component 8's `spectral` arm) need it. Two copies would let the two slots drift apart and make their band counts silently incomparable.
- **Related to**: Target Prior Modulation (target_prior.py) — Module G of the brief ("target-conditioned frequency selection") is implemented here in the prior slot rather than the backbone spectral slot, because the prior is what conditions it. The `spectral_feat` arm in TPM is the matched-capacity control (same network, same input width, but input is raw features instead of prior evidence).
- **Ablation arms** (from `VARIANTS`): `fr_none`, `fr_highpass`, `fr_static`, `fr_dct`, `fr_wavelet` — all keep every other v2 slot at its setting

---

## Component 10: Context Aggregation (CAG)

📍 `saryolo/nn/modules/context.py`

🎯 **Responsibility**: Additive multi-extent context, channel preserving and identity at init. Gathers context explicitly at two extents (local via dilated depth-wise convolutions, regional via global average pool) and adds it back as a residual. Context matters for SAR targets: a bright point return in open sea, a bright point in a harbour, and a speckle spike on grassland can have nearly identical local appearance; what separates them is the surrounding scene.

📥 **Primary inputs**:
- `x`: Input tensor (B, C, H, W)
- `source`: Index into `ch` for the layer actually consumed (required because this module is wired from an explicit layer index, not `-1`)

📤 **Primary outputs**:
- `forward(x)`: `x + alpha() * context(x)`, where `context(x)` combines local and regional context fields
- Mode `multi`: proposed — local + regional
- Mode `local`: dilated multi-extent spatial context only
- Mode `regional`: regionally pooled channel context only
- Mode `none`: returns `x` unchanged
- Identity at init: `alpha = 0` → `forward(x) == x`

🔗 **Callers**:
- Architecture builder (`arch.py`): Emitted per detection level: `b.add("head", heads[lvl], 1, "ContextAggregation", ["ch", heads[lvl], mode, reduction, dilations])`
- The `dilations` argument defaults to `(1, 2, 4)` — dilated depth-wise convs with these dilations

🔗 **Consumers**:
- The output feeds into the next per-level slot (refinement, SSAC) or the detection head

📋 **Important declarations**:
- `MODES = ("multi", "local", "regional", "none")`
- `version = 1`
- `self.c1 = resolve_c1(c1, source)` — resolved input channels from `ch` list and `source` index
- `self.mode = mode` — selects the CAG variant
- `self.dilations = tuple(int(d) for d in dilations)` — default `(1, 2, 4)` — dilated depth-wise convs widen receptive field without downsampling
- **Mode `multi`/`local` specifics**:
  - `self.branches = nn.ModuleList(nn.Sequential(
      nn.Conv2d(c1, c1, 3, 1, d, dilation=d, groups=c1, bias=False),
      nn.BatchNorm2d(c1),
      nn.SiLU(inplace=True)) for d in self.dilations)` — depth-wise dilated convs, one per dilation
  - `self.mix = nn.Conv2d(self.c1 * len(self.dilations), self.c1, 1, bias=False)` — mixes branched outputs back to c1
- **Mode `multi`/`regional` specifics**:
  - `hidden = max(c1 // max(reduction, 1), 8)`
  - `self.regional = nn.Sequential(ConvBNAct(c1, hidden, k=1, act=True), nn.Conv2d(hidden, c1, 1, bias=False))` — regional context: pool over space, project, broadcast back
- **Mode `forward` specifics**:
  - `mode == "none"`: `return x`
  - Local context: `ctx = ctx + mix(cat([b(x) for b in self.branches], dim=1))` — each dilated conv produces a feature map, they're concatenated and mixed back
  - Regional context: `pooled = x.mean(dim=(2, 3), keepdim=True)`; `ctx = ctx + regional(pooled)` — global pool, project, broadcast back to spatial size
  - `return x + alpha() * context(x)`
- `extra_repr()`: `f"c1={self.c1}, mode={self.mode}, dilations={self.dilations}"`

❌ **Error handling**:
- `mode not in MODES` → `ValueError`
- `dilations` must be a tuple of ints; handled by `tuple(int(d) for d in dilations)`

🔗 **Relationships**:
- **Upstream**: Per-level slots (prior, attention, refinement, SSAC) — CAG is after attention and context in the chain; actually the order is: prior → attention → context → refinement → SSAC. Wait, let me re-check from the architecture doc: the per-level chain is prior → attention → context → refinement → SSAC → head. So CAG (context) is before refinement and SSAC.
- **Downstream**: Refinement (TADR), SSAC, or detection head
- **Related to**: SAR-Adaptive Attention (attention.py) — overlaps in *purpose* (both reweight/aggregate features) but not in mechanism: attention produces a multiplicative gate on the existing feature, whereas CAG adds an additive context field derived from a wider neighbourhood. The ablation ladder measures this distinction.
- **Ablation arms** (from `VARIANTS`): `cx_none`, `cx_local`, `cx_regional` — all keep every other v2 slot at its setting
- **Mode name constraint**: The mode name must be a valid Python identifier (not a keyword like `global`), because `parse_model` resolves string arguments with `ast.literal_eval`, and `global` would raise `SyntaxError`. This is guarded by `tests/test_arch.py::test_module_mode_names_are_safe_for_parse_model`.

---

## Component 8: Target Prior Modulation (TPM)

📍 `saryolo/nn/modules/target_prior.py`

🎯 **Responsibility**: Signed, content-adaptive target prior applied multiplicatively to features. The project's central hypothesis: if a network is given an explicit, learned notion of "where target-like structure is likely to be", and that notion is used to modulate the detection features rather than merely plotted, then hard targets (small, low-contrast, speckle-obscured) become easier to detect.

📥 **Primary inputs**:
- `x`: Input tensor (B, C, H, W)

📤 **Primary outputs**:
- `forward(x)`: `x + alpha() * (x * (1 + m) - x)` (most modes) or `x + alpha() * (filtered * (1 + m) - x)` (spectral modes), where `m` is the signed prior in (-1, 1)
- `prior_map(x)`: Spatially-reduced prior as (B, 1, H, W) for every mode. Positive values mark target-like structure, negative values speckle/clutter-like. Uniform arms (static, channel) return a constant map by construction.
- Exact identity at init: `alpha = 0` → `forward(x) == x`

🔗 **Callers**:
- Architecture builder (`arch.py`): Emitted per detection level: `b.add("head", heads[lvl], 1, "TargetPriorModulation", ["ch", heads[lvl], mode, kernel, reduction, bands, cfar_threshold, cfar_gain, alpha_init])`

🔗 **Consumers**:
- The output feeds into the next per-level slot (attention, context, refinement, SSAC) or the detection head
- `prior_map(x)` is used by the visualization pipeline (SEC. 29-30 of the brief)

📋 **Important declarations**:
- `MODES = ("learned", "channel", "static", "cfar", "spectral", "spectral_feat", "none")`
- `version = 2` (bumped when prior-conditioned spectral arms and descriptor head were added)
- `self.c1 = resolve_c1(c1, source)` — resolved input channels
- `self.mode = mode` — selects the TPM variant
- `self.cfar_threshold = float(cfar_threshold)` — CFAR arm only: how many local std devs above local mean a response must sit to count as target-like
- `self.cfar_gain = float(cfar_gain)` — CFAR arm only: slope of logistic transfer
- `self.bands = int(bands)` — number of radial frequency bands for spectral arms; must be >= 2
- `self.stats = LocalStats(k=kernel)` — near-scale statistics (3x3 window)
- `self.wide = LocalStats(k=2 * kernel + 1)` — wide-scale statistics (for learned/channel/spectral modes)
- **Mode `learned`, `channel`, `spectral`, `spectral_feat` specifics**:
  - `hidden = max(c1 // max(reduction, 1), 8)`
  - `self.evidence = nn.Sequential(ConvBNAct(4 * c1, hidden, k=1), DWConvBNAct(hidden, hidden, k=3), nn.Conv2d(hidden, 1, 1))` — evidence network from [F; hp; sd_n; sd_w]
  - `nn.init.normal_(self.evidence[-1].weight, std=1e-2)`; `nn.init.zeros_(self.evidence[-1].bias)` — gentle init, not zero (zero would freeze the gate gradient)
  - If mode is `spectral` or `spectral_feat`:
    - `band_hidden = max(c1 // max(reduction, 1), 8)`
    - `self.band_head = nn.Sequential(nn.Linear(3, band_hidden), nn.SiLU(inplace=True), nn.Linear(band_hidden, c1 * self.bands))`; `nn.init.normal_(...)`; `nn.init.zeros_(...)`
  - If mode is `learned` or `channel`:
    - `e = e.mean(dim=(2, 3), keepdim=True)` — pool evidence over space, keep batch axis (prior loses spatial variation)
  - `nn.init.zeros_(self.modulate[-1].weight)`; `nn.init.zeros_(self.modulate[-1].bias)` — if mode is `sff` (wait, this is in frequency module, let me re-read)
  - Actually for TPM `learned`/`channel`/`spectral`/`spectral_feat`: the evidence head starts near but not at zero
- **Mode `static` specifics**:
  - `self.static_logits = nn.Parameter(torch.randn(1, self.c1, 1, 1) * 1e-2)` — per-channel learnable prior, spatially uniform
- **Mode `cfar` specifics**:
  - `excess = (x - mu) / sd` — how far above local background the response sits, in local standard deviations
  - `return tanh(cfar_gain * (excess - cfar_threshold))` — logistic transfer from excess to prior value
- **Mode `none`**: No gate created; returns zeros_like(x) for `_prior`

- `_evidence(x)`: Target evidence, shape (B, S, H', W'); S is 1 or C.
  - `cfar` mode: `excess = (x - mu) / sd`; `tanh(cfar_gain * (excess - cfar_threshold))`
  - `static` mode: `return self.static_logits`
  - `learned/channel/spectral/spectral_feat` modes: pass through `self.evidence` (and spatial pooling for `channel`)
  
- `_prior(x)`: Signed prior M in (-1, 1), broadcastable against x.
  - `cfar`, `learned`, `channel`, `spectral`, `spectral_feat`: `return 2.0 * sigmoid(e) - 1.0` — bounded in (-1, 1) and signed; M > 0 amplifies target-like structure, M < 0 suppresses speckle/clutter-like structure
  - `static`: `return tanh(e)` — already the modulation, squashed to (-1, 1)
  - `none`: `return torch.zeros_like(x)`

- `prior_map(x)`: Spatially-reduced prior as (B, 1, H, W) for every mode.
  - `m = self._prior(x)`; if `m.shape[1] != 1`: `m = m.mean(dim=1, keepdim=True)`; `return m.expand(x.shape[0], 1, *x.shape[-2:])` — uniform arms return a constant map by construction

- `forward(x)`:
  - `mode == "none"`: `return x`
  - `mode in ("spectral", "spectral_feat")`: 
    - `e = self._evidence(x)`; `m = 2.0 * sigmoid(e) - 1.0`; `filtered = self._spectral_filter(x, self._band_log_gain(x, e))`; `return x + alpha() * (filtered * (1.0 + m) - x)` — the prior does two jobs: selects radial frequency bands AND modulates the spatially filtered result. This is Module G of the brief ("target-conditioned frequency selection").
  - Other modes: `m = self._prior(x)`; `return x + alpha() * (x * (1.0 + m) - x)` — signed prior modulates the feature multiplicatively

- `extra_repr()`: `f"c1={self.c1}, mode={self.mode}"`

❌ **Error handling**:
- `mode not in MODES` → `ValueError`
- `bands < 2` → `ValueError`
- `cfar_threshold` and `cfar_gain` are floats, no range enforcement in __init__ (clamped/tanh'd at forward)

🔗 **Relationships**:
- **Upstream**: Per-level slots (attention, context, refinement, SSAC) — TPM is typically inserted before attention and context in the per-level chain. The architecture wires it after Component 8 and before the other slots: the prior modulates features, then attention/context/refinement/SSAC operate on the prior-modulated features.
- **Downstream**: Attention (SAA), Context (CAG), Refinement (TADR), SSAC, Detection head
- **Related to**: Spatial-Frequency Representation (frequency.py) — Module G ("target-conditioned frequency selection") is implemented here in the prior slot rather than the backbone spectral slot. The `spectral` arm of TPM also selects radial frequency bands, sharing the same `radial_band_index` and `interp_weights` infrastructure from `_common.py`. The `spectral_feat` arm is the matched-capacity control (same network, same band count, but input is raw features instead of prior evidence).
- **Related to**: CFAR front end (cfar.py) — both use local statistics (mu, sd) and dimensionless ratios. The CFAR arm of TPM generalises the CFAR statistic from intensity to features. The CFAR front end is the *first* representation; TPM's CFAR arm operates on backbone features.
- **Ablation arms** (from `VARIANTS`): `tp_none`, `tp_cfar`, `tp_static`, `tp_channel`, `tp_spectral`, `tp_spectral_feat` — all keep every other v2 slot at its setting, so differences are attributable to how the prior is produced alone.

---

## Component 11: Target-Aware Deformable Refinement (TADR)

📍 `saryolo/nn/modules/refinement.py`

🎯 **Responsibility**: Prior-conditioned deformable refinement, channel preserving and identity at init. Adds a lightweight second stage that lets each cell "look somewhere else" via sub-pixel resampling with predicted offsets, refining the feature's alignment for better target detection.

📥 **Primary inputs**:
- `x`: Input tensor (B, C, H, W)

📤 **Primary outputs**:
- `forward(x)`: `x + alpha() * (refined - x)`, where `refined` is the depth-wise mixed result after optional grid_sample with offsets
- Identity at init: `alpha = 0` → `forward(x) == x`
- `offset_map(x)`: Bounded offsets as (B, H, W, 2) in grid units, for figures

⚙️ **Key side effects**:
- Four modes: `deform` (proposed, per-sample offsets), `static` (learned but input-independent offsets), `local` (depth-wise conv only, no offsets — capacity control), `none` (identity)
- `local` mode: same sub-network as `deform`, so `deform - local` isolates the value of deformation rather than an extra conv
- `static` mode: learned but input-independent: one offset field, the same for every sample
- `deform` mode: offsets predicted per sample from the feature via a 3x3 conv
- All modes run in at least fp32 (grid_sample has no half-precision CPU kernel)
- `max_offset` bounds how far a cell may look (offsets are `max_offset * tanh(.)`, so 0.5 = half the feature map)

🔗 **Callers**:
- Architecture builder (`arch.py`): Emitted per detection level: `b.add("head", heads[lvl], 1, "TargetAwareRefinement", ["ch", heads[lvl], mode, kernel, max_offset])`
- The `max_offset` defaults to 0.5 (half the feature map); can be swept (`rf_off25`, `rf_off100` variant arms)

🔗 **Consumers**:
- The output feeds the detection head
- `offset_map(x)` is used by the visualization pipeline

📋 **Important declarations**:
- `MODES = ("deform", "static", "local", "none")`
- `version = 1`
- `self.c1 = resolve_c1(c1, source)` — resolved input channels
- `self.mode = mode` — selects the TADR variant
- `self.kernel = int(kernel)` — depth-wise mixing kernel applied after resampling; default 3
- `self.max_offset = float(max_offset)` — bounds offsets; default 0.5 (half the feature map); offsets are `max_offset * tanh(.)`
- **Mode `deform` specifics**:
  - `self.offset = nn.Conv2d(c1, 2, 3, padding=1, bias=True)` — predicts 2-channel offset field (x, y) from input
  - Offsets are `max_offset * tanh(self.offset(x))`
- **Mode `static` specifics**:
  - `self.static_offset = nn.Parameter(torch.randn(2) * 1e-2)` — learned but input-independent offset field; small random init rather than zeros (zero would freeze the gate gradient per _common docstring)
- **Mode `local`/`none`**: No offset convolution built (only `self.mix = DWConvBNAct(c1, c1, k=self.kernel)` when mode != "none")
- **Forward all modes**:
  - `compute_dtype = x.dtype if x.dtype in (float32, float64) else float32` — grid_sample has no half-precision CPU kernel
  - `xf = x.to(compute_dtype)`
  - `offsets = self._offsets(xf, h, w)`:
    - `mode == "local"` or `"none"`: `offsets = None`
    - `mode == "static"`: `d = self.static_offset.view(1, 2, 1, 1).expand(x.shape[0], 2, h, w)` — one offset field for all samples
    - `mode == "deform"`: `d = self.offset(xf)` — per-sample offsets
  - Bounded: `return (self.max_offset * torch.tanh(d)).permute(0, 2, 3, 1).contiguous()`
  - If `offsets is None`: `refined = self.mix(xf)` — just the depth-wise mix, no sampling
  - If `offsets is not None`:
    - `grid = self._base_grid(h, w, x.device, compute_dtype).unsqueeze(0) + offsets` — identity grid + bounded offsets
    - `sampled = grid_sample(xf, grid, mode="bilinear", padding_mode="border", align_corners=True)`
    - `refined = self.mix(sampled)`
  - `refined = refined.to(x.dtype)`; `return x + self.alpha().to(x.dtype) * (refined - x)`
- `offset_map(x)`: Bounded offsets as (B, H, W, 2) in grid units (1.0 = half the feature map). Raises `ValueError` if mode has no offsets ("local"/"none").
  - `offsets = self._offsets(x, x.shape[2], x.shape[3])`; if `offsets is None`: `raise ValueError(...)`
  - `return offsets`
- `extra_repr()`: `f"c1={self.c1}, mode={self.mode}, max_offset={self.max_offset}"`

❌ **Error handling**:
- `mode not in MODES` → `ValueError`
- `kernel` must be >= 1 (enforced by ... actually not in __init__, but DWConvBNAct handles it)
- `max_offset` must be > 0 (not explicitly enforced in __init__, but tanh bounded means it's always bounded)

🔗 **Relationships**:
- **Upstream**: Per-level slots (prior, attention, context, SSAC) — TADR is the last slot in the per-level chain, inserted immediately pre-head. The order is: prior → attention → context → refinement(=TADR) → SSAC → head. Wait, let me re-check from the architecture doc: the per-level chain is prior → attention → context → refinement → SSAC → head. So TADR (refinement) is before SSAC.
- **Downstream**: Detection head
- **Related to**: Target Prior Modulation (target_prior.py) — TADR consumes an already prior-modulated feature, because the architecture wires it after Component 8. The "target-aware" conditioning is structural, not a second prior network, deliberately so that this component does not duplicate Component 8's evidence computation.
- **Ablation arms** (from `VARIANTS`): `rf_none`, `rf_local`, `rf_static`, `rf_off25`, `rf_off100` — all keep every other v2 slot at its setting, so differences are attributable to the refinement mode alone
- **Cross-level routing limitation**: Cannot be expressed in this repository's model YAMLs because:
  - `parse_model` resolves unknown module output channels with `c2 = ch[f]`, which breaks if `f` is a list
  - The `Detect` head uses a `frozenset` identity membership test; a subclass falls through to the generic path
  - Cross-level routing would need either patching `parse_model` (avoided) or a model-level mechanism outside the YAML

---

## Shared Blocks: _common.py

📍 `saryolo/nn/modules/_common.py`

🎯 **Responsibility**: Shared building blocks for SAR-YOLO neural modules. Provides the primitives every module depends on: convolution blocks, local statistics, channel resolution, zero gates, and FFT/wavelet infrastructure.

📥 **Primary inputs** (varies by block):
- `x`: Input tensor (B, C, H, W)
- `c1`, `c2`: Channel counts
- `k`, `s`, `g`: Kernel size, stride, groups
- `eps`: Numerical floor

📤 **Primary outputs** (varies by block):
- Feature tensors, statistics tuples, gate values, interpolation weights

🔗 **Key blocks**:

### resolve_c1(c1, source)
- Resolves input channel count from explicit int or parser's `ch` list
- If `c1` is a list/tuple: returns `c1[-1]` if `source is None`, otherwise `c1[source]`
- If `c1` is an int: returns `c1`
- Raises `TypeError` if `c1` is `bool`; `TypeError` if other type; `ValueError` if empty list

### ConvBNAct(c_in, c_out, k=1, s=1, g=1, act=True)
- Standard Conv2d → BatchNorm2d → SiLU block
- Self-contained; no ultralytics coupling
- `p = k // 2` — padding to maintain spatial resolution
- `self.conv = nn.Conv2d(c_in, c_out, k, s, p, groups=g, bias=False)`
- `self.bn = nn.BatchNorm2d(c_out)`
- `self.act = nn.SiLU(inplace=True) if act else nn.Identity()`

### DWConvBNAct(c_in, c_out=None, k=3, s=1, act=True)
- Depth-wise separable conv: cheap structure extraction that preserves spatial detail
- `c_out = c_in if c_out is None else c_out`
- `self.dw = nn.Conv2d(c_in, c_in, k, s, k // 2, groups=c_in, bias=False)` — depth-wise conv
- `self.pw = nn.Conv2d(c_in, c_out, 1, bias=False)` — point-wise conv
- `self.bn = nn.BatchNorm2d(c_out)`
- `self.act = nn.SiLU(inplace=True) if act else nn.Identity()`
- Forward: `self.act(self.bn(self.pw(self.dw(x))))`

### LocalStats(k=3, eps=1e-5)
- Local first/second-order statistics
- `self.pool = nn.AvgPool2d(k, stride=1, padding=k // 2, count_include_pad=False)`
- Forward: `(mu, var, sd, highpass)` where:
  - `mu = self.pool(x)` — local mean
  - `var = (self.pool(x * x) - mu * mu).clamp_min(0.0)` — local variance
  - `sd = torch.sqrt(var + self.eps)` — local standard deviation
  - Returns `(mu, var, sd, x - mu)` — the last is the high-pass residual

### ChannelDescriptor()
- Global channel descriptor: (avg, max, std) over spatial dims
- Forward: `torch.cat((avg, mx, std), dim=1).view(b, 3 * c)` — (B, 3*C)

### interp_weights(radius, bands)
- Two-band interpolation weights for each coefficient
- `pos = (radius.clamp(0, 1) * (bands - 1)).clamp(0, bands - 1)`
- `lo = pos.floor()`; `hi = (lo + 1).clamp(max=bands - 1)`
- Returns `(lo.flatten().long(), hi.flatten().long(), (pos - lo).flatten())` — used for gain interpolation across radial bands

### radial_band_index(h, w, bands, device, dtype)
- Two-band interpolation weights for each rfft2 bin of an (h, w) map
- `fy = fftfreq(h)`; `fx = rfftfreq(w)`; `radius = sqrt(fy^2 + fx^2) / (0.5 * sqrt(2))`
- Returns `interp_weights(radius, bands)` — same function as above, but for rfft2 bins

### ZeroGate(init=0.0)
- Learnable scalar residual gate, bounded to (-1, 1) and initialised at 0
- `self.raw = nn.Parameter(torch.tensor(float(init)))`
- Forward: `return torch.tanh(self.raw)` — bounds the gate to (-1, 1)
- Identity at init: `raw = 0` → `tanh(0) = 0` → gate is 0 → residual term vanishes

❌ **Error handling**:
- Varies by block; generally: ValueError for invalid args, TypeError for wrong types

🔗 **Relationships**:
- **Consumed by**: Every SAR-YOLO neural module (input_adapter, speckle, enhancement, attention, fusion, cfar, ssac, frequency, context, target_prior, refinement)
- **Defines the interface contract**: Every module must satisfy the three constraints (single input/output, channel-preserving, identity at init) to be wireable through `parse_model`

---

## Module Relationship Summary

### Input Flow (one image → predictions)

```
SAR Image
    │
    ├──→ [SIA: Module A] (if enabled, row 0 of backbone)
    │     ↓
    │   [CFAR Front End] (if enabled, alternative to SIA, row 0 of backbone)
    │     ↓
    │   YOLO11 Backbone: P1→P5 stages
    │     ↓
    │   FPN Top-Down pathway (upsample + add + C3k2)
    │     ↓
    │   Per-Level Chain (for p2/p3/p4/p5, if enabled):
    │   │   ├──→ AcquisitionConditionedAdapter (if conditioned)
    │   │   ├──→ TargetPriorModulation (Component 8, if enabled)
    │   │   ├──→ SARAdaptiveAttention (Component 3, if enabled, gate: adaptive/static)
    │   │   ├──→ ContextAggregation (Component 10, if enabled, mode: multi/local/regional/none)
    │   │   ├──→ TargetAwareRefinement (Component 11, if enabled, mode: deform/static/local/none)
    │   │   └──→ ScatterSelectiveRefinement (SSAC, if enabled, mode: adaptive/adaptive_raw/fixed, execution: dense/sparse)
    │     ↓
    │   Detect Head
    │     ↓
    │   SAR-Aware Loss (criterion, if sar_loss block present in YAML)
    │     ↓
    │   Predictions (bounding boxes + scores + classes)
    ↓
Output: annotated image + detection JSON
```

### Key Cross-Module Dependencies

| Module | Consumes From | Produces For |
|---|---|---|
| **CFAR front end** | Input image (row 0) | Every backbone row |
| **SIA (Module A)** | Input image (row 0) | Every backbone row |
| **SFE (Component 1)** | Previous backbone feature | SFM, SAA, AMF, per-level slots |
| **SFM (Component 2)** | Enhanced features from SFE | SAA, AMF, per-level slots |
| **SAA (Component 3)** | Previous level feature | AMF, per-level slots (prior, context, refinement, SSAC), detection head |
| **AMF (Component 4)** | Concatenated FPN features | Next C3k2 in FPN pathway; per-level slots |
| **TPM (Component 8)** | Previous level feature | Prior map → per-level slots (attention, context, refinement, SSAC), detection head, prior_map() for visualization |
| **SFR (Component 9)** | P5/32 feature | Top-down pathway, per-level slots |
| **CAG (Component 10)** | Previous level feature | Next per-level slot (refinement, SSAC) or detection head |
| **TADR (Component 11)** | Previous level feature | Detection head |
| **SSAC** | Previous level feature | Detection head |
| **_common blocks** (all modules) | Various | Every module in the graph |

### Mode Ablation Summary

Each module has mode ablations kept in the `VARIANTS` dict, where every arm keeps every other v2 slot at its setting. This ensures that observed differences are attributable to the module's mechanism alone:

- **SFE**: `pre_identity`, `pre_log`, `pre_clahe`, `pre_standardize`
- **SFM**: `speckle_none`, `speckle_lee`, `speckle_denoise`
- **SAA**: `att_none`, `att_se`, `att_eca`, `att_cbam`, `att_saa_static`
- **AMF**: `fus_concat`, `fus_add`, `fus_static`
- **CFAR**: implicit via `cfar`/`conv`/`fixed` modes in `VARIANTS` (pre_* arms keep CFAR off)
- **TPM**: `tp_none`, `tp_cfar`, `tp_static`, `tp_channel`, `tp_spectral`, `tp_spectral_feat`
- **SFR**: `fr_none`, `fr_highpass`, `fr_static`, `fr_dct`, `fr_wavelet`
- **CAG**: `cx_none`, `cx_local`, `cx_regional`
- **TADR**: `rf_none`, `rf_local`, `rf_static`, `rf_off25`, `rf_off100`
- **SIA**: `in_identity`, `in_local`, `in_learned`, `in_hybrid`
- **SSAC**: `SSAC-001` through `SSAC-006` (master workflow arms) + `adaptive`/`adaptive_raw`/`fixed` × `dense`/`sparse` mode/execution combos

### Parameter Count Summary (at scale s, c1=first-row width)

| Module | `identity` Additional params | Proposed arm additional params |
|---|---|---|
| **SIA** | 0 | hybrid: +164, learned: +98, local: +43 |
| **SFE** | 0 (identity mode) | sfe: parameters from gamma/beta + reduce/expand; other modes vary |
| **SFM** | 0 (none mode) | sfm: parameters from noise head + structure + beta; sfm_clutter: + gamma; lee/denoise: minimal |
| **SAA** | 0 (identity mode) | proposed: parameters from 3 branches + gate MLP; se/eca/cbam: much fewer; static: 3 logits |
| **AMF** | 0 (concat mode) | amf: parameters from MLP + DWConv + channel gate; static: 3 logits + same mix/gate; add: proj_in + proj_out; concat: 0 |
| **CFAR** | 0 (fixed mode, by design) | cfar: parameters from gain MLP (n_stats → hidden → 1); conv: same layer shapes, same params; fixed: 0 params |
| **TPM** | 0 (none mode) | learned: parameters from evidence head + band_head (spectral); channel: same network, pooled; static: 1x1 logits; cfar: logistic transfer params |
| **SFR** | 0 (none mode) | sff: parameters from modulate head; static: band_gain (C × bands); highpass: bands buffer; dct: band_gain + cached basis; wavelet: band_gain (C × 4); none: 0 |
| **CAG** | 0 (none mode) | multi/local: dilated depth-wise branches + mix conv + regional proj; regional: hidden bottleneck; none: 0 |
| **TADR** | 0 (none mode) | deform: offset conv (2-output); static: 2-element parameter; local: DWConv only; none: 0 |
| **SSAC** | 0 (identity at init) | adaptive: scorer MLP (head_in → hidden → 1), cheap rich paths; adaptive_raw: same scorer but raw feature input; fixed: same layer shapes, constant allocation; sparse: tile/keep/halo params |

### Identity-at-Init Verification

Every module asserts `max|f(x) - x| = 0.0e+00` at init. This is verified by the test suite:

- `test_each_module_is_exactly_identity_at_init`: measures the property for every module
- `test_no_module_is_frozen_at_init`: measures that every learnable mode has a non-zero gate gradient after 2 epochs of real training
- `test_models_output_identically_to_baseline_at_init`: max abs diff = 0.0 (v1 and v2)

The contract is: every residual gate starts at `alpha = 0` (via `ZeroGate`), so a freshly built SAR-YOLO is numerically identical to its YOLO11 baseline. Any later difference is attributable to training, not initialization.