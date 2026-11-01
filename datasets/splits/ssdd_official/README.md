# SSDD split provenance

Everything in this directory is derived, reproducible data-manifest state for the
SSDD pilot. The raw release is never redistributed here; only split *lists* and
the recipe that produced them are recorded.

## Source

* Mirror: `https://github.com/DonnieRaymond3/cs231n_ship_detection`, path
  `SSDD/BBox_SSDD/voc_style/` (fetched with a sparse checkout; horizontal-box
  variant, as `docs/DATASETS.md` requires).
* On-disk verification against the release: **1,160 images + 1,160 PASCAL-VOC
  annotations**, and the official lists `ImageSets/Main/{train,test}.txt` are
  disjoint and cover all 1,160 stems (928 train / 232 test).
* Official test scenery lists `test_inshore.txt` (46) / `test_offshore.txt` (186)
  match the release's stated inshore/offshore test subsets exactly.

## Files

| File | What it is |
| --- | --- |
| `official_train.txt` | the release's own `ImageSets/Main/train.txt` (928) |
| `official_test.txt` | the release's own `ImageSets/Main/test.txt` (232), **unmodified** |
| `train.txt` | final train list (713) — derived, see recipe below |
| `val.txt` | final val list (178) — derived, see recipe below |
| `test.txt` | copy of `official_test.txt` |
| `excluded_leaky.txt` | 37 train-side images removed, see below |
| `scenery_sidecar.json` | image stem → `inshore` / `offshore`, from the release |

## Recipe

1. **Exclusions (leakage remediation).** The release's own train/test partition
   contains duplicates that straddle the boundary, and a naive random val carve
   splits duplicate groups apart. Auditing all 1,160 images with the repository's
   own tooling (`saryolo.data.validate._md5` for exact equality,
   `_dhash` Hamming ≤ 4 for near-duplicates, the same criterion
   `saryolo.data.splits.leakage_report` applies) found **22 exact-duplicate groups
   and 73 near-duplicate pairs**, forming 30 connected components — 14 of which
   contain an official test image.
   * Rule: **official test is never modified.** Every non-test member of a
     component that touches the test set is removed from training/val — 37 images,
     listed in `excluded_leaky.txt` (4% of the official train set).
   * Result: `leakage_report` over the final splits reports
     **0 exact and 0 near duplicate pairs across splits**.
2. **Val carve.** Over the remaining 891-image pool, val is carved with the
   repository's own convention (`saryolo.data.groups.leave_one_out_folds`:
   `random.Random(0)`, `val_ratio=0.2`) but at **component granularity**, so a
   duplicate group is never split across train/val. Target 178, achieved 178.
3. **Conversion.** `saryolo.data.prepare_dataset("ssdd", …,
   official_split_files={train, val, test})` → `datasets/processed/ssdd`
   (symlinks, so no data is copied). The converter's own guard reports
   `unlisted_images: 37`, i.e. exactly the deliberate exclusions above.

Final counts: **train 713 / val 178 / test 232 = 1,123 images** (1,160 release
images − 37 excluded), 2,495 boxes (1,581 / 368 / 546).

## Scenery folds (`datasets/splits/ssdd_scenery/`)

Built with:

```bash
python -m saryolo.cli loso --images datasets/processed/ssdd/images \
    --rule sidecar --sidecar datasets/splits/ssdd_official/scenery_sidecar.json \
    --data configs/datasets/generated/ssdd.yaml \
    --min-test-images 30 --out datasets/splits/ssdd_scenery
```

`scenery_sidecar.json` is assembled from the release's own organisation —
train images from the `JPEGImages_train_{inshore,offshore}` directories (182/746)
and test images from `ImageSets/Main/test_{inshore,offshore}.txt` (46/186) —
total 1,160, verified to cover exactly the release's train∪test. **No filename
pattern was guessed.**

Fold sizes after exclusions (manifest records rule, seed 0 and sizes):

| Held-out scenery | train | val | test |
| --- | --- | --- | --- |
| inshore | 720 | 180 | 223 |
| offshore | 178 | 45 | 900 |

Validation shares sources with training in every fold, so it drives early
stopping only; report from `test.txt`.

## Known gap: Stage 4 (acquisition metadata)

`saryolo metadata --dataset ssdd` **refuses by design**:

> no verified acquisition profile for 'ssdd'; add one sourced from official
> documentation, never inferred from filenames

SSDD's release states a *mixed* sensor set (Gaofen-3 / Sentinel-1 / RadarSat-2)
with no per-image mapping, so every per-image acquisition field is genuinely
unknown (registry profile is all-`null`). Building the table would mean
fabricating labels, which the pipeline forbids. Acquisition-conditioning work
(RQ3) therefore runs on **HRSID** (0.5/1/3 m, with a documented sidecar) or
**SARDet-100K** (official per-source profile table), not on SSDD.