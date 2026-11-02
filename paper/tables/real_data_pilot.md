**Pilot on a real HRSID subset: the official release, YOLO boxes, 200 train / 60 val / 60 test images, 40 epochs at 320 px, single seed, CPU only. Real and measured, but the release is 5,604 images -- this is not the benchmark, and a row here is not comparable with a full-release row. For REAL-003 the adapter is folded into the base weights when the checkpoint is written, so its parameter count equals the baseline's while what was optimised (a rank-8 adapter plus the head) is recorded separately in the ledger.**

| Model | mAP50 | mAP50:95 | Precision | Recall | Params (M) | GFLOPs@320 | FPS | Train (min) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| YOLO11n baseline -- HRSID subset | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| SARVO-Lite (s) -- HRSID subset | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| YOLO11n + LoRA r=8 -- HRSID subset | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| SARVO prototype: CFAR front end -- HRSID subset | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Prototype control: CFAR, fixed threshold -- HRSID subset | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Prototype control: matched-cost conv stem -- HRSID subset | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Baseline, SAR-augmented training (x1) -- HRSID subset | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Prototype, SAR-augmented training (x1) -- HRSID subset | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Baseline, SAR-augmented training (x2) -- HRSID subset | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Prototype, SAR-augmented training (x2) -- HRSID subset | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

`TBD` = not yet measured (run the corresponding experiment).
