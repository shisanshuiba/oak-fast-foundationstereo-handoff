# Report source and QA notes

- Audience: technical
- Delivery mode: portable self-contained HTML
- Main evaluation population: OAK `valid_mask > 0`, finite positive reference depth, `0.2 <= depth <= 2.0 m`
- Prediction-invalid policy: the saved `0 m` value remains in the main denominator and is scored as a failure
- Aggregation: pixel-weighted across the two reviewed frames

## Source inventory

- `output/oak_test/depth_meter.npy`
- `output/pseudocolor_smoke_00000034/depth_meter.npy`
- `FastFoundationStereo_OAK_Dataset/test/20260819_170600/{depth_mm,valid_mask}/00000032.png`
- `FastFoundationStereo_OAK_Dataset/train/20260819_172516/{depth_mm,valid_mask}/00000034.png`
- `output/depth_evaluation_report/depth_evaluation_analysis.py`
- `output/depth_evaluation_report/depth_evaluation_analysis.ipynb`

## Required technical-report structure mapping

1. Title: artifact title block
2. Technical summary: `技术摘要`
3. Key findings with visual evidence: error distribution, per-frame and depth-band sections, three qualitative figures
4. Scope, data and metric definitions: `评测回答的是与 OAK StereoDepth 的一致性`
5. Methodology: `两帧以统一尺度、单位和分母计算`
6. Limitations and robustness: erosion/full-range comparison and explicit limitations
7. Recommended next steps: independent ground truth and broader sampling plan
8. Further questions: three decision-relevant open questions

## Chart map

| Report section | Question | Family/type | Dataset | Supported claim |
|---|---|---|---|---|
| Error shape | How heavy is the long tail? | Distribution / histogram | `error_bins` | Half the pixels are within 5 mm while 8.90% exceed 10 cm |
| Frame stability | Is performance consistent across the two runs? | Comparison / bar | `frames` | Frame 00000034 has higher MAE despite a lower median error |
| Distance behavior | Does error change by reference distance? | Comparison / bar | `work_depth_bands` | Bias reverses between 0.5–1 m and 1–2 m |

A line/trend chart was intentionally omitted: two frames do not provide enough ordered observations for an honest trend. Exact values are retained in tables.

## QA record

- Prediction/source-image alignment: exact after the inference resize operation for both frames
- Analysis script: syntax check and complete execution passed
- Companion notebook: executed top-to-bottom; all code cells completed without error
- Artifact validation: passed
- Portable packaging: passed
- Portable verification: structural-only; the packaged verifier could not find a compatible Chromium headless-shell
- One targeted attempt with installed Chrome was rejected for chart-extraction environment mismatch, so no further browser retries or downloads were performed
