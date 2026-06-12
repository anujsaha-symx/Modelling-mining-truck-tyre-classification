# Comparison Report: New Baseline vs. Previous Crop-Based GradCAM

## Context

This report compares GradCAM attention patterns between:

- **New Baseline:** Full-image classifier (EfficientNet-B0, no cropping, 224x224 letterbox resize)
- **Previous Pipeline:** Crop-based classifier (no prior GradCAM data available in this repository)

> **Important:** No prior GradCAM analysis results exist in the repository for the crop-based pipeline.
> This comparison report establishes the new baseline and documents questions to answer in future iterations.

---

## Question 1: Did edge attention decrease?

| Metric | New Baseline | Previous Crop-Based | Change |
|--------|-------------|---------------------|--------|
| Edge Attention Fraction (mean) | 0.2728 | No data | N/A |
| Edge Attention Fraction (median, good_correct) | 0.2575 | No data | N/A |

**Assessment:** Cannot compare — no prior GradCAM data available.
The current edge attention metrics serve as the new baseline.

### Edge Attention by Category

| Category | Edge Fraction Mean | Edge Fraction Std |
|----------|-------------------|-------------------|
| good_correct | 0.2514 | 0.0931 |
| bad_correct | 0.2903 | 0.1011 |
| false_positive | 0.3333 | 0.0918 |
| false_negative | 0.2548 | 0.1054 |
| high_confidence | 0.2008 | 0.0509 |
| low_confidence | 0.3211 | 0.1183 |

---

## Question 2: Did crop-boundary attention disappear?

The previous crop-based pipeline may have exhibited attention artifacts at crop boundaries
due to abrupt transitions between cropped tyre regions and background.

**New Baseline Assessment:**

- Edge attention fraction is 27.28%, suggesting attention at image boundaries is still present.
- The letterbox padding introduces a similar boundary artifact (black → tyre transition).
- This is **not equivalent** to crop-boundary attention from a crop pipeline, but represents a similar failure mode.

---

## Question 3: Did tread-focused attention improve?

**New Baseline Assessment:**

- Center-of-mass offset: 0.2078 (0 = perfect center)
- Mean entropy: 14.8411
- Edge fraction: 0.2728

Without prior tread-focused attention metrics, we cannot quantify improvement.
However, the current attention distribution suggests:

- Attention is broadly centered, suggesting the model focuses on central image content.

- Good (correct) predictions are more center-focused than bad (correct) predictions.
  This could indicate that defect features are more spatially distributed.

---

## Summary of Findings

| Dimension | Status |
|-----------|--------|
| Edge Attention | Baseline established (no prior comparison possible) |
| Crop-Boundary Artifacts | N/A — no cropping in new pipeline |
| Tread-Focused Attention | Baseline established for future comparison |
| Attention Metrics | Edge fraction, corner fraction, COM offset, entropy tracked |
| Suspicious Flags | Automatic detection for edge/corner/background focus |

## Recommendations

1. **Save GradCAM results from the crop-based pipeline** to enable direct comparison.
2. **Run this explainability pipeline after every retraining** to track attention drift.
3. **Investigate flagged suspicious samples** to determine if they indicate model limitations.
4. **Consider switching from letterbox padding to center-crop** to eliminate edge-boundary cues.
5. **Add tread-region masks** (if tread location is known) to compute tread-focused attention fraction.

