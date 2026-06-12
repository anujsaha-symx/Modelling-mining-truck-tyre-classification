# GradCAM Explainability Summary

## Overview

- **Date:** Auto-generated
- **Model:** EfficientNet-B0 (full-image classifier, no cropping)
- **Test Samples:** 529
- **GradCAM Target:** Last Conv2d in `model.features` (7x7 spatial resolution)
- **GradCAM Samples per Category:** {'good_correct': 20, 'bad_correct': 20, 'false_positive': 15, 'false_negative': 20, 'high_confidence': 20, 'low_confidence': 20}

## Positive Findings

1. **Reasonable attention spread.** Mean entropy across all categories indicates the model is not focusing on a single pixel.
   - Global mean entropy: 14.841125

2. **Spatial attention diversity.** The center-of-mass offset varies across categories, suggesting the model adapts its attention region.
   - Global mean COM offset: 0.207779 (0 = center, 1 = edge)

3. **Most predictions are well-calibrated.** High-confidence correct predictions show focused attention on task-relevant regions.

## Remaining Biases

1. **Edge bias detected.** Average edge attention fraction is 27.28%, suggesting the model partially relies on letterbox padding boundaries or image edges.

2. **Corner bias minimal.** Average corner attention fraction is 7.73%.

3. **16 flaggings across 13 unique samples** (edge-focused, corner-focused, or background-focused). See `suspicious_attention/` for individual cases.

4. **False positive vs. false negative attention patterns.**
   - False positives (good predicted as bad): edge fraction = 33.33%
   - False negatives (bad predicted as good): edge fraction = 25.48%
   - Correct good: edge fraction = 25.14%
   - Correct bad: edge fraction = 29.03%

   - False positives show higher edge attention than correct goods, suggesting background/edge cues may trigger false alarms.

## Deployment Implications

1. **Edge/corner attention suggests letterbox padding may influence predictions.**
   - Consider center-cropping instead of letterbox padding in future iterations.
   - Alternatively, pad with image-mean color instead of black to reduce boundary contrast.

2. **False negative attention patterns should be investigated per-image.**
   - Check whether the model is looking at the correct region but failing to identify defects,
     or looking at irrelevant regions entirely.

3. **GradCAM provides spatial attribution, not causal evidence.**
   - High attention in a region means the model weights that region for its prediction,
     not that the region contains the actual defect.

4. **Recommend periodic GradCAM monitoring** after each retraining cycle to track attention shifts.

## Limitations

- GradCAM resolution is limited to 7x7 (EfficientNet-B0 feature map size), then upsampled to 224x224.
- Attention metrics are computed on the upsampled heatmap and may introduce interpolation artifacts.
- Thresholds for suspicious flags (edge > 40%, corner > 25%, COM > 50%) are heuristic and may need tuning.
- This analysis covers a subset of test samples per category (up to 20 each), not the full test set.

