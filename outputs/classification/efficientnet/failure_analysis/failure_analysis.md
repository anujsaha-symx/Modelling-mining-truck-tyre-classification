# Failure Analysis Report — EfficientNet-B0

## Overview

- **Model**: EfficientNet-B0 (head-only training, frozen backbone)
- **Test samples**: 529 (249 good, 280 bad)
- **Total misclassifications**: 60 (11.3% error rate)
- **Threshold**: 0.5

## Failure Distribution by Category

| Category | Count | % of Failures | Description |
|----------|-------|---------------|-------------|
| Blur | 17 | 28.3% | Low Laplacian variance (< 50) — out-of-focus or motion-blurred images |
| Low Texture | 5 | 8.3% | Low pixel std dev — uniform surfaces lacking discriminative features |
| Ambiguous Wear | 38 | 63.3% | No clear visual defect — wear boundary cases inherently hard to classify |
| Lighting | 0 | 0.0% | No extreme lighting failures detected |
| Side Angle | 0 | 0.0% | Not detected in current dataset |
| Background Clutter | 0 | 0.0% | Not detected in current dataset |

## Per-Class Failure Analysis

### False Negatives (Bad → Good): 36 cases

The model misses worn tyres most often when:

1. **Blur (12 cases)**: Motion blur or poor focus obscures tread wear patterns. These images would likely also confuse human inspectors.
2. **Ambiguous Wear (23 cases)**: Tyres with borderline wear levels where the visual difference from "good" is subtle. The model assigns low bad-class probabilities (typically 0.01–0.35), indicating low confidence rather than confident errors.
3. **Low Texture (1 case)**: Uniform surface with minimal tread variation.

**Key observation**: False negatives are predominantly low-confidence predictions. Only 3 of 36 FN cases had bad-confidence > 0.45 (near the decision boundary). This means a threshold adjustment can capture most of these.

### False Positives (Good → Bad): 24 cases

The model falsely flags good tyres as worn primarily when:

1. **Blur (5 cases)**: Blur on a good tyre creates artifacts that resemble wear patterns.
2. **Ambiguous Wear (15 cases)**: Good tyres with visual features (dirt, lighting shadows, mould marks) that the model interprets as wear. These are **high-confidence errors** — many with bad-confidence > 0.6, some > 0.8.
3. **Low Texture (4 cases)**: Good tyres with uniformly smooth surfaces.

**Key observation**: False positives tend to be higher confidence. The model's top-5 most confident errors are all false positives on good tyres (bad-confidence: 0.90, 0.82, 0.80, 0.77, 0.72). This suggests the model sometimes relies on spurious correlations (e.g., dirt, casting shadows) rather than true wear patterns.

## Low-Confidence Failure Analysis

The 20 lowest-confidence failures (closest to threshold 0.5) were saved to `confidence_analysis/low_confidence_examples/`. These represent cases where the model was most uncertain:

- **15 False Negatives** (bad tyres predicted good, conf 0.45–0.50)
- **5 False Positives** (good tyres predicted bad, conf 0.50–0.54)

These are the cases most likely to be corrected by threshold tuning or additional training data.

## Confusion Matrix (threshold=0.5)

```
              Predicted
              Good   Bad
True Good     225    24
True Bad       36   244
```

- Sensitivity (Recall, Bad): 87.1%
- Specificity (Good): 90.4%
- Positive Predictive Value (Precision, Bad): 91.0%
- Negative Predictive Value (Good): 86.2%

## Recommendations for FN Reduction

1. **Lower threshold to 0.36** (best F1 threshold): Increases bad recall to ~91.1% while maintaining F1=0.903
2. **Lower threshold to 0.10**: Achieves 98.2% recall on bad class, but precision drops to 74.7%
3. **For mining deployment (FN-critical)**: Threshold of **0.25–0.30** balances recall (~93–94%) with acceptable precision (~84–87%)

## Limitations

- Current dataset contains centered, clean tyre images — not true mining-environment data
- Blur category may be over-counted since single-image Laplacian variance is a noisy proxy for actual blur
- "Ambiguous wear" is a catch-all; many of these may actually belong to fine-grained subcategories that require expert labeling to distinguish
- No side-angle or extreme-lighting failures present in test set, but these will appear in real mining deployment
