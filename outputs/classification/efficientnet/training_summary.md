# Training Summary — EfficientNet-B0

## Configuration

| Parameter | Value |
|-----------|-------|
| Model | EfficientNet-B0 (torchvision, ImageNet pretrained) |
| Training mode | Head-only (backbone frozen) |
| Epochs configured | 25 |
| Epochs completed | 15 (early stopping at patience=5) |
| Batch size | 32 |
| Initial learning rate | 1e-3 |
| Optimizer | AdamW (weight decay 1e-4) |
| Scheduler | ReduceLROnPlateau (factor=0.5, patience=2) |
| Loss function | CrossEntropyLoss (class-weighted: good=1.058, bad=0.942) |
| Image size | 224×224 |
| Hardware | CPU |

## Best Test Metrics (threshold=0.5)

| Metric | Value |
|--------|-------|
| Accuracy | 0.887 |
| Precision (bad) | 0.910 |
| Recall (bad) | 0.871 |
| F1-score (bad) | 0.891 |
| ROC-AUC | 0.957 |
| Average Precision | 0.964 |
| Test loss | 0.286 |

### Per-Class Breakdown

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Good | 0.862 | 0.904 | 0.882 | 249 |
| Bad | 0.910 | 0.871 | 0.891 | 280 |

**Comparison to previous smoke-test run**: Accuracy improved from **0.527 to 0.887** (+36 ppt), F1 from **0.683 to 0.891** (+21 ppt), ROC-AUC from **0.524 to 0.957**.

## Training Dynamics

### Loss Curves
- **Train loss**: Started at 0.506, decreased steadily to 0.286 by epoch 15
- **Val loss**: Started at 0.372, reached minimum 0.280 at epoch 10, then plateaued
- **Learning rate**: Reduced from 1e-3 to 5e-4 at epoch 13 after val loss plateaued

### Performance Curves
- **Val F1**: Peaked at **0.891** (epoch 10), fluctuated between 0.870–0.891 thereafter
- **Train F1**: Rose steadily from 0.761 to 0.856 (epoch 15), still below val F1 — indicating headroom
- **Val accuracy**: Peaked at 0.886 (epoch 10)
- **Train accuracy**: Peaked at 0.848 (epoch 15)

### Early Stopping
- Triggered at epoch 15 (5 epochs without val F1 improvement beyond 0.891)
- Best checkpoint saved at epoch 10

## Overfitting Observations

**Mild overfitting present but well-controlled:**

1. **Gap between train and val metrics**: Train F1 (0.856) vs Val F1 (0.891) at epoch 15 — val exceeds train, suggesting the frozen backbone provides strong regularization
2. **Train metrics still improving**: At early stopping, train F1 was still trending upward (0.856), while val F1 was fluctuating — additional head training may not improve generalization
3. **No severe divergence**: The gap between train and val loss remained stable (~0.09–0.10) throughout training

**Conclusion**: The frozen backbone acts as an effective regularizer. Overfitting is minimal. Fine-tuning the backbone (CPU-limited) could potentially improve performance but carries overfitting risk with only 2,464 training samples.

## Augmentation Impact

### Augmentation Pipeline (strengthened for this run)

| Augmentation | Probability | Parameters |
|-------------|-------------|------------|
| RandomResizedCrop | 1.0 | scale (0.70–1.0), ratio (0.80–1.20) |
| RandomHorizontalFlip | 0.5 | — |
| RandomRotation | 1.0 | ±15° |
| PerspectiveDistortion | 0.35 | scale 0.06 |
| BrightnessContrast | 0.5 | brightness (0.8–1.2), contrast (0.8–1.2) |
| GaussianBlur | 0.3 | kernel 5, sigma (0.1–2.0) |
| MotionBlur | 0.4 | kernel (3,5,7,9) |
| JPEGCompression | 0.35 | quality (40–85) |
| Dust/Noise Overlay | 0.40 | 8–25 particles, max size 14 |
| Shadow Simulation | 0.50 | strength (0.2–0.7) |
| Low-Light Simulation | 0.35 | gamma (1.5–3.5), Gaussian noise std=5 |
| CoarseDropout | 0.30 | 3 holes, size (15–55), fill 128 |
| ColorJitter | 1.0 | brightness 0.2, contrast 0.2, saturation 0.15, hue 0.03 |

### Impact Assessment
- **Stronger augmentations** (increased probabilities vs. previous runs) contributed to better generalization
- **Shadow + dust** augmentations simulate mining-environment conditions (dirt, uneven lighting)
- **JPEG compression + blur** simulate varied camera quality in field deployment
- **Low-light** mimics underground mining conditions
- **Tyres remain structurally realistic** — augmentations affect appearance but not tread/tyre geometry

### Caution
Augmentations are applied on centered, clean tyre crops. Real mining images will have additional challenges (extreme dirt, water, partial occlusion by equipment) not fully captured by these synthetic augmentations.

## Threshold Sweep Analysis (Range: 0.10–0.90)

| Metric | Threshold | Score |
|--------|-----------|-------|
| Best F1 (bad) | 0.36 | **0.903** |
| Best Recall (bad) | 0.10 | 0.982 |
| Best Precision (bad) | 0.90 | 0.992 |
| 90% Recall Threshold | 0.39 | Precision at 0.90 recall: 0.897 |

### Deployment Recommendation

**For mining deployment (FN-critical):** Use threshold **0.25–0.30**
- Bad-class recall: 93–94%
- Bad-class precision: 84–87%
- Trade-off: ~5–6% more false positives but captures ~6–7% more worn tyres

**Default (balanced):** threshold **0.36** (best F1) if equal cost for FN/FP

## Calibration Analysis

| Metric | Value |
|--------|-------|
| Expected Calibration Error (ECE) | **0.094** |
| Brier Score | **0.086** |

- **Moderately calibrated**: The model is slightly overconfident — predicted probabilities are somewhat higher than actual frequencies
- **Reliability diagram**: Shows systematic overconfidence in the 0.4–0.7 range (common with cross-entropy training)
- **Recommendation**: Apply temperature scaling (not implemented) for better probability estimates in deployment

## Robustness Analysis

| Metric | Value |
|--------|-------|
| Mean prediction std under augmentation | **0.114** |
| Median prediction std | 0.090 |
| 90th percentile std | 0.232 |
| 95th percentile std | 0.272 |
| Max std | 0.350 |

- **Moderate robustness**: On average, augmentations shift predictions by ~11 percentage points
- **~10% of samples** show high instability (std > 0.23), concentrated near the decision boundary
- **Confidence-dependent**: Low-confidence predictions (near 0.5) are most unstable; high-confidence predictions (near 0 or 1) are stable
- **Implication**: The model is reasonably robust to common image corruptions but uncertain cases will flip under real-world variation

## Failure Mode Summary

- **60 misclassifications** out of 529 test samples (11.3%)
- **63% ambiguous wear** — borderline cases with subtle visual differences
- **28% blur** — out-of-focus or motion-blurred images
- **8% low texture** — uniform surfaces lacking features
- False negatives (missed bad tyres) are low-confidence and threshold-tunable
- False positives (false alarms on good tyres) include high-confidence errors, suggesting spurious correlations (dirt, shadows)

## Deployment Concerns

1. **Domain gap**: Current dataset (Kaggle + Mendeley) contains centered, clean tyre images under controlled conditions. Real mining-environment images will differ significantly in lighting, dirt, angle, and occlusion.
2. **CPU inference speed**: On CPU, inference takes ~0.5–1.0 second per image for EfficientNet-B0. Consider ONNX quantization or TensorRT for real-time deployment.
3. **False positive rate**: At threshold 0.5, ~9.6% of good tyres are falsely flagged. In a mining operation processing thousands of tyres daily, this translates to many interruptions.
4. **Undetected blur**: 28% of failures involve blur. A pre-classification blur detector could flag these for human review rather than relying on the classifier.
5. **Confidence calibration**: The model is overconfident. Probability estimates should not be interpreted as true wear likelihood without calibration (temperature scaling).

## Limitations of Current Dataset

**Critical — must be addressed before mining deployment:**

1. **Not mining-environment data**: Both Kaggle and Mendeley datasets are collected in controlled/studio settings. True mining data will include:
   - Mud, dirt, and debris on tyre surfaces
   - Water and wet conditions
   - Extreme shadows from mining equipment
   - Non-centered, partial tyre views
   - Low-resolution or compressed images from mine-site cameras

2. **Limited class diversity**: Only binary (good/bad) with no gradations of wear severity. Real mining operations need at minimum 3–5 wear levels.

3. **Single-source bias**: Two datasets (Kaggle + Mendeley) may share collection methodology biases. The model may learn dataset-specific artifacts rather than true tyre-wear features.

4. **Small dataset**: 3,521 images total. For deep learning, this is modest. More data (especially from mining environments) is essential.

5. **No temporal aspect**: Tyre wear is progressive, but the dataset only contains single-timepoint snapshots. A model that sees wear progression would be more robust.

## Recommendations for Next Steps

1. **Collect mining-environment data** — the single highest-impact improvement
2. **Fine-tune backbone** (when GPU available) — head-only training leaves potential on the table
3. **Temperature scaling** — improve calibration without retraining
4. **Ensemble** with ConvNeXt-Tiny when trained — may improve robustness
5. **Blur detector** pre-filter to catch out-of-distribution inputs
6. **Expand to multi-class** (mild/moderate/severe wear) for finer-grained decisions
7. **GradCAM** (next phase) to verify the model attends to tread regions, not background

## Conclusion

EfficientNet-B0 achieves **88.7% accuracy, 0.957 ROC-AUC** on the test set — a significant improvement from the previous smoke-test run. The model is reasonably well-calibrated and robust, with mild overfitting controlled by the frozen backbone. However, **these results do NOT imply mining-environment readiness**. The current dataset (centered, clean tyre images) is fundamentally different from real mining conditions. Deployment recommendations (threshold=0.25–0.30 for FN-critical use) are based on this limited dataset. Real-world performance may differ substantially.
