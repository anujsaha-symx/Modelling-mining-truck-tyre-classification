# Gatekeeper V3 Training Summary (Full)

## Dataset
- Total images: 19109
- Train: 13376 | Val: 1910 | Test: 3823
- Classes: Background (0), Tire (1), Non-Tire (2)
- Dataset variant: full

## Model
- Architecture: Faster R-CNN with MobileNetV3-Large-FPN
- Pretrained: COCO
- Backbone: Fine-tuned (unfrozen)

## Training
- Epochs trained: 12
- Batch size: 16
- Optimizer: AdamW (lr=0.0001, weight_decay=0.0001)
- Scheduler: CosineAnnealingLR
- Gradient clip: 1.0
- Mixed precision: Yes
- Best checkpoint selection: Validation F1
- Best val F1: 0.9972

## Gatekeeper Metrics (Test Set)
| Metric | Value |
|--------|-------|
| Precision | 0.9887 |
| Recall (Tire Recall) | 0.9972 |
| F1-Score | 0.9929 |
| Accuracy | 0.9974 |

## Per-Class Metrics
| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| Tire | 0.9874 | 0.9986 | 0.9929 |
| Non-Tire | 0.9994 | 0.9978 | 0.9986 |

## Confusion Matrix
| | Predicted Non-Tire | Predicted Tire |
|---|---|---|
| **Actual Non-Tire** | 3111 | 8 |
| **Actual Tire** | 2 | 702 |
