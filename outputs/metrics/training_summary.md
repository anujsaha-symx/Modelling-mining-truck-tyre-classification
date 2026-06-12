# Training Summary

## Dataset Statistics

- Dataset root: `datasets/processed`
- Total samples: 3521
- Train samples: 2464
- Validation samples: 528
- Test samples: 529
- Class distribution was preserved with stratified splitting and overlap verification.

## Training Configuration

- Model: EfficientNet-B0
- Input pipeline: aspect-ratio preserving resize to `224x224` with centered black padding
- Train augmentations: horizontal flip, small rotation, brightness/contrast jitter, color jitter, gaussian blur
- Loss: weighted cross entropy
- Optimizer: AdamW
- Scheduler: ReduceLROnPlateau
- Seed: 42
- Batch size: 32
- Initial learning rate: 0.001
- Weight decay: 0.0001
- Fine-tune flag: enabled
- Planned unfreeze epoch: 3

## Best Validation Metrics

- Best epoch: 1
- Accuracy: 0.8674
- Precision: 0.9035
- Recall: 0.8387
- F1-score: 0.8699
- ROC-AUC: 0.9347
- Average Precision: 0.9345

## Test Set Results

- Accuracy: 0.8922
- Precision: 0.9407
- Recall: 0.8500
- F1-score: 0.8931
- ROC-AUC: 0.9600
- Average Precision: 0.9662

## Confusion Matrix Interpretation

- True negatives (good -> good): 234
- False positives (good -> bad): 15
- False negatives (bad -> good): 42
- True positives (bad -> bad): 238
- The current baseline is conservative on `good` tyres and misses more `bad` tyres than it over-flags `good` tyres.

## Failure Analysis Summary

- Failure cases saved to `outputs/predictions/failure_cases/`
- False positives: 15
- False negatives: 42
- Failure analysis CSV: `outputs/predictions/failure_cases/failure_analysis.csv`
- Most errors are false negatives on damaged tyres, which is the main area to improve in later iterations.

## Notes

- This baseline uses only image classification on the processed dataset.
- No YOLO, cropping, severity estimation, GradCAM, or prior detection pipeline was used.
