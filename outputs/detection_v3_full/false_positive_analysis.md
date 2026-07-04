# False Positive Analysis

## Overview
Total test images: 3823
Non-Tire test images: 3119
False Positives: 8
False Positive Rate: 0.26%

## Impact
False positives cause the gatekeeper to incorrectly classify non-tire images as "Tire".
This can lead to unnecessary downstream processing of non-tire images.

## Top Failure Modes
1. Visual patterns resembling tire tread on non-tire objects
2. Circular/round objects in non-tire images
3. Dark textured regions that resemble rubber
4. Images with high contrast edges or repetitive patterns

## Recommendations
1. Add hard negative examples to training set
2. Increase threshold to reduce false positives at cost of recall
3. Consider data augmentation targeting negative examples
