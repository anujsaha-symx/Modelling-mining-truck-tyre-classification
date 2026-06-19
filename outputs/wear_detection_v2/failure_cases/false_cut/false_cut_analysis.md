# False Cut Analysis

Threshold: 0.001
Total false cut detections: 15

## Per-Image Details

| # | Filename | Max Cut Conf | Tire Dets | GT Classes |
|---|----------|-------------|----------|------------|
| 1 | good_1625.jpg | 0.0014 | 12 | Tire |
| 2 | good_1628.jpg | 0.0010 | 11 | Tire |
| 3 | good_290.jpg | 0.0025 | 16 | Tire |
| 4 | good_301.jpg | 0.0024 | 14 | Tire |
| 5 | good_549.jpg | 0.0020 | 9 | Tire |
| 6 | good_560.jpg | 0.0031 | 9 | Tire |
| 7 | good_781.jpg | 0.0050 | 14 | Tire |
| 8 | good_802.jpg | 0.0013 | 9 | Tire |
| 9 | negative_118.jpg | 0.0011 | 1 | Non-Tire |
| 10 | negative_132.jpg | 0.0012 | 2 | Non-Tire |
| 11 | negative_151.jpg | 0.0015 | 1 | Non-Tire |
| 12 | negative_156.jpg | 0.0025 | 2 | Non-Tire |
| 13 | negative_22.jpg | 0.0013 | 1 | Non-Tire |
| 14 | negative_62.jpg | 0.0016 | 2 | Non-Tire |
| 15 | negative_81.jpg | 0.0014 | 1 | Non-Tire |

## Analysis

False cut detections occur on:
1. **Clean tire images**: The model sometimes confuses tire tread patterns for cuts.
2. **Non-tire images**: Background patterns misclassified as cuts.

At threshold 0.001, there are false positives, but these can be reduced by raising the threshold.
