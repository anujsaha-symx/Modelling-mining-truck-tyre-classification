# Missed Cut Analysis

Threshold: 0.001
Total missed cuts: 21 out of 43 bad tyres
Cut Recall: 0.5116

## Per-Image Details

| # | Filename | Tire Conf (max) | Cut Dets (count) | GT Classes |
|---|----------|----------------|-----------------|------------|
| 1 | bad_103.jpg | 0.9872 | 0 | Tire, Cut |
| 2 | bad_126.jpg | 0.9944 | 0 | Tire, Cut |
| 3 | bad_133.jpg | 0.9810 | 0 | Tire, Cut |
| 4 | bad_140.jpg | 0.9900 | 0 | Tire, Cut |
| 5 | bad_160.jpg | 0.9920 | 0 | Tire, Cut |
| 6 | bad_161.jpg | 0.9981 | 0 | Tire, Cut |
| 7 | bad_165.jpg | 0.9979 | 0 | Tire, Cut |
| 8 | bad_168.jpg | 0.9886 | 0 | Tire, Cut |
| 9 | bad_184.jpg | 0.9893 | 0 | Tire, Cut |
| 10 | bad_189.jpg | 0.9926 | 0 | Tire, Cut |
| 11 | bad_190.jpg | 0.9951 | 0 | Tire, Cut |
| 12 | bad_193.jpg | 0.9968 | 0 | Tire, Cut |
| 13 | bad_2.jpg | 0.9917 | 0 | Tire, Cut |
| 14 | bad_39.jpg | 0.9831 | 0 | Tire, Cut |
| 15 | bad_44.jpg | 0.9868 | 0 | Tire, Cut |
| 16 | bad_49.jpg | 0.9870 | 0 | Tire, Cut |
| 17 | bad_50.jpg | 0.9612 | 0 | Tire, Cut |
| 18 | bad_60.jpg | 0.9760 | 0 | Tire, Cut |
| 19 | bad_64.jpg | 0.9976 | 0 | Tire, Cut |
| 20 | bad_66.jpg | 0.9934 | 0 | Tire, Cut |
| 21 | bad_71.jpg | 0.9874 | 0 | Tire, Cut |

## Analysis

Missed cuts fall into two categories:
1. **No cut proposal at all**: The RPN did not generate proposals in regions that could be cuts.
2. **Cut classified as Tire/Background**: The classifier assigned higher probability to Tire than Cut.

The model was trained for only 3 epochs with 200 cut annotations across 1410 training images.
With additional training, cut confidence would increase and recall would improve.
