# 2024 out-of-time: static vs monthly re-calibrated intervals (P10-P90, target 0.80)

Model trained on 2023 only, never refit.

| intervals | rows | coverage_p10_p90 | below_p10 | above_p90 | median_width_h | pinball_h |
|---|---|---|---|---|---|---|
| static (2023 held-out ships) | all | 0.750 | 0.079 | 0.171 | 160.6 | 24.7 |
| static (2023 held-out ships) | moving | 0.787 | 0.081 | 0.132 | 18.4 | 7.3 |
| static (2023 held-out ships) | waiting | 0.722 | 0.078 | 0.201 | 203.1 | 38.2 |
| rolling (previous month) | all | 0.787 | 0.100 | 0.114 | 189.9 | 24.1 |
| rolling (previous month) | moving | 0.801 | 0.097 | 0.102 | 21.4 | 7.3 |
| rolling (previous month) | waiting | 0.775 | 0.101 | 0.123 | 319.1 | 37.2 |

## Coverage by month

| month | static | rolling |
|---|---|---|
| 1 | 0.643 | 0.643 |
| 2 | 0.726 | 0.782 |
| 3 | 0.726 | 0.833 |
| 4 | 0.713 | 0.757 |
| 5 | 0.769 | 0.842 |
| 6 | 0.727 | 0.786 |
| 7 | 0.749 | 0.805 |
| 8 | 0.811 | 0.840 |
| 9 | 0.771 | 0.808 |
| 10 | 0.835 | 0.847 |
| 11 | 0.793 | 0.763 |
| 12 | 0.812 | 0.770 |
