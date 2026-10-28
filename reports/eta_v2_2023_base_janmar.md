# eta_v2_2023_base_janmar: decomposed probabilistic ETA, features: base

Train: arrivals before 2023-04-01; test: from 2023-10-01, 954 voyages. Hours. v1 is retrained on the same split.

## Point forecast (v2 = median of the predicted distribution)

| model | mae_h | median_ae_h | voyage_mae_h | median_bias_h |
|---|---|---|---|---|
| naive | 66.1 | 18.8 | 12.6 | -8.9 |
| calibrated | 57.7 | 21.2 | 17.1 | 4.7 |
| v1 | 55.9 | 14.8 | 12.1 | 0.137 |
| v2 | 54.5 | 23.3 | 13.5 | 0.041 |

## Ships never seen in training

| model | mae_h | median_ae_h | voyage_mae_h | median_bias_h |
|---|---|---|---|---|
| naive | 79.6 | 27.9 | 14.9 | -19.8 |
| calibrated | 68.1 | 26.4 | 18.3 | 3.3 |
| v1 | 67.7 | 20.2 | 14.5 | -1.5 |
| v2 | 65.4 | 30.2 | 15.1 | -0.099 |

## Intervals (P10-P90, target coverage 0.80)

| interval | coverage_p10_p90 | below_p10 | above_p90 | median_width_h | pinball_h |
|---|---|---|---|---|---|
| calibrated x residual quantiles | 0.823 | 0.099 | 0.078 | 259.8 | 20.1 |
| v2 raw | 0.799 | 0.086 | 0.115 | 242.6 | 18.5 |
| v2 conformal | 0.857 | 0.091 | 0.052 | 761.2 | 24.9 |

## Intervals, ships never seen

| interval | coverage_p10_p90 | below_p10 | above_p90 | median_width_h | pinball_h |
|---|---|---|---|---|---|
| calibrated x residual quantiles | 0.817 | 0.088 | 0.096 | 263.3 | 23.4 |
| v2 raw | 0.778 | 0.081 | 0.141 | 246.5 | 22.2 |
| v2 conformal | 0.857 | 0.086 | 0.057 | 765.8 | 28.3 |

## By state at prediction time (MAE; P10-P90 coverage and median width, v2 vs baseline)

| slice | n | naive | calibrated | v1 | v2 | v2_coverage | v2_width_h | base_coverage | base_width_h |
|---|---|---|---|---|---|---|---|---|---|
| moving, < 30 km from gate | 31,527 | 13.5 | 14.4 | 12.8 | 12.3 | 0.835 | 16.0 | 0.514 | 20.0 |
| moving, >= 30 km | 139,846 | 17.6 | 21.5 | 16.6 | 15.6 | 0.811 | 23.4 | 0.927 | 102.1 |
| slow (0.5-5 kn) | 27,050 | 81.9 | 82.7 | 73.4 | 77.4 | 0.834 | 768.8 | 0.683 | 258.3 |
| stationary (< 0.5 kn) | 192,852 | 107.6 | 87.4 | 89.0 | 86.5 | 0.898 | 770.7 | 0.817 | 286.2 |

## Will the ship wait before arriving? (moving ships)

| auc | brier | brier_base_rate | test_rate | train_rate |
|---|---|---|---|---|
| 0.811 | 0.126 | 0.180 | 0.223 | 0.307 |

Reliability (predicted vs observed rate, 5 bins):

| bin | n | predicted | observed |
|---|---|---|---|
| 0.000 | 102,417 | 0.103 | 0.089 |
| 0.200 | 44,981 | 0.267 | 0.235 |
| 0.400 | 13,049 | 0.481 | 0.422 |
| 0.600 | 10,519 | 0.697 | 0.718 |
| 0.800 | 8,992 | 0.864 | 0.830 |
