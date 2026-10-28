# eta_v2_2024_queue: decomposed probabilistic ETA, features: queue

Train: arrivals before 2024-01-01; test: from 2024-01-01, 4193 voyages. Hours. v1 is retrained on the same split.

## Point forecast (v2 = median of the predicted distribution)

| model | mae_h | median_ae_h | voyage_mae_h | median_bias_h |
|---|---|---|---|---|
| naive | 85.8 | 24.2 | 13.4 | -14.9 |
| calibrated | 76.4 | 25.5 | 15.8 | 1.9 |
| v1 | 72.9 | 18.7 | 15.2 | -0.086 |
| v2 | 70.9 | 22.8 | 12.5 | -0.218 |

## Ships never seen in training

| model | mae_h | median_ae_h | voyage_mae_h | median_bias_h |
|---|---|---|---|---|
| naive | 93.0 | 33.7 | 16.0 | -27.1 |
| calibrated | 80.6 | 31.4 | 17.2 | 0.967 |
| v1 | 77.8 | 25.2 | 15.2 | -2.0 |
| v2 | 75.7 | 30.0 | 14.8 | -0.706 |

## Intervals (P10-P90, target coverage 0.80)

| interval | coverage_p10_p90 | below_p10 | above_p90 | median_width_h | pinball_h |
|---|---|---|---|---|---|
| calibrated x residual quantiles | 0.782 | 0.090 | 0.128 | 239.2 | 26.2 |
| v2 raw | 0.735 | 0.083 | 0.183 | 149.5 | 25.8 |
| v2 conformal | 0.749 | 0.072 | 0.180 | 157.1 | 25.6 |

## Intervals, ships never seen

| interval | coverage_p10_p90 | below_p10 | above_p90 | median_width_h | pinball_h |
|---|---|---|---|---|---|
| calibrated x residual quantiles | 0.785 | 0.079 | 0.136 | 245.2 | 27.3 |
| v2 raw | 0.728 | 0.078 | 0.193 | 163.1 | 27.1 |
| v2 conformal | 0.743 | 0.068 | 0.189 | 172.0 | 26.8 |

## By state at prediction time (MAE; P10-P90 coverage and median width, v2 vs baseline)

| slice | n | naive | calibrated | v1 | v2 | v2_coverage | v2_width_h | base_coverage | base_width_h |
|---|---|---|---|---|---|---|---|---|---|
| moving, < 30 km from gate | 136,723 | 15.8 | 16.3 | 16.5 | 14.1 | 0.832 | 1.7 | 0.477 | 14.9 |
| moving, >= 30 km | 593,670 | 21.5 | 22.5 | 19.1 | 17.9 | 0.787 | 16.0 | 0.895 | 72.8 |
| slow (0.5-5 kn) | 178,487 | 86.0 | 82.4 | 74.3 | 74.4 | 0.797 | 188.9 | 0.721 | 195.3 |
| stationary (< 0.5 kn) | 904,006 | 138.5 | 119.6 | 116.5 | 113.7 | 0.701 | 200.7 | 0.766 | 282.7 |

## Will the ship wait before arriving? (moving ships)

| auc | brier | brier_base_rate | test_rate | train_rate |
|---|---|---|---|---|
| 0.852 | 0.120 | 0.190 | 0.254 | 0.253 |

Reliability (predicted vs observed rate, 5 bins):

| bin | n | predicted | observed |
|---|---|---|---|
| 0.000 | 522,306 | 0.085 | 0.102 |
| 0.200 | 120,182 | 0.273 | 0.312 |
| 0.400 | 38,695 | 0.497 | 0.561 |
| 0.600 | 39,292 | 0.707 | 0.725 |
| 0.800 | 64,795 | 0.895 | 0.904 |
