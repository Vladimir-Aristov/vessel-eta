# eta_v2_2024_base: decomposed probabilistic ETA, features: base

Train: arrivals before 2024-01-01; test: from 2024-01-01, 4193 voyages. Hours. v1 is retrained on the same split.

## Point forecast (v2 = median of the predicted distribution)

| model | mae_h | median_ae_h | voyage_mae_h | median_bias_h |
|---|---|---|---|---|
| naive | 85.8 | 24.2 | 13.4 | -14.9 |
| calibrated | 76.4 | 25.5 | 15.8 | 1.9 |
| v1 | 72.9 | 18.7 | 15.2 | -0.086 |
| v2 | 70.5 | 23.0 | 12.6 | -0.171 |

## Ships never seen in training

| model | mae_h | median_ae_h | voyage_mae_h | median_bias_h |
|---|---|---|---|---|
| naive | 93.0 | 33.7 | 16.0 | -27.1 |
| calibrated | 80.6 | 31.4 | 17.2 | 0.967 |
| v1 | 77.8 | 25.2 | 15.2 | -2.0 |
| v2 | 75.2 | 30.4 | 15.0 | -0.574 |

## Intervals (P10-P90, target coverage 0.80)

| interval | coverage_p10_p90 | below_p10 | above_p90 | median_width_h | pinball_h |
|---|---|---|---|---|---|
| calibrated x residual quantiles | 0.782 | 0.090 | 0.128 | 239.2 | 26.2 |
| v2 raw | 0.737 | 0.089 | 0.174 | 158.5 | 25.5 |
| v2 conformal | 0.743 | 0.079 | 0.178 | 162.6 | 25.4 |

## Intervals, ships never seen

| interval | coverage_p10_p90 | below_p10 | above_p90 | median_width_h | pinball_h |
|---|---|---|---|---|---|
| calibrated x residual quantiles | 0.785 | 0.079 | 0.136 | 245.2 | 27.3 |
| v2 raw | 0.732 | 0.085 | 0.183 | 170.9 | 26.7 |
| v2 conformal | 0.740 | 0.075 | 0.185 | 175.7 | 26.6 |

## By state at prediction time (MAE; P10-P90 coverage and median width, v2 vs baseline)

| slice | n | naive | calibrated | v1 | v2 | v2_coverage | v2_width_h | base_coverage | base_width_h |
|---|---|---|---|---|---|---|---|---|---|
| moving, < 30 km from gate | 136,723 | 15.8 | 16.3 | 16.5 | 14.0 | 0.825 | 2.6 | 0.477 | 14.9 |
| moving, >= 30 km | 593,670 | 21.5 | 22.5 | 19.1 | 17.8 | 0.766 | 18.3 | 0.895 | 72.8 |
| slow (0.5-5 kn) | 178,487 | 86.0 | 82.4 | 74.3 | 74.3 | 0.795 | 186.5 | 0.721 | 195.3 |
| stationary (< 0.5 kn) | 904,006 | 138.5 | 119.6 | 116.5 | 112.9 | 0.706 | 204.4 | 0.766 | 282.7 |

## Will the ship wait before arriving? (moving ships)

| auc | brier | brier_base_rate | test_rate | train_rate |
|---|---|---|---|---|
| 0.854 | 0.119 | 0.190 | 0.254 | 0.253 |

Reliability (predicted vs observed rate, 5 bins):

| bin | n | predicted | observed |
|---|---|---|---|
| 0.000 | 512,977 | 0.089 | 0.099 |
| 0.200 | 124,844 | 0.272 | 0.301 |
| 0.400 | 38,837 | 0.489 | 0.553 |
| 0.600 | 41,397 | 0.709 | 0.700 |
| 0.800 | 67,218 | 0.895 | 0.906 |
