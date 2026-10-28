# eta_v2_2024_base: decomposed probabilistic ETA, features: base

Train: arrivals before 2024-01-01; test: from 2024-01-01, 4193 voyages. Hours. v1 is retrained on the same split.

## Point forecast (v2 = median of the predicted distribution)

| model | mae_h | median_ae_h | voyage_mae_h | median_bias_h |
|---|---|---|---|---|
| naive | 85.8 | 24.2 | 13.4 | -14.9 |
| calibrated | 76.4 | 25.5 | 15.8 | 1.9 |
| v1 | 72.9 | 18.7 | 15.2 | -0.086 |
| v2 | 70.5 | 23.1 | 12.7 | -0.160 |

## Ships never seen in training

| model | mae_h | median_ae_h | voyage_mae_h | median_bias_h |
|---|---|---|---|---|
| naive | 93.0 | 33.7 | 16.0 | -27.1 |
| calibrated | 80.6 | 31.4 | 17.2 | 0.967 |
| v1 | 77.8 | 25.2 | 15.2 | -2.0 |
| v2 | 75.2 | 30.4 | 15.0 | -0.561 |

## Intervals (P10-P90, target coverage 0.80)

| interval | coverage_p10_p90 | below_p10 | above_p90 | median_width_h | pinball_h |
|---|---|---|---|---|---|
| calibrated x residual quantiles | 0.782 | 0.090 | 0.128 | 239.2 | 26.2 |
| v2 raw | 0.743 | 0.084 | 0.173 | 158.2 | 25.5 |
| v2 conformal | 0.745 | 0.078 | 0.176 | 162.4 | 25.4 |

## Intervals, ships never seen

| interval | coverage_p10_p90 | below_p10 | above_p90 | median_width_h | pinball_h |
|---|---|---|---|---|---|
| calibrated x residual quantiles | 0.785 | 0.079 | 0.136 | 245.2 | 27.3 |
| v2 raw | 0.736 | 0.081 | 0.183 | 170.8 | 26.7 |
| v2 conformal | 0.742 | 0.074 | 0.184 | 175.5 | 26.6 |

## By state at prediction time (MAE; P10-P90 coverage and median width, v2 vs baseline)

| slice | n | naive | calibrated | v1 | v2 | v2_coverage | v2_width_h | base_coverage | base_width_h |
|---|---|---|---|---|---|---|---|---|---|
| moving, < 30 km from gate | 136,723 | 15.8 | 16.3 | 16.5 | 14.1 | 0.826 | 2.5 | 0.477 | 14.9 |
| moving, >= 30 km | 593,670 | 21.5 | 22.5 | 19.1 | 17.7 | 0.772 | 18.2 | 0.895 | 72.8 |
| slow (0.5-5 kn) | 178,487 | 86.0 | 82.4 | 74.3 | 74.3 | 0.796 | 186.0 | 0.721 | 195.3 |
| stationary (< 0.5 kn) | 904,006 | 138.5 | 119.6 | 116.5 | 112.9 | 0.706 | 204.4 | 0.766 | 282.7 |

## Will the ship wait before arriving? (moving ships)

| auc | brier | brier_base_rate | test_rate | train_rate |
|---|---|---|---|---|
| 0.854 | 0.119 | 0.190 | 0.254 | 0.253 |

Reliability (predicted vs observed rate, 5 bins):

| bin | n | predicted | observed |
|---|---|---|---|
| 0.000 | 512,794 | 0.089 | 0.098 |
| 0.200 | 124,726 | 0.273 | 0.304 |
| 0.400 | 38,909 | 0.490 | 0.550 |
| 0.600 | 39,468 | 0.709 | 0.711 |
| 0.800 | 69,373 | 0.894 | 0.899 |
