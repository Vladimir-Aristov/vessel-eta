# eta_v2_2023_base: decomposed probabilistic ETA, features: base

Train: arrivals before 2023-10-01; test: from 2023-10-01, 954 voyages. Hours. v1 is retrained on the same split.

## Point forecast (v2 = median of the predicted distribution)

| model | mae_h | median_ae_h | voyage_mae_h | median_bias_h |
|---|---|---|---|---|
| naive | 66.1 | 18.8 | 12.6 | -8.9 |
| calibrated | 57.3 | 21.4 | 15.5 | 3.2 |
| v1 | 52.0 | 15.3 | 11.9 | 0.805 |
| v2 | 50.7 | 17.2 | 11.4 | -0.017 |

## Ships never seen in training

| model | mae_h | median_ae_h | voyage_mae_h | median_bias_h |
|---|---|---|---|---|
| naive | 63.8 | 30.3 | 16.1 | -23.1 |
| calibrated | 51.9 | 26.2 | 17.0 | 2.4 |
| v1 | 47.7 | 21.2 | 14.0 | 0.455 |
| v2 | 46.8 | 24.9 | 13.5 | -0.097 |

## Intervals (P10-P90, target coverage 0.80)

| interval | coverage_p10_p90 | below_p10 | above_p90 | median_width_h | pinball_h |
|---|---|---|---|---|---|
| calibrated x residual quantiles | 0.805 | 0.105 | 0.089 | 228.3 | 20.1 |
| v2 raw | 0.784 | 0.091 | 0.125 | 164.7 | 17.5 |
| v2 conformal | 0.803 | 0.079 | 0.118 | 189.5 | 17.3 |

## Intervals, ships never seen

| interval | coverage_p10_p90 | below_p10 | above_p90 | median_width_h | pinball_h |
|---|---|---|---|---|---|
| calibrated x residual quantiles | 0.826 | 0.098 | 0.076 | 242.5 | 17.7 |
| v2 raw | 0.801 | 0.086 | 0.113 | 177.2 | 15.1 |
| v2 conformal | 0.836 | 0.067 | 0.097 | 202.7 | 15.2 |

## By state at prediction time (MAE; P10-P90 coverage and median width, v2 vs baseline)

| slice | n | naive | calibrated | v1 | v2 | v2_coverage | v2_width_h | base_coverage | base_width_h |
|---|---|---|---|---|---|---|---|---|---|
| moving, < 30 km from gate | 31,527 | 13.5 | 14.1 | 13.2 | 13.0 | 0.823 | 25.5 | 0.464 | 15.6 |
| moving, >= 30 km | 139,846 | 17.6 | 19.5 | 16.1 | 15.0 | 0.765 | 21.2 | 0.904 | 77.7 |
| slow (0.5-5 kn) | 27,050 | 81.9 | 80.3 | 69.3 | 69.9 | 0.798 | 193.6 | 0.674 | 206.0 |
| stationary (< 0.5 kn) | 192,852 | 107.6 | 88.5 | 81.9 | 80.0 | 0.828 | 237.2 | 0.808 | 272.8 |

## Will the ship wait before arriving? (moving ships)

| auc | brier | brier_base_rate | test_rate | train_rate |
|---|---|---|---|---|
| 0.827 | 0.123 | 0.175 | 0.223 | 0.260 |

Reliability (predicted vs observed rate, 5 bins):

| bin | n | predicted | observed |
|---|---|---|---|
| 0.000 | 104,416 | 0.119 | 0.082 |
| 0.200 | 45,530 | 0.259 | 0.246 |
| 0.400 | 12,433 | 0.513 | 0.525 |
| 0.600 | 16,876 | 0.717 | 0.785 |
| 0.800 | 703 | 0.805 | 0.973 |
