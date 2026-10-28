# eta_v2: decomposed probabilistic ETA (same split as eta_v1)

Test: arrivals from 2023-03-01, 310 voyages. Hours.

## Point forecast (v2 = median of the predicted distribution)

| model | mae_h | median_ae_h | voyage_mae_h | median_bias_h |
|---|---|---|---|---|
| naive | 97.1 | 33.8 | 16.1 | -28.0 |
| calibrated | 93.3 | 34.3 | 19.7 | -0.557 |
| v1 | 99.4 | 27.8 | 16.0 | -13.1 |
| v2 | 93.7 | 35.0 | 17.1 | -1.3 |

## Ships never seen in training

| model | mae_h | median_ae_h | voyage_mae_h | median_bias_h |
|---|---|---|---|---|
| naive | 123.5 | 48.9 | 20.1 | -43.4 |
| calibrated | 118.8 | 41.9 | 21.8 | -14.9 |
| v1 | 128.5 | 40.2 | 19.6 | -31.2 |
| v2 | 120.2 | 42.2 | 19.4 | -14.5 |

## Intervals (P10-P90, target coverage 0.80)

| interval | coverage_p10_p90 | below_p10 | above_p90 | median_width_h | pinball_h |
|---|---|---|---|---|---|
| calibrated x residual quantiles | 0.793 | 0.090 | 0.117 | 413.4 | 33.0 |
| v2 raw | 0.747 | 0.074 | 0.179 | 252.2 | 34.0 |
| v2 conformal | 0.769 | 0.106 | 0.124 | 275.2 | 33.4 |

## Intervals, ships never seen

| interval | coverage_p10_p90 | below_p10 | above_p90 | median_width_h | pinball_h |
|---|---|---|---|---|---|
| calibrated x residual quantiles | 0.798 | 0.069 | 0.133 | 418.0 | 40.2 |
| v2 raw | 0.724 | 0.057 | 0.220 | 252.8 | 43.8 |
| v2 conformal | 0.752 | 0.084 | 0.164 | 275.8 | 42.9 |

## By state at prediction time (MAE; P10-P90 coverage and median width, v2 vs baseline)

| slice | n | naive | calibrated | v1 | v2 | v2_coverage | v2_width_h | base_coverage | base_width_h |
|---|---|---|---|---|---|---|---|---|---|
| moving, < 30 km from gate | 9,219 | 12.5 | 13.6 | 11.9 | 11.2 | 0.831 | 18.1 | 0.467 | 17.6 |
| moving, >= 30 km | 38,134 | 28.8 | 28.7 | 26.8 | 24.7 | 0.793 | 39.8 | 0.857 | 66.0 |
| slow (0.5-5 kn) | 26,772 | 75.1 | 74.6 | 85.7 | 77.0 | 0.877 | 276.7 | 0.823 | 257.3 |
| stationary (< 0.5 kn) | 77,858 | 148.2 | 140.8 | 150.1 | 143.1 | 0.714 | 277.4 | 0.789 | 429.7 |

## Will the ship wait before arriving? (moving ships)

| auc | brier | brier_base_rate | test_rate | train_rate |
|---|---|---|---|---|
| 0.850 | 0.142 | 0.245 | 0.365 | 0.249 |

Reliability (predicted vs observed rate, 5 bins):

| bin | n | predicted | observed |
|---|---|---|---|
| 0.000 | 30,352 | 0.088 | 0.154 |
| 0.200 | 8,670 | 0.278 | 0.285 |
| 0.400 | 4,450 | 0.492 | 0.589 |
| 0.600 | 5,936 | 0.716 | 0.881 |
| 0.800 | 4,880 | 0.852 | 0.990 |
