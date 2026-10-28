# eta_v1: ETA to LA/Long Beach

Train: arrivals before 2023-03-01 (early stopping on 20% held-out ships); test: arrivals from 2023-03-01. Errors in hours. Test: 310 voyages, 151,983 rows (one per 5 min).

## Test, all

| model | mae_h | median_ae_h | voyage_mae_h | median_bias_h |
|---|---|---|---|---|
| median | 103.9 | 26.0 | 27.1 | -18.5 |
| naive | 97.1 | 33.8 | 16.1 | -28.0 |
| calibrated | 93.3 | 34.3 | 19.7 | -0.557 |
| lgbm | 99.4 | 27.8 | 16.0 | -13.1 |

## Test, ships seen in training (140 voyages)

| model | mae_h | median_ae_h | voyage_mae_h | median_bias_h |
|---|---|---|---|---|
| median | 43.5 | 22.3 | 22.9 | 1.9 |
| naive | 42.5 | 17.5 | 11.1 | -10.3 |
| calibrated | 40.3 | 22.1 | 17.1 | 4.6 |
| lgbm | 39.2 | 15.4 | 11.6 | 0.832 |

## Test, ships never seen (170 voyages)

| model | mae_h | median_ae_h | voyage_mae_h | median_bias_h |
|---|---|---|---|---|
| median | 133.0 | 38.6 | 30.6 | -38.6 |
| naive | 123.5 | 48.9 | 20.1 | -43.4 |
| calibrated | 118.8 | 41.9 | 21.8 | -14.9 |
| lgbm | 128.5 | 40.2 | 19.6 | -31.2 |

## MAE by state at prediction time

| slice | n | median_true_eta_h | pred_median | pred_naive | pred_calibrated | pred_lgbm |
|---|---|---|---|---|---|---|
| moving, < 30 km from gate | 9,219 | 1.6 | 30.0 | 12.5 | 13.6 | 11.9 |
| moving, >= 30 km | 38,134 | 11.6 | 34.8 | 28.8 | 28.7 | 26.8 |
| slow (0.5-5 kn) | 26,772 | 94.2 | 87.4 | 75.1 | 74.6 | 85.7 |
| stationary (< 0.5 kn) | 77,858 | 75.1 | 152.1 | 148.2 | 140.8 | 150.1 |

## MAE by true horizon (diagnostic: conditions on the outcome)

| slice | n | median_true_eta_h | pred_median | pred_naive | pred_calibrated | pred_lgbm |
|---|---|---|---|---|---|---|
| 0-2 h | 7,016 | 0.997 | 26.4 | 0.465 | 9.1 | 5.7 |
| 2-6 h | 13,627 | 4.0 | 23.4 | 2.6 | 18.4 | 11.3 |
| 6-12 h | 18,276 | 8.8 | 18.5 | 4.8 | 18.7 | 11.6 |
| 12-24 h | 19,062 | 16.2 | 10.5 | 14.6 | 25.5 | 12.5 |
| 24-72 h | 32,996 | 43.6 | 18.2 | 35.0 | 20.4 | 22.8 |
| > 72 h | 61,006 | 166.0 | 231.9 | 216.4 | 202.7 | 224.9 |

## Top features (gain share)

| feature | share |
|---|---|
| closing_kmh_1h | 0.219 |
| sog | 0.179 |
| draft | 0.142 |
| naive_eta_min | 0.099 |
| status | 0.072 |
| angels_km | 0.044 |
| width | 0.036 |
| naive_eta_2h_min | 0.033 |
| min_since_start | 0.028 |
| length | 0.026 |
| stationary_min_total | 0.024 |
| align | 0.022 |
| lat | 0.019 |
| min_since_moving | 0.016 |
| lon | 0.012 |
