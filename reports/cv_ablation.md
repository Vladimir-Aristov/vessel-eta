# Grouped-by-ship 5-fold CV, Jan-Feb 2023 (MAE, hours)

| config | mae_h | voyage_mae_h | moving, < 30 km from gate | moving, >= 30 km | slow (0.5-5 kn) | stationary (< 0.5 kn) |
|---|---|---|---|---|---|---|
| naive distance / speed | 73.9 | 13.4 | 15.9 | 19.1 | 83.7 | 112.0 |
| calibrated naive | 62.9 | 17.0 | 16.9 | 21.1 | 80.6 | 89.2 |
| wide trees | 64.7 | 14.1 | 15.6 | 18.4 | 71.2 | 97.4 |
| wide, no ship size/type | 67.4 | 15.6 | 16.3 | 19.2 | 73.0 | 101.7 |
| regularised (chosen) | 62.8 | 13.9 | 15.6 | 18.7 | 67.9 | 94.3 |
| regularised, no ship size/type | 64.4 | 14.6 | 15.8 | 18.7 | 70.8 | 96.6 |
