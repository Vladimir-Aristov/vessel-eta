# v2 intervals, grouped-by-ship 5-fold CV inside Jan-Feb 2023

| intervals | slice | coverage_p10_p90 | below_p10 | above_p90 | median_width_h | pinball_h | mae_h |
|---|---|---|---|---|---|---|---|
| raw quantiles | all | 0.765 | 0.097 | 0.138 | 196.0 | 20.4 | 59.4 |
| raw quantiles | moving, < 30 km from gate | 0.818 | 0.090 | 0.092 | 27.4 | 6.7 | 15.4 |
| raw quantiles | moving, >= 30 km | 0.772 | 0.098 | 0.130 | 26.7 | 7.3 | 17.3 |
| raw quantiles | slow (0.5-5 kn) | 0.755 | 0.087 | 0.158 | 201.1 | 22.8 | 69.9 |
| raw quantiles | stationary (< 0.5 kn) | 0.755 | 0.101 | 0.144 | 211.8 | 29.5 | 87.8 |
| conformalised | all | 0.789 | 0.114 | 0.097 | 183.3 | 22.4 | 59.4 |
| conformalised | moving, < 30 km from gate | 0.808 | 0.110 | 0.082 | 34.2 | 6.7 | 15.4 |
| conformalised | moving, >= 30 km | 0.748 | 0.139 | 0.113 | 32.5 | 7.2 | 17.3 |
| conformalised | slow (0.5-5 kn) | 0.797 | 0.102 | 0.101 | 216.6 | 26.1 | 69.9 |
| conformalised | stationary (< 0.5 kn) | 0.808 | 0.103 | 0.090 | 235.1 | 32.5 | 87.8 |
