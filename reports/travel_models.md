# Travel time to the next event (moving ships), test Oct-Dec 2023

pinball on log1p(minutes) over 7 quantiles; MAE of the median in hours; near = within 30 km of a gate.

| model | fit_s | pinball_log | mae_h | median_ae_h | mae_h_near | mae_h_far | coverage_p10_p90 | valid_pinball_log |
|---|---|---|---|---|---|---|---|---|
| LightGBM (tabular) | 67.2 | 0.084 | 1.1 | 0.432 | 0.521 | 1.2 | 0.792 | 0.098 |
| MLP (tabular) | 124.5 | 0.097 | 1.3 | 0.504 | 0.596 | 1.4 | 0.761 | 0.120 |
| GRU (2 h track) + tabular | 86.7 | 0.094 | 1.3 | 0.502 | 0.556 | 1.4 | 0.781 | 0.109 |
| GRU + tabular, regularised | 119.2 | 0.116 | 1.4 | 0.656 | 0.627 | 1.6 | 0.938 | 0.127 |
| LightGBM + GRU (2 h track) + tabular (w=0.25) | nan | 0.083 | 1.1 | 0.415 | 0.516 | 1.2 | 0.816 | 0.097 |
