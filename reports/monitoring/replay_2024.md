# A year of monitoring, replayed (model trained on 2023, never refit)

Interval re-calibration each month: window. Per-month reports: replay_2024/.

| month | voyages | model MAE h | physics MAE h | coverage | worst input PSI | prediction PSI | status | drifted inputs |
|---|---|---|---|---|---|---|---|---|
| 2024-01 | 341 | 79.0 | 97.0 | 0.708 | port_waiting 0.49 | 0.145 | OK | - |
| 2024-02 | 317 | 127.2 | 149.1 | 0.710 | length 0.45 | 0.159 | WARN | draft 0.36 > 0.27, length 0.45 > 0.26 |
| 2024-03 | 356 | 65.0 | 83.5 | 0.830 | port_waiting 0.49 | 0.133 | OK | - |
| 2024-04 | 330 | 100.6 | 118.1 | 0.733 | port_waiting 0.30 | 0.092 | WARN | draft 0.29 > 0.27 |
| 2024-05 | 337 | 84.0 | 98.8 | 0.816 | port_waiting 0.46 | 0.124 | WARN | length 0.43 > 0.26 |
| 2024-06 | 348 | 67.5 | 84.0 | 0.817 | port_waiting 0.57 | 0.055 | OK | - |
| 2024-07 | 378 | 82.0 | 99.5 | 0.779 | port_waiting 1.12 | 0.257 | OK | - |
| 2024-08 | 383 | 48.3 | 62.4 | 0.844 | port_waiting 0.86 | 0.094 | WARN | draft 0.28 > 0.27 |
| 2024-09 | 350 | 47.2 | 61.4 | 0.798 | port_waiting 0.35 | 0.014 | OK | - |
| 2024-10 | 335 | 25.9 | 31.5 | 0.836 | port_waiting 0.76 | 0.070 | WARN | draft 0.31 > 0.27 |
| 2024-11 | 363 | 24.5 | 30.5 | 0.805 | port_waiting 0.73 | 0.058 | WARN | draft 0.28 > 0.27 |
| 2024-12 | 355 | 57.8 | 68.9 | 0.732 | port_waiting 0.86 | 0.028 | WARN | draft 0.32 > 0.27 |
