# Interval calibration over time (P10-P90, target 0.80)

## Step chosen on Oct-Dec 2023 (model trained Jan-Sep 2023)

| step | coverage_p10_p90 | below_p10 | above_p90 | median_width_h | pinball_h |
|---|---|---|---|---|---|
| 0.010 | 0.811 | 0.081 | 0.107 | 176.7 | 14.2 |
| 0.020 | 0.809 | 0.081 | 0.111 | 171.3 | 14.2 |
| 0.050 | 0.801 | 0.077 | 0.122 | 157.1 | 14.3 |
| 0.100 | 0.790 | 0.074 | 0.137 | 139.5 | 14.5 |
| 0.200 | 0.778 | 0.070 | 0.153 | 119.3 | 14.9 |
| 0.500 | 0.772 | 0.067 | 0.161 | 104.9 | 15.2 |
| 1.0 | 0.775 | 0.066 | 0.159 | 103.7 | 15.3 |

chosen step: 0.01

## Test: 2024, model trained on 2023, never refit

| method | rows | coverage_p10_p90 | below_p10 | above_p90 | median_width_h | pinball_h |
|---|---|---|---|---|---|---|
| static | all | 0.748 | 0.080 | 0.172 | 161.1 | 24.7 |
| static | under way | 0.782 | 0.083 | 0.135 | 18.5 | 7.3 |
| static | waiting | 0.722 | 0.078 | 0.201 | 203.1 | 38.2 |
| monthly re-calibration | all | 0.784 | 0.098 | 0.118 | 188.7 | 24.2 |
| monthly re-calibration | under way | 0.799 | 0.097 | 0.104 | 21.7 | 7.3 |
| monthly re-calibration | waiting | 0.772 | 0.099 | 0.129 | 307.3 | 37.3 |
| daily tracking (step 0.01) | all | 0.753 | 0.081 | 0.166 | 159.3 | 24.7 |
| daily tracking (step 0.01) | under way | 0.788 | 0.095 | 0.117 | 19.5 | 7.3 |
| daily tracking (step 0.01) | waiting | 0.726 | 0.069 | 0.205 | 199.2 | 38.3 |

## Coverage by month of 2024

| method | 01 | 02 | 03 | 04 | 05 | 06 | 07 | 08 | 09 | 10 | 11 | 12 | worst month | spread |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| static | 0.640 | 0.721 | 0.726 | 0.711 | 0.766 | 0.724 | 0.747 | 0.809 | 0.770 | 0.834 | 0.791 | 0.811 | 0.640 | 0.052 |
| monthly re-calibration | 0.640 | 0.753 | 0.836 | 0.754 | 0.843 | 0.785 | 0.805 | 0.839 | 0.809 | 0.844 | 0.762 | 0.772 | 0.640 | 0.056 |
| daily tracking (step 0.01) | 0.639 | 0.723 | 0.732 | 0.717 | 0.777 | 0.732 | 0.751 | 0.814 | 0.773 | 0.848 | 0.794 | 0.814 | 0.639 | 0.054 |
