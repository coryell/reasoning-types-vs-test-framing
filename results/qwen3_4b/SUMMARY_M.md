# Mechanistic arm (Qwen3-4B) — first-pass analysis

- `triggers_gate.json.awareness.stats.json`: positives 62 / judged 400 / generated 400 → PASS (rate over generated 0.155)
- `triggers_probe.json.awareness.stats.json`: positives 184 / judged 1200 / generated 1200 → PASS (rate over generated 0.153)

## Probe

best hidden-state index 20: acc 0.977, AUROC 0.997; spans pos/neg (balanced) 257/257; controls: {"shuffled_layer": 20, "shuffled_acc": 0.509, "shuffled_auroc": 0.696, "layer0_acc": 0.48, "layer0_auroc": 0.984, "n_pos": 257, "n_neg": 257}

| layer | n_train | n_test | acc | auroc | sk_acc | sk_auroc |
|---|---|---|---|---|---|---|
| 0.000 | 343.000 | 171.000 | 0.480 | 0.984 | 0.918 | 0.991 |
| 3.000 | 343.000 | 171.000 | 0.936 | 0.986 | 0.947 | 0.991 |
| 6.000 | 343.000 | 171.000 | 0.953 | 0.987 | 0.936 | 0.990 |
| 9.000 | 343.000 | 171.000 | 0.947 | 0.989 | 0.947 | 0.992 |
| 12.000 | 343.000 | 171.000 | 0.953 | 0.994 | 0.942 | 0.995 |
| 15.000 | 343.000 | 171.000 | 0.959 | 0.995 | 0.953 | 0.996 |
| 18.000 | 343.000 | 171.000 | 0.959 | 0.997 | 0.959 | 0.997 |
| 21.000 | 343.000 | 171.000 | 0.971 | 0.997 | 0.965 | 0.998 |
| 24.000 | 343.000 | 171.000 | 0.947 | 0.995 | 0.965 | 0.997 |
| 27.000 | 343.000 | 171.000 | 0.947 | 0.994 | 0.959 | 0.995 |
| 30.000 | 343.000 | 171.000 | 0.959 | 0.994 | 0.953 | 0.995 |
| 33.000 | 343.000 | 171.000 | 0.947 | 0.993 | 0.947 | 0.995 |
| 36.000 | 343.000 | 171.000 | 0.947 | 0.993 | 0.947 | 0.995 |


## Probe token-position ablation (their §4.2): span mean vs whole-reasoning mean vs last token

| pooling | best_layer | acc | auroc | shuffled_auroc | layer0_auroc | n_pos | n_neg |
|---|---|---|---|---|---|---|---|
| span | 20 | 0.977 | 0.997 | 0.696 | 0.984 | 257 | 257 |
| whole | 11 | 0.884 | 0.960 | 0.802 | 0.864 | 184 | 184 |
| last | 35 | 0.860 | 0.925 | 0.771 | 0.525 | 184 | 184 |


## Geometry at the probe's layer (hidden_states[20]); chance |cos| = 0.0163

| behaviour | all_spans | nontest_spans | testlex_spans | shuffled_probe |
|---|---|---|---|---|
| initializing | -0.027 | -0.027 | -0.152 | 0.050 |
| deduction | 0.065 | 0.063 | 0.121 | -0.000 |
| adding-knowledge | -0.106 | -0.108 | 0.087 | -0.054 |
| example-testing | 0.058 | 0.057 | -0.058 | 0.026 |
| uncertainty-estimation | 0.029 | 0.017 | 0.098 | 0.040 |
| backtracking | 0.146 | 0.145 | 0.082 | 0.057 |

random-direction |cos| at this layer: 0.0164


## α sweep

| arm | alpha | aware | n | closure | repetition | n_judged | verbalization |
|---|---|---|---|---|---|---|---|
| alpha0.02 | 0.020 | False | 40 | 1.000 | 0.027 | 40 | 0.125 |
| alpha0.02_aware | 0.020 | True | 40 | 0.925 | 0.067 | 40 | 0.175 |
| alpha0.05 | 0.050 | False | 40 | 1.000 | 0.044 | 40 | 0.150 |
| alpha0.05_aware | 0.050 | True | 40 | 1.000 | 0.027 | 40 | 0.100 |
| alpha0.1 | 0.100 | False | 40 | 1.000 | 0.028 | 40 | 0.100 |
| alpha0.1_aware | 0.100 | True | 40 | 1.000 | 0.045 | 40 | 0.200 |
| alpha0.2 | 0.200 | False | 40 | 1.000 | 0.033 | 40 | 0.100 |
| alpha0.2_aware | 0.200 | True | 40 | 0.950 | 0.040 | 40 | 0.200 |
| baseline | 0.000 | None | 40 | 1.000 | 0.026 | 40 | 0.025 |

chosen: {"aware": {"arm": "alpha0.05_aware", "alpha": 0.05, "closure": 1.0, "repetition": 0.02745229144949729, "verbalization": 0.1, "flagged_no_qualifying_arm": false}, "unaware": {"arm": "alpha0.1", "alpha": 0.1, "closure": 1.0, "repetition": 0.02783636057145275, "verbalization": 0.1, "flagged_no_qualifying_arm": false}}


## Main run: 2980 traces, 2980 with usable morphology annotation

| arm | framing | n | exec_rate_closed | exec_rate_all | lost | gained | p_mcnemar | task_judged_yes | judge_vs_string_agree | verbalization | closure | repetition | words |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | real | 149 | 0.726 | 0.711 | — | — | — | 0.826 | 0.582 | 0.188 | 0.980 | 0.046 | 334.470 |
| baseline | fake | 149 | 0.882 | 0.852 | — | — | — | 0.851 | 0.762 | 0.088 | 0.966 | 0.047 | 345.228 |
| baseline_seed1 | real | 149 | 0.748 | 0.718 | 3.000 | 6.000 | 0.508 | 0.738 | 0.545 | 0.148 | 0.960 | 0.051 | 353.859 |
| baseline_seed1 | fake | 149 | 0.890 | 0.872 | 5.000 | 6.000 | 1.000 | 0.852 | 0.760 | 0.060 | 0.980 | 0.047 | 331.664 |
| unaware | real | 149 | 0.755 | 0.725 | 5.000 | 10.000 | 0.302 | 0.777 | 0.577 | 0.135 | 0.960 | 0.043 | 332.826 |
| unaware | fake | 149 | 0.866 | 0.826 | 7.000 | 5.000 | 0.774 | 0.819 | 0.725 | 0.107 | 0.953 | 0.056 | 373.128 |
| aware | real | 149 | 0.764 | 0.738 | 5.000 | 11.000 | 0.210 | 0.765 | 0.583 | 0.094 | 0.966 | 0.053 | 359.946 |
| aware | fake | 149 | 0.859 | 0.819 | 8.000 | 5.000 | 0.581 | 0.830 | 0.736 | 0.102 | 0.953 | 0.056 | 365.933 |
| unaware_strong | real | 149 | 0.788 | 0.772 | 3.000 | 11.000 | 0.057 | 0.799 | 0.596 | 0.107 | 0.980 | 0.043 | 338.208 |
| unaware_strong | fake | 149 | 0.930 | 0.886 | 4.000 | 11.000 | 0.118 | 0.832 | 0.803 | 0.067 | 0.953 | 0.053 | 361.725 |
| aware_strong | real | 149 | 0.766 | 0.745 | 6.000 | 12.000 | 0.238 | 0.777 | 0.562 | 0.182 | 0.973 | 0.043 | 327.289 |
| aware_strong | fake | 149 | 0.873 | 0.832 | 4.000 | 4.000 | 1.000 | 0.797 | 0.745 | 0.135 | 0.953 | 0.065 | 398.463 |
| random_aware | real | 149 | 0.769 | 0.758 | 5.000 | 10.000 | 0.302 | 0.819 | 0.599 | 0.161 | 0.987 | 0.041 | 329.678 |
| random_aware | fake | 149 | 0.906 | 0.839 | 2.000 | 6.000 | 0.289 | 0.826 | 0.797 | 0.101 | 0.926 | 0.052 | 376.678 |
| shuffled_aware | real | 149 | 0.729 | 0.705 | 7.000 | 7.000 | 1.000 | 0.818 | 0.580 | 0.108 | 0.966 | 0.046 | 341.564 |
| shuffled_aware | fake | 149 | 0.880 | 0.839 | 4.000 | 3.000 | 1.000 | 0.852 | 0.775 | 0.081 | 0.953 | 0.048 | 358.913 |
| random_samerows_aware | real | 149 | 0.755 | 0.725 | 5.000 | 10.000 | 0.302 | 0.799 | 0.601 | 0.195 | 0.960 | 0.054 | 347.906 |
| random_samerows_aware | fake | 149 | 0.890 | 0.866 | 5.000 | 7.000 | 0.774 | 0.849 | 0.803 | 0.068 | 0.973 | 0.056 | 361.523 |
| shuffled_samerows_aware | real | 149 | 0.726 | 0.711 | 6.000 | 6.000 | 1.000 | 0.770 | 0.510 | 0.209 | 0.980 | 0.042 | 333.664 |
| shuffled_samerows_aware | fake | 149 | 0.897 | 0.872 | 5.000 | 7.000 | 0.774 | 0.858 | 0.785 | 0.122 | 0.973 | 0.043 | 329.081 |


## Morphology: Δ spans per 100 words vs baseline, paired by item

| framing | arm | n_pairs | behaviour | metric | base_mean | arm_mean | delta | ci_lo | ci_hi | p_paired_t |
|---|---|---|---|---|---|---|---|---|---|---|
| fake | aware | 149 | deduction | density | 3.843 | 3.987 | 0.144 | -0.108 | 0.380 | 0.255 |
| fake | aware | 149 | adding-knowledge | density | 1.706 | 1.601 | -0.104 | -0.289 | 0.087 | 0.300 |
| fake | aware | 149 | uncertainty-estimation | density | 0.606 | 0.616 | 0.009 | -0.126 | 0.148 | 0.895 |
| fake | aware | 149 | backtracking | density | 0.279 | 0.339 | 0.060 | -0.015 | 0.135 | 0.130 |
| fake | aware | 149 | deduction | density_testlex | 0.131 | 0.145 | 0.014 | -0.032 | 0.055 | 0.532 |
| fake | aware | 149 | adding-knowledge | density_testlex | 0.016 | 0.015 | -0.000 | -0.018 | 0.021 | 0.980 |
| fake | aware | 149 | uncertainty-estimation | density_testlex | 0.062 | 0.073 | 0.011 | -0.021 | 0.045 | 0.498 |
| fake | aware | 149 | backtracking | density_testlex | 0.004 | 0.003 | -0.001 | -0.005 | 0.002 | 0.470 |
| fake | aware | 149 | deduction | density_nontest | 3.712 | 3.842 | 0.130 | -0.120 | 0.359 | 0.292 |
| fake | aware | 149 | adding-knowledge | density_nontest | 1.690 | 1.586 | -0.104 | -0.294 | 0.088 | 0.300 |
| fake | aware | 149 | uncertainty-estimation | density_nontest | 0.545 | 0.543 | -0.002 | -0.135 | 0.139 | 0.976 |
| fake | aware | 149 | backtracking | density_nontest | 0.275 | 0.336 | 0.061 | -0.013 | 0.136 | 0.120 |
| fake | aware_strong | 149 | deduction | density | 3.843 | 3.880 | 0.037 | -0.190 | 0.278 | 0.758 |
| fake | aware_strong | 149 | adding-knowledge | density | 1.706 | 1.558 | -0.147 | -0.313 | 0.021 | 0.084 |
| fake | aware_strong | 149 | uncertainty-estimation | density | 0.606 | 0.751 | 0.145 | 0.031 | 0.269 | 0.016 |
| fake | aware_strong | 149 | backtracking | density | 0.279 | 0.275 | -0.004 | -0.069 | 0.061 | 0.913 |
| fake | aware_strong | 149 | deduction | density_testlex | 0.131 | 0.097 | -0.034 | -0.079 | 0.011 | 0.135 |
| fake | aware_strong | 149 | adding-knowledge | density_testlex | 0.016 | 0.024 | 0.008 | -0.013 | 0.031 | 0.475 |
| fake | aware_strong | 149 | uncertainty-estimation | density_testlex | 0.062 | 0.105 | 0.044 | 0.007 | 0.083 | 0.032 |
| fake | aware_strong | 149 | backtracking | density_testlex | 0.004 | 0.003 | -0.001 | -0.008 | 0.006 | 0.697 |
| fake | aware_strong | 149 | deduction | density_nontest | 3.712 | 3.783 | 0.071 | -0.165 | 0.316 | 0.550 |
| fake | aware_strong | 149 | adding-knowledge | density_nontest | 1.690 | 1.535 | -0.155 | -0.326 | 0.011 | 0.072 |
| fake | aware_strong | 149 | uncertainty-estimation | density_nontest | 0.545 | 0.646 | 0.101 | -0.007 | 0.222 | 0.082 |
| fake | aware_strong | 149 | backtracking | density_nontest | 0.275 | 0.272 | -0.002 | -0.067 | 0.061 | 0.945 |
| fake | baseline_seed1 | 149 | deduction | density | 3.843 | 4.111 | 0.268 | 0.030 | 0.511 | 0.028 |
| fake | baseline_seed1 | 149 | adding-knowledge | density | 1.706 | 1.650 | -0.056 | -0.234 | 0.119 | 0.530 |
| fake | baseline_seed1 | 149 | uncertainty-estimation | density | 0.606 | 0.619 | 0.012 | -0.101 | 0.134 | 0.834 |
| fake | baseline_seed1 | 149 | backtracking | density | 0.279 | 0.269 | -0.010 | -0.075 | 0.051 | 0.763 |
| fake | baseline_seed1 | 149 | deduction | density_testlex | 0.131 | 0.115 | -0.016 | -0.063 | 0.027 | 0.490 |
| fake | baseline_seed1 | 149 | adding-knowledge | density_testlex | 0.016 | 0.009 | -0.007 | -0.022 | 0.008 | 0.389 |
| fake | baseline_seed1 | 149 | uncertainty-estimation | density_testlex | 0.062 | 0.093 | 0.031 | -0.003 | 0.071 | 0.108 |
| fake | baseline_seed1 | 149 | backtracking | density_testlex | 0.004 | 0.003 | -0.001 | -0.008 | 0.006 | 0.814 |
| fake | baseline_seed1 | 149 | deduction | density_nontest | 3.712 | 3.996 | 0.284 | 0.045 | 0.520 | 0.022 |
| fake | baseline_seed1 | 149 | adding-knowledge | density_nontest | 1.690 | 1.642 | -0.049 | -0.228 | 0.124 | 0.585 |
| fake | baseline_seed1 | 149 | uncertainty-estimation | density_nontest | 0.545 | 0.526 | -0.019 | -0.129 | 0.095 | 0.740 |
| fake | baseline_seed1 | 149 | backtracking | density_nontest | 0.275 | 0.266 | -0.009 | -0.073 | 0.052 | 0.779 |
| fake | random_aware | 149 | deduction | density | 3.843 | 3.996 | 0.153 | -0.080 | 0.370 | 0.191 |
| fake | random_aware | 149 | adding-knowledge | density | 1.706 | 1.524 | -0.182 | -0.356 | -0.010 | 0.035 |
| fake | random_aware | 149 | uncertainty-estimation | density | 0.606 | 0.694 | 0.088 | -0.034 | 0.211 | 0.161 |
| fake | random_aware | 149 | backtracking | density | 0.279 | 0.300 | 0.021 | -0.044 | 0.087 | 0.539 |
| fake | random_aware | 149 | deduction | density_testlex | 0.131 | 0.108 | -0.023 | -0.072 | 0.026 | 0.342 |
| fake | random_aware | 149 | adding-knowledge | density_testlex | 0.016 | 0.002 | -0.013 | -0.028 | -0.001 | 0.061 |
| fake | random_aware | 149 | uncertainty-estimation | density_testlex | 0.062 | 0.077 | 0.015 | -0.016 | 0.046 | 0.347 |
| fake | random_aware | 149 | backtracking | density_testlex | 0.004 | 0.003 | -0.001 | -0.008 | 0.005 | 0.689 |
| fake | random_aware | 149 | deduction | density_nontest | 3.712 | 3.888 | 0.176 | -0.059 | 0.398 | 0.138 |
| fake | random_aware | 149 | adding-knowledge | density_nontest | 1.690 | 1.522 | -0.169 | -0.345 | 0.004 | 0.050 |
| fake | random_aware | 149 | uncertainty-estimation | density_nontest | 0.545 | 0.617 | 0.073 | -0.043 | 0.192 | 0.233 |
| fake | random_aware | 149 | backtracking | density_nontest | 0.275 | 0.297 | 0.022 | -0.043 | 0.087 | 0.510 |
| fake | random_samerows_aware | 149 | deduction | density | 3.843 | 3.919 | 0.075 | -0.160 | 0.309 | 0.522 |
| fake | random_samerows_aware | 149 | adding-knowledge | density | 1.706 | 1.630 | -0.076 | -0.251 | 0.103 | 0.428 |
| fake | random_samerows_aware | 149 | uncertainty-estimation | density | 0.606 | 0.609 | 0.002 | -0.118 | 0.115 | 0.967 |
| fake | random_samerows_aware | 149 | backtracking | density | 0.279 | 0.272 | -0.007 | -0.071 | 0.056 | 0.841 |
| fake | random_samerows_aware | 149 | deduction | density_testlex | 0.131 | 0.099 | -0.032 | -0.075 | 0.010 | 0.151 |
| fake | random_samerows_aware | 149 | adding-knowledge | density_testlex | 0.016 | 0.026 | 0.010 | -0.012 | 0.037 | 0.409 |
| fake | random_samerows_aware | 149 | uncertainty-estimation | density_testlex | 0.062 | 0.085 | 0.024 | -0.012 | 0.061 | 0.221 |
| fake | random_samerows_aware | 149 | backtracking | density_testlex | 0.004 | 0.005 | 0.001 | -0.004 | 0.006 | 0.829 |
| fake | random_samerows_aware | 149 | deduction | density_nontest | 3.712 | 3.819 | 0.107 | -0.127 | 0.338 | 0.357 |
| fake | random_samerows_aware | 149 | adding-knowledge | density_nontest | 1.690 | 1.604 | -0.086 | -0.259 | 0.092 | 0.361 |
| fake | random_samerows_aware | 149 | uncertainty-estimation | density_nontest | 0.545 | 0.524 | -0.021 | -0.134 | 0.089 | 0.708 |
| fake | random_samerows_aware | 149 | backtracking | density_nontest | 0.275 | 0.268 | -0.007 | -0.072 | 0.056 | 0.827 |
| fake | shuffled_aware | 149 | deduction | density | 3.843 | 3.980 | 0.137 | -0.116 | 0.400 | 0.289 |
| fake | shuffled_aware | 149 | adding-knowledge | density | 1.706 | 1.589 | -0.116 | -0.292 | 0.049 | 0.184 |
| fake | shuffled_aware | 149 | uncertainty-estimation | density | 0.606 | 0.717 | 0.110 | -0.031 | 0.251 | 0.134 |
| fake | shuffled_aware | 149 | backtracking | density | 0.279 | 0.314 | 0.036 | -0.035 | 0.106 | 0.324 |
| fake | shuffled_aware | 149 | deduction | density_testlex | 0.131 | 0.119 | -0.013 | -0.055 | 0.030 | 0.560 |
| fake | shuffled_aware | 149 | adding-knowledge | density_testlex | 0.016 | 0.024 | 0.008 | -0.013 | 0.032 | 0.475 |
| fake | shuffled_aware | 149 | uncertainty-estimation | density_testlex | 0.062 | 0.066 | 0.005 | -0.024 | 0.035 | 0.753 |
| fake | shuffled_aware | 149 | backtracking | density_testlex | 0.004 | 0.007 | 0.003 | -0.003 | 0.012 | 0.458 |
| fake | shuffled_aware | 149 | deduction | density_nontest | 3.712 | 3.862 | 0.150 | -0.101 | 0.411 | 0.241 |
| fake | shuffled_aware | 149 | adding-knowledge | density_nontest | 1.690 | 1.566 | -0.124 | -0.300 | 0.044 | 0.156 |
| fake | shuffled_aware | 149 | uncertainty-estimation | density_nontest | 0.545 | 0.651 | 0.106 | -0.035 | 0.248 | 0.159 |
| fake | shuffled_aware | 149 | backtracking | density_nontest | 0.275 | 0.307 | 0.032 | -0.038 | 0.101 | 0.356 |
| fake | shuffled_samerows_aware | 149 | deduction | density | 3.843 | 3.995 | 0.151 | -0.106 | 0.403 | 0.222 |
| fake | shuffled_samerows_aware | 149 | adding-knowledge | density | 1.706 | 1.626 | -0.080 | -0.255 | 0.093 | 0.376 |
| fake | shuffled_samerows_aware | 149 | uncertainty-estimation | density | 0.606 | 0.612 | 0.006 | -0.109 | 0.118 | 0.924 |
| fake | shuffled_samerows_aware | 149 | backtracking | density | 0.279 | 0.284 | 0.005 | -0.063 | 0.071 | 0.887 |
| fake | shuffled_samerows_aware | 149 | deduction | density_testlex | 0.131 | 0.135 | 0.004 | -0.044 | 0.052 | 0.876 |
| fake | shuffled_samerows_aware | 149 | adding-knowledge | density_testlex | 0.016 | 0.011 | -0.004 | -0.020 | 0.012 | 0.615 |
| fake | shuffled_samerows_aware | 149 | uncertainty-estimation | density_testlex | 0.062 | 0.068 | 0.006 | -0.024 | 0.038 | 0.713 |
| fake | shuffled_samerows_aware | 149 | backtracking | density_testlex | 0.004 | 0.003 | -0.001 | -0.005 | 0.003 | 0.701 |
| fake | shuffled_samerows_aware | 149 | deduction | density_nontest | 3.712 | 3.859 | 0.147 | -0.114 | 0.401 | 0.236 |
| fake | shuffled_samerows_aware | 149 | adding-knowledge | density_nontest | 1.690 | 1.614 | -0.076 | -0.254 | 0.095 | 0.400 |
| fake | shuffled_samerows_aware | 149 | uncertainty-estimation | density_nontest | 0.545 | 0.545 | -0.000 | -0.112 | 0.113 | 0.995 |
| fake | shuffled_samerows_aware | 149 | backtracking | density_nontest | 0.275 | 0.280 | 0.006 | -0.063 | 0.071 | 0.869 |
| fake | unaware | 149 | deduction | density | 3.843 | 3.951 | 0.107 | -0.147 | 0.348 | 0.391 |
| fake | unaware | 149 | adding-knowledge | density | 1.706 | 1.506 | -0.200 | -0.369 | -0.038 | 0.023 |
| fake | unaware | 149 | uncertainty-estimation | density | 0.606 | 0.720 | 0.113 | -0.011 | 0.238 | 0.075 |
| fake | unaware | 149 | backtracking | density | 0.279 | 0.287 | 0.009 | -0.067 | 0.078 | 0.817 |
| fake | unaware | 149 | deduction | density_testlex | 0.131 | 0.128 | -0.003 | -0.051 | 0.044 | 0.904 |
| fake | unaware | 149 | adding-knowledge | density_testlex | 0.016 | 0.010 | -0.006 | -0.021 | 0.009 | 0.458 |
| fake | unaware | 149 | uncertainty-estimation | density_testlex | 0.062 | 0.082 | 0.020 | -0.013 | 0.056 | 0.260 |
| fake | unaware | 149 | backtracking | density_testlex | 0.004 | 0.002 | -0.002 | -0.008 | 0.003 | 0.402 |
| fake | unaware | 149 | deduction | density_nontest | 3.712 | 3.822 | 0.110 | -0.140 | 0.350 | 0.375 |
| fake | unaware | 149 | adding-knowledge | density_nontest | 1.690 | 1.496 | -0.194 | -0.362 | -0.032 | 0.027 |
| fake | unaware | 149 | uncertainty-estimation | density_nontest | 0.545 | 0.638 | 0.093 | -0.026 | 0.218 | 0.137 |
| fake | unaware | 149 | backtracking | density_nontest | 0.275 | 0.286 | 0.011 | -0.063 | 0.081 | 0.766 |
| fake | unaware_strong | 149 | deduction | density | 3.843 | 3.984 | 0.141 | -0.089 | 0.374 | 0.239 |
| fake | unaware_strong | 149 | adding-knowledge | density | 1.706 | 1.524 | -0.182 | -0.338 | -0.014 | 0.033 |
| fake | unaware_strong | 149 | uncertainty-estimation | density | 0.606 | 0.644 | 0.038 | -0.083 | 0.155 | 0.534 |
| fake | unaware_strong | 149 | backtracking | density | 0.279 | 0.281 | 0.002 | -0.071 | 0.071 | 0.946 |
| fake | unaware_strong | 149 | deduction | density_testlex | 0.131 | 0.128 | -0.003 | -0.053 | 0.045 | 0.892 |
| fake | unaware_strong | 149 | adding-knowledge | density_testlex | 0.016 | 0.007 | -0.008 | -0.023 | 0.004 | 0.241 |
| fake | unaware_strong | 149 | uncertainty-estimation | density_testlex | 0.062 | 0.059 | -0.003 | -0.035 | 0.029 | 0.875 |
| fake | unaware_strong | 149 | backtracking | density_testlex | 0.004 | 0.002 | -0.002 | -0.009 | 0.004 | 0.530 |
| fake | unaware_strong | 149 | deduction | density_nontest | 3.712 | 3.856 | 0.144 | -0.085 | 0.372 | 0.220 |
| fake | unaware_strong | 149 | adding-knowledge | density_nontest | 1.690 | 1.517 | -0.173 | -0.331 | -0.009 | 0.043 |
| fake | unaware_strong | 149 | uncertainty-estimation | density_nontest | 0.545 | 0.585 | 0.041 | -0.078 | 0.152 | 0.495 |
| fake | unaware_strong | 149 | backtracking | density_nontest | 0.275 | 0.279 | 0.004 | -0.069 | 0.073 | 0.902 |
| real | aware | 149 | deduction | density | 4.160 | 4.020 | -0.140 | -0.362 | 0.071 | 0.201 |
| real | aware | 149 | adding-knowledge | density | 1.570 | 1.700 | 0.130 | -0.032 | 0.308 | 0.153 |
| real | aware | 149 | uncertainty-estimation | density | 0.721 | 0.721 | 0.001 | -0.145 | 0.146 | 0.995 |
| real | aware | 149 | backtracking | density | 0.279 | 0.284 | 0.005 | -0.067 | 0.083 | 0.902 |
| real | aware | 149 | deduction | density_testlex | 0.157 | 0.164 | 0.007 | -0.037 | 0.055 | 0.763 |
| real | aware | 149 | adding-knowledge | density_testlex | 0.016 | 0.003 | -0.013 | -0.027 | -0.000 | 0.056 |
| real | aware | 149 | uncertainty-estimation | density_testlex | 0.132 | 0.086 | -0.046 | -0.093 | -0.002 | 0.042 |
| real | aware | 149 | backtracking | density_testlex | 0.004 | 0.000 | -0.004 | -0.009 | 0.000 | 0.158 |
| real | aware | 149 | deduction | density_nontest | 4.003 | 3.856 | -0.147 | -0.368 | 0.070 | 0.186 |
| real | aware | 149 | adding-knowledge | density_nontest | 1.554 | 1.697 | 0.143 | -0.023 | 0.320 | 0.117 |
| real | aware | 149 | uncertainty-estimation | density_nontest | 0.589 | 0.635 | 0.047 | -0.100 | 0.190 | 0.530 |
| real | aware | 149 | backtracking | density_nontest | 0.276 | 0.284 | 0.009 | -0.064 | 0.087 | 0.828 |
| real | aware_strong | 149 | deduction | density | 4.160 | 3.965 | -0.195 | -0.428 | 0.016 | 0.088 |
| real | aware_strong | 149 | adding-knowledge | density | 1.570 | 1.811 | 0.242 | 0.080 | 0.397 | 0.006 |
| real | aware_strong | 149 | uncertainty-estimation | density | 0.721 | 0.637 | -0.084 | -0.224 | 0.042 | 0.229 |
| real | aware_strong | 149 | backtracking | density | 0.279 | 0.232 | -0.047 | -0.121 | 0.021 | 0.192 |
| real | aware_strong | 149 | deduction | density_testlex | 0.157 | 0.158 | 0.001 | -0.041 | 0.046 | 0.972 |
| real | aware_strong | 149 | adding-knowledge | density_testlex | 0.016 | 0.022 | 0.006 | -0.015 | 0.028 | 0.579 |
| real | aware_strong | 149 | uncertainty-estimation | density_testlex | 0.132 | 0.110 | -0.022 | -0.069 | 0.024 | 0.368 |
| real | aware_strong | 149 | backtracking | density_testlex | 0.004 | 0.004 | 0.000 | -0.006 | 0.007 | 0.945 |
| real | aware_strong | 149 | deduction | density_nontest | 4.003 | 3.808 | -0.195 | -0.425 | 0.015 | 0.092 |
| real | aware_strong | 149 | adding-knowledge | density_nontest | 1.554 | 1.789 | 0.236 | 0.069 | 0.396 | 0.008 |
| real | aware_strong | 149 | uncertainty-estimation | density_nontest | 0.589 | 0.526 | -0.062 | -0.196 | 0.055 | 0.344 |
| real | aware_strong | 149 | backtracking | density_nontest | 0.276 | 0.228 | -0.047 | -0.121 | 0.018 | 0.189 |
| real | baseline_seed1 | 149 | deduction | density | 4.160 | 3.920 | -0.240 | -0.448 | -0.048 | 0.024 |
| real | baseline_seed1 | 149 | adding-knowledge | density | 1.570 | 1.619 | 0.049 | -0.112 | 0.203 | 0.552 |
| real | baseline_seed1 | 149 | uncertainty-estimation | density | 0.721 | 0.753 | 0.032 | -0.097 | 0.161 | 0.632 |
| real | baseline_seed1 | 149 | backtracking | density | 0.279 | 0.269 | -0.010 | -0.086 | 0.059 | 0.794 |
| real | baseline_seed1 | 149 | deduction | density_testlex | 0.157 | 0.143 | -0.013 | -0.055 | 0.025 | 0.524 |
| real | baseline_seed1 | 149 | adding-knowledge | density_testlex | 0.016 | 0.023 | 0.006 | -0.010 | 0.025 | 0.477 |
| real | baseline_seed1 | 149 | uncertainty-estimation | density_testlex | 0.132 | 0.117 | -0.015 | -0.059 | 0.029 | 0.504 |
| real | baseline_seed1 | 149 | backtracking | density_testlex | 0.004 | 0.006 | 0.002 | -0.006 | 0.014 | 0.673 |
| real | baseline_seed1 | 149 | deduction | density_nontest | 4.003 | 3.776 | -0.227 | -0.440 | -0.032 | 0.036 |
| real | baseline_seed1 | 149 | adding-knowledge | density_nontest | 1.554 | 1.596 | 0.043 | -0.120 | 0.193 | 0.601 |
| real | baseline_seed1 | 149 | uncertainty-estimation | density_nontest | 0.589 | 0.636 | 0.047 | -0.079 | 0.168 | 0.468 |
| real | baseline_seed1 | 149 | backtracking | density_nontest | 0.276 | 0.263 | -0.012 | -0.089 | 0.059 | 0.751 |
| real | random_aware | 149 | deduction | density | 4.160 | 4.125 | -0.035 | -0.283 | 0.202 | 0.777 |
| real | random_aware | 149 | adding-knowledge | density | 1.570 | 1.580 | 0.010 | -0.142 | 0.167 | 0.902 |
| real | random_aware | 149 | uncertainty-estimation | density | 0.721 | 0.689 | -0.032 | -0.150 | 0.082 | 0.600 |
| real | random_aware | 149 | backtracking | density | 0.279 | 0.270 | -0.010 | -0.087 | 0.064 | 0.806 |
| real | random_aware | 149 | deduction | density_testlex | 0.157 | 0.136 | -0.021 | -0.063 | 0.019 | 0.317 |
| real | random_aware | 149 | adding-knowledge | density_testlex | 0.016 | 0.011 | -0.005 | -0.023 | 0.013 | 0.551 |
| real | random_aware | 149 | uncertainty-estimation | density_testlex | 0.132 | 0.136 | 0.004 | -0.042 | 0.046 | 0.861 |
| real | random_aware | 149 | backtracking | density_testlex | 0.004 | 0.002 | -0.002 | -0.008 | 0.003 | 0.501 |
| real | random_aware | 149 | deduction | density_nontest | 4.003 | 3.990 | -0.013 | -0.258 | 0.226 | 0.914 |
| real | random_aware | 149 | adding-knowledge | density_nontest | 1.554 | 1.569 | 0.016 | -0.140 | 0.173 | 0.849 |
| real | random_aware | 149 | uncertainty-estimation | density_nontest | 0.589 | 0.553 | -0.036 | -0.151 | 0.072 | 0.540 |
| real | random_aware | 149 | backtracking | density_nontest | 0.276 | 0.268 | -0.008 | -0.087 | 0.066 | 0.846 |
| real | random_samerows_aware | 149 | deduction | density | 4.160 | 3.986 | -0.174 | -0.385 | 0.045 | 0.121 |
| real | random_samerows_aware | 149 | adding-knowledge | density | 1.570 | 1.586 | 0.016 | -0.139 | 0.166 | 0.846 |
| real | random_samerows_aware | 149 | uncertainty-estimation | density | 0.721 | 0.745 | 0.024 | -0.115 | 0.153 | 0.725 |
| real | random_samerows_aware | 149 | backtracking | density | 0.279 | 0.244 | -0.036 | -0.113 | 0.038 | 0.355 |
| real | random_samerows_aware | 149 | deduction | density_testlex | 0.157 | 0.157 | 0.000 | -0.045 | 0.045 | 0.998 |
| real | random_samerows_aware | 149 | adding-knowledge | density_testlex | 0.016 | 0.024 | 0.008 | -0.011 | 0.028 | 0.457 |
| real | random_samerows_aware | 149 | uncertainty-estimation | density_testlex | 0.132 | 0.108 | -0.024 | -0.073 | 0.021 | 0.297 |
| real | random_samerows_aware | 149 | backtracking | density_testlex | 0.004 | 0.006 | 0.002 | -0.006 | 0.011 | 0.633 |
| real | random_samerows_aware | 149 | deduction | density_nontest | 4.003 | 3.829 | -0.174 | -0.385 | 0.040 | 0.123 |
| real | random_samerows_aware | 149 | adding-knowledge | density_nontest | 1.554 | 1.562 | 0.009 | -0.148 | 0.157 | 0.918 |
| real | random_samerows_aware | 149 | uncertainty-estimation | density_nontest | 0.589 | 0.637 | 0.049 | -0.082 | 0.174 | 0.472 |
| real | random_samerows_aware | 149 | backtracking | density_nontest | 0.276 | 0.238 | -0.038 | -0.114 | 0.034 | 0.317 |
| real | shuffled_aware | 149 | deduction | density | 4.160 | 4.117 | -0.043 | -0.264 | 0.157 | 0.698 |
| real | shuffled_aware | 149 | adding-knowledge | density | 1.570 | 1.630 | 0.060 | -0.108 | 0.229 | 0.497 |
| real | shuffled_aware | 149 | uncertainty-estimation | density | 0.721 | 0.697 | -0.024 | -0.144 | 0.092 | 0.688 |
| real | shuffled_aware | 149 | backtracking | density | 0.279 | 0.273 | -0.006 | -0.086 | 0.074 | 0.882 |
| real | shuffled_aware | 149 | deduction | density_testlex | 0.157 | 0.168 | 0.011 | -0.034 | 0.052 | 0.640 |
| real | shuffled_aware | 149 | adding-knowledge | density_testlex | 0.016 | 0.009 | -0.007 | -0.020 | 0.005 | 0.256 |
| real | shuffled_aware | 149 | uncertainty-estimation | density_testlex | 0.132 | 0.093 | -0.039 | -0.085 | 0.006 | 0.086 |
| real | shuffled_aware | 149 | backtracking | density_testlex | 0.004 | 0.001 | -0.003 | -0.009 | 0.001 | 0.238 |
| real | shuffled_aware | 149 | deduction | density_nontest | 4.003 | 3.949 | -0.054 | -0.266 | 0.151 | 0.627 |
| real | shuffled_aware | 149 | adding-knowledge | density_nontest | 1.554 | 1.621 | 0.068 | -0.103 | 0.236 | 0.445 |
| real | shuffled_aware | 149 | uncertainty-estimation | density_nontest | 0.589 | 0.604 | 0.015 | -0.095 | 0.123 | 0.790 |
| real | shuffled_aware | 149 | backtracking | density_nontest | 0.276 | 0.273 | -0.003 | -0.085 | 0.076 | 0.943 |
| real | shuffled_samerows_aware | 149 | deduction | density | 4.160 | 3.890 | -0.270 | -0.492 | -0.066 | 0.015 |
| real | shuffled_samerows_aware | 149 | adding-knowledge | density | 1.570 | 1.643 | 0.073 | -0.087 | 0.224 | 0.377 |
| real | shuffled_samerows_aware | 149 | uncertainty-estimation | density | 0.721 | 0.733 | 0.013 | -0.123 | 0.142 | 0.858 |
| real | shuffled_samerows_aware | 149 | backtracking | density | 0.279 | 0.252 | -0.027 | -0.103 | 0.043 | 0.463 |
| real | shuffled_samerows_aware | 149 | deduction | density_testlex | 0.157 | 0.179 | 0.022 | -0.017 | 0.057 | 0.279 |
| real | shuffled_samerows_aware | 149 | adding-knowledge | density_testlex | 0.016 | 0.017 | 0.001 | -0.015 | 0.017 | 0.929 |
| real | shuffled_samerows_aware | 149 | uncertainty-estimation | density_testlex | 0.132 | 0.150 | 0.018 | -0.018 | 0.057 | 0.353 |
| real | shuffled_samerows_aware | 149 | backtracking | density_testlex | 0.004 | 0.002 | -0.002 | -0.008 | 0.004 | 0.570 |
| real | shuffled_samerows_aware | 149 | deduction | density_nontest | 4.003 | 3.711 | -0.292 | -0.516 | -0.087 | 0.009 |
| real | shuffled_samerows_aware | 149 | adding-knowledge | density_nontest | 1.554 | 1.626 | 0.072 | -0.086 | 0.223 | 0.380 |
| real | shuffled_samerows_aware | 149 | uncertainty-estimation | density_nontest | 0.589 | 0.583 | -0.006 | -0.140 | 0.126 | 0.932 |
| real | shuffled_samerows_aware | 149 | backtracking | density_nontest | 0.276 | 0.250 | -0.025 | -0.101 | 0.044 | 0.494 |
| real | unaware | 149 | deduction | density | 4.160 | 4.015 | -0.145 | -0.383 | 0.083 | 0.233 |
| real | unaware | 149 | adding-knowledge | density | 1.570 | 1.588 | 0.018 | -0.141 | 0.178 | 0.825 |
| real | unaware | 149 | uncertainty-estimation | density | 0.721 | 0.707 | -0.014 | -0.141 | 0.108 | 0.832 |
| real | unaware | 149 | backtracking | density | 0.279 | 0.300 | 0.021 | -0.056 | 0.099 | 0.598 |
| real | unaware | 149 | deduction | density_testlex | 0.157 | 0.138 | -0.019 | -0.061 | 0.022 | 0.391 |
| real | unaware | 149 | adding-knowledge | density_testlex | 0.016 | 0.007 | -0.009 | -0.024 | 0.005 | 0.246 |
| real | unaware | 149 | uncertainty-estimation | density_testlex | 0.132 | 0.123 | -0.010 | -0.053 | 0.034 | 0.674 |
| real | unaware | 149 | backtracking | density_testlex | 0.004 | 0.003 | -0.001 | -0.008 | 0.006 | 0.759 |
| real | unaware | 149 | deduction | density_nontest | 4.003 | 3.877 | -0.126 | -0.370 | 0.105 | 0.308 |
| real | unaware | 149 | adding-knowledge | density_nontest | 1.554 | 1.581 | 0.027 | -0.132 | 0.184 | 0.744 |
| real | unaware | 149 | uncertainty-estimation | density_nontest | 0.589 | 0.585 | -0.004 | -0.127 | 0.112 | 0.947 |
| real | unaware | 149 | backtracking | density_nontest | 0.276 | 0.298 | 0.022 | -0.054 | 0.100 | 0.576 |
| real | unaware_strong | 149 | deduction | density | 4.160 | 4.123 | -0.037 | -0.248 | 0.175 | 0.729 |
| real | unaware_strong | 149 | adding-knowledge | density | 1.570 | 1.648 | 0.078 | -0.091 | 0.246 | 0.371 |
| real | unaware_strong | 149 | uncertainty-estimation | density | 0.721 | 0.710 | -0.011 | -0.147 | 0.119 | 0.879 |
| real | unaware_strong | 149 | backtracking | density | 0.279 | 0.209 | -0.070 | -0.148 | 0.000 | 0.059 |
| real | unaware_strong | 149 | deduction | density_testlex | 0.157 | 0.131 | -0.026 | -0.065 | 0.016 | 0.227 |
| real | unaware_strong | 149 | adding-knowledge | density_testlex | 0.016 | 0.024 | 0.007 | -0.015 | 0.030 | 0.519 |
| real | unaware_strong | 149 | uncertainty-estimation | density_testlex | 0.132 | 0.090 | -0.042 | -0.084 | -0.000 | 0.059 |
| real | unaware_strong | 149 | backtracking | density_testlex | 0.004 | 0.003 | -0.001 | -0.008 | 0.007 | 0.842 |
| real | unaware_strong | 149 | deduction | density_nontest | 4.003 | 3.992 | -0.011 | -0.227 | 0.200 | 0.917 |
| real | unaware_strong | 149 | adding-knowledge | density_nontest | 1.554 | 1.624 | 0.071 | -0.099 | 0.245 | 0.422 |
| real | unaware_strong | 149 | uncertainty-estimation | density_nontest | 0.589 | 0.620 | 0.031 | -0.098 | 0.152 | 0.645 |
| real | unaware_strong | 149 | backtracking | density_nontest | 0.276 | 0.206 | -0.069 | -0.146 | 0.004 | 0.063 |


## Morphology Δ by compliance class (decision-change confound): `refuse_both` / `comply_both` hold the decision fixed

| framing | arm | flip_class | n | behaviour | metric | delta_density | ci_lo | ci_hi |
|---|---|---|---|---|---|---|---|---|
| fake | aware | gained | 5 | deduction | density | 1.388 | -0.258 | 3.034 |
| fake | aware | gained | 5 | deduction | density_nontest | 1.620 | -0.115 | 3.354 |
| fake | aware | gained | 5 | deduction | density_testlex | -0.232 | -0.497 | -0.023 |
| fake | aware | gained | 5 | uncertainty-estimation | density | 0.376 | 0.047 | 0.705 |
| fake | aware | gained | 5 | uncertainty-estimation | density_nontest | 0.511 | -0.081 | 1.127 |
| fake | aware | gained | 5 | uncertainty-estimation | density_testlex | -0.135 | -0.538 | 0.269 |
| fake | aware | gained | 5 | backtracking | density | -0.084 | -0.251 | 0.000 |
| fake | aware | gained | 5 | backtracking | density_nontest | -0.084 | -0.251 | 0.000 |
| fake | aware | gained | 5 | backtracking | density_testlex | 0.000 | 0.000 | 0.000 |
| fake | aware | lost | 8 | deduction | density | 0.894 | 0.076 | 1.678 |
| fake | aware | lost | 8 | deduction | density_nontest | 0.705 | -0.111 | 1.505 |
| fake | aware | lost | 8 | deduction | density_testlex | 0.189 | -0.029 | 0.415 |
| fake | aware | lost | 8 | uncertainty-estimation | density | -0.528 | -0.998 | -0.005 |
| fake | aware | lost | 8 | uncertainty-estimation | density_nontest | -0.391 | -0.814 | 0.076 |
| fake | aware | lost | 8 | uncertainty-estimation | density_testlex | -0.137 | -0.262 | -0.041 |
| fake | aware | lost | 8 | backtracking | density | 0.058 | -0.180 | 0.305 |
| fake | aware | lost | 8 | backtracking | density_nontest | 0.058 | -0.180 | 0.305 |
| fake | aware | lost | 8 | backtracking | density_testlex | 0.000 | 0.000 | 0.000 |
| fake | aware | comply_both | 112 | deduction | density | 0.058 | -0.226 | 0.328 |
| fake | aware | comply_both | 112 | deduction | density_nontest | 0.057 | -0.225 | 0.324 |
| fake | aware | comply_both | 112 | deduction | density_testlex | 0.001 | -0.046 | 0.046 |
| fake | aware | comply_both | 112 | uncertainty-estimation | density | 0.044 | -0.106 | 0.206 |
| fake | aware | comply_both | 112 | uncertainty-estimation | density_nontest | 0.017 | -0.122 | 0.172 |
| fake | aware | comply_both | 112 | uncertainty-estimation | density_testlex | 0.026 | -0.007 | 0.061 |
| fake | aware | comply_both | 112 | backtracking | density | 0.065 | -0.009 | 0.141 |
| fake | aware | comply_both | 112 | backtracking | density_nontest | 0.066 | -0.008 | 0.143 |
| fake | aware | comply_both | 112 | backtracking | density_testlex | -0.001 | -0.005 | 0.002 |
| fake | aware | refuse_both | 12 | deduction | density | 0.232 | -0.392 | 0.800 |
| fake | aware | refuse_both | 12 | deduction | density_nontest | 0.100 | -0.424 | 0.591 |
| fake | aware | refuse_both | 12 | deduction | density_testlex | 0.132 | -0.058 | 0.313 |
| fake | aware | refuse_both | 12 | uncertainty-estimation | density | -0.200 | -0.553 | 0.156 |
| fake | aware | refuse_both | 12 | uncertainty-estimation | density_nontest | -0.244 | -0.506 | 0.040 |
| fake | aware | refuse_both | 12 | uncertainty-estimation | density_testlex | 0.044 | -0.114 | 0.222 |
| fake | aware | refuse_both | 12 | backtracking | density | -0.164 | -0.424 | 0.110 |
| fake | aware | refuse_both | 12 | backtracking | density_nontest | -0.150 | -0.392 | 0.115 |
| fake | aware | refuse_both | 12 | backtracking | density_testlex | -0.014 | -0.042 | 0.000 |
| fake | aware_strong | comply_both | 118 | deduction | density | -0.021 | -0.294 | 0.260 |
| fake | aware_strong | comply_both | 118 | deduction | density_nontest | 0.015 | -0.262 | 0.298 |
| fake | aware_strong | comply_both | 118 | deduction | density_testlex | -0.037 | -0.083 | 0.011 |
| fake | aware_strong | comply_both | 118 | uncertainty-estimation | density | 0.153 | 0.032 | 0.278 |
| fake | aware_strong | comply_both | 118 | uncertainty-estimation | density_nontest | 0.121 | 0.003 | 0.246 |
| fake | aware_strong | comply_both | 118 | uncertainty-estimation | density_testlex | 0.033 | -0.009 | 0.075 |
| fake | aware_strong | comply_both | 118 | backtracking | density | 0.054 | -0.012 | 0.118 |
| fake | aware_strong | comply_both | 118 | backtracking | density_nontest | 0.055 | -0.011 | 0.118 |
| fake | aware_strong | comply_both | 118 | backtracking | density_testlex | -0.000 | -0.008 | 0.009 |
| fake | aware_strong | refuse_both | 13 | deduction | density | 0.170 | -0.274 | 0.632 |
| fake | aware_strong | refuse_both | 13 | deduction | density_nontest | 0.176 | -0.308 | 0.714 |
| fake | aware_strong | refuse_both | 13 | deduction | density_testlex | -0.006 | -0.169 | 0.149 |
| fake | aware_strong | refuse_both | 13 | uncertainty-estimation | density | 0.082 | -0.297 | 0.462 |
| fake | aware_strong | refuse_both | 13 | uncertainty-estimation | density_nontest | -0.009 | -0.322 | 0.354 |
| fake | aware_strong | refuse_both | 13 | uncertainty-estimation | density_testlex | 0.091 | -0.144 | 0.327 |
| fake | aware_strong | refuse_both | 13 | backtracking | density | -0.230 | -0.379 | -0.093 |
| fake | aware_strong | refuse_both | 13 | backtracking | density_nontest | -0.217 | -0.354 | -0.089 |
| fake | aware_strong | refuse_both | 13 | backtracking | density_testlex | -0.013 | -0.039 | 0.000 |
| fake | baseline_seed1 | gained | 6 | deduction | density | 1.033 | 0.286 | 1.868 |
| fake | baseline_seed1 | gained | 6 | deduction | density_nontest | 1.010 | 0.076 | 2.033 |
| fake | baseline_seed1 | gained | 6 | deduction | density_testlex | 0.023 | -0.205 | 0.223 |
| fake | baseline_seed1 | gained | 6 | uncertainty-estimation | density | 0.178 | -0.130 | 0.433 |
| fake | baseline_seed1 | gained | 6 | uncertainty-estimation | density_nontest | 0.093 | -0.176 | 0.367 |
| fake | baseline_seed1 | gained | 6 | uncertainty-estimation | density_testlex | 0.085 | -0.262 | 0.384 |
| fake | baseline_seed1 | gained | 6 | backtracking | density | -0.298 | -0.627 | -0.052 |
| fake | baseline_seed1 | gained | 6 | backtracking | density_nontest | -0.270 | -0.543 | -0.052 |
| fake | baseline_seed1 | gained | 6 | backtracking | density_testlex | -0.028 | -0.084 | 0.000 |
| fake | baseline_seed1 | lost | 5 | deduction | density | -0.516 | -1.771 | 0.850 |
| fake | baseline_seed1 | lost | 5 | deduction | density_nontest | -0.729 | -2.137 | 0.737 |
| fake | baseline_seed1 | lost | 5 | deduction | density_testlex | 0.213 | -0.100 | 0.539 |
| fake | baseline_seed1 | lost | 5 | uncertainty-estimation | density | -0.200 | -0.884 | 0.380 |
| fake | baseline_seed1 | lost | 5 | uncertainty-estimation | density_nontest | -0.231 | -0.915 | 0.323 |
| fake | baseline_seed1 | lost | 5 | uncertainty-estimation | density_testlex | 0.031 | 0.000 | 0.093 |
| fake | baseline_seed1 | lost | 5 | backtracking | density | -0.146 | -0.360 | 0.131 |
| fake | baseline_seed1 | lost | 5 | backtracking | density_nontest | -0.146 | -0.360 | 0.131 |
| fake | baseline_seed1 | lost | 5 | backtracking | density_testlex | 0.000 | 0.000 | 0.000 |
| fake | baseline_seed1 | comply_both | 119 | deduction | density | 0.268 | 0.017 | 0.543 |
| fake | baseline_seed1 | comply_both | 119 | deduction | density_nontest | 0.288 | 0.038 | 0.575 |
| fake | baseline_seed1 | comply_both | 119 | deduction | density_testlex | -0.020 | -0.072 | 0.032 |
| fake | baseline_seed1 | comply_both | 119 | uncertainty-estimation | density | 0.041 | -0.099 | 0.174 |
| fake | baseline_seed1 | comply_both | 119 | uncertainty-estimation | density_nontest | 0.014 | -0.113 | 0.147 |
| fake | baseline_seed1 | comply_both | 119 | uncertainty-estimation | density_testlex | 0.027 | -0.012 | 0.068 |
| fake | baseline_seed1 | comply_both | 119 | backtracking | density | 0.025 | -0.037 | 0.088 |
| fake | baseline_seed1 | comply_both | 119 | backtracking | density_nontest | 0.027 | -0.037 | 0.089 |
| fake | baseline_seed1 | comply_both | 119 | backtracking | density_testlex | -0.001 | -0.008 | 0.005 |
| fake | baseline_seed1 | refuse_both | 11 | deduction | density | 0.397 | -0.297 | 1.076 |
| fake | baseline_seed1 | refuse_both | 11 | deduction | density_nontest | 0.515 | -0.249 | 1.229 |
| fake | baseline_seed1 | refuse_both | 11 | deduction | density_testlex | -0.118 | -0.312 | 0.060 |
| fake | baseline_seed1 | refuse_both | 11 | uncertainty-estimation | density | -0.019 | -0.362 | 0.317 |
| fake | baseline_seed1 | refuse_both | 11 | uncertainty-estimation | density_nontest | -0.022 | -0.263 | 0.225 |
| fake | baseline_seed1 | refuse_both | 11 | uncertainty-estimation | density_testlex | 0.002 | -0.185 | 0.190 |
| fake | baseline_seed1 | refuse_both | 11 | backtracking | density | 0.033 | -0.305 | 0.374 |
| fake | baseline_seed1 | refuse_both | 11 | backtracking | density_nontest | 0.013 | -0.305 | 0.314 |
| fake | baseline_seed1 | refuse_both | 11 | backtracking | density_testlex | 0.020 | 0.000 | 0.061 |
| fake | random_aware | gained | 6 | deduction | density | 0.084 | -0.320 | 0.461 |
| fake | random_aware | gained | 6 | deduction | density_nontest | 0.529 | 0.187 | 0.812 |
| fake | random_aware | gained | 6 | deduction | density_testlex | -0.445 | -0.719 | -0.189 |
| fake | random_aware | gained | 6 | uncertainty-estimation | density | 0.596 | -0.059 | 1.228 |
| fake | random_aware | gained | 6 | uncertainty-estimation | density_nontest | 0.455 | -0.185 | 1.082 |
| fake | random_aware | gained | 6 | uncertainty-estimation | density_testlex | 0.141 | -0.003 | 0.266 |
| fake | random_aware | gained | 6 | backtracking | density | -0.281 | -0.608 | 0.033 |
| fake | random_aware | gained | 6 | backtracking | density_nontest | -0.253 | -0.576 | 0.050 |
| fake | random_aware | gained | 6 | backtracking | density_testlex | -0.028 | -0.084 | 0.000 |
| fake | random_aware | comply_both | 118 | deduction | density | 0.229 | -0.042 | 0.492 |
| fake | random_aware | comply_both | 118 | deduction | density_nontest | 0.240 | -0.034 | 0.506 |
| fake | random_aware | comply_both | 118 | deduction | density_testlex | -0.011 | -0.056 | 0.035 |
| fake | random_aware | comply_both | 118 | uncertainty-estimation | density | 0.038 | -0.091 | 0.164 |
| fake | random_aware | comply_both | 118 | uncertainty-estimation | density_nontest | 0.027 | -0.099 | 0.156 |
| fake | random_aware | comply_both | 118 | uncertainty-estimation | density_testlex | 0.011 | -0.024 | 0.045 |
| fake | random_aware | comply_both | 118 | backtracking | density | 0.039 | -0.027 | 0.111 |
| fake | random_aware | comply_both | 118 | backtracking | density_nontest | 0.040 | -0.026 | 0.110 |
| fake | random_aware | comply_both | 118 | backtracking | density_testlex | -0.000 | -0.008 | 0.007 |
| fake | random_aware | refuse_both | 11 | deduction | density | -0.254 | -0.699 | 0.172 |
| fake | random_aware | refuse_both | 11 | deduction | density_nontest | -0.346 | -0.937 | 0.217 |
| fake | random_aware | refuse_both | 11 | deduction | density_testlex | 0.092 | -0.139 | 0.331 |
| fake | random_aware | refuse_both | 11 | uncertainty-estimation | density | 0.038 | -0.465 | 0.585 |
| fake | random_aware | refuse_both | 11 | uncertainty-estimation | density_nontest | 0.054 | -0.357 | 0.532 |
| fake | random_aware | refuse_both | 11 | uncertainty-estimation | density_testlex | -0.016 | -0.179 | 0.141 |
| fake | random_aware | refuse_both | 11 | backtracking | density | 0.039 | -0.146 | 0.249 |
| fake | random_aware | refuse_both | 11 | backtracking | density_nontest | 0.039 | -0.146 | 0.249 |
| fake | random_aware | refuse_both | 11 | backtracking | density_testlex | 0.000 | 0.000 | 0.000 |
| fake | random_samerows_aware | gained | 7 | deduction | density | 0.656 | -0.022 | 1.463 |
| fake | random_samerows_aware | gained | 7 | deduction | density_nontest | 0.959 | 0.156 | 1.879 |
| fake | random_samerows_aware | gained | 7 | deduction | density_testlex | -0.303 | -0.530 | -0.043 |
| fake | random_samerows_aware | gained | 7 | uncertainty-estimation | density | 0.133 | -0.203 | 0.353 |
| fake | random_samerows_aware | gained | 7 | uncertainty-estimation | density_nontest | 0.146 | -0.218 | 0.514 |
| fake | random_samerows_aware | gained | 7 | uncertainty-estimation | density_testlex | -0.013 | -0.264 | 0.204 |
| fake | random_samerows_aware | gained | 7 | backtracking | density | -0.000 | -0.434 | 0.441 |
| fake | random_samerows_aware | gained | 7 | backtracking | density_nontest | -0.005 | -0.409 | 0.368 |
| fake | random_samerows_aware | gained | 7 | backtracking | density_testlex | 0.005 | -0.072 | 0.088 |
| fake | random_samerows_aware | lost | 5 | deduction | density | -0.050 | -1.594 | 1.393 |
| fake | random_samerows_aware | lost | 5 | deduction | density_nontest | -0.145 | -1.327 | 1.027 |
| fake | random_samerows_aware | lost | 5 | deduction | density_testlex | 0.094 | -0.322 | 0.477 |
| fake | random_samerows_aware | lost | 5 | uncertainty-estimation | density | -0.134 | -0.974 | 0.669 |
| fake | random_samerows_aware | lost | 5 | uncertainty-estimation | density_nontest | -0.178 | -0.793 | 0.220 |
| fake | random_samerows_aware | lost | 5 | uncertainty-estimation | density_testlex | 0.043 | -0.422 | 0.607 |
| fake | random_samerows_aware | lost | 5 | backtracking | density | 0.109 | -0.158 | 0.413 |
| fake | random_samerows_aware | lost | 5 | backtracking | density_nontest | 0.109 | -0.158 | 0.413 |
| fake | random_samerows_aware | lost | 5 | backtracking | density_testlex | 0.000 | 0.000 | 0.000 |
| fake | random_samerows_aware | comply_both | 119 | deduction | density | 0.081 | -0.173 | 0.339 |
| fake | random_samerows_aware | comply_both | 119 | deduction | density_nontest | 0.110 | -0.142 | 0.363 |
| fake | random_samerows_aware | comply_both | 119 | deduction | density_testlex | -0.030 | -0.073 | 0.017 |
| fake | random_samerows_aware | comply_both | 119 | uncertainty-estimation | density | 0.011 | -0.117 | 0.143 |
| fake | random_samerows_aware | comply_both | 119 | uncertainty-estimation | density_nontest | -0.012 | -0.136 | 0.110 |
| fake | random_samerows_aware | comply_both | 119 | uncertainty-estimation | density_testlex | 0.023 | -0.012 | 0.060 |
| fake | random_samerows_aware | comply_both | 119 | backtracking | density | 0.005 | -0.059 | 0.070 |
| fake | random_samerows_aware | comply_both | 119 | backtracking | density_nontest | 0.005 | -0.058 | 0.071 |
| fake | random_samerows_aware | comply_both | 119 | backtracking | density_testlex | 0.000 | -0.004 | 0.005 |
| fake | random_samerows_aware | refuse_both | 10 | deduction | density | -0.455 | -0.966 | 0.115 |
| fake | random_samerows_aware | refuse_both | 10 | deduction | density_nontest | -0.534 | -1.090 | 0.065 |
| fake | random_samerows_aware | refuse_both | 10 | deduction | density_testlex | 0.079 | -0.108 | 0.231 |
| fake | random_samerows_aware | refuse_both | 10 | uncertainty-estimation | density | 0.064 | -0.261 | 0.389 |
| fake | random_samerows_aware | refuse_both | 10 | uncertainty-estimation | density_nontest | -0.002 | -0.308 | 0.342 |
| fake | random_samerows_aware | refuse_both | 10 | uncertainty-estimation | density_testlex | 0.065 | -0.079 | 0.217 |
| fake | random_samerows_aware | refuse_both | 10 | backtracking | density | -0.016 | -0.199 | 0.173 |
| fake | random_samerows_aware | refuse_both | 10 | backtracking | density_nontest | -0.016 | -0.199 | 0.173 |
| fake | random_samerows_aware | refuse_both | 10 | backtracking | density_testlex | 0.000 | 0.000 | 0.000 |
| fake | shuffled_aware | comply_both | 117 | deduction | density | 0.236 | -0.049 | 0.516 |
| fake | shuffled_aware | comply_both | 117 | deduction | density_nontest | 0.251 | -0.032 | 0.524 |
| fake | shuffled_aware | comply_both | 117 | deduction | density_testlex | -0.014 | -0.060 | 0.027 |
| fake | shuffled_aware | comply_both | 117 | uncertainty-estimation | density | 0.032 | -0.115 | 0.171 |
| fake | shuffled_aware | comply_both | 117 | uncertainty-estimation | density_nontest | 0.021 | -0.125 | 0.164 |
| fake | shuffled_aware | comply_both | 117 | uncertainty-estimation | density_testlex | 0.012 | -0.019 | 0.043 |
| fake | shuffled_aware | comply_both | 117 | backtracking | density | 0.073 | 0.008 | 0.146 |
| fake | shuffled_aware | comply_both | 117 | backtracking | density_nontest | 0.068 | 0.003 | 0.138 |
| fake | shuffled_aware | comply_both | 117 | backtracking | density_testlex | 0.005 | -0.002 | 0.017 |
| fake | shuffled_aware | refuse_both | 13 | deduction | density | 0.032 | -0.658 | 0.796 |
| fake | shuffled_aware | refuse_both | 13 | deduction | density_nontest | 0.040 | -0.619 | 0.781 |
| fake | shuffled_aware | refuse_both | 13 | deduction | density_testlex | -0.008 | -0.213 | 0.210 |
| fake | shuffled_aware | refuse_both | 13 | uncertainty-estimation | density | 0.301 | -0.176 | 0.806 |
| fake | shuffled_aware | refuse_both | 13 | uncertainty-estimation | density_nontest | 0.238 | -0.269 | 0.802 |
| fake | shuffled_aware | refuse_both | 13 | uncertainty-estimation | density_testlex | 0.063 | -0.080 | 0.200 |
| fake | shuffled_aware | refuse_both | 13 | backtracking | density | -0.015 | -0.268 | 0.201 |
| fake | shuffled_aware | refuse_both | 13 | backtracking | density_nontest | -0.015 | -0.268 | 0.201 |
| fake | shuffled_aware | refuse_both | 13 | backtracking | density_testlex | 0.000 | 0.000 | 0.000 |
| fake | shuffled_samerows_aware | gained | 7 | deduction | density | -0.202 | -0.797 | 0.514 |
| fake | shuffled_samerows_aware | gained | 7 | deduction | density_nontest | -0.090 | -0.949 | 0.867 |
| fake | shuffled_samerows_aware | gained | 7 | deduction | density_testlex | -0.112 | -0.459 | 0.213 |
| fake | shuffled_samerows_aware | gained | 7 | uncertainty-estimation | density | 0.509 | 0.330 | 0.694 |
| fake | shuffled_samerows_aware | gained | 7 | uncertainty-estimation | density_nontest | 0.441 | 0.139 | 0.701 |
| fake | shuffled_samerows_aware | gained | 7 | uncertainty-estimation | density_testlex | 0.068 | -0.153 | 0.324 |
| fake | shuffled_samerows_aware | gained | 7 | backtracking | density | -0.195 | -0.589 | 0.221 |
| fake | shuffled_samerows_aware | gained | 7 | backtracking | density_nontest | -0.171 | -0.565 | 0.230 |
| fake | shuffled_samerows_aware | gained | 7 | backtracking | density_testlex | -0.024 | -0.072 | 0.000 |
| fake | shuffled_samerows_aware | lost | 5 | deduction | density | -0.606 | -2.342 | 1.129 |
| fake | shuffled_samerows_aware | lost | 5 | deduction | density_nontest | -0.632 | -2.270 | 1.005 |
| fake | shuffled_samerows_aware | lost | 5 | deduction | density_testlex | 0.027 | -0.107 | 0.187 |
| fake | shuffled_samerows_aware | lost | 5 | uncertainty-estimation | density | -0.131 | -0.734 | 0.485 |
| fake | shuffled_samerows_aware | lost | 5 | uncertainty-estimation | density_nontest | 0.042 | -0.727 | 0.824 |
| fake | shuffled_samerows_aware | lost | 5 | uncertainty-estimation | density_testlex | -0.173 | -0.387 | -0.007 |
| fake | shuffled_samerows_aware | lost | 5 | backtracking | density | 0.271 | 0.106 | 0.437 |
| fake | shuffled_samerows_aware | lost | 5 | backtracking | density_nontest | 0.245 | 0.058 | 0.437 |
| fake | shuffled_samerows_aware | lost | 5 | backtracking | density_testlex | 0.026 | 0.000 | 0.078 |
| fake | shuffled_samerows_aware | comply_both | 119 | deduction | density | 0.200 | -0.064 | 0.474 |
| fake | shuffled_samerows_aware | comply_both | 119 | deduction | density_nontest | 0.199 | -0.058 | 0.475 |
| fake | shuffled_samerows_aware | comply_both | 119 | deduction | density_testlex | 0.001 | -0.052 | 0.052 |
| fake | shuffled_samerows_aware | comply_both | 119 | uncertainty-estimation | density | -0.032 | -0.165 | 0.091 |
| fake | shuffled_samerows_aware | comply_both | 119 | uncertainty-estimation | density_nontest | -0.038 | -0.165 | 0.079 |
| fake | shuffled_samerows_aware | comply_both | 119 | uncertainty-estimation | density_testlex | 0.005 | -0.025 | 0.037 |
| fake | shuffled_samerows_aware | comply_both | 119 | backtracking | density | 0.007 | -0.063 | 0.076 |
| fake | shuffled_samerows_aware | comply_both | 119 | backtracking | density_nontest | 0.009 | -0.062 | 0.077 |
| fake | shuffled_samerows_aware | comply_both | 119 | backtracking | density_testlex | -0.001 | -0.004 | 0.000 |
| fake | shuffled_samerows_aware | refuse_both | 10 | deduction | density | 0.684 | -0.122 | 1.663 |
| fake | shuffled_samerows_aware | refuse_both | 10 | deduction | density_nontest | 0.553 | -0.182 | 1.614 |
| fake | shuffled_samerows_aware | refuse_both | 10 | deduction | density_testlex | 0.131 | -0.146 | 0.409 |
| fake | shuffled_samerows_aware | refuse_both | 10 | uncertainty-estimation | density | 0.005 | -0.386 | 0.380 |
| fake | shuffled_samerows_aware | refuse_both | 10 | uncertainty-estimation | density_nontest | -0.061 | -0.423 | 0.259 |
| fake | shuffled_samerows_aware | refuse_both | 10 | uncertainty-estimation | density_testlex | 0.066 | -0.109 | 0.225 |
| fake | shuffled_samerows_aware | refuse_both | 10 | backtracking | density | -0.116 | -0.304 | 0.112 |
| fake | shuffled_samerows_aware | refuse_both | 10 | backtracking | density_nontest | -0.116 | -0.304 | 0.112 |
| fake | shuffled_samerows_aware | refuse_both | 10 | backtracking | density_testlex | 0.000 | 0.000 | 0.000 |
| fake | unaware | gained | 5 | deduction | density | 0.871 | -0.760 | 2.501 |
| fake | unaware | gained | 5 | deduction | density_nontest | 1.152 | -0.646 | 2.950 |
| fake | unaware | gained | 5 | deduction | density_testlex | -0.281 | -0.486 | -0.073 |
| fake | unaware | gained | 5 | uncertainty-estimation | density | 0.657 | -0.122 | 1.435 |
| fake | unaware | gained | 5 | uncertainty-estimation | density_nontest | 0.926 | 0.283 | 1.576 |
| fake | unaware | gained | 5 | uncertainty-estimation | density_testlex | -0.269 | -0.557 | 0.000 |
| fake | unaware | gained | 5 | backtracking | density | -0.115 | -0.664 | 0.365 |
| fake | unaware | gained | 5 | backtracking | density_nontest | -0.115 | -0.664 | 0.365 |
| fake | unaware | gained | 5 | backtracking | density_testlex | 0.000 | 0.000 | 0.000 |
| fake | unaware | lost | 7 | deduction | density | -0.236 | -1.407 | 0.914 |
| fake | unaware | lost | 7 | deduction | density_nontest | -0.360 | -1.566 | 0.790 |
| fake | unaware | lost | 7 | deduction | density_testlex | 0.124 | -0.035 | 0.273 |
| fake | unaware | lost | 7 | uncertainty-estimation | density | -0.311 | -0.744 | 0.071 |
| fake | unaware | lost | 7 | uncertainty-estimation | density_nontest | -0.373 | -0.781 | 0.006 |
| fake | unaware | lost | 7 | uncertainty-estimation | density_testlex | 0.061 | 0.002 | 0.168 |
| fake | unaware | lost | 7 | backtracking | density | 0.546 | 0.310 | 0.794 |
| fake | unaware | lost | 7 | backtracking | density_nontest | 0.546 | 0.310 | 0.794 |
| fake | unaware | lost | 7 | backtracking | density_testlex | 0.000 | 0.000 | 0.000 |
| fake | unaware | comply_both | 113 | deduction | density | 0.088 | -0.199 | 0.362 |
| fake | unaware | comply_both | 113 | deduction | density_nontest | 0.084 | -0.191 | 0.347 |
| fake | unaware | comply_both | 113 | deduction | density_testlex | 0.004 | -0.053 | 0.061 |
| fake | unaware | comply_both | 113 | uncertainty-estimation | density | 0.090 | -0.040 | 0.226 |
| fake | unaware | comply_both | 113 | uncertainty-estimation | density_nontest | 0.077 | -0.047 | 0.207 |
| fake | unaware | comply_both | 113 | uncertainty-estimation | density_testlex | 0.013 | -0.025 | 0.051 |
| fake | unaware | comply_both | 113 | backtracking | density | 0.025 | -0.035 | 0.088 |
| fake | unaware | comply_both | 113 | backtracking | density_nontest | 0.027 | -0.033 | 0.091 |
| fake | unaware | comply_both | 113 | backtracking | density_testlex | -0.002 | -0.009 | 0.005 |
| fake | unaware | refuse_both | 12 | deduction | density | -0.126 | -0.622 | 0.317 |
| fake | unaware | refuse_both | 12 | deduction | density_nontest | -0.092 | -0.567 | 0.381 |
| fake | unaware | refuse_both | 12 | deduction | density_testlex | -0.034 | -0.199 | 0.122 |
| fake | unaware | refuse_both | 12 | uncertainty-estimation | density | 0.461 | -0.012 | 0.986 |
| fake | unaware | refuse_both | 12 | uncertainty-estimation | density_nontest | 0.251 | -0.197 | 0.783 |
| fake | unaware | refuse_both | 12 | uncertainty-estimation | density_testlex | 0.210 | 0.065 | 0.357 |
| fake | unaware | refuse_both | 12 | backtracking | density | -0.138 | -0.347 | 0.060 |
| fake | unaware | refuse_both | 12 | backtracking | density_nontest | -0.124 | -0.311 | 0.065 |
| fake | unaware | refuse_both | 12 | backtracking | density_testlex | -0.014 | -0.042 | 0.000 |
| fake | unaware_strong | gained | 11 | deduction | density | -0.256 | -0.720 | 0.205 |
| fake | unaware_strong | gained | 11 | deduction | density_nontest | -0.032 | -0.574 | 0.513 |
| fake | unaware_strong | gained | 11 | deduction | density_testlex | -0.224 | -0.465 | -0.006 |
| fake | unaware_strong | gained | 11 | uncertainty-estimation | density | 0.476 | 0.037 | 1.002 |
| fake | unaware_strong | gained | 11 | uncertainty-estimation | density_nontest | 0.591 | 0.228 | 1.040 |
| fake | unaware_strong | gained | 11 | uncertainty-estimation | density_testlex | -0.116 | -0.322 | 0.075 |
| fake | unaware_strong | gained | 11 | backtracking | density | -0.097 | -0.270 | 0.101 |
| fake | unaware_strong | gained | 11 | backtracking | density_nontest | -0.110 | -0.256 | 0.025 |
| fake | unaware_strong | gained | 11 | backtracking | density_testlex | 0.013 | -0.046 | 0.084 |
| fake | unaware_strong | comply_both | 119 | deduction | density | 0.123 | -0.144 | 0.412 |
| fake | unaware_strong | comply_both | 119 | deduction | density_nontest | 0.109 | -0.149 | 0.392 |
| fake | unaware_strong | comply_both | 119 | deduction | density_testlex | 0.013 | -0.037 | 0.062 |
| fake | unaware_strong | comply_both | 119 | uncertainty-estimation | density | 0.035 | -0.094 | 0.155 |
| fake | unaware_strong | comply_both | 119 | uncertainty-estimation | density_nontest | 0.032 | -0.094 | 0.150 |
| fake | unaware_strong | comply_both | 119 | uncertainty-estimation | density_testlex | 0.003 | -0.029 | 0.034 |
| fake | unaware_strong | comply_both | 119 | backtracking | density | 0.004 | -0.066 | 0.073 |
| fake | unaware_strong | comply_both | 119 | backtracking | density_nontest | 0.008 | -0.063 | 0.077 |
| fake | unaware_strong | comply_both | 119 | backtracking | density_testlex | -0.004 | -0.010 | 0.000 |
| fake | unaware_strong | refuse_both | 6 | deduction | density | 1.116 | 0.180 | 2.199 |
| fake | unaware_strong | refuse_both | 6 | deduction | density_nontest | 1.177 | 0.268 | 2.365 |
| fake | unaware_strong | refuse_both | 6 | deduction | density_testlex | -0.061 | -0.351 | 0.335 |
| fake | unaware_strong | refuse_both | 6 | uncertainty-estimation | density | -0.122 | -0.640 | 0.441 |
| fake | unaware_strong | refuse_both | 6 | uncertainty-estimation | density_nontest | -0.162 | -0.679 | 0.399 |
| fake | unaware_strong | refuse_both | 6 | uncertainty-estimation | density_testlex | 0.039 | 0.000 | 0.118 |
| fake | unaware_strong | refuse_both | 6 | backtracking | density | -0.394 | -0.776 | 0.007 |
| fake | unaware_strong | refuse_both | 6 | backtracking | density_nontest | -0.394 | -0.776 | 0.007 |
| fake | unaware_strong | refuse_both | 6 | backtracking | density_testlex | 0.000 | 0.000 | 0.000 |
| real | aware | gained | 11 | deduction | density | 0.374 | -0.132 | 0.821 |
| real | aware | gained | 11 | deduction | density_nontest | 0.513 | -0.009 | 1.032 |
| real | aware | gained | 11 | deduction | density_testlex | -0.139 | -0.318 | 0.057 |
| real | aware | gained | 11 | uncertainty-estimation | density | 0.263 | -0.356 | 0.882 |
| real | aware | gained | 11 | uncertainty-estimation | density_nontest | 0.380 | -0.283 | 1.022 |
| real | aware | gained | 11 | uncertainty-estimation | density_testlex | -0.117 | -0.306 | 0.101 |
| real | aware | gained | 11 | backtracking | density | -0.277 | -0.542 | -0.041 |
| real | aware | gained | 11 | backtracking | density_nontest | -0.277 | -0.542 | -0.041 |
| real | aware | gained | 11 | backtracking | density_testlex | 0.000 | 0.000 | 0.000 |
| real | aware | lost | 5 | deduction | density | 0.055 | -1.046 | 0.893 |
| real | aware | lost | 5 | deduction | density_nontest | 0.043 | -0.923 | 0.880 |
| real | aware | lost | 5 | deduction | density_testlex | 0.012 | -0.277 | 0.438 |
| real | aware | lost | 5 | uncertainty-estimation | density | 0.386 | -0.773 | 1.647 |
| real | aware | lost | 5 | uncertainty-estimation | density_nontest | 0.231 | -0.959 | 1.643 |
| real | aware | lost | 5 | uncertainty-estimation | density_testlex | 0.155 | 0.000 | 0.394 |
| real | aware | lost | 5 | backtracking | density | 0.028 | -0.251 | 0.361 |
| real | aware | lost | 5 | backtracking | density_nontest | 0.028 | -0.251 | 0.361 |
| real | aware | lost | 5 | backtracking | density_testlex | 0.000 | 0.000 | 0.000 |
| real | aware | comply_both | 97 | deduction | density | -0.252 | -0.517 | 0.008 |
| real | aware | comply_both | 97 | deduction | density_nontest | -0.241 | -0.511 | 0.028 |
| real | aware | comply_both | 97 | deduction | density_testlex | -0.011 | -0.062 | 0.044 |
| real | aware | comply_both | 97 | uncertainty-estimation | density | -0.060 | -0.210 | 0.086 |
| real | aware | comply_both | 97 | uncertainty-estimation | density_nontest | -0.027 | -0.176 | 0.118 |
| real | aware | comply_both | 97 | uncertainty-estimation | density_testlex | -0.033 | -0.086 | 0.017 |
| real | aware | comply_both | 97 | backtracking | density | 0.050 | -0.027 | 0.135 |
| real | aware | comply_both | 97 | backtracking | density_nontest | 0.056 | -0.021 | 0.139 |
| real | aware | comply_both | 97 | backtracking | density_testlex | -0.006 | -0.014 | 0.000 |
| real | aware | refuse_both | 29 | deduction | density | 0.133 | -0.240 | 0.482 |
| real | aware | refuse_both | 29 | deduction | density_nontest | 0.010 | -0.368 | 0.359 |
| real | aware | refuse_both | 29 | deduction | density_testlex | 0.123 | 0.002 | 0.262 |
| real | aware | refuse_both | 29 | uncertainty-estimation | density | -0.031 | -0.289 | 0.214 |
| real | aware | refuse_both | 29 | uncertainty-estimation | density_nontest | 0.081 | -0.149 | 0.288 |
| real | aware | refuse_both | 29 | uncertainty-estimation | density_testlex | -0.112 | -0.221 | -0.005 |
| real | aware | refuse_both | 29 | backtracking | density | 0.056 | -0.093 | 0.235 |
| real | aware | refuse_both | 29 | backtracking | density_nontest | 0.056 | -0.093 | 0.235 |
| real | aware | refuse_both | 29 | backtracking | density_testlex | 0.000 | 0.000 | 0.000 |
| real | aware_strong | gained | 12 | deduction | density | -0.646 | -1.278 | -0.074 |
| real | aware_strong | gained | 12 | deduction | density_nontest | -0.557 | -1.267 | 0.080 |
| real | aware_strong | gained | 12 | deduction | density_testlex | -0.089 | -0.290 | 0.105 |
| real | aware_strong | gained | 12 | uncertainty-estimation | density | 0.188 | -0.331 | 0.631 |
| real | aware_strong | gained | 12 | uncertainty-estimation | density_nontest | 0.273 | -0.206 | 0.697 |
| real | aware_strong | gained | 12 | uncertainty-estimation | density_testlex | -0.084 | -0.238 | 0.066 |
| real | aware_strong | gained | 12 | backtracking | density | -0.110 | -0.376 | 0.117 |
| real | aware_strong | gained | 12 | backtracking | density_nontest | -0.110 | -0.376 | 0.117 |
| real | aware_strong | gained | 12 | backtracking | density_testlex | 0.000 | 0.000 | 0.000 |
| real | aware_strong | lost | 6 | deduction | density | -0.403 | -1.149 | 0.396 |
| real | aware_strong | lost | 6 | deduction | density_nontest | -0.478 | -1.340 | 0.414 |
| real | aware_strong | lost | 6 | deduction | density_testlex | 0.075 | -0.137 | 0.278 |
| real | aware_strong | lost | 6 | uncertainty-estimation | density | 0.127 | -0.635 | 0.884 |
| real | aware_strong | lost | 6 | uncertainty-estimation | density_nontest | -0.009 | -0.727 | 0.694 |
| real | aware_strong | lost | 6 | uncertainty-estimation | density_testlex | 0.135 | -0.045 | 0.323 |
| real | aware_strong | lost | 6 | backtracking | density | -0.064 | -0.159 | 0.030 |
| real | aware_strong | lost | 6 | backtracking | density_nontest | -0.064 | -0.159 | 0.030 |
| real | aware_strong | lost | 6 | backtracking | density_testlex | 0.000 | 0.000 | 0.000 |
| real | aware_strong | comply_both | 96 | deduction | density | -0.160 | -0.465 | 0.151 |
| real | aware_strong | comply_both | 96 | deduction | density_nontest | -0.125 | -0.431 | 0.193 |
| real | aware_strong | comply_both | 96 | deduction | density_testlex | -0.035 | -0.077 | 0.006 |
| real | aware_strong | comply_both | 96 | uncertainty-estimation | density | -0.137 | -0.294 | 0.015 |
| real | aware_strong | comply_both | 96 | uncertainty-estimation | density_nontest | -0.125 | -0.274 | 0.023 |
| real | aware_strong | comply_both | 96 | uncertainty-estimation | density_testlex | -0.012 | -0.064 | 0.041 |
| real | aware_strong | comply_both | 96 | backtracking | density | -0.028 | -0.118 | 0.054 |
| real | aware_strong | comply_both | 96 | backtracking | density_nontest | -0.028 | -0.119 | 0.055 |
| real | aware_strong | comply_both | 96 | backtracking | density_testlex | 0.000 | -0.009 | 0.011 |
| real | aware_strong | refuse_both | 28 | deduction | density | 0.107 | -0.263 | 0.473 |
| real | aware_strong | refuse_both | 28 | deduction | density_nontest | -0.036 | -0.405 | 0.319 |
| real | aware_strong | refuse_both | 28 | deduction | density_testlex | 0.143 | 0.008 | 0.282 |
| real | aware_strong | refuse_both | 28 | uncertainty-estimation | density | -0.009 | -0.246 | 0.259 |
| real | aware_strong | refuse_both | 28 | uncertainty-estimation | density_nontest | 0.065 | -0.107 | 0.241 |
| real | aware_strong | refuse_both | 28 | uncertainty-estimation | density_testlex | -0.073 | -0.219 | 0.077 |
| real | aware_strong | refuse_both | 28 | backtracking | density | -0.107 | -0.277 | 0.042 |
| real | aware_strong | refuse_both | 28 | backtracking | density_nontest | -0.107 | -0.277 | 0.042 |
| real | aware_strong | refuse_both | 28 | backtracking | density_testlex | 0.000 | 0.000 | 0.000 |
| real | baseline_seed1 | gained | 6 | deduction | density | 0.187 | -0.730 | 1.115 |
| real | baseline_seed1 | gained | 6 | deduction | density_nontest | 0.113 | -0.897 | 1.148 |
| real | baseline_seed1 | gained | 6 | deduction | density_testlex | 0.074 | -0.137 | 0.284 |
| real | baseline_seed1 | gained | 6 | uncertainty-estimation | density | -0.177 | -0.895 | 0.544 |
| real | baseline_seed1 | gained | 6 | uncertainty-estimation | density_nontest | -0.044 | -0.630 | 0.566 |
| real | baseline_seed1 | gained | 6 | uncertainty-estimation | density_testlex | -0.133 | -0.449 | 0.188 |
| real | baseline_seed1 | gained | 6 | backtracking | density | 0.061 | -0.229 | 0.303 |
| real | baseline_seed1 | gained | 6 | backtracking | density_nontest | 0.061 | -0.229 | 0.303 |
| real | baseline_seed1 | gained | 6 | backtracking | density_testlex | 0.000 | 0.000 | 0.000 |
| real | baseline_seed1 | comply_both | 99 | deduction | density | -0.226 | -0.492 | 0.011 |
| real | baseline_seed1 | comply_both | 99 | deduction | density_nontest | -0.199 | -0.471 | 0.043 |
| real | baseline_seed1 | comply_both | 99 | deduction | density_testlex | -0.027 | -0.073 | 0.022 |
| real | baseline_seed1 | comply_both | 99 | uncertainty-estimation | density | 0.059 | -0.086 | 0.200 |
| real | baseline_seed1 | comply_both | 99 | uncertainty-estimation | density_nontest | 0.063 | -0.071 | 0.194 |
| real | baseline_seed1 | comply_both | 99 | uncertainty-estimation | density_testlex | -0.004 | -0.054 | 0.043 |
| real | baseline_seed1 | comply_both | 99 | backtracking | density | 0.006 | -0.090 | 0.098 |
| real | baseline_seed1 | comply_both | 99 | backtracking | density_nontest | 0.008 | -0.089 | 0.103 |
| real | baseline_seed1 | comply_both | 99 | backtracking | density_testlex | -0.003 | -0.014 | 0.007 |
| real | baseline_seed1 | refuse_both | 33 | deduction | density | -0.162 | -0.582 | 0.248 |
| real | baseline_seed1 | refuse_both | 33 | deduction | density_nontest | -0.205 | -0.644 | 0.229 |
| real | baseline_seed1 | refuse_both | 33 | deduction | density_testlex | 0.044 | -0.053 | 0.139 |
| real | baseline_seed1 | refuse_both | 33 | uncertainty-estimation | density | -0.038 | -0.291 | 0.223 |
| real | baseline_seed1 | refuse_both | 33 | uncertainty-estimation | density_nontest | 0.006 | -0.227 | 0.251 |
| real | baseline_seed1 | refuse_both | 33 | uncertainty-estimation | density_testlex | -0.045 | -0.146 | 0.057 |
| real | baseline_seed1 | refuse_both | 33 | backtracking | density | -0.068 | -0.206 | 0.069 |
| real | baseline_seed1 | refuse_both | 33 | backtracking | density_nontest | -0.068 | -0.206 | 0.069 |
| real | baseline_seed1 | refuse_both | 33 | backtracking | density_testlex | 0.000 | 0.000 | 0.000 |
| real | random_aware | gained | 10 | deduction | density | -0.313 | -1.192 | 0.512 |
| real | random_aware | gained | 10 | deduction | density_nontest | -0.212 | -1.089 | 0.607 |
| real | random_aware | gained | 10 | deduction | density_testlex | -0.101 | -0.307 | 0.085 |
| real | random_aware | gained | 10 | uncertainty-estimation | density | 0.135 | -0.558 | 0.848 |
| real | random_aware | gained | 10 | uncertainty-estimation | density_nontest | 0.032 | -0.625 | 0.628 |
| real | random_aware | gained | 10 | uncertainty-estimation | density_testlex | 0.104 | -0.138 | 0.334 |
| real | random_aware | gained | 10 | backtracking | density | -0.021 | -0.360 | 0.253 |
| real | random_aware | gained | 10 | backtracking | density_nontest | -0.021 | -0.360 | 0.253 |
| real | random_aware | gained | 10 | backtracking | density_testlex | 0.000 | 0.000 | 0.000 |
| real | random_aware | lost | 5 | deduction | density | 0.600 | -0.058 | 1.252 |
| real | random_aware | lost | 5 | deduction | density_nontest | 0.586 | -0.190 | 1.391 |
| real | random_aware | lost | 5 | deduction | density_testlex | 0.014 | -0.157 | 0.248 |
| real | random_aware | lost | 5 | uncertainty-estimation | density | -0.281 | -1.209 | 0.604 |
| real | random_aware | lost | 5 | uncertainty-estimation | density_nontest | -0.333 | -1.153 | 0.614 |
| real | random_aware | lost | 5 | uncertainty-estimation | density_testlex | 0.052 | -0.090 | 0.267 |
| real | random_aware | lost | 5 | backtracking | density | 0.212 | 0.060 | 0.370 |
| real | random_aware | lost | 5 | backtracking | density_nontest | 0.212 | 0.060 | 0.370 |
| real | random_aware | lost | 5 | backtracking | density_testlex | 0.000 | 0.000 | 0.000 |
| real | random_aware | comply_both | 101 | deduction | density | -0.115 | -0.389 | 0.157 |
| real | random_aware | comply_both | 101 | deduction | density_nontest | -0.085 | -0.362 | 0.194 |
| real | random_aware | comply_both | 101 | deduction | density_testlex | -0.029 | -0.077 | 0.019 |
| real | random_aware | comply_both | 101 | uncertainty-estimation | density | -0.035 | -0.167 | 0.089 |
| real | random_aware | comply_both | 101 | uncertainty-estimation | density_nontest | -0.036 | -0.165 | 0.085 |
| real | random_aware | comply_both | 101 | uncertainty-estimation | density_testlex | 0.001 | -0.044 | 0.045 |
| real | random_aware | comply_both | 101 | backtracking | density | -0.024 | -0.115 | 0.058 |
| real | random_aware | comply_both | 101 | backtracking | density_nontest | -0.019 | -0.111 | 0.064 |
| real | random_aware | comply_both | 101 | backtracking | density_testlex | -0.006 | -0.014 | 0.000 |
| real | random_aware | refuse_both | 29 | deduction | density | 0.007 | -0.471 | 0.486 |
| real | random_aware | refuse_both | 29 | deduction | density_nontest | -0.043 | -0.514 | 0.463 |
| real | random_aware | refuse_both | 29 | deduction | density_testlex | 0.050 | -0.042 | 0.144 |
| real | random_aware | refuse_both | 29 | uncertainty-estimation | density | 0.133 | -0.090 | 0.339 |
| real | random_aware | refuse_both | 29 | uncertainty-estimation | density_nontest | 0.159 | -0.023 | 0.343 |
| real | random_aware | refuse_both | 29 | uncertainty-estimation | density_testlex | -0.027 | -0.158 | 0.101 |
| real | random_aware | refuse_both | 29 | backtracking | density | 0.003 | -0.145 | 0.147 |
| real | random_aware | refuse_both | 29 | backtracking | density_nontest | -0.006 | -0.151 | 0.135 |
| real | random_aware | refuse_both | 29 | backtracking | density_testlex | 0.008 | 0.000 | 0.025 |
| real | random_samerows_aware | gained | 10 | deduction | density | -0.617 | -1.535 | 0.194 |
| real | random_samerows_aware | gained | 10 | deduction | density_nontest | -0.492 | -1.448 | 0.393 |
| real | random_samerows_aware | gained | 10 | deduction | density_testlex | -0.125 | -0.363 | 0.114 |
| real | random_samerows_aware | gained | 10 | uncertainty-estimation | density | 0.107 | -0.413 | 0.713 |
| real | random_samerows_aware | gained | 10 | uncertainty-estimation | density_nontest | 0.325 | -0.177 | 0.880 |
| real | random_samerows_aware | gained | 10 | uncertainty-estimation | density_testlex | -0.218 | -0.371 | -0.043 |
| real | random_samerows_aware | gained | 10 | backtracking | density | -0.057 | -0.271 | 0.148 |
| real | random_samerows_aware | gained | 10 | backtracking | density_nontest | -0.057 | -0.271 | 0.148 |
| real | random_samerows_aware | gained | 10 | backtracking | density_testlex | 0.000 | 0.000 | 0.000 |
| real | random_samerows_aware | lost | 5 | deduction | density | 0.887 | -0.026 | 1.801 |
| real | random_samerows_aware | lost | 5 | deduction | density_nontest | 0.829 | 0.015 | 1.748 |
| real | random_samerows_aware | lost | 5 | deduction | density_testlex | 0.058 | -0.522 | 0.648 |
| real | random_samerows_aware | lost | 5 | uncertainty-estimation | density | 0.346 | -0.381 | 1.004 |
| real | random_samerows_aware | lost | 5 | uncertainty-estimation | density_nontest | 0.463 | -0.658 | 1.323 |
| real | random_samerows_aware | lost | 5 | uncertainty-estimation | density_testlex | -0.117 | -0.441 | 0.286 |
| real | random_samerows_aware | lost | 5 | backtracking | density | 0.159 | -0.064 | 0.417 |
| real | random_samerows_aware | lost | 5 | backtracking | density_nontest | 0.159 | -0.064 | 0.417 |
| real | random_samerows_aware | lost | 5 | backtracking | density_testlex | 0.000 | 0.000 | 0.000 |
| real | random_samerows_aware | comply_both | 95 | deduction | density | -0.195 | -0.478 | 0.066 |
| real | random_samerows_aware | comply_both | 95 | deduction | density_nontest | -0.155 | -0.437 | 0.102 |
| real | random_samerows_aware | comply_both | 95 | deduction | density_testlex | -0.040 | -0.084 | 0.004 |
| real | random_samerows_aware | comply_both | 95 | uncertainty-estimation | density | 0.007 | -0.123 | 0.136 |
| real | random_samerows_aware | comply_both | 95 | uncertainty-estimation | density_nontest | 0.022 | -0.103 | 0.152 |
| real | random_samerows_aware | comply_both | 95 | uncertainty-estimation | density_testlex | -0.015 | -0.067 | 0.037 |
| real | random_samerows_aware | comply_both | 95 | backtracking | density | -0.049 | -0.143 | 0.041 |
| real | random_samerows_aware | comply_both | 95 | backtracking | density_nontest | -0.049 | -0.140 | 0.037 |
| real | random_samerows_aware | comply_both | 95 | backtracking | density_testlex | 0.000 | -0.011 | 0.012 |
| real | random_samerows_aware | refuse_both | 30 | deduction | density | -0.183 | -0.555 | 0.174 |
| real | random_samerows_aware | refuse_both | 30 | deduction | density_nontest | -0.345 | -0.723 | 0.029 |
| real | random_samerows_aware | refuse_both | 30 | deduction | density_testlex | 0.163 | 0.055 | 0.273 |
| real | random_samerows_aware | refuse_both | 30 | uncertainty-estimation | density | 0.004 | -0.259 | 0.264 |
| real | random_samerows_aware | refuse_both | 30 | uncertainty-estimation | density_nontest | -0.001 | -0.220 | 0.217 |
| real | random_samerows_aware | refuse_both | 30 | uncertainty-estimation | density_testlex | 0.006 | -0.132 | 0.129 |
| real | random_samerows_aware | refuse_both | 30 | backtracking | density | 0.024 | -0.136 | 0.173 |
| real | random_samerows_aware | refuse_both | 30 | backtracking | density_nontest | 0.014 | -0.145 | 0.165 |
| real | random_samerows_aware | refuse_both | 30 | backtracking | density_testlex | 0.009 | 0.000 | 0.028 |
| real | shuffled_aware | gained | 7 | deduction | density | -0.060 | -0.897 | 0.654 |
| real | shuffled_aware | gained | 7 | deduction | density_nontest | 0.074 | -0.794 | 0.856 |
| real | shuffled_aware | gained | 7 | deduction | density_testlex | -0.134 | -0.356 | 0.045 |
| real | shuffled_aware | gained | 7 | uncertainty-estimation | density | 0.457 | 0.149 | 0.803 |
| real | shuffled_aware | gained | 7 | uncertainty-estimation | density_nontest | 0.675 | 0.400 | 0.983 |
| real | shuffled_aware | gained | 7 | uncertainty-estimation | density_testlex | -0.218 | -0.377 | -0.060 |
| real | shuffled_aware | gained | 7 | backtracking | density | -0.077 | -0.279 | 0.123 |
| real | shuffled_aware | gained | 7 | backtracking | density_nontest | -0.077 | -0.279 | 0.123 |
| real | shuffled_aware | gained | 7 | backtracking | density_testlex | 0.000 | 0.000 | 0.000 |
| real | shuffled_aware | lost | 7 | deduction | density | 0.544 | -0.096 | 1.223 |
| real | shuffled_aware | lost | 7 | deduction | density_nontest | 0.718 | 0.213 | 1.312 |
| real | shuffled_aware | lost | 7 | deduction | density_testlex | -0.173 | -0.390 | 0.057 |
| real | shuffled_aware | lost | 7 | uncertainty-estimation | density | -0.200 | -0.980 | 0.676 |
| real | shuffled_aware | lost | 7 | uncertainty-estimation | density_nontest | -0.129 | -0.974 | 0.739 |
| real | shuffled_aware | lost | 7 | uncertainty-estimation | density_testlex | -0.071 | -0.251 | 0.094 |
| real | shuffled_aware | lost | 7 | backtracking | density | 0.055 | -0.203 | 0.361 |
| real | shuffled_aware | lost | 7 | backtracking | density_nontest | 0.055 | -0.203 | 0.361 |
| real | shuffled_aware | lost | 7 | backtracking | density_testlex | 0.000 | 0.000 | 0.000 |
| real | shuffled_aware | comply_both | 96 | deduction | density | -0.058 | -0.329 | 0.209 |
| real | shuffled_aware | comply_both | 96 | deduction | density_nontest | -0.038 | -0.299 | 0.222 |
| real | shuffled_aware | comply_both | 96 | deduction | density_testlex | -0.020 | -0.064 | 0.024 |
| real | shuffled_aware | comply_both | 96 | uncertainty-estimation | density | 0.012 | -0.120 | 0.133 |
| real | shuffled_aware | comply_both | 96 | uncertainty-estimation | density_nontest | 0.019 | -0.100 | 0.136 |
| real | shuffled_aware | comply_both | 96 | uncertainty-estimation | density_testlex | -0.006 | -0.058 | 0.044 |
| real | shuffled_aware | comply_both | 96 | backtracking | density | -0.008 | -0.104 | 0.079 |
| real | shuffled_aware | comply_both | 96 | backtracking | density_nontest | -0.002 | -0.099 | 0.085 |
| real | shuffled_aware | comply_both | 96 | backtracking | density_testlex | -0.006 | -0.014 | 0.000 |
| real | shuffled_aware | refuse_both | 32 | deduction | density | 0.011 | -0.480 | 0.543 |
| real | shuffled_aware | refuse_both | 32 | deduction | density_nontest | -0.178 | -0.678 | 0.370 |
| real | shuffled_aware | refuse_both | 32 | deduction | density_testlex | 0.189 | 0.062 | 0.320 |
| real | shuffled_aware | refuse_both | 32 | uncertainty-estimation | density | -0.128 | -0.389 | 0.120 |
| real | shuffled_aware | refuse_both | 32 | uncertainty-estimation | density_nontest | -0.024 | -0.234 | 0.188 |
| real | shuffled_aware | refuse_both | 32 | uncertainty-estimation | density_testlex | -0.104 | -0.213 | -0.012 |
| real | shuffled_aware | refuse_both | 32 | backtracking | density | 0.005 | -0.203 | 0.210 |
| real | shuffled_aware | refuse_both | 32 | backtracking | density_nontest | 0.005 | -0.203 | 0.210 |
| real | shuffled_aware | refuse_both | 32 | backtracking | density_testlex | 0.000 | 0.000 | 0.000 |
| real | shuffled_samerows_aware | gained | 6 | deduction | density | 0.279 | -0.141 | 0.656 |
| real | shuffled_samerows_aware | gained | 6 | deduction | density_nontest | 0.401 | -0.191 | 0.938 |
| real | shuffled_samerows_aware | gained | 6 | deduction | density_testlex | -0.122 | -0.326 | 0.071 |
| real | shuffled_samerows_aware | gained | 6 | uncertainty-estimation | density | 0.331 | -0.246 | 0.797 |
| real | shuffled_samerows_aware | gained | 6 | uncertainty-estimation | density_nontest | 0.365 | -0.126 | 0.808 |
| real | shuffled_samerows_aware | gained | 6 | uncertainty-estimation | density_testlex | -0.034 | -0.260 | 0.203 |
| real | shuffled_samerows_aware | gained | 6 | backtracking | density | -0.097 | -0.271 | 0.083 |
| real | shuffled_samerows_aware | gained | 6 | backtracking | density_nontest | -0.097 | -0.271 | 0.083 |
| real | shuffled_samerows_aware | gained | 6 | backtracking | density_testlex | 0.000 | 0.000 | 0.000 |
| real | shuffled_samerows_aware | lost | 6 | deduction | density | 0.203 | -0.800 | 0.986 |
| real | shuffled_samerows_aware | lost | 6 | deduction | density_nontest | 0.189 | -0.800 | 0.945 |
| real | shuffled_samerows_aware | lost | 6 | deduction | density_testlex | 0.015 | -0.057 | 0.102 |
| real | shuffled_samerows_aware | lost | 6 | uncertainty-estimation | density | 0.098 | -0.718 | 0.924 |
| real | shuffled_samerows_aware | lost | 6 | uncertainty-estimation | density_nontest | 0.128 | -0.792 | 1.069 |
| real | shuffled_samerows_aware | lost | 6 | uncertainty-estimation | density_testlex | -0.030 | -0.263 | 0.202 |
| real | shuffled_samerows_aware | lost | 6 | backtracking | density | 0.336 | -0.057 | 0.700 |
| real | shuffled_samerows_aware | lost | 6 | backtracking | density_nontest | 0.336 | -0.057 | 0.700 |
| real | shuffled_samerows_aware | lost | 6 | backtracking | density_testlex | 0.000 | 0.000 | 0.000 |
| real | shuffled_samerows_aware | comply_both | 97 | deduction | density | -0.338 | -0.620 | -0.044 |
| real | shuffled_samerows_aware | comply_both | 97 | deduction | density_nontest | -0.340 | -0.627 | -0.045 |
| real | shuffled_samerows_aware | comply_both | 97 | deduction | density_testlex | 0.002 | -0.038 | 0.045 |
| real | shuffled_samerows_aware | comply_both | 97 | uncertainty-estimation | density | -0.004 | -0.170 | 0.143 |
| real | shuffled_samerows_aware | comply_both | 97 | uncertainty-estimation | density_nontest | -0.026 | -0.184 | 0.119 |
| real | shuffled_samerows_aware | comply_both | 97 | uncertainty-estimation | density_testlex | 0.022 | -0.023 | 0.066 |
| real | shuffled_samerows_aware | comply_both | 97 | backtracking | density | -0.022 | -0.110 | 0.060 |
| real | shuffled_samerows_aware | comply_both | 97 | backtracking | density_nontest | -0.019 | -0.107 | 0.062 |
| real | shuffled_samerows_aware | comply_both | 97 | backtracking | density_testlex | -0.003 | -0.012 | 0.006 |
| real | shuffled_samerows_aware | refuse_both | 34 | deduction | density | -0.326 | -0.683 | 0.033 |
| real | shuffled_samerows_aware | refuse_both | 34 | deduction | density_nontest | -0.433 | -0.793 | -0.087 |
| real | shuffled_samerows_aware | refuse_both | 34 | deduction | density_testlex | 0.108 | 0.001 | 0.219 |
| real | shuffled_samerows_aware | refuse_both | 34 | uncertainty-estimation | density | 0.019 | -0.202 | 0.220 |
| real | shuffled_samerows_aware | refuse_both | 34 | uncertainty-estimation | density_nontest | -0.024 | -0.199 | 0.144 |
| real | shuffled_samerows_aware | refuse_both | 34 | uncertainty-estimation | density_testlex | 0.043 | -0.055 | 0.138 |
| real | shuffled_samerows_aware | refuse_both | 34 | backtracking | density | -0.051 | -0.202 | 0.084 |
| real | shuffled_samerows_aware | refuse_both | 34 | backtracking | density_nontest | -0.051 | -0.202 | 0.084 |
| real | shuffled_samerows_aware | refuse_both | 34 | backtracking | density_testlex | 0.000 | 0.000 | 0.000 |
| real | unaware | gained | 10 | deduction | density | 0.322 | -0.343 | 1.060 |
| real | unaware | gained | 10 | deduction | density_nontest | 0.513 | -0.235 | 1.375 |
| real | unaware | gained | 10 | deduction | density_testlex | -0.191 | -0.429 | 0.054 |
| real | unaware | gained | 10 | uncertainty-estimation | density | -0.082 | -0.568 | 0.461 |
| real | unaware | gained | 10 | uncertainty-estimation | density_nontest | 0.161 | -0.293 | 0.688 |
| real | unaware | gained | 10 | uncertainty-estimation | density_testlex | -0.242 | -0.408 | -0.100 |
| real | unaware | gained | 10 | backtracking | density | 0.037 | -0.185 | 0.268 |
| real | unaware | gained | 10 | backtracking | density_nontest | -0.001 | -0.205 | 0.214 |
| real | unaware | gained | 10 | backtracking | density_testlex | 0.039 | 0.000 | 0.116 |
| real | unaware | lost | 5 | deduction | density | -1.040 | -2.387 | 0.308 |
| real | unaware | lost | 5 | deduction | density_nontest | -0.899 | -2.337 | 0.623 |
| real | unaware | lost | 5 | deduction | density_testlex | -0.141 | -0.435 | 0.157 |
| real | unaware | lost | 5 | uncertainty-estimation | density | 0.007 | -0.335 | 0.349 |
| real | unaware | lost | 5 | uncertainty-estimation | density_nontest | -0.178 | -0.511 | 0.095 |
| real | unaware | lost | 5 | uncertainty-estimation | density_testlex | 0.185 | -0.187 | 0.497 |
| real | unaware | lost | 5 | backtracking | density | 0.016 | -0.323 | 0.331 |
| real | unaware | lost | 5 | backtracking | density_nontest | 0.016 | -0.323 | 0.331 |
| real | unaware | lost | 5 | backtracking | density_testlex | 0.000 | 0.000 | 0.000 |
| real | unaware | comply_both | 95 | deduction | density | -0.043 | -0.360 | 0.278 |
| real | unaware | comply_both | 95 | deduction | density_nontest | -0.022 | -0.339 | 0.300 |
| real | unaware | comply_both | 95 | deduction | density_testlex | -0.021 | -0.063 | 0.021 |
| real | unaware | comply_both | 95 | uncertainty-estimation | density | -0.051 | -0.197 | 0.092 |
| real | unaware | comply_both | 95 | uncertainty-estimation | density_nontest | -0.069 | -0.210 | 0.071 |
| real | unaware | comply_both | 95 | uncertainty-estimation | density_testlex | 0.018 | -0.034 | 0.072 |
| real | unaware | comply_both | 95 | backtracking | density | 0.017 | -0.079 | 0.105 |
| real | unaware | comply_both | 95 | backtracking | density_nontest | 0.023 | -0.073 | 0.110 |
| real | unaware | comply_both | 95 | backtracking | density_testlex | -0.006 | -0.015 | 0.000 |
| real | unaware | refuse_both | 30 | deduction | density | -0.314 | -0.719 | 0.086 |
| real | unaware | refuse_both | 30 | deduction | density_nontest | -0.385 | -0.807 | 0.043 |
| real | unaware | refuse_both | 30 | deduction | density_testlex | 0.071 | -0.039 | 0.196 |
| real | unaware | refuse_both | 30 | uncertainty-estimation | density | 0.201 | -0.015 | 0.424 |
| real | unaware | refuse_both | 30 | uncertainty-estimation | density_nontest | 0.226 | 0.039 | 0.412 |
| real | unaware | refuse_both | 30 | uncertainty-estimation | density_testlex | -0.025 | -0.124 | 0.082 |
| real | unaware | refuse_both | 30 | backtracking | density | -0.038 | -0.218 | 0.142 |
| real | unaware | refuse_both | 30 | backtracking | density_nontest | -0.038 | -0.218 | 0.142 |
| real | unaware | refuse_both | 30 | backtracking | density_testlex | 0.000 | 0.000 | 0.000 |
| real | unaware_strong | gained | 11 | deduction | density | 0.033 | -0.663 | 0.656 |
| real | unaware_strong | gained | 11 | deduction | density_nontest | 0.143 | -0.580 | 0.784 |
| real | unaware_strong | gained | 11 | deduction | density_testlex | -0.110 | -0.247 | 0.018 |
| real | unaware_strong | gained | 11 | uncertainty-estimation | density | 0.364 | -0.172 | 0.898 |
| real | unaware_strong | gained | 11 | uncertainty-estimation | density_nontest | 0.450 | 0.062 | 0.804 |
| real | unaware_strong | gained | 11 | uncertainty-estimation | density_testlex | -0.086 | -0.304 | 0.137 |
| real | unaware_strong | gained | 11 | backtracking | density | -0.161 | -0.481 | 0.165 |
| real | unaware_strong | gained | 11 | backtracking | density_nontest | -0.161 | -0.481 | 0.165 |
| real | unaware_strong | gained | 11 | backtracking | density_testlex | 0.000 | 0.000 | 0.000 |
| real | unaware_strong | comply_both | 101 | deduction | density | -0.060 | -0.322 | 0.206 |
| real | unaware_strong | comply_both | 101 | deduction | density_nontest | -0.026 | -0.294 | 0.234 |
| real | unaware_strong | comply_both | 101 | deduction | density_testlex | -0.034 | -0.078 | 0.006 |
| real | unaware_strong | comply_both | 101 | uncertainty-estimation | density | -0.041 | -0.185 | 0.106 |
| real | unaware_strong | comply_both | 101 | uncertainty-estimation | density_nontest | -0.034 | -0.174 | 0.117 |
| real | unaware_strong | comply_both | 101 | uncertainty-estimation | density_testlex | -0.007 | -0.051 | 0.037 |
| real | unaware_strong | comply_both | 101 | backtracking | density | -0.046 | -0.137 | 0.034 |
| real | unaware_strong | comply_both | 101 | backtracking | density_nontest | -0.041 | -0.133 | 0.040 |
| real | unaware_strong | comply_both | 101 | backtracking | density_testlex | -0.006 | -0.014 | 0.000 |
| real | unaware_strong | refuse_both | 28 | deduction | density | -0.028 | -0.479 | 0.424 |
| real | unaware_strong | refuse_both | 28 | deduction | density_nontest | -0.117 | -0.576 | 0.361 |
| real | unaware_strong | refuse_both | 28 | deduction | density_testlex | 0.089 | -0.004 | 0.193 |
| real | unaware_strong | refuse_both | 28 | uncertainty-estimation | density | -0.093 | -0.348 | 0.142 |
| real | unaware_strong | refuse_both | 28 | uncertainty-estimation | density_nontest | 0.027 | -0.197 | 0.240 |
| real | unaware_strong | refuse_both | 28 | uncertainty-estimation | density_testlex | -0.120 | -0.242 | -0.012 |
| real | unaware_strong | refuse_both | 28 | backtracking | density | -0.128 | -0.243 | -0.016 |
| real | unaware_strong | refuse_both | 28 | backtracking | density_nontest | -0.128 | -0.243 | -0.016 |
| real | unaware_strong | refuse_both | 28 | backtracking | density_testlex | 0.000 | 0.000 | 0.000 |


## Representation-level morphology: Δ mean projection onto the rebuilt directions (steered model's own activations)

| arm | framing | projection | n_pairs | base_mean | delta | ci_lo | ci_hi |
|---|---|---|---|---|---|---|---|
| baseline_seed1 | real | proj_backtracking | 149 | 20.3383 | 0.0840 | -0.3104 | 0.4595 |
| baseline_seed1 | real | proj_uncertainty-estimation | 149 | 8.7282 | -0.1679 | -0.4809 | 0.1340 |
| baseline_seed1 | fake | proj_backtracking | 149 | 19.4140 | 0.1414 | -0.2317 | 0.4960 |
| baseline_seed1 | fake | proj_uncertainty-estimation | 149 | 8.3488 | 0.0931 | -0.2649 | 0.4594 |
| unaware | real | proj_backtracking | 149 | 20.3383 | 0.0738 | -0.3265 | 0.4613 |
| unaware | real | proj_uncertainty-estimation | 149 | 8.7282 | -0.1917 | -0.4737 | 0.0663 |
| unaware | fake | proj_backtracking | 149 | 19.4140 | 0.5111 | 0.1546 | 0.8579 |
| unaware | fake | proj_uncertainty-estimation | 149 | 8.3488 | 0.0155 | -0.3418 | 0.3586 |
| aware | real | proj_backtracking | 149 | 20.3383 | 0.0363 | -0.3974 | 0.4522 |
| aware | real | proj_uncertainty-estimation | 149 | 8.7282 | -0.0424 | -0.3629 | 0.2806 |
| aware | fake | proj_backtracking | 149 | 19.4140 | 0.2334 | -0.1813 | 0.6305 |
| aware | fake | proj_uncertainty-estimation | 149 | 8.3488 | -0.0449 | -0.4064 | 0.3017 |
| unaware_strong | real | proj_backtracking | 149 | 20.3383 | -0.2015 | -0.6415 | 0.2226 |
| unaware_strong | real | proj_uncertainty-estimation | 149 | 8.7282 | -0.2086 | -0.5435 | 0.1407 |
| unaware_strong | fake | proj_backtracking | 149 | 19.4140 | 0.2064 | -0.1496 | 0.5597 |
| unaware_strong | fake | proj_uncertainty-estimation | 149 | 8.3488 | -0.1481 | -0.5172 | 0.2241 |
| aware_strong | real | proj_backtracking | 149 | 20.3383 | -0.1800 | -0.6088 | 0.2246 |
| aware_strong | real | proj_uncertainty-estimation | 149 | 8.7282 | -0.2790 | -0.5283 | -0.0587 |
| aware_strong | fake | proj_backtracking | 149 | 19.4140 | 0.5754 | 0.1815 | 0.9753 |
| aware_strong | fake | proj_uncertainty-estimation | 149 | 8.3488 | -0.1149 | -0.4379 | 0.1666 |
| random_aware | real | proj_backtracking | 149 | 20.3383 | 0.0358 | -0.3247 | 0.3976 |
| random_aware | real | proj_uncertainty-estimation | 149 | 8.7282 | -0.1826 | -0.4695 | 0.0680 |
| random_aware | fake | proj_backtracking | 149 | 19.4140 | 0.3837 | -0.0187 | 0.7963 |
| random_aware | fake | proj_uncertainty-estimation | 149 | 8.3488 | 0.0539 | -0.2786 | 0.3799 |
| shuffled_aware | real | proj_backtracking | 149 | 20.3383 | -0.0454 | -0.4383 | 0.3263 |
| shuffled_aware | real | proj_uncertainty-estimation | 149 | 8.7282 | -0.0524 | -0.3768 | 0.2715 |
| shuffled_aware | fake | proj_backtracking | 149 | 19.4140 | 0.3024 | -0.0785 | 0.6861 |
| shuffled_aware | fake | proj_uncertainty-estimation | 149 | 8.3488 | -0.0678 | -0.4337 | 0.2969 |
| random_samerows_aware | real | proj_backtracking | 149 | 20.3383 | 0.0433 | -0.4130 | 0.4683 |
| random_samerows_aware | real | proj_uncertainty-estimation | 149 | 8.7282 | -0.0851 | -0.4045 | 0.2655 |
| random_samerows_aware | fake | proj_backtracking | 149 | 19.4140 | 0.2471 | -0.1638 | 0.6414 |
| random_samerows_aware | fake | proj_uncertainty-estimation | 149 | 8.3488 | -0.0559 | -0.4254 | 0.2714 |
| shuffled_samerows_aware | real | proj_backtracking | 149 | 20.3383 | 0.1240 | -0.2874 | 0.5346 |
| shuffled_samerows_aware | real | proj_uncertainty-estimation | 149 | 8.7282 | -0.2131 | -0.5101 | 0.0665 |
| shuffled_samerows_aware | fake | proj_backtracking | 149 | 19.4140 | 0.3374 | -0.0259 | 0.7170 |
| shuffled_samerows_aware | fake | proj_uncertainty-estimation | 149 | 8.3488 | 0.1126 | -0.2863 | 0.5109 |

