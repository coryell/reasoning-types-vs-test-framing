# Mechanistic arm (Qwen3-4B) — first-pass analysis: f2_uncertainty

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


## f2_uncertainty: 840 traces, 840 with usable morphology annotation

| arm | framing | n | exec_rate_closed | exec_rate_all | lost | gained | p_mcnemar | task_judged_yes | judge_vs_string_agree | verbalization | closure | repetition | words |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | real | 60 | — | 0.000 | — | — | — | 0.367 | — | 0.167 | 0.950 | 0.039 | 503.133 |
| baseline | hypothetical | 60 | — | 0.000 | — | — | — | 0.617 | — | 0.050 | 0.950 | 0.027 | 513.283 |
| unc_all_neg | real | 60 | — | 0.000 | 0.000 | 0.000 | — | 0.383 | — | 0.050 | 0.983 | 0.022 | 433.233 |
| unc_all_neg | hypothetical | 60 | — | 0.000 | 0.000 | 0.000 | — | 0.617 | — | 0.017 | 0.950 | 0.032 | 471.767 |
| unc_all_pos | real | 60 | — | 0.000 | 0.000 | 0.000 | — | 0.383 | — | 0.267 | 0.950 | 0.063 | 561.133 |
| unc_all_pos | hypothetical | 60 | — | 0.000 | 0.000 | 0.000 | — | 0.500 | — | 0.167 | 0.917 | 0.062 | 639.200 |
| unc_nontest_neg | real | 60 | — | 0.000 | 0.000 | 0.000 | — | 0.483 | — | 0.050 | 0.983 | 0.022 | 435.000 |
| unc_nontest_neg | hypothetical | 60 | — | 0.000 | 0.000 | 0.000 | — | 0.633 | — | 0.033 | 0.950 | 0.030 | 471.317 |
| unc_nontest_pos | real | 60 | — | 0.000 | 0.000 | 0.000 | — | 0.383 | — | 0.300 | 0.933 | 0.059 | 571.000 |
| unc_nontest_pos | hypothetical | 60 | — | 0.000 | 0.000 | 0.000 | — | 0.483 | — | 0.183 | 0.917 | 0.062 | 604.667 |
| unc_testlex_neg | real | 60 | — | 0.000 | 0.000 | 0.000 | — | 0.467 | — | 0.033 | 0.983 | 0.022 | 350.400 |
| unc_testlex_neg | hypothetical | 60 | — | 0.000 | 0.000 | 0.000 | — | 0.667 | — | 0.083 | 0.950 | 0.036 | 394.617 |
| unc_testlex_pos | real | 60 | — | 0.000 | 0.000 | 0.000 | — | 0.300 | — | 0.183 | 0.933 | 0.065 | 721.000 |
| unc_testlex_pos | hypothetical | 60 | — | 0.000 | 0.000 | 0.000 | — | 0.450 | — | 0.150 | 0.800 | 0.064 | 782.833 |


## Morphology: Δ spans per 100 words vs baseline, paired by item

| framing | arm | n_pairs | behaviour | metric | base_mean | arm_mean | delta | ci_lo | ci_hi | p_paired_t |
|---|---|---|---|---|---|---|---|---|---|---|
| hypothetical | unc_all_neg | 60 | deduction | density | 3.742 | 4.006 | 0.264 | -0.133 | 0.670 | 0.218 |
| hypothetical | unc_all_neg | 60 | adding-knowledge | density | 1.325 | 1.113 | -0.212 | -0.474 | 0.056 | 0.127 |
| hypothetical | unc_all_neg | 60 | uncertainty-estimation | density | 0.537 | 0.335 | -0.202 | -0.330 | -0.084 | 0.002 |
| hypothetical | unc_all_neg | 60 | backtracking | density | 0.170 | 0.167 | -0.004 | -0.079 | 0.085 | 0.932 |
| hypothetical | unc_all_neg | 60 | deduction | density_testlex | 0.244 | 0.165 | -0.079 | -0.161 | -0.001 | 0.057 |
| hypothetical | unc_all_neg | 60 | adding-knowledge | density_testlex | 0.107 | 0.027 | -0.079 | -0.127 | -0.033 | 0.002 |
| hypothetical | unc_all_neg | 60 | uncertainty-estimation | density_testlex | 0.081 | 0.036 | -0.045 | -0.082 | -0.006 | 0.026 |
| hypothetical | unc_all_neg | 60 | backtracking | density_testlex | 0.033 | 0.028 | -0.005 | -0.034 | 0.024 | 0.749 |
| hypothetical | unc_all_neg | 60 | deduction | density_nontest | 3.498 | 3.841 | 0.343 | -0.072 | 0.779 | 0.124 |
| hypothetical | unc_all_neg | 60 | adding-knowledge | density_nontest | 1.218 | 1.085 | -0.133 | -0.386 | 0.125 | 0.308 |
| hypothetical | unc_all_neg | 60 | uncertainty-estimation | density_nontest | 0.456 | 0.299 | -0.157 | -0.276 | -0.053 | 0.008 |
| hypothetical | unc_all_neg | 60 | backtracking | density_nontest | 0.137 | 0.138 | 0.001 | -0.071 | 0.080 | 0.976 |
| hypothetical | unc_all_pos | 60 | deduction | density | 3.742 | 3.444 | -0.299 | -0.680 | 0.082 | 0.134 |
| hypothetical | unc_all_pos | 60 | adding-knowledge | density | 1.325 | 0.961 | -0.364 | -0.584 | -0.147 | 0.002 |
| hypothetical | unc_all_pos | 60 | uncertainty-estimation | density | 0.537 | 0.848 | 0.311 | 0.151 | 0.466 | 0.000 |
| hypothetical | unc_all_pos | 60 | backtracking | density | 0.170 | 0.138 | -0.033 | -0.089 | 0.031 | 0.301 |
| hypothetical | unc_all_pos | 60 | deduction | density_testlex | 0.244 | 0.273 | 0.029 | -0.059 | 0.121 | 0.541 |
| hypothetical | unc_all_pos | 60 | adding-knowledge | density_testlex | 0.107 | 0.081 | -0.025 | -0.074 | 0.029 | 0.340 |
| hypothetical | unc_all_pos | 60 | uncertainty-estimation | density_testlex | 0.081 | 0.180 | 0.099 | 0.028 | 0.175 | 0.009 |
| hypothetical | unc_all_pos | 60 | backtracking | density_testlex | 0.033 | 0.025 | -0.008 | -0.034 | 0.017 | 0.520 |
| hypothetical | unc_all_pos | 60 | deduction | density_nontest | 3.498 | 3.170 | -0.328 | -0.712 | 0.052 | 0.094 |
| hypothetical | unc_all_pos | 60 | adding-knowledge | density_nontest | 1.218 | 0.880 | -0.338 | -0.549 | -0.131 | 0.003 |
| hypothetical | unc_all_pos | 60 | uncertainty-estimation | density_nontest | 0.456 | 0.668 | 0.212 | 0.063 | 0.355 | 0.005 |
| hypothetical | unc_all_pos | 60 | backtracking | density_nontest | 0.137 | 0.113 | -0.024 | -0.075 | 0.031 | 0.383 |
| hypothetical | unc_nontest_neg | 60 | deduction | density | 3.742 | 4.050 | 0.307 | -0.041 | 0.638 | 0.082 |
| hypothetical | unc_nontest_neg | 60 | adding-knowledge | density | 1.325 | 1.262 | -0.062 | -0.306 | 0.197 | 0.644 |
| hypothetical | unc_nontest_neg | 60 | uncertainty-estimation | density | 0.537 | 0.312 | -0.225 | -0.353 | -0.108 | 0.001 |
| hypothetical | unc_nontest_neg | 60 | backtracking | density | 0.170 | 0.171 | 0.001 | -0.064 | 0.073 | 0.973 |
| hypothetical | unc_nontest_neg | 60 | deduction | density_testlex | 0.244 | 0.198 | -0.047 | -0.137 | 0.043 | 0.315 |
| hypothetical | unc_nontest_neg | 60 | adding-knowledge | density_testlex | 0.107 | 0.088 | -0.018 | -0.078 | 0.046 | 0.560 |
| hypothetical | unc_nontest_neg | 60 | uncertainty-estimation | density_testlex | 0.081 | 0.017 | -0.064 | -0.103 | -0.027 | 0.001 |
| hypothetical | unc_nontest_neg | 60 | backtracking | density_testlex | 0.033 | 0.027 | -0.006 | -0.034 | 0.023 | 0.686 |
| hypothetical | unc_nontest_neg | 60 | deduction | density_nontest | 3.498 | 3.852 | 0.354 | 0.001 | 0.679 | 0.047 |
| hypothetical | unc_nontest_neg | 60 | adding-knowledge | density_nontest | 1.218 | 1.174 | -0.044 | -0.286 | 0.206 | 0.736 |
| hypothetical | unc_nontest_neg | 60 | uncertainty-estimation | density_nontest | 0.456 | 0.295 | -0.161 | -0.280 | -0.050 | 0.009 |
| hypothetical | unc_nontest_neg | 60 | backtracking | density_nontest | 0.137 | 0.144 | 0.007 | -0.052 | 0.071 | 0.828 |
| hypothetical | unc_nontest_pos | 60 | deduction | density | 3.742 | 3.777 | 0.034 | -0.342 | 0.405 | 0.859 |
| hypothetical | unc_nontest_pos | 60 | adding-knowledge | density | 1.325 | 0.890 | -0.434 | -0.676 | -0.171 | 0.002 |
| hypothetical | unc_nontest_pos | 60 | uncertainty-estimation | density | 0.537 | 0.867 | 0.330 | 0.153 | 0.506 | 0.001 |
| hypothetical | unc_nontest_pos | 60 | backtracking | density | 0.170 | 0.120 | -0.050 | -0.123 | 0.047 | 0.249 |
| hypothetical | unc_nontest_pos | 60 | deduction | density_testlex | 0.244 | 0.233 | -0.011 | -0.105 | 0.090 | 0.821 |
| hypothetical | unc_nontest_pos | 60 | adding-knowledge | density_testlex | 0.107 | 0.126 | 0.019 | -0.041 | 0.084 | 0.552 |
| hypothetical | unc_nontest_pos | 60 | uncertainty-estimation | density_testlex | 0.081 | 0.219 | 0.138 | 0.045 | 0.239 | 0.007 |
| hypothetical | unc_nontest_pos | 60 | backtracking | density_testlex | 0.033 | 0.009 | -0.024 | -0.047 | -0.004 | 0.036 |
| hypothetical | unc_nontest_pos | 60 | deduction | density_nontest | 3.498 | 3.543 | 0.045 | -0.311 | 0.395 | 0.804 |
| hypothetical | unc_nontest_pos | 60 | adding-knowledge | density_nontest | 1.218 | 0.764 | -0.454 | -0.686 | -0.213 | 0.000 |
| hypothetical | unc_nontest_pos | 60 | uncertainty-estimation | density_nontest | 0.456 | 0.648 | 0.192 | 0.023 | 0.356 | 0.025 |
| hypothetical | unc_nontest_pos | 60 | backtracking | density_nontest | 0.137 | 0.111 | -0.026 | -0.094 | 0.063 | 0.523 |
| hypothetical | unc_testlex_neg | 60 | deduction | density | 3.742 | 4.226 | 0.484 | 0.086 | 0.877 | 0.020 |
| hypothetical | unc_testlex_neg | 60 | adding-knowledge | density | 1.325 | 1.521 | 0.197 | -0.089 | 0.469 | 0.181 |
| hypothetical | unc_testlex_neg | 60 | uncertainty-estimation | density | 0.537 | 0.333 | -0.204 | -0.324 | -0.090 | 0.001 |
| hypothetical | unc_testlex_neg | 60 | backtracking | density | 0.170 | 0.144 | -0.026 | -0.092 | 0.042 | 0.477 |
| hypothetical | unc_testlex_neg | 60 | deduction | density_testlex | 0.244 | 0.148 | -0.097 | -0.197 | 0.010 | 0.067 |
| hypothetical | unc_testlex_neg | 60 | adding-knowledge | density_testlex | 0.107 | 0.035 | -0.072 | -0.122 | -0.021 | 0.010 |
| hypothetical | unc_testlex_neg | 60 | uncertainty-estimation | density_testlex | 0.081 | 0.064 | -0.017 | -0.060 | 0.029 | 0.455 |
| hypothetical | unc_testlex_neg | 60 | backtracking | density_testlex | 0.033 | 0.008 | -0.025 | -0.047 | -0.005 | 0.027 |
| hypothetical | unc_testlex_neg | 60 | deduction | density_nontest | 3.498 | 4.078 | 0.581 | 0.179 | 0.978 | 0.006 |
| hypothetical | unc_testlex_neg | 60 | adding-knowledge | density_nontest | 1.218 | 1.486 | 0.268 | -0.013 | 0.530 | 0.060 |
| hypothetical | unc_testlex_neg | 60 | uncertainty-estimation | density_nontest | 0.456 | 0.268 | -0.187 | -0.308 | -0.076 | 0.002 |
| hypothetical | unc_testlex_neg | 60 | backtracking | density_nontest | 0.137 | 0.136 | -0.001 | -0.069 | 0.065 | 0.987 |
| hypothetical | unc_testlex_pos | 60 | deduction | density | 3.742 | 3.564 | -0.178 | -0.564 | 0.202 | 0.375 |
| hypothetical | unc_testlex_pos | 60 | adding-knowledge | density | 1.325 | 0.912 | -0.413 | -0.615 | -0.202 | 0.000 |
| hypothetical | unc_testlex_pos | 60 | uncertainty-estimation | density | 0.537 | 0.750 | 0.213 | 0.052 | 0.366 | 0.011 |
| hypothetical | unc_testlex_pos | 60 | backtracking | density | 0.170 | 0.173 | 0.002 | -0.061 | 0.065 | 0.944 |
| hypothetical | unc_testlex_pos | 60 | deduction | density_testlex | 0.244 | 0.256 | 0.012 | -0.074 | 0.095 | 0.787 |
| hypothetical | unc_testlex_pos | 60 | adding-knowledge | density_testlex | 0.107 | 0.069 | -0.038 | -0.093 | 0.021 | 0.184 |
| hypothetical | unc_testlex_pos | 60 | uncertainty-estimation | density_testlex | 0.081 | 0.121 | 0.040 | -0.015 | 0.094 | 0.158 |
| hypothetical | unc_testlex_pos | 60 | backtracking | density_testlex | 0.033 | 0.018 | -0.015 | -0.032 | 0.001 | 0.084 |
| hypothetical | unc_testlex_pos | 60 | deduction | density_nontest | 3.498 | 3.308 | -0.190 | -0.565 | 0.183 | 0.335 |
| hypothetical | unc_testlex_pos | 60 | adding-knowledge | density_nontest | 1.218 | 0.843 | -0.375 | -0.573 | -0.162 | 0.001 |
| hypothetical | unc_testlex_pos | 60 | uncertainty-estimation | density_nontest | 0.456 | 0.629 | 0.173 | 0.032 | 0.299 | 0.015 |
| hypothetical | unc_testlex_pos | 60 | backtracking | density_nontest | 0.137 | 0.154 | 0.017 | -0.041 | 0.076 | 0.578 |
| real | unc_all_neg | 60 | deduction | density | 3.708 | 3.844 | 0.135 | -0.297 | 0.594 | 0.551 |
| real | unc_all_neg | 60 | adding-knowledge | density | 1.232 | 1.389 | 0.156 | -0.136 | 0.423 | 0.282 |
| real | unc_all_neg | 60 | uncertainty-estimation | density | 0.740 | 0.363 | -0.377 | -0.523 | -0.231 | 0.000 |
| real | unc_all_neg | 60 | backtracking | density | 0.129 | 0.178 | 0.050 | -0.027 | 0.127 | 0.212 |
| real | unc_all_neg | 60 | deduction | density_testlex | 0.123 | 0.070 | -0.053 | -0.096 | -0.011 | 0.020 |
| real | unc_all_neg | 60 | adding-knowledge | density_testlex | 0.015 | 0.000 | -0.015 | -0.030 | -0.003 | 0.033 |
| real | unc_all_neg | 60 | uncertainty-estimation | density_testlex | 0.105 | 0.034 | -0.071 | -0.113 | -0.033 | 0.001 |
| real | unc_all_neg | 60 | backtracking | density_testlex | 0.007 | 0.001 | -0.006 | -0.017 | 0.000 | 0.324 |
| real | unc_all_neg | 60 | deduction | density_nontest | 3.586 | 3.774 | 0.188 | -0.218 | 0.629 | 0.398 |
| real | unc_all_neg | 60 | adding-knowledge | density_nontest | 1.217 | 1.389 | 0.171 | -0.121 | 0.442 | 0.242 |
| real | unc_all_neg | 60 | uncertainty-estimation | density_nontest | 0.635 | 0.329 | -0.306 | -0.446 | -0.169 | 0.000 |
| real | unc_all_neg | 60 | backtracking | density_nontest | 0.121 | 0.177 | 0.055 | -0.012 | 0.131 | 0.143 |
| real | unc_all_pos | 60 | deduction | density | 3.708 | 3.660 | -0.048 | -0.456 | 0.364 | 0.821 |
| real | unc_all_pos | 60 | adding-knowledge | density | 1.232 | 0.910 | -0.322 | -0.577 | -0.075 | 0.014 |
| real | unc_all_pos | 60 | uncertainty-estimation | density | 0.740 | 1.021 | 0.281 | 0.064 | 0.522 | 0.017 |
| real | unc_all_pos | 60 | backtracking | density | 0.129 | 0.080 | -0.048 | -0.119 | 0.022 | 0.200 |
| real | unc_all_pos | 60 | deduction | density_testlex | 0.123 | 0.181 | 0.058 | -0.004 | 0.118 | 0.084 |
| real | unc_all_pos | 60 | adding-knowledge | density_testlex | 0.015 | 0.020 | 0.005 | -0.015 | 0.027 | 0.639 |
| real | unc_all_pos | 60 | uncertainty-estimation | density_testlex | 0.105 | 0.166 | 0.061 | -0.006 | 0.131 | 0.092 |
| real | unc_all_pos | 60 | backtracking | density_testlex | 0.007 | 0.000 | -0.007 | -0.021 | 0.000 | 0.227 |
| real | unc_all_pos | 60 | deduction | density_nontest | 3.586 | 3.479 | -0.106 | -0.524 | 0.302 | 0.618 |
| real | unc_all_pos | 60 | adding-knowledge | density_nontest | 1.217 | 0.890 | -0.327 | -0.579 | -0.078 | 0.014 |
| real | unc_all_pos | 60 | uncertainty-estimation | density_nontest | 0.635 | 0.855 | 0.221 | 0.042 | 0.415 | 0.022 |
| real | unc_all_pos | 60 | backtracking | density_nontest | 0.121 | 0.080 | -0.041 | -0.108 | 0.027 | 0.244 |
| real | unc_nontest_neg | 60 | deduction | density | 3.708 | 3.740 | 0.032 | -0.395 | 0.441 | 0.882 |
| real | unc_nontest_neg | 60 | adding-knowledge | density | 1.232 | 1.431 | 0.199 | -0.049 | 0.451 | 0.128 |
| real | unc_nontest_neg | 60 | uncertainty-estimation | density | 0.740 | 0.444 | -0.296 | -0.449 | -0.143 | 0.000 |
| real | unc_nontest_neg | 60 | backtracking | density | 0.129 | 0.164 | 0.035 | -0.040 | 0.104 | 0.338 |
| real | unc_nontest_neg | 60 | deduction | density_testlex | 0.123 | 0.058 | -0.064 | -0.112 | -0.016 | 0.012 |
| real | unc_nontest_neg | 60 | adding-knowledge | density_testlex | 0.015 | 0.020 | 0.005 | -0.019 | 0.033 | 0.722 |
| real | unc_nontest_neg | 60 | uncertainty-estimation | density_testlex | 0.105 | 0.035 | -0.071 | -0.112 | -0.032 | 0.001 |
| real | unc_nontest_neg | 60 | backtracking | density_testlex | 0.007 | 0.005 | -0.002 | -0.018 | 0.014 | 0.816 |
| real | unc_nontest_neg | 60 | deduction | density_nontest | 3.586 | 3.681 | 0.096 | -0.326 | 0.498 | 0.650 |
| real | unc_nontest_neg | 60 | adding-knowledge | density_nontest | 1.217 | 1.411 | 0.194 | -0.061 | 0.449 | 0.143 |
| real | unc_nontest_neg | 60 | uncertainty-estimation | density_nontest | 0.635 | 0.410 | -0.225 | -0.370 | -0.077 | 0.004 |
| real | unc_nontest_neg | 60 | backtracking | density_nontest | 0.121 | 0.159 | 0.037 | -0.029 | 0.101 | 0.274 |
| real | unc_nontest_pos | 60 | deduction | density | 3.708 | 3.611 | -0.098 | -0.467 | 0.270 | 0.610 |
| real | unc_nontest_pos | 60 | adding-knowledge | density | 1.232 | 0.960 | -0.272 | -0.513 | -0.011 | 0.041 |
| real | unc_nontest_pos | 60 | uncertainty-estimation | density | 0.740 | 1.178 | 0.438 | 0.222 | 0.653 | 0.000 |
| real | unc_nontest_pos | 60 | backtracking | density | 0.129 | 0.121 | -0.008 | -0.084 | 0.069 | 0.852 |
| real | unc_nontest_pos | 60 | deduction | density_testlex | 0.123 | 0.151 | 0.028 | -0.034 | 0.092 | 0.402 |
| real | unc_nontest_pos | 60 | adding-knowledge | density_testlex | 0.015 | 0.022 | 0.007 | -0.014 | 0.031 | 0.528 |
| real | unc_nontest_pos | 60 | uncertainty-estimation | density_testlex | 0.105 | 0.199 | 0.093 | 0.030 | 0.158 | 0.006 |
| real | unc_nontest_pos | 60 | backtracking | density_testlex | 0.007 | 0.007 | -0.000 | -0.017 | 0.017 | 0.974 |
| real | unc_nontest_pos | 60 | deduction | density_nontest | 3.586 | 3.460 | -0.126 | -0.493 | 0.240 | 0.511 |
| real | unc_nontest_pos | 60 | adding-knowledge | density_nontest | 1.217 | 0.938 | -0.279 | -0.519 | -0.017 | 0.036 |
| real | unc_nontest_pos | 60 | uncertainty-estimation | density_nontest | 0.635 | 0.980 | 0.345 | 0.149 | 0.543 | 0.001 |
| real | unc_nontest_pos | 60 | backtracking | density_nontest | 0.121 | 0.114 | -0.007 | -0.078 | 0.062 | 0.841 |
| real | unc_testlex_neg | 60 | deduction | density | 3.708 | 4.251 | 0.543 | 0.154 | 0.971 | 0.011 |
| real | unc_testlex_neg | 60 | adding-knowledge | density | 1.232 | 1.435 | 0.202 | -0.089 | 0.494 | 0.184 |
| real | unc_testlex_neg | 60 | uncertainty-estimation | density | 0.740 | 0.415 | -0.325 | -0.461 | -0.192 | 0.000 |
| real | unc_testlex_neg | 60 | backtracking | density | 0.129 | 0.176 | 0.048 | -0.023 | 0.125 | 0.213 |
| real | unc_testlex_neg | 60 | deduction | density_testlex | 0.123 | 0.021 | -0.102 | -0.146 | -0.059 | 0.000 |
| real | unc_testlex_neg | 60 | adding-knowledge | density_testlex | 0.015 | 0.001 | -0.014 | -0.029 | -0.001 | 0.060 |
| real | unc_testlex_neg | 60 | uncertainty-estimation | density_testlex | 0.105 | 0.023 | -0.082 | -0.124 | -0.043 | 0.000 |
| real | unc_testlex_neg | 60 | backtracking | density_testlex | 0.007 | 0.003 | -0.004 | -0.019 | 0.008 | 0.568 |
| real | unc_testlex_neg | 60 | deduction | density_nontest | 3.586 | 4.230 | 0.645 | 0.253 | 1.074 | 0.003 |
| real | unc_testlex_neg | 60 | adding-knowledge | density_nontest | 1.217 | 1.433 | 0.216 | -0.076 | 0.508 | 0.158 |
| real | unc_testlex_neg | 60 | uncertainty-estimation | density_nontest | 0.635 | 0.392 | -0.242 | -0.364 | -0.120 | 0.000 |
| real | unc_testlex_neg | 60 | backtracking | density_nontest | 0.121 | 0.173 | 0.052 | -0.015 | 0.125 | 0.155 |
| real | unc_testlex_pos | 60 | deduction | density | 3.708 | 3.419 | -0.289 | -0.627 | 0.055 | 0.104 |
| real | unc_testlex_pos | 60 | adding-knowledge | density | 1.232 | 1.045 | -0.188 | -0.437 | 0.066 | 0.149 |
| real | unc_testlex_pos | 60 | uncertainty-estimation | density | 0.740 | 0.933 | 0.193 | -0.024 | 0.420 | 0.088 |
| real | unc_testlex_pos | 60 | backtracking | density | 0.129 | 0.205 | 0.076 | -0.010 | 0.172 | 0.118 |
| real | unc_testlex_pos | 60 | deduction | density_testlex | 0.123 | 0.214 | 0.091 | 0.025 | 0.155 | 0.008 |
| real | unc_testlex_pos | 60 | adding-knowledge | density_testlex | 0.015 | 0.034 | 0.019 | -0.003 | 0.043 | 0.107 |
| real | unc_testlex_pos | 60 | uncertainty-estimation | density_testlex | 0.105 | 0.126 | 0.020 | -0.032 | 0.074 | 0.450 |
| real | unc_testlex_pos | 60 | backtracking | density_testlex | 0.007 | 0.011 | 0.004 | -0.015 | 0.025 | 0.675 |
| real | unc_testlex_pos | 60 | deduction | density_nontest | 3.586 | 3.206 | -0.380 | -0.708 | -0.050 | 0.029 |
| real | unc_testlex_pos | 60 | adding-knowledge | density_nontest | 1.217 | 1.011 | -0.207 | -0.452 | 0.044 | 0.109 |
| real | unc_testlex_pos | 60 | uncertainty-estimation | density_nontest | 0.635 | 0.807 | 0.173 | -0.024 | 0.382 | 0.100 |
| real | unc_testlex_pos | 60 | backtracking | density_nontest | 0.121 | 0.193 | 0.072 | -0.004 | 0.159 | 0.099 |

