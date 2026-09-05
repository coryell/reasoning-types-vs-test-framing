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


## Representation-level morphology: Δ mean projection onto the rebuilt directions (steered model's own activations)

| arm | framing | projection | n_pairs | base_mean | delta | ci_lo | ci_hi |
|---|---|---|---|---|---|---|---|
| unc_all_neg | real | proj_probe | 60 | -3.4419 | -0.4239 | -0.7013 | -0.1341 |
| unc_all_neg | real | proj_lastprompt_backtracking | 60 | 8.7861 | -3.0235 | -3.1437 | -2.9001 |
| unc_all_neg | real | proj_lastprompt_uncertainty-estimation | 60 | 11.5538 | -10.1413 | -10.1433 | -10.1395 |
| unc_all_neg | real | proj_lastprompt_probe | 60 | -2.4264 | -0.7048 | -0.7544 | -0.6546 |
| unc_all_neg | real | proj_incontext_backtracking | 60 | 20.8345 | -7.2259 | -7.6293 | -6.8260 |
| unc_all_neg | real | proj_incontext_uncertainty-estimation | 60 | 10.3384 | -10.2533 | -11.0213 | -9.4852 |
| unc_all_neg | real | proj_incontext_probe | 60 | -3.4669 | -0.9357 | -1.2102 | -0.6648 |
| unc_all_neg | hypothetical | proj_probe | 60 | -3.8292 | -0.1281 | -0.4050 | 0.1298 |
| unc_all_neg | hypothetical | proj_lastprompt_backtracking | 60 | 8.9752 | -3.0300 | -3.1302 | -2.9269 |
| unc_all_neg | hypothetical | proj_lastprompt_uncertainty-estimation | 60 | 11.4298 | -10.1411 | -10.1430 | -10.1394 |
| unc_all_neg | hypothetical | proj_lastprompt_probe | 60 | -2.5859 | -0.7046 | -0.7477 | -0.6634 |
| unc_all_neg | hypothetical | proj_incontext_backtracking | 60 | 20.3364 | -7.0871 | -7.4632 | -6.7167 |
| unc_all_neg | hypothetical | proj_incontext_uncertainty-estimation | 60 | 10.6492 | -11.0244 | -11.7451 | -10.3342 |
| unc_all_neg | hypothetical | proj_incontext_probe | 60 | -3.7889 | -0.6942 | -0.9700 | -0.4415 |
| unc_all_pos | real | proj_probe | 60 | -3.4419 | 0.5734 | 0.1923 | 0.9810 |
| unc_all_pos | real | proj_lastprompt_backtracking | 60 | 8.7861 | 6.2645 | 6.0493 | 6.4698 |
| unc_all_pos | real | proj_lastprompt_uncertainty-estimation | 60 | 11.5538 | 10.1368 | 10.1348 | 10.1391 |
| unc_all_pos | real | proj_lastprompt_probe | 60 | -2.4264 | 0.3044 | 0.2600 | 0.3504 |
| unc_all_pos | real | proj_incontext_backtracking | 60 | 20.8345 | 8.4534 | 7.9454 | 8.9804 |
| unc_all_pos | real | proj_incontext_uncertainty-estimation | 60 | 10.3384 | 10.8996 | 10.1445 | 11.6181 |
| unc_all_pos | real | proj_incontext_probe | 60 | -3.4669 | 0.8013 | 0.4359 | 1.1979 |
| unc_all_pos | hypothetical | proj_probe | 60 | -3.8292 | 0.5422 | 0.3378 | 0.7830 |
| unc_all_pos | hypothetical | proj_lastprompt_backtracking | 60 | 8.9752 | 6.1433 | 5.9165 | 6.3483 |
| unc_all_pos | hypothetical | proj_lastprompt_uncertainty-estimation | 60 | 11.4298 | 10.1362 | 10.1345 | 10.1381 |
| unc_all_pos | hypothetical | proj_lastprompt_probe | 60 | -2.5859 | 0.3323 | 0.2865 | 0.3782 |
| unc_all_pos | hypothetical | proj_incontext_backtracking | 60 | 20.3364 | 8.3441 | 7.9936 | 8.6916 |
| unc_all_pos | hypothetical | proj_incontext_uncertainty-estimation | 60 | 10.6492 | 10.4679 | 9.7860 | 11.0717 |
| unc_all_pos | hypothetical | proj_incontext_probe | 60 | -3.7889 | 0.7392 | 0.5268 | 0.9816 |
| unc_nontest_neg | real | proj_probe | 60 | -3.4419 | -0.6731 | -0.9472 | -0.3652 |
| unc_nontest_neg | real | proj_lastprompt_backtracking | 60 | 8.7861 | -2.8729 | -2.9977 | -2.7453 |
| unc_nontest_neg | real | proj_lastprompt_uncertainty-estimation | 60 | 11.5538 | -10.1270 | -10.1287 | -10.1248 |
| unc_nontest_neg | real | proj_lastprompt_probe | 60 | -2.4264 | -0.5425 | -0.5906 | -0.4939 |
| unc_nontest_neg | real | proj_incontext_backtracking | 60 | 20.8345 | -7.1221 | -7.6466 | -6.6476 |
| unc_nontest_neg | real | proj_incontext_uncertainty-estimation | 60 | 10.3384 | -10.3301 | -11.2406 | -9.4127 |
| unc_nontest_neg | real | proj_incontext_probe | 60 | -3.4669 | -0.9878 | -1.2510 | -0.6939 |
| unc_nontest_neg | hypothetical | proj_probe | 60 | -3.8292 | -0.1195 | -0.4710 | 0.1999 |
| unc_nontest_neg | hypothetical | proj_lastprompt_backtracking | 60 | 8.9752 | -2.8789 | -2.9769 | -2.7786 |
| unc_nontest_neg | hypothetical | proj_lastprompt_uncertainty-estimation | 60 | 11.4298 | -10.1278 | -10.1292 | -10.1260 |
| unc_nontest_neg | hypothetical | proj_lastprompt_probe | 60 | -2.5859 | -0.5473 | -0.5879 | -0.5079 |
| unc_nontest_neg | hypothetical | proj_incontext_backtracking | 60 | 20.3364 | -6.7750 | -7.1523 | -6.4206 |
| unc_nontest_neg | hypothetical | proj_incontext_uncertainty-estimation | 60 | 10.6492 | -10.6251 | -11.3937 | -9.8653 |
| unc_nontest_neg | hypothetical | proj_incontext_probe | 60 | -3.7889 | -0.5679 | -0.9086 | -0.2628 |
| unc_nontest_pos | real | proj_probe | 60 | -3.4419 | 0.5044 | 0.1165 | 0.8853 |
| unc_nontest_pos | real | proj_lastprompt_backtracking | 60 | 8.7861 | 6.0060 | 5.7946 | 6.2021 |
| unc_nontest_pos | real | proj_lastprompt_uncertainty-estimation | 60 | 11.5538 | 10.1235 | 10.1213 | 10.1254 |
| unc_nontest_pos | real | proj_lastprompt_probe | 60 | -2.4264 | 0.1297 | 0.0867 | 0.1750 |
| unc_nontest_pos | real | proj_incontext_backtracking | 60 | 20.8345 | 8.1606 | 7.6186 | 8.6539 |
| unc_nontest_pos | real | proj_incontext_uncertainty-estimation | 60 | 10.3384 | 11.3632 | 10.5823 | 12.1338 |
| unc_nontest_pos | real | proj_incontext_probe | 60 | -3.4669 | 0.5436 | 0.1644 | 0.9259 |
| unc_nontest_pos | hypothetical | proj_probe | 60 | -3.8292 | 0.6839 | 0.4091 | 0.9611 |
| unc_nontest_pos | hypothetical | proj_lastprompt_backtracking | 60 | 8.9752 | 5.8713 | 5.6479 | 6.0753 |
| unc_nontest_pos | hypothetical | proj_lastprompt_uncertainty-estimation | 60 | 11.4298 | 10.1237 | 10.1213 | 10.1256 |
| unc_nontest_pos | hypothetical | proj_lastprompt_probe | 60 | -2.5859 | 0.1576 | 0.1125 | 0.2032 |
| unc_nontest_pos | hypothetical | proj_incontext_backtracking | 60 | 20.3364 | 8.0268 | 7.5710 | 8.4619 |
| unc_nontest_pos | hypothetical | proj_incontext_uncertainty-estimation | 60 | 10.6492 | 10.1424 | 9.3498 | 10.9173 |
| unc_nontest_pos | hypothetical | proj_incontext_probe | 60 | -3.7889 | 0.7100 | 0.4389 | 0.9955 |
| unc_testlex_neg | real | proj_probe | 60 | -3.4419 | -0.4659 | -0.7101 | -0.2154 |
| unc_testlex_neg | real | proj_lastprompt_backtracking | 60 | 8.7861 | -2.1700 | -2.2726 | -2.0735 |
| unc_testlex_neg | real | proj_lastprompt_uncertainty-estimation | 60 | 11.5538 | -3.1342 | -3.1423 | -3.1262 |
| unc_testlex_neg | real | proj_lastprompt_probe | 60 | -2.4264 | -1.5166 | -1.5638 | -1.4680 |
| unc_testlex_neg | real | proj_incontext_backtracking | 60 | 20.8345 | -5.9550 | -6.3061 | -5.5881 |
| unc_testlex_neg | real | proj_incontext_uncertainty-estimation | 60 | 10.3384 | -3.2142 | -3.9701 | -2.4568 |
| unc_testlex_neg | real | proj_incontext_probe | 60 | -3.4669 | -1.8281 | -2.0798 | -1.5647 |
| unc_testlex_neg | hypothetical | proj_probe | 60 | -3.8292 | -0.0962 | -0.4050 | 0.1930 |
| unc_testlex_neg | hypothetical | proj_lastprompt_backtracking | 60 | 8.9752 | -2.1577 | -2.2488 | -2.0712 |
| unc_testlex_neg | hypothetical | proj_lastprompt_uncertainty-estimation | 60 | 11.4298 | -3.1361 | -3.1440 | -3.1277 |
| unc_testlex_neg | hypothetical | proj_lastprompt_probe | 60 | -2.5859 | -1.4778 | -1.5214 | -1.4349 |
| unc_testlex_neg | hypothetical | proj_incontext_backtracking | 60 | 20.3364 | -6.1088 | -6.5413 | -5.6809 |
| unc_testlex_neg | hypothetical | proj_incontext_uncertainty-estimation | 60 | 10.6492 | -4.0374 | -4.8421 | -3.2310 |
| unc_testlex_neg | hypothetical | proj_incontext_probe | 60 | -3.7889 | -1.5097 | -1.7987 | -1.2333 |
| unc_testlex_pos | real | proj_probe | 60 | -3.4419 | 0.2411 | -0.1027 | 0.6162 |
| unc_testlex_pos | real | proj_lastprompt_backtracking | 60 | 8.7861 | 3.4298 | 3.3081 | 3.5355 |
| unc_testlex_pos | real | proj_lastprompt_uncertainty-estimation | 60 | 11.5538 | 3.1366 | 3.1355 | 3.1376 |
| unc_testlex_pos | real | proj_lastprompt_probe | 60 | -2.4264 | 1.4130 | 1.3721 | 1.4554 |
| unc_testlex_pos | real | proj_incontext_backtracking | 60 | 20.8345 | 6.8010 | 6.2660 | 7.3046 |
| unc_testlex_pos | real | proj_incontext_uncertainty-estimation | 60 | 10.3384 | 3.9202 | 3.2572 | 4.5650 |
| unc_testlex_pos | real | proj_incontext_probe | 60 | -3.4669 | 1.5007 | 1.1877 | 1.8381 |
| unc_testlex_pos | hypothetical | proj_probe | 60 | -3.8292 | 0.3568 | 0.0186 | 0.6937 |
| unc_testlex_pos | hypothetical | proj_lastprompt_backtracking | 60 | 8.9752 | 3.2307 | 3.0847 | 3.3593 |
| unc_testlex_pos | hypothetical | proj_lastprompt_uncertainty-estimation | 60 | 11.4298 | 3.1356 | 3.1345 | 3.1368 |
| unc_testlex_pos | hypothetical | proj_lastprompt_probe | 60 | -2.5859 | 1.4008 | 1.3595 | 1.4429 |
| unc_testlex_pos | hypothetical | proj_incontext_backtracking | 60 | 20.3364 | 6.6131 | 6.1648 | 7.0699 |
| unc_testlex_pos | hypothetical | proj_incontext_uncertainty-estimation | 60 | 10.6492 | 3.3761 | 2.6895 | 4.0165 |
| unc_testlex_pos | hypothetical | proj_incontext_probe | 60 | -3.7889 | 1.5572 | 1.2238 | 1.8982 |


## Awareness-probe score of each arm's reasoning (unedited model, the probe's own input regime and hidden-state index): per arm, and verbalizing vs non-verbalizing traces within the arm

| arm | framing | n | mean | ci_lo | ci_hi | n_verbalizing | mean_verbalizing | mean_not | p_welch |
|---|---|---|---|---|---|---|---|---|---|
| baseline | real | 60 | -3.4419 | -3.7663 | -3.0811 | 10 | -1.8753 | -3.7553 | 0.0021 |
| baseline | hypothetical | 60 | -3.8292 | -4.0863 | -3.5514 | 3 | -2.7066 | -3.8883 | 0.2269 |
| unc_all_neg | real | 60 | -3.8658 | -4.1395 | -3.5940 | 3 | -1.7614 | -3.9766 | 0.0317 |
| unc_all_neg | hypothetical | 60 | -3.9573 | -4.1757 | -3.7297 | 1 | -4.1656 | -3.9538 | — |
| unc_all_pos | real | 60 | -2.8685 | -3.2449 | -2.4897 | 16 | -0.9745 | -3.5573 | 0.0000 |
| unc_all_pos | hypothetical | 60 | -3.2870 | -3.5438 | -2.9941 | 10 | -2.4500 | -3.4544 | 0.0352 |
| unc_nontest_neg | real | 60 | -4.1151 | -4.3822 | -3.8469 | 3 | -1.6666 | -4.2440 | 0.0002 |
| unc_nontest_neg | hypothetical | 60 | -3.9487 | -4.2115 | -3.6698 | 2 | -2.8131 | -3.9879 | 0.5318 |
| unc_nontest_pos | real | 60 | -2.9376 | -3.3080 | -2.5602 | 18 | -1.6186 | -3.5028 | 0.0000 |
| unc_nontest_pos | hypothetical | 60 | -3.1453 | -3.4590 | -2.8239 | 11 | -1.7663 | -3.4549 | 0.0048 |
| unc_testlex_neg | real | 60 | -3.9078 | -4.2079 | -3.5881 | 2 | -1.1389 | -4.0033 | 0.0010 |
| unc_testlex_neg | hypothetical | 60 | -3.9254 | -4.1899 | -3.6445 | 5 | -2.3696 | -4.0669 | 0.0973 |
| unc_testlex_pos | real | 60 | -3.2008 | -3.5303 | -2.8485 | 11 | -1.4472 | -3.5945 | 0.0005 |
| unc_testlex_pos | hypothetical | 60 | -3.4724 | -3.7434 | -3.1588 | 9 | -2.2333 | -3.6911 | 0.0259 |

