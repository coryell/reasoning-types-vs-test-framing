# Mechanistic arm (Qwen3-4B) — first-pass analysis: f4_reverse

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


## f4_reverse: 1788 traces, 1788 with usable morphology annotation

| arm | framing | n | exec_rate_closed | exec_rate_all | lost | gained | p_mcnemar | task_judged_yes | judge_vs_string_agree | verbalization | closure | repetition | words |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | real | 149 | 0.726 | 0.711 | — | — | — | 0.826 | 0.582 | 0.188 | 0.980 | 0.046 | 334.470 |
| baseline | fake | 149 | 0.882 | 0.852 | — | — | — | 0.851 | 0.762 | 0.088 | 0.966 | 0.047 | 345.228 |
| baseline_seed1 | real | 149 | 0.748 | 0.718 | 3.000 | 6.000 | 0.508 | 0.738 | 0.545 | 0.148 | 0.960 | 0.051 | 353.859 |
| baseline_seed1 | fake | 149 | 0.890 | 0.872 | 5.000 | 6.000 | 1.000 | 0.852 | 0.760 | 0.060 | 0.980 | 0.047 | 331.664 |
| bt_neg | real | 149 | 0.711 | 0.711 | 14.000 | 11.000 | 0.690 | 0.866 | 0.604 | 0.013 | 1.000 | 0.009 | 186.349 |
| bt_neg | fake | 0 | — | — | 0.000 | 0.000 | — | — | — | — | — | — | — |
| bt_pos | real | 149 | 0.660 | 0.416 | 5.000 | 3.000 | 0.727 | 0.497 | 0.404 | 0.436 | 0.631 | 0.216 | 789.893 |
| bt_pos | fake | 0 | — | — | 0.000 | 0.000 | — | — | — | — | — | — | — |
| random_bt | real | 149 | 0.782 | 0.772 | 7.000 | 15.000 | 0.134 | 0.804 | 0.596 | 0.182 | 0.987 | 0.046 | 336.369 |
| random_bt | fake | 0 | — | — | 0.000 | 0.000 | — | — | — | — | — | — | — |
| random_unc | real | 149 | 0.803 | 0.792 | 3.000 | 14.000 | 0.013 | 0.735 | 0.545 | 0.204 | 0.987 | 0.049 | 333.987 |
| random_unc | fake | 0 | — | — | 0.000 | 0.000 | — | — | — | — | — | — | — |
| shuffled_bt | real | 149 | 0.688 | 0.517 | 4.000 | 4.000 | 1.000 | 0.584 | 0.482 | 0.295 | 0.752 | 0.183 | 688.705 |
| shuffled_bt | fake | 0 | — | — | 0.000 | 0.000 | — | — | — | — | — | — | — |
| shuffled_unc | real | 149 | 0.742 | 0.638 | 8.000 | 9.000 | 1.000 | 0.611 | 0.500 | 0.275 | 0.859 | 0.105 | 564.745 |
| shuffled_unc | fake | 0 | — | — | 0.000 | 0.000 | — | — | — | — | — | — | — |
| unc_neg | real | 149 | 0.788 | 0.772 | 8.000 | 16.000 | 0.152 | 0.784 | 0.600 | 0.007 | 0.980 | 0.028 | 265.866 |
| unc_neg | fake | 0 | — | — | 0.000 | 0.000 | — | — | — | — | — | — | — |
| unc_pos | real | 149 | 0.683 | 0.638 | 10.000 | 7.000 | 0.629 | 0.685 | 0.468 | 0.369 | 0.933 | 0.103 | 500.685 |
| unc_pos | fake | 0 | — | — | 0.000 | 0.000 | — | — | — | — | — | — | — |


## Morphology: Δ spans per 100 words vs baseline, paired by item

| framing | arm | n_pairs | behaviour | metric | base_mean | arm_mean | delta | ci_lo | ci_hi | p_paired_t |
|---|---|---|---|---|---|---|---|---|---|---|
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
| real | bt_neg | 149 | deduction | density | 4.160 | 4.682 | 0.522 | 0.266 | 0.773 | 0.000 |
| real | bt_neg | 149 | adding-knowledge | density | 1.570 | 1.949 | 0.379 | 0.191 | 0.564 | 0.000 |
| real | bt_neg | 149 | uncertainty-estimation | density | 0.721 | 0.303 | -0.418 | -0.553 | -0.294 | 0.000 |
| real | bt_neg | 149 | backtracking | density | 0.279 | 0.122 | -0.158 | -0.229 | -0.093 | 0.000 |
| real | bt_neg | 149 | deduction | density_testlex | 0.157 | 0.234 | 0.078 | 0.012 | 0.142 | 0.021 |
| real | bt_neg | 149 | adding-knowledge | density_testlex | 0.016 | 0.003 | -0.013 | -0.028 | -0.001 | 0.053 |
| real | bt_neg | 149 | uncertainty-estimation | density_testlex | 0.132 | 0.031 | -0.101 | -0.142 | -0.062 | 0.000 |
| real | bt_neg | 149 | backtracking | density_testlex | 0.004 | 0.000 | -0.004 | -0.009 | 0.000 | 0.158 |
| real | bt_neg | 149 | deduction | density_nontest | 4.003 | 4.448 | 0.444 | 0.181 | 0.706 | 0.002 |
| real | bt_neg | 149 | adding-knowledge | density_nontest | 1.554 | 1.946 | 0.392 | 0.200 | 0.578 | 0.000 |
| real | bt_neg | 149 | uncertainty-estimation | density_nontest | 0.589 | 0.272 | -0.317 | -0.446 | -0.194 | 0.000 |
| real | bt_neg | 149 | backtracking | density_nontest | 0.276 | 0.122 | -0.154 | -0.225 | -0.089 | 0.000 |
| real | bt_pos | 149 | deduction | density | 4.160 | 3.497 | -0.663 | -0.909 | -0.408 | 0.000 |
| real | bt_pos | 149 | adding-knowledge | density | 1.570 | 1.219 | -0.350 | -0.485 | -0.210 | 0.000 |
| real | bt_pos | 149 | uncertainty-estimation | density | 0.721 | 1.399 | 0.678 | 0.520 | 0.841 | 0.000 |
| real | bt_pos | 149 | backtracking | density | 0.279 | 0.342 | 0.062 | -0.022 | 0.150 | 0.158 |
| real | bt_pos | 149 | deduction | density_testlex | 0.157 | 0.097 | -0.060 | -0.102 | -0.019 | 0.006 |
| real | bt_pos | 149 | adding-knowledge | density_testlex | 0.016 | 0.016 | -0.000 | -0.018 | 0.017 | 0.967 |
| real | bt_pos | 149 | uncertainty-estimation | density_testlex | 0.132 | 0.168 | 0.036 | -0.012 | 0.084 | 0.141 |
| real | bt_pos | 149 | backtracking | density_testlex | 0.004 | 0.004 | 0.000 | -0.007 | 0.008 | 0.993 |
| real | bt_pos | 149 | deduction | density_nontest | 4.003 | 3.400 | -0.603 | -0.853 | -0.340 | 0.000 |
| real | bt_pos | 149 | adding-knowledge | density_nontest | 1.554 | 1.204 | -0.350 | -0.487 | -0.209 | 0.000 |
| real | bt_pos | 149 | uncertainty-estimation | density_nontest | 0.589 | 1.231 | 0.643 | 0.487 | 0.804 | 0.000 |
| real | bt_pos | 149 | backtracking | density_nontest | 0.276 | 0.338 | 0.062 | -0.020 | 0.150 | 0.156 |
| real | random_bt | 149 | deduction | density | 4.160 | 4.162 | 0.002 | -0.230 | 0.238 | 0.989 |
| real | random_bt | 149 | adding-knowledge | density | 1.570 | 1.584 | 0.014 | -0.147 | 0.168 | 0.864 |
| real | random_bt | 149 | uncertainty-estimation | density | 0.721 | 0.705 | -0.016 | -0.164 | 0.128 | 0.832 |
| real | random_bt | 149 | backtracking | density | 0.279 | 0.234 | -0.046 | -0.116 | 0.015 | 0.187 |
| real | random_bt | 149 | deduction | density_testlex | 0.157 | 0.141 | -0.016 | -0.062 | 0.029 | 0.496 |
| real | random_bt | 149 | adding-knowledge | density_testlex | 0.016 | 0.011 | -0.005 | -0.020 | 0.010 | 0.503 |
| real | random_bt | 149 | uncertainty-estimation | density_testlex | 0.132 | 0.085 | -0.047 | -0.087 | -0.008 | 0.021 |
| real | random_bt | 149 | backtracking | density_testlex | 0.004 | 0.003 | -0.000 | -0.007 | 0.008 | 0.927 |
| real | random_bt | 149 | deduction | density_nontest | 4.003 | 4.021 | 0.018 | -0.205 | 0.255 | 0.883 |
| real | random_bt | 149 | adding-knowledge | density_nontest | 1.554 | 1.573 | 0.019 | -0.143 | 0.171 | 0.816 |
| real | random_bt | 149 | uncertainty-estimation | density_nontest | 0.589 | 0.620 | 0.031 | -0.112 | 0.174 | 0.680 |
| real | random_bt | 149 | backtracking | density_nontest | 0.276 | 0.230 | -0.045 | -0.114 | 0.015 | 0.188 |
| real | random_unc | 149 | deduction | density | 4.160 | 4.072 | -0.088 | -0.316 | 0.146 | 0.467 |
| real | random_unc | 149 | adding-knowledge | density | 1.570 | 1.624 | 0.054 | -0.109 | 0.216 | 0.521 |
| real | random_unc | 149 | uncertainty-estimation | density | 0.721 | 0.761 | 0.040 | -0.111 | 0.168 | 0.588 |
| real | random_unc | 149 | backtracking | density | 0.279 | 0.244 | -0.035 | -0.100 | 0.028 | 0.306 |
| real | random_unc | 149 | deduction | density_testlex | 0.157 | 0.124 | -0.032 | -0.073 | 0.005 | 0.118 |
| real | random_unc | 149 | adding-knowledge | density_testlex | 0.016 | 0.010 | -0.006 | -0.022 | 0.010 | 0.470 |
| real | random_unc | 149 | uncertainty-estimation | density_testlex | 0.132 | 0.119 | -0.013 | -0.060 | 0.034 | 0.588 |
| real | random_unc | 149 | backtracking | density_testlex | 0.004 | 0.006 | 0.003 | -0.006 | 0.012 | 0.560 |
| real | random_unc | 149 | deduction | density_nontest | 4.003 | 3.948 | -0.055 | -0.285 | 0.183 | 0.651 |
| real | random_unc | 149 | adding-knowledge | density_nontest | 1.554 | 1.614 | 0.060 | -0.104 | 0.221 | 0.478 |
| real | random_unc | 149 | uncertainty-estimation | density_nontest | 0.589 | 0.641 | 0.053 | -0.096 | 0.178 | 0.473 |
| real | random_unc | 149 | backtracking | density_nontest | 0.276 | 0.238 | -0.038 | -0.101 | 0.025 | 0.271 |
| real | shuffled_bt | 149 | deduction | density | 4.160 | 3.523 | -0.638 | -0.844 | -0.417 | 0.000 |
| real | shuffled_bt | 149 | adding-knowledge | density | 1.570 | 1.262 | -0.308 | -0.460 | -0.157 | 0.000 |
| real | shuffled_bt | 149 | uncertainty-estimation | density | 0.721 | 1.056 | 0.335 | 0.187 | 0.474 | 0.000 |
| real | shuffled_bt | 149 | backtracking | density | 0.279 | 0.399 | 0.120 | 0.020 | 0.219 | 0.019 |
| real | shuffled_bt | 149 | deduction | density_testlex | 0.157 | 0.080 | -0.077 | -0.113 | -0.042 | 0.000 |
| real | shuffled_bt | 149 | adding-knowledge | density_testlex | 0.016 | 0.004 | -0.012 | -0.026 | 0.001 | 0.096 |
| real | shuffled_bt | 149 | uncertainty-estimation | density_testlex | 0.132 | 0.143 | 0.011 | -0.038 | 0.058 | 0.638 |
| real | shuffled_bt | 149 | backtracking | density_testlex | 0.004 | 0.002 | -0.001 | -0.008 | 0.005 | 0.673 |
| real | shuffled_bt | 149 | deduction | density_nontest | 4.003 | 3.443 | -0.561 | -0.772 | -0.332 | 0.000 |
| real | shuffled_bt | 149 | adding-knowledge | density_nontest | 1.554 | 1.257 | -0.296 | -0.448 | -0.145 | 0.000 |
| real | shuffled_bt | 149 | uncertainty-estimation | density_nontest | 0.589 | 0.913 | 0.324 | 0.177 | 0.459 | 0.000 |
| real | shuffled_bt | 149 | backtracking | density_nontest | 0.276 | 0.396 | 0.121 | 0.024 | 0.220 | 0.017 |
| real | shuffled_unc | 149 | deduction | density | 4.160 | 3.240 | -0.920 | -1.130 | -0.703 | 0.000 |
| real | shuffled_unc | 149 | adding-knowledge | density | 1.570 | 1.104 | -0.466 | -0.622 | -0.308 | 0.000 |
| real | shuffled_unc | 149 | uncertainty-estimation | density | 0.721 | 0.966 | 0.245 | 0.106 | 0.375 | 0.001 |
| real | shuffled_unc | 149 | backtracking | density | 0.279 | 0.231 | -0.048 | -0.139 | 0.054 | 0.326 |
| real | shuffled_unc | 149 | deduction | density_testlex | 0.157 | 0.128 | -0.029 | -0.082 | 0.027 | 0.305 |
| real | shuffled_unc | 149 | adding-knowledge | density_testlex | 0.016 | 0.013 | -0.003 | -0.020 | 0.013 | 0.698 |
| real | shuffled_unc | 149 | uncertainty-estimation | density_testlex | 0.132 | 0.185 | 0.053 | 0.005 | 0.103 | 0.038 |
| real | shuffled_unc | 149 | backtracking | density_testlex | 0.004 | 0.003 | -0.000 | -0.007 | 0.006 | 0.896 |
| real | shuffled_unc | 149 | deduction | density_nontest | 4.003 | 3.112 | -0.891 | -1.112 | -0.669 | 0.000 |
| real | shuffled_unc | 149 | adding-knowledge | density_nontest | 1.554 | 1.091 | -0.462 | -0.619 | -0.304 | 0.000 |
| real | shuffled_unc | 149 | uncertainty-estimation | density_nontest | 0.589 | 0.781 | 0.192 | 0.057 | 0.317 | 0.006 |
| real | shuffled_unc | 149 | backtracking | density_nontest | 0.276 | 0.228 | -0.047 | -0.139 | 0.054 | 0.329 |
| real | unc_neg | 149 | deduction | density | 4.160 | 4.320 | 0.160 | -0.091 | 0.417 | 0.236 |
| real | unc_neg | 149 | adding-knowledge | density | 1.570 | 1.628 | 0.059 | -0.133 | 0.245 | 0.546 |
| real | unc_neg | 149 | uncertainty-estimation | density | 0.721 | 0.417 | -0.304 | -0.446 | -0.170 | 0.000 |
| real | unc_neg | 149 | backtracking | density | 0.279 | 0.208 | -0.072 | -0.149 | 0.006 | 0.077 |
| real | unc_neg | 149 | deduction | density_testlex | 0.157 | 0.097 | -0.060 | -0.109 | -0.012 | 0.017 |
| real | unc_neg | 149 | adding-knowledge | density_testlex | 0.016 | 0.005 | -0.011 | -0.026 | 0.004 | 0.154 |
| real | unc_neg | 149 | uncertainty-estimation | density_testlex | 0.132 | 0.030 | -0.102 | -0.143 | -0.064 | 0.000 |
| real | unc_neg | 149 | backtracking | density_testlex | 0.004 | 0.000 | -0.004 | -0.009 | 0.000 | 0.158 |
| real | unc_neg | 149 | deduction | density_nontest | 4.003 | 4.223 | 0.220 | -0.038 | 0.483 | 0.115 |
| real | unc_neg | 149 | adding-knowledge | density_nontest | 1.554 | 1.623 | 0.070 | -0.123 | 0.256 | 0.476 |
| real | unc_neg | 149 | uncertainty-estimation | density_nontest | 0.589 | 0.387 | -0.202 | -0.338 | -0.069 | 0.005 |
| real | unc_neg | 149 | backtracking | density_nontest | 0.276 | 0.208 | -0.068 | -0.145 | 0.010 | 0.093 |
| real | unc_pos | 149 | deduction | density | 4.160 | 3.959 | -0.201 | -0.476 | 0.053 | 0.143 |
| real | unc_pos | 149 | adding-knowledge | density | 1.570 | 1.401 | -0.169 | -0.323 | -0.012 | 0.040 |
| real | unc_pos | 149 | uncertainty-estimation | density | 0.721 | 1.292 | 0.571 | 0.409 | 0.722 | 0.000 |
| real | unc_pos | 149 | backtracking | density | 0.279 | 0.218 | -0.061 | -0.137 | 0.008 | 0.111 |
| real | unc_pos | 149 | deduction | density_testlex | 0.157 | 0.146 | -0.011 | -0.058 | 0.035 | 0.660 |
| real | unc_pos | 149 | adding-knowledge | density_testlex | 0.016 | 0.027 | 0.011 | -0.008 | 0.031 | 0.297 |
| real | unc_pos | 149 | uncertainty-estimation | density_testlex | 0.132 | 0.246 | 0.114 | 0.056 | 0.174 | 0.000 |
| real | unc_pos | 149 | backtracking | density_testlex | 0.004 | 0.003 | -0.001 | -0.007 | 0.006 | 0.874 |
| real | unc_pos | 149 | deduction | density_nontest | 4.003 | 3.813 | -0.190 | -0.473 | 0.070 | 0.166 |
| real | unc_pos | 149 | adding-knowledge | density_nontest | 1.554 | 1.373 | -0.180 | -0.336 | -0.025 | 0.031 |
| real | unc_pos | 149 | uncertainty-estimation | density_nontest | 0.589 | 1.046 | 0.457 | 0.300 | 0.605 | 0.000 |
| real | unc_pos | 149 | backtracking | density_nontest | 0.276 | 0.215 | -0.060 | -0.135 | 0.009 | 0.111 |


## Morphology Δ by compliance class (decision-change confound): `refuse_both` / `comply_both` hold the decision fixed

| framing | arm | flip_class | n | behaviour | metric | delta_density | ci_lo | ci_hi |
|---|---|---|---|---|---|---|---|---|
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
| real | bt_neg | gained | 11 | deduction | density | -0.044 | -0.747 | 0.649 |
| real | bt_neg | gained | 11 | deduction | density_nontest | 0.261 | -0.512 | 0.990 |
| real | bt_neg | gained | 11 | deduction | density_testlex | -0.305 | -0.468 | -0.131 |
| real | bt_neg | gained | 11 | uncertainty-estimation | density | 0.176 | -0.252 | 0.560 |
| real | bt_neg | gained | 11 | uncertainty-estimation | density_nontest | 0.400 | 0.090 | 0.711 |
| real | bt_neg | gained | 11 | uncertainty-estimation | density_testlex | -0.224 | -0.379 | -0.093 |
| real | bt_neg | gained | 11 | backtracking | density | -0.203 | -0.480 | 0.054 |
| real | bt_neg | gained | 11 | backtracking | density_nontest | -0.203 | -0.480 | 0.054 |
| real | bt_neg | gained | 11 | backtracking | density_testlex | 0.000 | 0.000 | 0.000 |
| real | bt_neg | lost | 14 | deduction | density | 0.526 | -0.694 | 1.820 |
| real | bt_neg | lost | 14 | deduction | density_nontest | 0.186 | -1.153 | 1.615 |
| real | bt_neg | lost | 14 | deduction | density_testlex | 0.340 | 0.137 | 0.546 |
| real | bt_neg | lost | 14 | uncertainty-estimation | density | -0.983 | -1.529 | -0.458 |
| real | bt_neg | lost | 14 | uncertainty-estimation | density_nontest | -0.915 | -1.416 | -0.428 |
| real | bt_neg | lost | 14 | uncertainty-estimation | density_testlex | -0.067 | -0.131 | -0.012 |
| real | bt_neg | lost | 14 | backtracking | density | -0.085 | -0.205 | 0.021 |
| real | bt_neg | lost | 14 | backtracking | density_nontest | -0.085 | -0.205 | 0.021 |
| real | bt_neg | lost | 14 | backtracking | density_testlex | 0.000 | 0.000 | 0.000 |
| real | bt_neg | comply_both | 92 | deduction | density | 0.537 | 0.175 | 0.876 |
| real | bt_neg | comply_both | 92 | deduction | density_nontest | 0.528 | 0.155 | 0.860 |
| real | bt_neg | comply_both | 92 | deduction | density_testlex | 0.009 | -0.054 | 0.070 |
| real | bt_neg | comply_both | 92 | uncertainty-estimation | density | -0.339 | -0.473 | -0.210 |
| real | bt_neg | comply_both | 92 | uncertainty-estimation | density_nontest | -0.252 | -0.385 | -0.123 |
| real | bt_neg | comply_both | 92 | uncertainty-estimation | density_testlex | -0.087 | -0.132 | -0.040 |
| real | bt_neg | comply_both | 92 | backtracking | density | -0.150 | -0.242 | -0.062 |
| real | bt_neg | comply_both | 92 | backtracking | density_nontest | -0.144 | -0.236 | -0.057 |
| real | bt_neg | comply_both | 92 | backtracking | density_testlex | -0.006 | -0.015 | 0.000 |
| real | bt_neg | refuse_both | 29 | deduction | density | 0.577 | 0.141 | 1.000 |
| real | bt_neg | refuse_both | 29 | deduction | density_nontest | 0.276 | -0.188 | 0.730 |
| real | bt_neg | refuse_both | 29 | deduction | density_testlex | 0.301 | 0.123 | 0.483 |
| real | bt_neg | refuse_both | 29 | uncertainty-estimation | density | -0.411 | -0.708 | -0.145 |
| real | bt_neg | refuse_both | 29 | uncertainty-estimation | density_nontest | -0.285 | -0.525 | -0.080 |
| real | bt_neg | refuse_both | 29 | uncertainty-estimation | density_testlex | -0.126 | -0.261 | 0.008 |
| real | bt_neg | refuse_both | 29 | backtracking | density | -0.129 | -0.290 | 0.034 |
| real | bt_neg | refuse_both | 29 | backtracking | density_nontest | -0.129 | -0.290 | 0.034 |
| real | bt_neg | refuse_both | 29 | backtracking | density_testlex | 0.000 | 0.000 | 0.000 |
| real | bt_pos | lost | 5 | deduction | density | -0.525 | -1.256 | 0.232 |
| real | bt_pos | lost | 5 | deduction | density_nontest | -0.559 | -1.101 | -0.049 |
| real | bt_pos | lost | 5 | deduction | density_testlex | 0.034 | -0.452 | 0.414 |
| real | bt_pos | lost | 5 | uncertainty-estimation | density | 0.443 | -0.389 | 1.190 |
| real | bt_pos | lost | 5 | uncertainty-estimation | density_nontest | 0.281 | -0.629 | 1.269 |
| real | bt_pos | lost | 5 | uncertainty-estimation | density_testlex | 0.162 | -0.117 | 0.447 |
| real | bt_pos | lost | 5 | backtracking | density | -0.104 | -0.217 | 0.000 |
| real | bt_pos | lost | 5 | backtracking | density_nontest | -0.104 | -0.217 | 0.000 |
| real | bt_pos | lost | 5 | backtracking | density_testlex | 0.000 | 0.000 | 0.000 |
| real | bt_pos | comply_both | 59 | deduction | density | -0.779 | -1.196 | -0.371 |
| real | bt_pos | comply_both | 59 | deduction | density_nontest | -0.728 | -1.152 | -0.312 |
| real | bt_pos | comply_both | 59 | deduction | density_testlex | -0.051 | -0.105 | 0.001 |
| real | bt_pos | comply_both | 59 | uncertainty-estimation | density | 0.558 | 0.366 | 0.730 |
| real | bt_pos | comply_both | 59 | uncertainty-estimation | density_nontest | 0.459 | 0.270 | 0.640 |
| real | bt_pos | comply_both | 59 | uncertainty-estimation | density_testlex | 0.099 | 0.027 | 0.174 |
| real | bt_pos | comply_both | 59 | backtracking | density | 0.092 | -0.041 | 0.218 |
| real | bt_pos | comply_both | 59 | backtracking | density_nontest | 0.092 | -0.041 | 0.218 |
| real | bt_pos | comply_both | 59 | backtracking | density_testlex | -0.000 | -0.014 | 0.014 |
| real | bt_pos | refuse_both | 27 | deduction | density | -0.713 | -1.158 | -0.275 |
| real | bt_pos | refuse_both | 27 | deduction | density_nontest | -0.731 | -1.171 | -0.255 |
| real | bt_pos | refuse_both | 27 | deduction | density_testlex | 0.018 | -0.114 | 0.145 |
| real | bt_pos | refuse_both | 27 | uncertainty-estimation | density | 0.409 | 0.110 | 0.718 |
| real | bt_pos | refuse_both | 27 | uncertainty-estimation | density_nontest | 0.348 | 0.076 | 0.616 |
| real | bt_pos | refuse_both | 27 | uncertainty-estimation | density_testlex | 0.061 | -0.096 | 0.200 |
| real | bt_pos | refuse_both | 27 | backtracking | density | 0.005 | -0.171 | 0.181 |
| real | bt_pos | refuse_both | 27 | backtracking | density_nontest | -0.005 | -0.184 | 0.172 |
| real | bt_pos | refuse_both | 27 | backtracking | density_testlex | 0.011 | 0.000 | 0.033 |
| real | random_bt | gained | 15 | deduction | density | -0.185 | -0.849 | 0.471 |
| real | random_bt | gained | 15 | deduction | density_nontest | -0.064 | -0.764 | 0.630 |
| real | random_bt | gained | 15 | deduction | density_testlex | -0.121 | -0.259 | 0.000 |
| real | random_bt | gained | 15 | uncertainty-estimation | density | 0.150 | -0.338 | 0.668 |
| real | random_bt | gained | 15 | uncertainty-estimation | density_nontest | 0.318 | -0.161 | 0.772 |
| real | random_bt | gained | 15 | uncertainty-estimation | density_testlex | -0.169 | -0.294 | -0.044 |
| real | random_bt | gained | 15 | backtracking | density | -0.100 | -0.405 | 0.160 |
| real | random_bt | gained | 15 | backtracking | density_nontest | -0.134 | -0.432 | 0.125 |
| real | random_bt | gained | 15 | backtracking | density_testlex | 0.034 | 0.000 | 0.093 |
| real | random_bt | lost | 7 | deduction | density | 0.388 | -0.141 | 0.957 |
| real | random_bt | lost | 7 | deduction | density_nontest | 0.059 | -0.391 | 0.580 |
| real | random_bt | lost | 7 | deduction | density_testlex | 0.328 | 0.068 | 0.558 |
| real | random_bt | lost | 7 | uncertainty-estimation | density | -0.129 | -1.035 | 0.775 |
| real | random_bt | lost | 7 | uncertainty-estimation | density_nontest | -0.183 | -0.983 | 0.565 |
| real | random_bt | lost | 7 | uncertainty-estimation | density_testlex | 0.054 | -0.215 | 0.307 |
| real | random_bt | lost | 7 | backtracking | density | -0.222 | -0.453 | -0.059 |
| real | random_bt | lost | 7 | backtracking | density_nontest | -0.222 | -0.453 | -0.059 |
| real | random_bt | lost | 7 | backtracking | density_testlex | 0.000 | 0.000 | 0.000 |
| real | random_bt | comply_both | 97 | deduction | density | -0.071 | -0.359 | 0.233 |
| real | random_bt | comply_both | 97 | deduction | density_nontest | -0.025 | -0.323 | 0.273 |
| real | random_bt | comply_both | 97 | deduction | density_testlex | -0.046 | -0.093 | 0.003 |
| real | random_bt | comply_both | 97 | uncertainty-estimation | density | -0.027 | -0.186 | 0.121 |
| real | random_bt | comply_both | 97 | uncertainty-estimation | density_nontest | 0.009 | -0.161 | 0.172 |
| real | random_bt | comply_both | 97 | uncertainty-estimation | density_testlex | -0.036 | -0.082 | 0.006 |
| real | random_bt | comply_both | 97 | backtracking | density | -0.010 | -0.094 | 0.073 |
| real | random_bt | comply_both | 97 | backtracking | density_nontest | -0.004 | -0.086 | 0.078 |
| real | random_bt | comply_both | 97 | backtracking | density_testlex | -0.006 | -0.014 | 0.000 |
| real | random_bt | refuse_both | 25 | deduction | density | 0.161 | -0.397 | 0.832 |
| real | random_bt | refuse_both | 25 | deduction | density_nontest | 0.096 | -0.480 | 0.767 |
| real | random_bt | refuse_both | 25 | deduction | density_testlex | 0.064 | -0.081 | 0.218 |
| real | random_bt | refuse_both | 25 | uncertainty-estimation | density | 0.168 | -0.158 | 0.550 |
| real | random_bt | refuse_both | 25 | uncertainty-estimation | density_nontest | 0.239 | -0.074 | 0.584 |
| real | random_bt | refuse_both | 25 | uncertainty-estimation | density_testlex | -0.071 | -0.176 | 0.033 |
| real | random_bt | refuse_both | 25 | backtracking | density | -0.044 | -0.176 | 0.081 |
| real | random_bt | refuse_both | 25 | backtracking | density_nontest | -0.044 | -0.176 | 0.081 |
| real | random_bt | refuse_both | 25 | backtracking | density_testlex | 0.000 | 0.000 | 0.000 |
| real | random_unc | gained | 14 | deduction | density | 0.050 | -0.418 | 0.494 |
| real | random_unc | gained | 14 | deduction | density_nontest | 0.158 | -0.307 | 0.577 |
| real | random_unc | gained | 14 | deduction | density_testlex | -0.108 | -0.256 | 0.015 |
| real | random_unc | gained | 14 | uncertainty-estimation | density | 0.211 | -0.365 | 0.794 |
| real | random_unc | gained | 14 | uncertainty-estimation | density_nontest | 0.349 | -0.195 | 0.905 |
| real | random_unc | gained | 14 | uncertainty-estimation | density_testlex | -0.138 | -0.311 | 0.042 |
| real | random_unc | gained | 14 | backtracking | density | -0.014 | -0.198 | 0.183 |
| real | random_unc | gained | 14 | backtracking | density_nontest | -0.014 | -0.198 | 0.183 |
| real | random_unc | gained | 14 | backtracking | density_testlex | 0.000 | 0.000 | 0.000 |
| real | random_unc | comply_both | 101 | deduction | density | -0.095 | -0.397 | 0.206 |
| real | random_unc | comply_both | 101 | deduction | density_nontest | -0.083 | -0.382 | 0.228 |
| real | random_unc | comply_both | 101 | deduction | density_testlex | -0.013 | -0.052 | 0.033 |
| real | random_unc | comply_both | 101 | uncertainty-estimation | density | 0.036 | -0.119 | 0.180 |
| real | random_unc | comply_both | 101 | uncertainty-estimation | density_nontest | 0.015 | -0.133 | 0.146 |
| real | random_unc | comply_both | 101 | uncertainty-estimation | density_testlex | 0.021 | -0.035 | 0.076 |
| real | random_unc | comply_both | 101 | backtracking | density | -0.008 | -0.091 | 0.072 |
| real | random_unc | comply_both | 101 | backtracking | density_nontest | -0.008 | -0.091 | 0.073 |
| real | random_unc | comply_both | 101 | backtracking | density_testlex | 0.000 | -0.011 | 0.012 |
| real | random_unc | refuse_both | 26 | deduction | density | -0.148 | -0.648 | 0.356 |
| real | random_unc | refuse_both | 26 | deduction | density_nontest | -0.097 | -0.592 | 0.422 |
| real | random_unc | refuse_both | 26 | deduction | density_testlex | -0.051 | -0.163 | 0.058 |
| real | random_unc | refuse_both | 26 | uncertainty-estimation | density | 0.176 | -0.151 | 0.522 |
| real | random_unc | refuse_both | 26 | uncertainty-estimation | density_nontest | 0.245 | -0.049 | 0.548 |
| real | random_unc | refuse_both | 26 | uncertainty-estimation | density_testlex | -0.069 | -0.154 | 0.015 |
| real | random_unc | refuse_both | 26 | backtracking | density | -0.115 | -0.262 | 0.029 |
| real | random_unc | refuse_both | 26 | backtracking | density_nontest | -0.130 | -0.278 | 0.016 |
| real | random_unc | refuse_both | 26 | backtracking | density_testlex | 0.015 | 0.000 | 0.044 |
| real | shuffled_bt | comply_both | 70 | deduction | density | -0.742 | -1.071 | -0.421 |
| real | shuffled_bt | comply_both | 70 | deduction | density_nontest | -0.665 | -0.993 | -0.337 |
| real | shuffled_bt | comply_both | 70 | deduction | density_testlex | -0.077 | -0.126 | -0.029 |
| real | shuffled_bt | comply_both | 70 | uncertainty-estimation | density | 0.422 | 0.250 | 0.592 |
| real | shuffled_bt | comply_both | 70 | uncertainty-estimation | density_nontest | 0.377 | 0.194 | 0.554 |
| real | shuffled_bt | comply_both | 70 | uncertainty-estimation | density_testlex | 0.045 | -0.024 | 0.108 |
| real | shuffled_bt | comply_both | 70 | backtracking | density | 0.051 | -0.040 | 0.141 |
| real | shuffled_bt | comply_both | 70 | backtracking | density_nontest | 0.055 | -0.032 | 0.144 |
| real | shuffled_bt | comply_both | 70 | backtracking | density_testlex | -0.004 | -0.012 | 0.000 |
| real | shuffled_bt | refuse_both | 31 | deduction | density | -0.691 | -1.056 | -0.317 |
| real | shuffled_bt | refuse_both | 31 | deduction | density_nontest | -0.638 | -1.005 | -0.235 |
| real | shuffled_bt | refuse_both | 31 | deduction | density_testlex | -0.054 | -0.145 | 0.031 |
| real | shuffled_bt | refuse_both | 31 | uncertainty-estimation | density | 0.307 | 0.038 | 0.551 |
| real | shuffled_bt | refuse_both | 31 | uncertainty-estimation | density_nontest | 0.247 | 0.014 | 0.473 |
| real | shuffled_bt | refuse_both | 31 | uncertainty-estimation | density_testlex | 0.060 | -0.056 | 0.186 |
| real | shuffled_bt | refuse_both | 31 | backtracking | density | 0.090 | -0.097 | 0.278 |
| real | shuffled_bt | refuse_both | 31 | backtracking | density_nontest | 0.085 | -0.100 | 0.269 |
| real | shuffled_bt | refuse_both | 31 | backtracking | density_testlex | 0.006 | 0.000 | 0.017 |
| real | shuffled_unc | gained | 9 | deduction | density | -0.329 | -1.029 | 0.359 |
| real | shuffled_unc | gained | 9 | deduction | density_nontest | 0.003 | -0.622 | 0.641 |
| real | shuffled_unc | gained | 9 | deduction | density_testlex | -0.332 | -0.475 | -0.194 |
| real | shuffled_unc | gained | 9 | uncertainty-estimation | density | 1.063 | 0.550 | 1.643 |
| real | shuffled_unc | gained | 9 | uncertainty-estimation | density_nontest | 0.931 | 0.433 | 1.445 |
| real | shuffled_unc | gained | 9 | uncertainty-estimation | density_testlex | 0.132 | -0.021 | 0.333 |
| real | shuffled_unc | gained | 9 | backtracking | density | -0.059 | -0.232 | 0.095 |
| real | shuffled_unc | gained | 9 | backtracking | density_nontest | -0.059 | -0.232 | 0.095 |
| real | shuffled_unc | gained | 9 | backtracking | density_testlex | 0.000 | 0.000 | 0.000 |
| real | shuffled_unc | lost | 8 | deduction | density | -0.356 | -1.276 | 0.725 |
| real | shuffled_unc | lost | 8 | deduction | density_nontest | -0.491 | -1.218 | 0.283 |
| real | shuffled_unc | lost | 8 | deduction | density_testlex | 0.135 | -0.120 | 0.525 |
| real | shuffled_unc | lost | 8 | uncertainty-estimation | density | 0.446 | 0.049 | 0.826 |
| real | shuffled_unc | lost | 8 | uncertainty-estimation | density_nontest | 0.284 | 0.037 | 0.542 |
| real | shuffled_unc | lost | 8 | uncertainty-estimation | density_testlex | 0.163 | -0.041 | 0.339 |
| real | shuffled_unc | lost | 8 | backtracking | density | 0.055 | -0.208 | 0.267 |
| real | shuffled_unc | lost | 8 | backtracking | density_nontest | 0.055 | -0.208 | 0.267 |
| real | shuffled_unc | lost | 8 | backtracking | density_testlex | 0.000 | 0.000 | 0.000 |
| real | shuffled_unc | comply_both | 84 | deduction | density | -1.003 | -1.303 | -0.704 |
| real | shuffled_unc | comply_both | 84 | deduction | density_nontest | -0.966 | -1.277 | -0.658 |
| real | shuffled_unc | comply_both | 84 | deduction | density_testlex | -0.037 | -0.096 | 0.029 |
| real | shuffled_unc | comply_both | 84 | uncertainty-estimation | density | 0.139 | -0.022 | 0.292 |
| real | shuffled_unc | comply_both | 84 | uncertainty-estimation | density_nontest | 0.113 | -0.060 | 0.277 |
| real | shuffled_unc | comply_both | 84 | uncertainty-estimation | density_testlex | 0.026 | -0.030 | 0.083 |
| real | shuffled_unc | comply_both | 84 | backtracking | density | -0.074 | -0.170 | 0.018 |
| real | shuffled_unc | comply_both | 84 | backtracking | density_nontest | -0.073 | -0.169 | 0.018 |
| real | shuffled_unc | comply_both | 84 | backtracking | density_testlex | -0.002 | -0.014 | 0.009 |
| real | shuffled_unc | refuse_both | 25 | deduction | density | -0.785 | -1.258 | -0.332 |
| real | shuffled_unc | refuse_both | 25 | deduction | density_nontest | -0.896 | -1.392 | -0.400 |
| real | shuffled_unc | refuse_both | 25 | deduction | density_testlex | 0.111 | -0.070 | 0.288 |
| real | shuffled_unc | refuse_both | 25 | uncertainty-estimation | density | 0.276 | -0.048 | 0.604 |
| real | shuffled_unc | refuse_both | 25 | uncertainty-estimation | density_nontest | 0.095 | -0.143 | 0.337 |
| real | shuffled_unc | refuse_both | 25 | uncertainty-estimation | density_testlex | 0.181 | 0.022 | 0.372 |
| real | shuffled_unc | refuse_both | 25 | backtracking | density | -0.147 | -0.348 | 0.036 |
| real | shuffled_unc | refuse_both | 25 | backtracking | density_nontest | -0.147 | -0.348 | 0.036 |
| real | shuffled_unc | refuse_both | 25 | backtracking | density_testlex | 0.000 | 0.000 | 0.000 |
| real | unc_neg | gained | 16 | deduction | density | 0.355 | -0.348 | 1.055 |
| real | unc_neg | gained | 16 | deduction | density_nontest | 0.677 | -0.182 | 1.475 |
| real | unc_neg | gained | 16 | deduction | density_testlex | -0.322 | -0.544 | -0.087 |
| real | unc_neg | gained | 16 | uncertainty-estimation | density | 0.017 | -0.282 | 0.310 |
| real | unc_neg | gained | 16 | uncertainty-estimation | density_nontest | 0.174 | -0.110 | 0.456 |
| real | unc_neg | gained | 16 | uncertainty-estimation | density_testlex | -0.156 | -0.278 | -0.052 |
| real | unc_neg | gained | 16 | backtracking | density | -0.098 | -0.265 | 0.076 |
| real | unc_neg | gained | 16 | backtracking | density_nontest | -0.098 | -0.265 | 0.076 |
| real | unc_neg | gained | 16 | backtracking | density_testlex | 0.000 | 0.000 | 0.000 |
| real | unc_neg | lost | 8 | deduction | density | 0.172 | -0.534 | 0.915 |
| real | unc_neg | lost | 8 | deduction | density_nontest | 0.029 | -0.771 | 0.869 |
| real | unc_neg | lost | 8 | deduction | density_testlex | 0.143 | -0.018 | 0.336 |
| real | unc_neg | lost | 8 | uncertainty-estimation | density | -0.710 | -1.436 | 0.044 |
| real | unc_neg | lost | 8 | uncertainty-estimation | density_nontest | -0.754 | -1.427 | -0.083 |
| real | unc_neg | lost | 8 | uncertainty-estimation | density_testlex | 0.044 | -0.084 | 0.186 |
| real | unc_neg | lost | 8 | backtracking | density | 0.122 | -0.071 | 0.397 |
| real | unc_neg | lost | 8 | backtracking | density_nontest | 0.122 | -0.071 | 0.397 |
| real | unc_neg | lost | 8 | backtracking | density_testlex | 0.000 | 0.000 | 0.000 |
| real | unc_neg | comply_both | 96 | deduction | density | 0.225 | -0.120 | 0.558 |
| real | unc_neg | comply_both | 96 | deduction | density_nontest | 0.249 | -0.111 | 0.586 |
| real | unc_neg | comply_both | 96 | deduction | density_testlex | -0.024 | -0.075 | 0.029 |
| real | unc_neg | comply_both | 96 | uncertainty-estimation | density | -0.358 | -0.502 | -0.219 |
| real | unc_neg | comply_both | 96 | uncertainty-estimation | density_nontest | -0.263 | -0.405 | -0.125 |
| real | unc_neg | comply_both | 96 | uncertainty-estimation | density_testlex | -0.094 | -0.136 | -0.057 |
| real | unc_neg | comply_both | 96 | backtracking | density | -0.106 | -0.196 | -0.019 |
| real | unc_neg | comply_both | 96 | backtracking | density_nontest | -0.101 | -0.190 | -0.015 |
| real | unc_neg | comply_both | 96 | backtracking | density_testlex | -0.006 | -0.015 | 0.000 |
| real | unc_neg | refuse_both | 23 | deduction | density | -0.073 | -0.629 | 0.511 |
| real | unc_neg | refuse_both | 23 | deduction | density_nontest | 0.036 | -0.522 | 0.623 |
| real | unc_neg | refuse_both | 23 | deduction | density_testlex | -0.109 | -0.200 | -0.021 |
| real | unc_neg | refuse_both | 23 | uncertainty-estimation | density | -0.023 | -0.402 | 0.400 |
| real | unc_neg | refuse_both | 23 | uncertainty-estimation | density_nontest | 0.136 | -0.194 | 0.492 |
| real | unc_neg | refuse_both | 23 | uncertainty-estimation | density_testlex | -0.158 | -0.301 | -0.036 |
| real | unc_neg | refuse_both | 23 | backtracking | density | 0.017 | -0.192 | 0.215 |
| real | unc_neg | refuse_both | 23 | backtracking | density_nontest | 0.017 | -0.192 | 0.215 |
| real | unc_neg | refuse_both | 23 | backtracking | density_testlex | 0.000 | 0.000 | 0.000 |
| real | unc_pos | gained | 7 | deduction | density | -0.679 | -1.461 | 0.067 |
| real | unc_pos | gained | 7 | deduction | density_nontest | -0.597 | -1.423 | 0.106 |
| real | unc_pos | gained | 7 | deduction | density_testlex | -0.082 | -0.313 | 0.106 |
| real | unc_pos | gained | 7 | uncertainty-estimation | density | 1.244 | 0.830 | 1.599 |
| real | unc_pos | gained | 7 | uncertainty-estimation | density_nontest | 1.084 | 0.557 | 1.581 |
| real | unc_pos | gained | 7 | uncertainty-estimation | density_testlex | 0.160 | -0.215 | 0.614 |
| real | unc_pos | gained | 7 | backtracking | density | -0.277 | -0.652 | 0.045 |
| real | unc_pos | gained | 7 | backtracking | density_nontest | -0.277 | -0.652 | 0.045 |
| real | unc_pos | gained | 7 | backtracking | density_testlex | 0.000 | 0.000 | 0.000 |
| real | unc_pos | lost | 10 | deduction | density | -0.232 | -1.146 | 0.728 |
| real | unc_pos | lost | 10 | deduction | density_nontest | -0.316 | -1.264 | 0.719 |
| real | unc_pos | lost | 10 | deduction | density_testlex | 0.084 | -0.151 | 0.284 |
| real | unc_pos | lost | 10 | uncertainty-estimation | density | 0.373 | -0.263 | 0.975 |
| real | unc_pos | lost | 10 | uncertainty-estimation | density_nontest | 0.125 | -0.571 | 0.751 |
| real | unc_pos | lost | 10 | uncertainty-estimation | density_testlex | 0.248 | 0.032 | 0.447 |
| real | unc_pos | lost | 10 | backtracking | density | -0.060 | -0.187 | 0.054 |
| real | unc_pos | lost | 10 | backtracking | density_nontest | -0.060 | -0.187 | 0.054 |
| real | unc_pos | lost | 10 | backtracking | density_testlex | 0.000 | 0.000 | 0.000 |
| real | unc_pos | comply_both | 86 | deduction | density | -0.234 | -0.620 | 0.155 |
| real | unc_pos | comply_both | 86 | deduction | density_nontest | -0.188 | -0.576 | 0.201 |
| real | unc_pos | comply_both | 86 | deduction | density_testlex | -0.045 | -0.092 | 0.002 |
| real | unc_pos | comply_both | 86 | uncertainty-estimation | density | 0.547 | 0.364 | 0.740 |
| real | unc_pos | comply_both | 86 | uncertainty-estimation | density_nontest | 0.468 | 0.280 | 0.669 |
| real | unc_pos | comply_both | 86 | uncertainty-estimation | density_testlex | 0.079 | 0.015 | 0.147 |
| real | unc_pos | comply_both | 86 | backtracking | density | -0.025 | -0.126 | 0.063 |
| real | unc_pos | comply_both | 86 | backtracking | density_nontest | -0.022 | -0.125 | 0.066 |
| real | unc_pos | comply_both | 86 | backtracking | density_testlex | -0.003 | -0.016 | 0.008 |
| real | unc_pos | refuse_both | 33 | deduction | density | -0.001 | -0.392 | 0.408 |
| real | unc_pos | refuse_both | 33 | deduction | density_nontest | -0.080 | -0.476 | 0.340 |
| real | unc_pos | refuse_both | 33 | deduction | density_testlex | 0.079 | -0.066 | 0.227 |
| real | unc_pos | refuse_both | 33 | uncertainty-estimation | density | 0.602 | 0.306 | 0.876 |
| real | unc_pos | refuse_both | 33 | uncertainty-estimation | density_nontest | 0.387 | 0.181 | 0.578 |
| real | unc_pos | refuse_both | 33 | uncertainty-estimation | density_testlex | 0.215 | 0.066 | 0.364 |
| real | unc_pos | refuse_both | 33 | backtracking | density | -0.106 | -0.244 | 0.031 |
| real | unc_pos | refuse_both | 33 | backtracking | density_nontest | -0.106 | -0.244 | 0.031 |
| real | unc_pos | refuse_both | 33 | backtracking | density_testlex | 0.000 | 0.000 | 0.000 |


## Representation-level morphology: Δ mean projection onto the rebuilt directions (steered model's own activations)

| arm | framing | projection | n_pairs | base_mean | delta | ci_lo | ci_hi |
|---|---|---|---|---|---|---|---|
| baseline_seed1 | real | proj_backtracking | 149 | 20.3383 | 0.0840 | -0.3104 | 0.4595 |
| baseline_seed1 | real | proj_uncertainty-estimation | 149 | 8.7282 | -0.1679 | -0.4809 | 0.1340 |
| baseline_seed1 | real | proj_probe | 149 | -2.4006 | 0.0435 | -0.1405 | 0.2261 |
| baseline_seed1 | real | proj_lastprompt_backtracking | 149 | 11.9075 | 0.0000 | 0.0000 | 0.0000 |
| baseline_seed1 | real | proj_lastprompt_uncertainty-estimation | 149 | 9.7791 | 0.0000 | 0.0000 | 0.0000 |
| baseline_seed1 | real | proj_lastprompt_probe | 149 | -1.9825 | 0.0000 | 0.0000 | 0.0000 |
| baseline_seed1 | real | proj_incontext_backtracking | 149 | 18.8387 | 0.0492 | -0.3461 | 0.4324 |
| baseline_seed1 | real | proj_incontext_uncertainty-estimation | 149 | 8.5692 | -0.3027 | -0.7517 | 0.1046 |
| baseline_seed1 | real | proj_incontext_probe | 149 | -2.9817 | 0.0752 | -0.0959 | 0.2403 |
| baseline_seed1 | fake | proj_backtracking | 149 | 19.4140 | 0.1414 | -0.2317 | 0.4960 |
| baseline_seed1 | fake | proj_uncertainty-estimation | 149 | 8.3488 | 0.0931 | -0.2649 | 0.4594 |
| baseline_seed1 | fake | proj_probe | 149 | -2.7419 | -0.0196 | -0.2233 | 0.1764 |
| baseline_seed1 | fake | proj_lastprompt_backtracking | 149 | 11.7580 | 0.0000 | 0.0000 | 0.0000 |
| baseline_seed1 | fake | proj_lastprompt_uncertainty-estimation | 149 | 9.6792 | 0.0000 | 0.0000 | 0.0000 |
| baseline_seed1 | fake | proj_lastprompt_probe | 149 | -2.0660 | 0.0000 | 0.0000 | 0.0000 |
| baseline_seed1 | fake | proj_incontext_backtracking | 149 | 17.9669 | 0.1430 | -0.2352 | 0.4983 |
| baseline_seed1 | fake | proj_incontext_uncertainty-estimation | 149 | 7.7193 | 0.1819 | -0.2166 | 0.6130 |
| baseline_seed1 | fake | proj_incontext_probe | 149 | -3.2032 | -0.0272 | -0.2094 | 0.1516 |
| bt_neg | real | proj_backtracking | 0 | 20.3383 | — | — | — |
| bt_neg | real | proj_uncertainty-estimation | 0 | 8.7282 | — | — | — |
| bt_neg | real | proj_probe | 149 | -2.4006 | -0.8945 | -1.0828 | -0.7179 |
| bt_neg | real | proj_lastprompt_backtracking | 149 | 11.9075 | -18.2858 | -18.2869 | -18.2849 |
| bt_neg | real | proj_lastprompt_uncertainty-estimation | 149 | 9.7791 | 0.0000 | 0.0000 | 0.0000 |
| bt_neg | real | proj_lastprompt_probe | 149 | -1.9825 | 0.0000 | 0.0000 | 0.0000 |
| bt_neg | real | proj_incontext_backtracking | 149 | 18.8387 | -20.1982 | -20.6182 | -19.7816 |
| bt_neg | real | proj_incontext_uncertainty-estimation | 149 | 8.5692 | -1.2246 | -1.6265 | -0.8377 |
| bt_neg | real | proj_incontext_probe | 149 | -2.9817 | -0.6625 | -0.8351 | -0.4949 |
| bt_neg | fake | proj_backtracking | 0 | — | — | — | — |
| bt_neg | fake | proj_uncertainty-estimation | 0 | — | — | — | — |
| bt_neg | fake | proj_probe | 0 | — | — | — | — |
| bt_neg | fake | proj_lastprompt_backtracking | 0 | — | — | — | — |
| bt_neg | fake | proj_lastprompt_uncertainty-estimation | 0 | — | — | — | — |
| bt_neg | fake | proj_lastprompt_probe | 0 | — | — | — | — |
| bt_neg | fake | proj_incontext_backtracking | 0 | — | — | — | — |
| bt_neg | fake | proj_incontext_uncertainty-estimation | 0 | — | — | — | — |
| bt_neg | fake | proj_incontext_probe | 0 | — | — | — | — |
| bt_pos | real | proj_backtracking | 0 | 20.3383 | — | — | — |
| bt_pos | real | proj_uncertainty-estimation | 0 | 8.7282 | — | — | — |
| bt_pos | real | proj_probe | 149 | -2.4006 | 1.3888 | 1.1714 | 1.6151 |
| bt_pos | real | proj_lastprompt_backtracking | 149 | 11.9075 | 18.2675 | 18.2651 | 18.2701 |
| bt_pos | real | proj_lastprompt_uncertainty-estimation | 149 | 9.7791 | 0.0000 | 0.0000 | 0.0000 |
| bt_pos | real | proj_lastprompt_probe | 149 | -1.9825 | 0.0000 | 0.0000 | 0.0000 |
| bt_pos | real | proj_incontext_backtracking | 149 | 18.8387 | 21.1344 | 20.7021 | 21.5678 |
| bt_pos | real | proj_incontext_uncertainty-estimation | 149 | 8.5692 | -0.2447 | -0.6854 | 0.1848 |
| bt_pos | real | proj_incontext_probe | 149 | -2.9817 | 1.3009 | 1.1144 | 1.4913 |
| bt_pos | fake | proj_backtracking | 0 | — | — | — | — |
| bt_pos | fake | proj_uncertainty-estimation | 0 | — | — | — | — |
| bt_pos | fake | proj_probe | 0 | — | — | — | — |
| bt_pos | fake | proj_lastprompt_backtracking | 0 | — | — | — | — |
| bt_pos | fake | proj_lastprompt_uncertainty-estimation | 0 | — | — | — | — |
| bt_pos | fake | proj_lastprompt_probe | 0 | — | — | — | — |
| bt_pos | fake | proj_incontext_backtracking | 0 | — | — | — | — |
| bt_pos | fake | proj_incontext_uncertainty-estimation | 0 | — | — | — | — |
| bt_pos | fake | proj_incontext_probe | 0 | — | — | — | — |
| random_bt | real | proj_backtracking | 0 | 20.3383 | — | — | — |
| random_bt | real | proj_uncertainty-estimation | 0 | 8.7282 | — | — | — |
| random_bt | real | proj_probe | 149 | -2.4006 | -0.1129 | -0.3263 | 0.0847 |
| random_bt | real | proj_lastprompt_backtracking | 149 | 11.9075 | -0.4130 | -0.4138 | -0.4123 |
| random_bt | real | proj_lastprompt_uncertainty-estimation | 149 | 9.7791 | 0.0000 | 0.0000 | 0.0000 |
| random_bt | real | proj_lastprompt_probe | 149 | -1.9825 | 0.0000 | 0.0000 | 0.0000 |
| random_bt | real | proj_incontext_backtracking | 149 | 18.8387 | -0.6042 | -1.0623 | -0.1922 |
| random_bt | real | proj_incontext_uncertainty-estimation | 149 | 8.5692 | -0.6765 | -1.0655 | -0.3098 |
| random_bt | real | proj_incontext_probe | 149 | -2.9817 | 0.0538 | -0.1457 | 0.2312 |
| random_bt | fake | proj_backtracking | 0 | — | — | — | — |
| random_bt | fake | proj_uncertainty-estimation | 0 | — | — | — | — |
| random_bt | fake | proj_probe | 0 | — | — | — | — |
| random_bt | fake | proj_lastprompt_backtracking | 0 | — | — | — | — |
| random_bt | fake | proj_lastprompt_uncertainty-estimation | 0 | — | — | — | — |
| random_bt | fake | proj_lastprompt_probe | 0 | — | — | — | — |
| random_bt | fake | proj_incontext_backtracking | 0 | — | — | — | — |
| random_bt | fake | proj_incontext_uncertainty-estimation | 0 | — | — | — | — |
| random_bt | fake | proj_incontext_probe | 0 | — | — | — | — |
| random_unc | real | proj_backtracking | 0 | 20.3383 | — | — | — |
| random_unc | real | proj_uncertainty-estimation | 0 | 8.7282 | — | — | — |
| random_unc | real | proj_probe | 149 | -2.4006 | -0.0012 | -0.2320 | 0.2181 |
| random_unc | real | proj_lastprompt_backtracking | 149 | 11.9075 | -0.6820 | -0.7137 | -0.6493 |
| random_unc | real | proj_lastprompt_uncertainty-estimation | 149 | 9.7791 | -0.1146 | -0.1158 | -0.1133 |
| random_unc | real | proj_lastprompt_probe | 149 | -1.9825 | -0.0932 | -0.0999 | -0.0867 |
| random_unc | real | proj_incontext_backtracking | 149 | 18.8387 | -0.3494 | -0.7798 | 0.0666 |
| random_unc | real | proj_incontext_uncertainty-estimation | 149 | 8.5692 | -0.6320 | -1.0338 | -0.2431 |
| random_unc | real | proj_incontext_probe | 149 | -2.9817 | -0.0890 | -0.2792 | 0.0993 |
| random_unc | fake | proj_backtracking | 0 | — | — | — | — |
| random_unc | fake | proj_uncertainty-estimation | 0 | — | — | — | — |
| random_unc | fake | proj_probe | 0 | — | — | — | — |
| random_unc | fake | proj_lastprompt_backtracking | 0 | — | — | — | — |
| random_unc | fake | proj_lastprompt_uncertainty-estimation | 0 | — | — | — | — |
| random_unc | fake | proj_lastprompt_probe | 0 | — | — | — | — |
| random_unc | fake | proj_incontext_backtracking | 0 | — | — | — | — |
| random_unc | fake | proj_incontext_uncertainty-estimation | 0 | — | — | — | — |
| random_unc | fake | proj_incontext_probe | 0 | — | — | — | — |
| shuffled_bt | real | proj_backtracking | 0 | 20.3383 | — | — | — |
| shuffled_bt | real | proj_uncertainty-estimation | 0 | 8.7282 | — | — | — |
| shuffled_bt | real | proj_probe | 149 | -2.4006 | 1.0616 | 0.8456 | 1.2687 |
| shuffled_bt | real | proj_lastprompt_backtracking | 149 | 11.9075 | 16.4233 | 16.4206 | 16.4259 |
| shuffled_bt | real | proj_lastprompt_uncertainty-estimation | 149 | 9.7791 | 0.0000 | 0.0000 | 0.0000 |
| shuffled_bt | real | proj_lastprompt_probe | 149 | -1.9825 | 0.0000 | 0.0000 | 0.0000 |
| shuffled_bt | real | proj_incontext_backtracking | 149 | 18.8387 | 17.9904 | 17.5303 | 18.4180 |
| shuffled_bt | real | proj_incontext_uncertainty-estimation | 149 | 8.5692 | -0.5863 | -1.0410 | -0.1458 |
| shuffled_bt | real | proj_incontext_probe | 149 | -2.9817 | 0.9775 | 0.7765 | 1.1753 |
| shuffled_bt | fake | proj_backtracking | 0 | — | — | — | — |
| shuffled_bt | fake | proj_uncertainty-estimation | 0 | — | — | — | — |
| shuffled_bt | fake | proj_probe | 0 | — | — | — | — |
| shuffled_bt | fake | proj_lastprompt_backtracking | 0 | — | — | — | — |
| shuffled_bt | fake | proj_lastprompt_uncertainty-estimation | 0 | — | — | — | — |
| shuffled_bt | fake | proj_lastprompt_probe | 0 | — | — | — | — |
| shuffled_bt | fake | proj_incontext_backtracking | 0 | — | — | — | — |
| shuffled_bt | fake | proj_incontext_uncertainty-estimation | 0 | — | — | — | — |
| shuffled_bt | fake | proj_incontext_probe | 0 | — | — | — | — |
| shuffled_unc | real | proj_backtracking | 0 | 20.3383 | — | — | — |
| shuffled_unc | real | proj_uncertainty-estimation | 0 | 8.7282 | — | — | — |
| shuffled_unc | real | proj_probe | 149 | -2.4006 | 0.0932 | -0.1062 | 0.2870 |
| shuffled_unc | real | proj_lastprompt_backtracking | 149 | 11.9075 | 8.9474 | 8.9018 | 8.9934 |
| shuffled_unc | real | proj_lastprompt_uncertainty-estimation | 149 | 9.7791 | 3.5322 | 3.5309 | 3.5332 |
| shuffled_unc | real | proj_lastprompt_probe | 149 | -1.9825 | 1.3275 | 1.3183 | 1.3375 |
| shuffled_unc | real | proj_incontext_backtracking | 149 | 18.8387 | 7.5677 | 7.1092 | 8.0149 |
| shuffled_unc | real | proj_incontext_uncertainty-estimation | 149 | 8.5692 | 2.6348 | 2.2268 | 3.0420 |
| shuffled_unc | real | proj_incontext_probe | 149 | -2.9817 | 1.4734 | 1.2851 | 1.6590 |
| shuffled_unc | fake | proj_backtracking | 0 | — | — | — | — |
| shuffled_unc | fake | proj_uncertainty-estimation | 0 | — | — | — | — |
| shuffled_unc | fake | proj_probe | 0 | — | — | — | — |
| shuffled_unc | fake | proj_lastprompt_backtracking | 0 | — | — | — | — |
| shuffled_unc | fake | proj_lastprompt_uncertainty-estimation | 0 | — | — | — | — |
| shuffled_unc | fake | proj_lastprompt_probe | 0 | — | — | — | — |
| shuffled_unc | fake | proj_incontext_backtracking | 0 | — | — | — | — |
| shuffled_unc | fake | proj_incontext_uncertainty-estimation | 0 | — | — | — | — |
| shuffled_unc | fake | proj_incontext_probe | 0 | — | — | — | — |
| unc_neg | real | proj_backtracking | 0 | 20.3383 | — | — | — |
| unc_neg | real | proj_uncertainty-estimation | 0 | 8.7282 | — | — | — |
| unc_neg | real | proj_probe | 149 | -2.4006 | -0.5946 | -0.8138 | -0.3793 |
| unc_neg | real | proj_lastprompt_backtracking | 149 | 11.9075 | -4.9205 | -4.9877 | -4.8521 |
| unc_neg | real | proj_lastprompt_uncertainty-estimation | 149 | 9.7791 | -10.1403 | -10.1415 | -10.1392 |
| unc_neg | real | proj_lastprompt_probe | 149 | -1.9825 | -0.3083 | -0.3249 | -0.2928 |
| unc_neg | real | proj_incontext_backtracking | 149 | 18.8387 | -7.9556 | -8.4172 | -7.5033 |
| unc_neg | real | proj_incontext_uncertainty-estimation | 149 | 8.5692 | -11.4990 | -11.9411 | -11.0716 |
| unc_neg | real | proj_incontext_probe | 149 | -2.9817 | -0.8615 | -1.0458 | -0.6754 |
| unc_neg | fake | proj_backtracking | 0 | — | — | — | — |
| unc_neg | fake | proj_uncertainty-estimation | 0 | — | — | — | — |
| unc_neg | fake | proj_probe | 0 | — | — | — | — |
| unc_neg | fake | proj_lastprompt_backtracking | 0 | — | — | — | — |
| unc_neg | fake | proj_lastprompt_uncertainty-estimation | 0 | — | — | — | — |
| unc_neg | fake | proj_lastprompt_probe | 0 | — | — | — | — |
| unc_neg | fake | proj_incontext_backtracking | 0 | — | — | — | — |
| unc_neg | fake | proj_incontext_uncertainty-estimation | 0 | — | — | — | — |
| unc_neg | fake | proj_incontext_probe | 0 | — | — | — | — |
| unc_pos | real | proj_backtracking | 0 | 20.3383 | — | — | — |
| unc_pos | real | proj_uncertainty-estimation | 0 | 8.7282 | — | — | — |
| unc_pos | real | proj_probe | 149 | -2.4006 | 0.5483 | 0.3359 | 0.7512 |
| unc_pos | real | proj_lastprompt_backtracking | 149 | 11.9075 | 8.5304 | 8.4946 | 8.5660 |
| unc_pos | real | proj_lastprompt_uncertainty-estimation | 149 | 9.7791 | 10.1402 | 10.1390 | 10.1415 |
| unc_pos | real | proj_lastprompt_probe | 149 | -1.9825 | 0.2358 | 0.2257 | 0.2461 |
| unc_pos | real | proj_incontext_backtracking | 149 | 18.8387 | 9.2524 | 8.7937 | 9.7017 |
| unc_pos | real | proj_incontext_uncertainty-estimation | 149 | 8.5692 | 10.3405 | 9.9208 | 10.7436 |
| unc_pos | real | proj_incontext_probe | 149 | -2.9817 | 0.7990 | 0.6115 | 0.9776 |
| unc_pos | fake | proj_backtracking | 0 | — | — | — | — |
| unc_pos | fake | proj_uncertainty-estimation | 0 | — | — | — | — |
| unc_pos | fake | proj_probe | 0 | — | — | — | — |
| unc_pos | fake | proj_lastprompt_backtracking | 0 | — | — | — | — |
| unc_pos | fake | proj_lastprompt_uncertainty-estimation | 0 | — | — | — | — |
| unc_pos | fake | proj_lastprompt_probe | 0 | — | — | — | — |
| unc_pos | fake | proj_incontext_backtracking | 0 | — | — | — | — |
| unc_pos | fake | proj_incontext_uncertainty-estimation | 0 | — | — | — | — |
| unc_pos | fake | proj_incontext_probe | 0 | — | — | — | — |


## Awareness-probe score of each arm's reasoning (unedited model, the probe's own input regime and hidden-state index): per arm, and verbalizing vs non-verbalizing traces within the arm

| arm | framing | n | mean | ci_lo | ci_hi | n_verbalizing | mean_verbalizing | mean_not | p_welch |
|---|---|---|---|---|---|---|---|---|---|
| baseline | real | 149 | -2.4006 | -2.5992 | -2.1838 | 28 | -1.1496 | -2.6900 | 0.0000 |
| baseline | fake | 149 | -2.7419 | -2.9424 | -2.5154 | 13 | -1.2038 | -2.8878 | 0.0000 |
| baseline_seed1 | real | 149 | -2.3571 | -2.5585 | -2.1620 | 22 | -1.4930 | -2.5068 | 0.0004 |
| baseline_seed1 | fake | 149 | -2.7615 | -2.9646 | -2.5511 | 9 | -0.5367 | -2.9046 | 0.0010 |
| bt_neg | real | 149 | -3.2950 | -3.4503 | -3.1295 | 2 | -2.5390 | -3.3053 | 0.3947 |
| bt_neg | fake | 0 | — | — | — | 0 | — | — | — |
| bt_pos | real | 149 | -1.0118 | -1.2417 | -0.7847 | 65 | -0.4032 | -1.4827 | 0.0000 |
| bt_pos | fake | 0 | — | — | — | 0 | — | — | — |
| random_bt | real | 149 | -2.5135 | -2.7112 | -2.3245 | 27 | -1.4335 | -2.7591 | 0.0000 |
| random_bt | fake | 0 | — | — | — | 0 | — | — | — |
| random_unc | real | 149 | -2.4018 | -2.5934 | -2.2102 | 30 | -1.5062 | -2.6267 | 0.0000 |
| random_unc | fake | 0 | — | — | — | 0 | — | — | — |
| shuffled_bt | real | 149 | -1.3390 | -1.5634 | -1.1297 | 44 | -0.7287 | -1.5947 | 0.0001 |
| shuffled_bt | fake | 0 | — | — | — | 0 | — | — | — |
| shuffled_unc | real | 149 | -2.3074 | -2.5237 | -2.0924 | 41 | -1.6319 | -2.5638 | 0.0001 |
| shuffled_unc | fake | 0 | — | — | — | 0 | — | — | — |
| unc_neg | real | 149 | -2.9952 | -3.1833 | -2.8039 | 1 | -2.1406 | -3.0001 | — |
| unc_neg | fake | 0 | — | — | — | 0 | — | — | — |
| unc_pos | real | 149 | -1.8523 | -2.0877 | -1.6208 | 55 | -1.0191 | -2.3397 | 0.0000 |
| unc_pos | fake | 0 | — | — | — | 0 | — | — | — |

