# Mechanistic arm (Qwen3-4B) — first-pass analysis: f4_prompt

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


## f4_prompt: 1192 traces, 1192 with usable morphology annotation

| arm | framing | n | exec_rate_closed | exec_rate_all | lost | gained | p_mcnemar | task_judged_yes | judge_vs_string_agree | verbalization | closure | repetition | words |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | real | 149 | 0.726 | 0.711 | — | — | — | 0.826 | 0.582 | 0.188 | 0.980 | 0.046 | 334.470 |
| baseline | fake | 149 | 0.882 | 0.852 | — | — | — | 0.851 | 0.762 | 0.088 | 0.966 | 0.047 | 345.228 |
| baseline_seed1 | real | 149 | 0.748 | 0.718 | 3.000 | 6.000 | 0.508 | 0.738 | 0.545 | 0.148 | 0.960 | 0.051 | 353.859 |
| baseline_seed1 | fake | 149 | 0.890 | 0.872 | 5.000 | 6.000 | 1.000 | 0.852 | 0.760 | 0.060 | 0.980 | 0.047 | 331.664 |
| ask_bt_neg | real | 149 | 0.773 | 0.732 | 6.000 | 13.000 | 0.167 | 0.769 | 0.597 | 0.136 | 0.946 | 0.045 | 354.651 |
| ask_bt_neg | fake | 0 | — | — | 0.000 | 0.000 | — | — | — | — | — | — | — |
| ask_bt_pos | real | 149 | 0.769 | 0.758 | 3.000 | 9.000 | 0.146 | 0.792 | 0.585 | 0.168 | 0.987 | 0.060 | 392.611 |
| ask_bt_pos | fake | 0 | — | — | 0.000 | 0.000 | — | — | — | — | — | — | — |
| ask_unc_neg | real | 149 | 0.789 | 0.779 | 4.000 | 13.000 | 0.049 | 0.818 | 0.616 | 0.095 | 0.987 | 0.040 | 335.960 |
| ask_unc_neg | fake | 0 | — | — | 0.000 | 0.000 | — | — | — | — | — | — | — |
| ask_unc_pos | real | 149 | 0.735 | 0.725 | 9.000 | 10.000 | 1.000 | 0.832 | 0.592 | 0.154 | 0.987 | 0.052 | 364.544 |
| ask_unc_pos | fake | 0 | — | — | 0.000 | 0.000 | — | — | — | — | — | — | — |


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
| real | ask_bt_neg | 149 | deduction | density | 4.160 | 4.035 | -0.125 | -0.372 | 0.108 | 0.318 |
| real | ask_bt_neg | 149 | adding-knowledge | density | 1.570 | 1.579 | 0.010 | -0.137 | 0.154 | 0.900 |
| real | ask_bt_neg | 149 | uncertainty-estimation | density | 0.721 | 0.686 | -0.035 | -0.189 | 0.099 | 0.644 |
| real | ask_bt_neg | 149 | backtracking | density | 0.279 | 0.277 | -0.002 | -0.081 | 0.074 | 0.956 |
| real | ask_bt_neg | 149 | deduction | density_testlex | 0.157 | 0.155 | -0.002 | -0.046 | 0.042 | 0.936 |
| real | ask_bt_neg | 149 | adding-knowledge | density_testlex | 0.016 | 0.010 | -0.006 | -0.021 | 0.010 | 0.468 |
| real | ask_bt_neg | 149 | uncertainty-estimation | density_testlex | 0.132 | 0.092 | -0.040 | -0.082 | 0.001 | 0.060 |
| real | ask_bt_neg | 149 | backtracking | density_testlex | 0.004 | 0.004 | 0.000 | -0.007 | 0.008 | 0.905 |
| real | ask_bt_neg | 149 | deduction | density_nontest | 4.003 | 3.880 | -0.123 | -0.366 | 0.121 | 0.337 |
| real | ask_bt_neg | 149 | adding-knowledge | density_nontest | 1.554 | 1.569 | 0.016 | -0.131 | 0.159 | 0.838 |
| real | ask_bt_neg | 149 | uncertainty-estimation | density_nontest | 0.589 | 0.594 | 0.005 | -0.150 | 0.141 | 0.944 |
| real | ask_bt_neg | 149 | backtracking | density_nontest | 0.276 | 0.273 | -0.003 | -0.083 | 0.074 | 0.946 |
| real | ask_bt_pos | 149 | deduction | density | 4.160 | 3.884 | -0.276 | -0.499 | -0.046 | 0.021 |
| real | ask_bt_pos | 149 | adding-knowledge | density | 1.570 | 1.457 | -0.113 | -0.271 | 0.044 | 0.174 |
| real | ask_bt_pos | 149 | uncertainty-estimation | density | 0.721 | 0.796 | 0.075 | -0.054 | 0.194 | 0.239 |
| real | ask_bt_pos | 149 | backtracking | density | 0.279 | 0.291 | 0.012 | -0.069 | 0.086 | 0.752 |
| real | ask_bt_pos | 149 | deduction | density_testlex | 0.157 | 0.154 | -0.003 | -0.047 | 0.039 | 0.901 |
| real | ask_bt_pos | 149 | adding-knowledge | density_testlex | 0.016 | 0.008 | -0.008 | -0.022 | 0.006 | 0.278 |
| real | ask_bt_pos | 149 | uncertainty-estimation | density_testlex | 0.132 | 0.147 | 0.014 | -0.032 | 0.062 | 0.545 |
| real | ask_bt_pos | 149 | backtracking | density_testlex | 0.004 | 0.006 | 0.002 | -0.006 | 0.010 | 0.629 |
| real | ask_bt_pos | 149 | deduction | density_nontest | 4.003 | 3.730 | -0.273 | -0.506 | -0.040 | 0.023 |
| real | ask_bt_pos | 149 | adding-knowledge | density_nontest | 1.554 | 1.449 | -0.105 | -0.264 | 0.052 | 0.199 |
| real | ask_bt_pos | 149 | uncertainty-estimation | density_nontest | 0.589 | 0.649 | 0.060 | -0.065 | 0.182 | 0.336 |
| real | ask_bt_pos | 149 | backtracking | density_nontest | 0.276 | 0.286 | 0.010 | -0.069 | 0.084 | 0.789 |
| real | ask_unc_neg | 149 | deduction | density | 4.160 | 4.186 | 0.026 | -0.188 | 0.233 | 0.812 |
| real | ask_unc_neg | 149 | adding-knowledge | density | 1.570 | 1.613 | 0.043 | -0.132 | 0.215 | 0.629 |
| real | ask_unc_neg | 149 | uncertainty-estimation | density | 0.721 | 0.653 | -0.068 | -0.199 | 0.054 | 0.300 |
| real | ask_unc_neg | 149 | backtracking | density | 0.279 | 0.269 | -0.010 | -0.093 | 0.070 | 0.799 |
| real | ask_unc_neg | 149 | deduction | density_testlex | 0.157 | 0.137 | -0.020 | -0.063 | 0.023 | 0.374 |
| real | ask_unc_neg | 149 | adding-knowledge | density_testlex | 0.016 | 0.015 | -0.001 | -0.021 | 0.020 | 0.894 |
| real | ask_unc_neg | 149 | uncertainty-estimation | density_testlex | 0.132 | 0.084 | -0.048 | -0.095 | -0.007 | 0.033 |
| real | ask_unc_neg | 149 | backtracking | density_testlex | 0.004 | 0.003 | -0.001 | -0.008 | 0.006 | 0.751 |
| real | ask_unc_neg | 149 | deduction | density_nontest | 4.003 | 4.049 | 0.046 | -0.168 | 0.255 | 0.680 |
| real | ask_unc_neg | 149 | adding-knowledge | density_nontest | 1.554 | 1.598 | 0.045 | -0.134 | 0.216 | 0.618 |
| real | ask_unc_neg | 149 | uncertainty-estimation | density_nontest | 0.589 | 0.569 | -0.019 | -0.154 | 0.099 | 0.762 |
| real | ask_unc_neg | 149 | backtracking | density_nontest | 0.276 | 0.266 | -0.009 | -0.090 | 0.070 | 0.821 |
| real | ask_unc_pos | 149 | deduction | density | 4.160 | 4.134 | -0.026 | -0.237 | 0.185 | 0.813 |
| real | ask_unc_pos | 149 | adding-knowledge | density | 1.570 | 1.484 | -0.086 | -0.247 | 0.077 | 0.296 |
| real | ask_unc_pos | 149 | uncertainty-estimation | density | 0.721 | 0.732 | 0.011 | -0.120 | 0.134 | 0.858 |
| real | ask_unc_pos | 149 | backtracking | density | 0.279 | 0.244 | -0.036 | -0.107 | 0.034 | 0.337 |
| real | ask_unc_pos | 149 | deduction | density_testlex | 0.157 | 0.212 | 0.055 | 0.011 | 0.099 | 0.019 |
| real | ask_unc_pos | 149 | adding-knowledge | density_testlex | 0.016 | 0.010 | -0.006 | -0.024 | 0.011 | 0.471 |
| real | ask_unc_pos | 149 | uncertainty-estimation | density_testlex | 0.132 | 0.107 | -0.025 | -0.065 | 0.016 | 0.212 |
| real | ask_unc_pos | 149 | backtracking | density_testlex | 0.004 | 0.001 | -0.002 | -0.009 | 0.003 | 0.444 |
| real | ask_unc_pos | 149 | deduction | density_nontest | 4.003 | 3.922 | -0.082 | -0.304 | 0.142 | 0.476 |
| real | ask_unc_pos | 149 | adding-knowledge | density_nontest | 1.554 | 1.474 | -0.079 | -0.242 | 0.082 | 0.336 |
| real | ask_unc_pos | 149 | uncertainty-estimation | density_nontest | 0.589 | 0.625 | 0.036 | -0.088 | 0.147 | 0.542 |
| real | ask_unc_pos | 149 | backtracking | density_nontest | 0.276 | 0.242 | -0.033 | -0.106 | 0.036 | 0.368 |
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
| real | ask_bt_neg | gained | 13 | deduction | density | 0.935 | 0.115 | 1.948 |
| real | ask_bt_neg | gained | 13 | deduction | density_nontest | 1.118 | 0.278 | 2.145 |
| real | ask_bt_neg | gained | 13 | deduction | density_testlex | -0.183 | -0.298 | -0.065 |
| real | ask_bt_neg | gained | 13 | uncertainty-estimation | density | 0.308 | -0.107 | 0.732 |
| real | ask_bt_neg | gained | 13 | uncertainty-estimation | density_nontest | 0.406 | -0.035 | 0.884 |
| real | ask_bt_neg | gained | 13 | uncertainty-estimation | density_testlex | -0.098 | -0.288 | 0.094 |
| real | ask_bt_neg | gained | 13 | backtracking | density | -0.132 | -0.298 | 0.040 |
| real | ask_bt_neg | gained | 13 | backtracking | density_nontest | -0.132 | -0.298 | 0.040 |
| real | ask_bt_neg | gained | 13 | backtracking | density_testlex | 0.000 | 0.000 | 0.000 |
| real | ask_bt_neg | lost | 6 | deduction | density | -0.580 | -1.447 | 0.294 |
| real | ask_bt_neg | lost | 6 | deduction | density_nontest | -0.727 | -1.578 | 0.152 |
| real | ask_bt_neg | lost | 6 | deduction | density_testlex | 0.147 | 0.035 | 0.288 |
| real | ask_bt_neg | lost | 6 | uncertainty-estimation | density | 0.086 | -0.400 | 0.571 |
| real | ask_bt_neg | lost | 6 | uncertainty-estimation | density_nontest | 0.138 | -0.302 | 0.528 |
| real | ask_bt_neg | lost | 6 | uncertainty-estimation | density_testlex | -0.052 | -0.264 | 0.106 |
| real | ask_bt_neg | lost | 6 | backtracking | density | 0.037 | -0.232 | 0.305 |
| real | ask_bt_neg | lost | 6 | backtracking | density_nontest | 0.037 | -0.232 | 0.305 |
| real | ask_bt_neg | lost | 6 | backtracking | density_testlex | 0.000 | 0.000 | 0.000 |
| real | ask_bt_neg | comply_both | 93 | deduction | density | -0.115 | -0.409 | 0.178 |
| real | ask_bt_neg | comply_both | 93 | deduction | density_nontest | -0.087 | -0.399 | 0.212 |
| real | ask_bt_neg | comply_both | 93 | deduction | density_testlex | -0.028 | -0.080 | 0.025 |
| real | ask_bt_neg | comply_both | 93 | uncertainty-estimation | density | -0.051 | -0.215 | 0.099 |
| real | ask_bt_neg | comply_both | 93 | uncertainty-estimation | density_nontest | -0.050 | -0.207 | 0.094 |
| real | ask_bt_neg | comply_both | 93 | uncertainty-estimation | density_testlex | -0.001 | -0.053 | 0.049 |
| real | ask_bt_neg | comply_both | 93 | backtracking | density | -0.009 | -0.105 | 0.075 |
| real | ask_bt_neg | comply_both | 93 | backtracking | density_nontest | -0.006 | -0.102 | 0.076 |
| real | ask_bt_neg | comply_both | 93 | backtracking | density_testlex | -0.003 | -0.012 | 0.006 |
| real | ask_bt_neg | refuse_both | 26 | deduction | density | -0.022 | -0.428 | 0.375 |
| real | ask_bt_neg | refuse_both | 26 | deduction | density_nontest | -0.160 | -0.576 | 0.266 |
| real | ask_bt_neg | refuse_both | 26 | deduction | density_testlex | 0.138 | 0.012 | 0.268 |
| real | ask_bt_neg | refuse_both | 26 | uncertainty-estimation | density | -0.212 | -0.477 | 0.035 |
| real | ask_bt_neg | refuse_both | 26 | uncertainty-estimation | density_nontest | -0.078 | -0.316 | 0.146 |
| real | ask_bt_neg | refuse_both | 26 | uncertainty-estimation | density_testlex | -0.133 | -0.217 | -0.047 |
| real | ask_bt_neg | refuse_both | 26 | backtracking | density | 0.011 | -0.198 | 0.192 |
| real | ask_bt_neg | refuse_both | 26 | backtracking | density_nontest | -0.002 | -0.213 | 0.179 |
| real | ask_bt_neg | refuse_both | 26 | backtracking | density_testlex | 0.013 | 0.000 | 0.039 |
| real | ask_bt_pos | gained | 9 | deduction | density | -0.736 | -1.997 | 0.318 |
| real | ask_bt_pos | gained | 9 | deduction | density_nontest | -0.638 | -1.757 | 0.363 |
| real | ask_bt_pos | gained | 9 | deduction | density_testlex | -0.098 | -0.324 | 0.127 |
| real | ask_bt_pos | gained | 9 | uncertainty-estimation | density | 0.365 | -0.183 | 1.119 |
| real | ask_bt_pos | gained | 9 | uncertainty-estimation | density_nontest | 0.251 | -0.363 | 1.093 |
| real | ask_bt_pos | gained | 9 | uncertainty-estimation | density_testlex | 0.114 | -0.145 | 0.375 |
| real | ask_bt_pos | gained | 9 | backtracking | density | 0.039 | -0.151 | 0.222 |
| real | ask_bt_pos | gained | 9 | backtracking | density_nontest | 0.018 | -0.165 | 0.184 |
| real | ask_bt_pos | gained | 9 | backtracking | density_testlex | 0.021 | 0.000 | 0.064 |
| real | ask_bt_pos | comply_both | 101 | deduction | density | -0.232 | -0.501 | 0.048 |
| real | ask_bt_pos | comply_both | 101 | deduction | density_nontest | -0.222 | -0.495 | 0.062 |
| real | ask_bt_pos | comply_both | 101 | deduction | density_testlex | -0.010 | -0.058 | 0.039 |
| real | ask_bt_pos | comply_both | 101 | uncertainty-estimation | density | 0.108 | -0.025 | 0.241 |
| real | ask_bt_pos | comply_both | 101 | uncertainty-estimation | density_nontest | 0.095 | -0.036 | 0.225 |
| real | ask_bt_pos | comply_both | 101 | uncertainty-estimation | density_testlex | 0.013 | -0.036 | 0.062 |
| real | ask_bt_pos | comply_both | 101 | backtracking | density | 0.014 | -0.077 | 0.101 |
| real | ask_bt_pos | comply_both | 101 | backtracking | density_nontest | 0.014 | -0.079 | 0.099 |
| real | ask_bt_pos | comply_both | 101 | backtracking | density_testlex | 0.001 | -0.010 | 0.011 |
| real | ask_bt_pos | refuse_both | 31 | deduction | density | -0.305 | -0.771 | 0.141 |
| real | ask_bt_pos | refuse_both | 31 | deduction | density_nontest | -0.356 | -0.840 | 0.091 |
| real | ask_bt_pos | refuse_both | 31 | deduction | density_testlex | 0.051 | -0.067 | 0.158 |
| real | ask_bt_pos | refuse_both | 31 | uncertainty-estimation | density | 0.103 | -0.126 | 0.354 |
| real | ask_bt_pos | refuse_both | 31 | uncertainty-estimation | density_nontest | 0.078 | -0.133 | 0.304 |
| real | ask_bt_pos | refuse_both | 31 | uncertainty-estimation | density_testlex | 0.024 | -0.104 | 0.144 |
| real | ask_bt_pos | refuse_both | 31 | backtracking | density | 0.004 | -0.144 | 0.155 |
| real | ask_bt_pos | refuse_both | 31 | backtracking | density_nontest | 0.004 | -0.144 | 0.155 |
| real | ask_bt_pos | refuse_both | 31 | backtracking | density_testlex | 0.000 | 0.000 | 0.000 |
| real | ask_unc_neg | gained | 13 | deduction | density | -0.220 | -0.878 | 0.461 |
| real | ask_unc_neg | gained | 13 | deduction | density_nontest | -0.003 | -0.736 | 0.763 |
| real | ask_unc_neg | gained | 13 | deduction | density_testlex | -0.216 | -0.408 | -0.037 |
| real | ask_unc_neg | gained | 13 | uncertainty-estimation | density | 0.699 | 0.316 | 1.047 |
| real | ask_unc_neg | gained | 13 | uncertainty-estimation | density_nontest | 0.781 | 0.406 | 1.136 |
| real | ask_unc_neg | gained | 13 | uncertainty-estimation | density_testlex | -0.082 | -0.223 | 0.065 |
| real | ask_unc_neg | gained | 13 | backtracking | density | -0.255 | -0.542 | 0.004 |
| real | ask_unc_neg | gained | 13 | backtracking | density_nontest | -0.255 | -0.542 | 0.004 |
| real | ask_unc_neg | gained | 13 | backtracking | density_testlex | 0.000 | 0.000 | 0.000 |
| real | ask_unc_neg | comply_both | 101 | deduction | density | 0.082 | -0.191 | 0.339 |
| real | ask_unc_neg | comply_both | 101 | deduction | density_nontest | 0.116 | -0.159 | 0.375 |
| real | ask_unc_neg | comply_both | 101 | deduction | density_testlex | -0.034 | -0.072 | 0.005 |
| real | ask_unc_neg | comply_both | 101 | uncertainty-estimation | density | -0.070 | -0.206 | 0.066 |
| real | ask_unc_neg | comply_both | 101 | uncertainty-estimation | density_nontest | -0.039 | -0.170 | 0.092 |
| real | ask_unc_neg | comply_both | 101 | uncertainty-estimation | density_testlex | -0.031 | -0.077 | 0.014 |
| real | ask_unc_neg | comply_both | 101 | backtracking | density | 0.022 | -0.074 | 0.107 |
| real | ask_unc_neg | comply_both | 101 | backtracking | density_nontest | 0.027 | -0.068 | 0.114 |
| real | ask_unc_neg | comply_both | 101 | backtracking | density_testlex | -0.006 | -0.014 | 0.000 |
| real | ask_unc_neg | refuse_both | 27 | deduction | density | 0.020 | -0.355 | 0.374 |
| real | ask_unc_neg | refuse_both | 27 | deduction | density_nontest | -0.134 | -0.512 | 0.233 |
| real | ask_unc_neg | refuse_both | 27 | deduction | density_testlex | 0.154 | 0.014 | 0.296 |
| real | ask_unc_neg | refuse_both | 27 | uncertainty-estimation | density | -0.247 | -0.488 | -0.010 |
| real | ask_unc_neg | refuse_both | 27 | uncertainty-estimation | density_nontest | -0.134 | -0.354 | 0.071 |
| real | ask_unc_neg | refuse_both | 27 | uncertainty-estimation | density_testlex | -0.113 | -0.261 | 0.024 |
| real | ask_unc_neg | refuse_both | 27 | backtracking | density | 0.015 | -0.120 | 0.154 |
| real | ask_unc_neg | refuse_both | 27 | backtracking | density_nontest | 0.001 | -0.131 | 0.136 |
| real | ask_unc_neg | refuse_both | 27 | backtracking | density_testlex | 0.014 | 0.000 | 0.042 |
| real | ask_unc_pos | gained | 10 | deduction | density | 0.041 | -0.494 | 0.554 |
| real | ask_unc_pos | gained | 10 | deduction | density_nontest | 0.192 | -0.455 | 0.820 |
| real | ask_unc_pos | gained | 10 | deduction | density_testlex | -0.151 | -0.369 | 0.057 |
| real | ask_unc_pos | gained | 10 | uncertainty-estimation | density | 0.258 | -0.207 | 0.735 |
| real | ask_unc_pos | gained | 10 | uncertainty-estimation | density_nontest | 0.153 | -0.204 | 0.535 |
| real | ask_unc_pos | gained | 10 | uncertainty-estimation | density_testlex | 0.105 | -0.108 | 0.344 |
| real | ask_unc_pos | gained | 10 | backtracking | density | -0.318 | -0.597 | -0.102 |
| real | ask_unc_pos | gained | 10 | backtracking | density_nontest | -0.339 | -0.603 | -0.139 |
| real | ask_unc_pos | gained | 10 | backtracking | density_testlex | 0.021 | 0.000 | 0.064 |
| real | ask_unc_pos | lost | 9 | deduction | density | 0.340 | -0.697 | 1.440 |
| real | ask_unc_pos | lost | 9 | deduction | density_nontest | 0.137 | -0.979 | 1.266 |
| real | ask_unc_pos | lost | 9 | deduction | density_testlex | 0.203 | -0.074 | 0.477 |
| real | ask_unc_pos | lost | 9 | uncertainty-estimation | density | -0.509 | -1.014 | -0.046 |
| real | ask_unc_pos | lost | 9 | uncertainty-estimation | density_nontest | -0.449 | -0.884 | -0.074 |
| real | ask_unc_pos | lost | 9 | uncertainty-estimation | density_testlex | -0.060 | -0.223 | 0.097 |
| real | ask_unc_pos | lost | 9 | backtracking | density | 0.122 | -0.200 | 0.520 |
| real | ask_unc_pos | lost | 9 | backtracking | density_nontest | 0.122 | -0.200 | 0.520 |
| real | ask_unc_pos | lost | 9 | backtracking | density_testlex | 0.000 | 0.000 | 0.000 |
| real | ask_unc_pos | comply_both | 97 | deduction | density | -0.088 | -0.371 | 0.194 |
| real | ask_unc_pos | comply_both | 97 | deduction | density_nontest | -0.117 | -0.409 | 0.172 |
| real | ask_unc_pos | comply_both | 97 | deduction | density_testlex | 0.029 | -0.017 | 0.075 |
| real | ask_unc_pos | comply_both | 97 | uncertainty-estimation | density | 0.145 | 0.006 | 0.291 |
| real | ask_unc_pos | comply_both | 97 | uncertainty-estimation | density_nontest | 0.147 | 0.011 | 0.290 |
| real | ask_unc_pos | comply_both | 97 | uncertainty-estimation | density_testlex | -0.002 | -0.043 | 0.040 |
| real | ask_unc_pos | comply_both | 97 | backtracking | density | -0.013 | -0.107 | 0.071 |
| real | ask_unc_pos | comply_both | 97 | backtracking | density_nontest | -0.008 | -0.101 | 0.078 |
| real | ask_unc_pos | comply_both | 97 | backtracking | density_testlex | -0.006 | -0.014 | 0.000 |
| real | ask_unc_pos | refuse_both | 30 | deduction | density | -0.001 | -0.387 | 0.395 |
| real | ask_unc_pos | refuse_both | 30 | deduction | density_nontest | -0.173 | -0.584 | 0.248 |
| real | ask_unc_pos | refuse_both | 30 | deduction | density_testlex | 0.173 | 0.063 | 0.280 |
| real | ask_unc_pos | refuse_both | 30 | uncertainty-estimation | density | -0.246 | -0.490 | -0.018 |
| real | ask_unc_pos | refuse_both | 30 | uncertainty-estimation | density_nontest | -0.109 | -0.317 | 0.098 |
| real | ask_unc_pos | refuse_both | 30 | uncertainty-estimation | density_testlex | -0.136 | -0.232 | -0.037 |
| real | ask_unc_pos | refuse_both | 30 | backtracking | density | -0.007 | -0.115 | 0.108 |
| real | ask_unc_pos | refuse_both | 30 | backtracking | density_nontest | -0.007 | -0.115 | 0.108 |
| real | ask_unc_pos | refuse_both | 30 | backtracking | density_testlex | 0.000 | 0.000 | 0.000 |
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
| ask_bt_neg | real | proj_backtracking | 0 | 20.3383 | — | — | — |
| ask_bt_neg | real | proj_uncertainty-estimation | 0 | 8.7282 | — | — | — |
| ask_bt_neg | real | proj_probe | 149 | -2.4006 | -0.0964 | -0.3169 | 0.1208 |
| ask_bt_neg | real | proj_lastprompt_backtracking | 149 | 11.9075 | -0.2411 | -0.2819 | -0.1985 |
| ask_bt_neg | real | proj_lastprompt_uncertainty-estimation | 149 | 9.7791 | -0.0498 | -0.0642 | -0.0357 |
| ask_bt_neg | real | proj_lastprompt_probe | 149 | -1.9825 | -0.2363 | -0.2548 | -0.2174 |
| ask_bt_neg | real | proj_incontext_backtracking | 149 | 18.8387 | -0.0891 | -0.5369 | 0.3419 |
| ask_bt_neg | real | proj_incontext_uncertainty-estimation | 149 | 8.5692 | -0.0394 | -0.5233 | 0.4397 |
| ask_bt_neg | real | proj_incontext_probe | 149 | -2.9817 | -0.0958 | -0.2913 | 0.0956 |
| ask_bt_neg | fake | proj_backtracking | 0 | — | — | — | — |
| ask_bt_neg | fake | proj_uncertainty-estimation | 0 | — | — | — | — |
| ask_bt_neg | fake | proj_probe | 0 | — | — | — | — |
| ask_bt_neg | fake | proj_lastprompt_backtracking | 0 | — | — | — | — |
| ask_bt_neg | fake | proj_lastprompt_uncertainty-estimation | 0 | — | — | — | — |
| ask_bt_neg | fake | proj_lastprompt_probe | 0 | — | — | — | — |
| ask_bt_neg | fake | proj_incontext_backtracking | 0 | — | — | — | — |
| ask_bt_neg | fake | proj_incontext_uncertainty-estimation | 0 | — | — | — | — |
| ask_bt_neg | fake | proj_incontext_probe | 0 | — | — | — | — |
| ask_bt_pos | real | proj_backtracking | 0 | 20.3383 | — | — | — |
| ask_bt_pos | real | proj_uncertainty-estimation | 0 | 8.7282 | — | — | — |
| ask_bt_pos | real | proj_probe | 149 | -2.4006 | 0.1350 | -0.0563 | 0.3248 |
| ask_bt_pos | real | proj_lastprompt_backtracking | 149 | 11.9075 | 0.0856 | 0.0423 | 0.1292 |
| ask_bt_pos | real | proj_lastprompt_uncertainty-estimation | 149 | 9.7791 | 0.2536 | 0.2393 | 0.2676 |
| ask_bt_pos | real | proj_lastprompt_probe | 149 | -1.9825 | -0.5095 | -0.5240 | -0.4949 |
| ask_bt_pos | real | proj_incontext_backtracking | 149 | 18.8387 | 0.4842 | 0.0545 | 0.9199 |
| ask_bt_pos | real | proj_incontext_uncertainty-estimation | 149 | 8.5692 | -0.3517 | -0.7809 | 0.0688 |
| ask_bt_pos | real | proj_incontext_probe | 149 | -2.9817 | 0.1613 | -0.0184 | 0.3358 |
| ask_bt_pos | fake | proj_backtracking | 0 | — | — | — | — |
| ask_bt_pos | fake | proj_uncertainty-estimation | 0 | — | — | — | — |
| ask_bt_pos | fake | proj_probe | 0 | — | — | — | — |
| ask_bt_pos | fake | proj_lastprompt_backtracking | 0 | — | — | — | — |
| ask_bt_pos | fake | proj_lastprompt_uncertainty-estimation | 0 | — | — | — | — |
| ask_bt_pos | fake | proj_lastprompt_probe | 0 | — | — | — | — |
| ask_bt_pos | fake | proj_incontext_backtracking | 0 | — | — | — | — |
| ask_bt_pos | fake | proj_incontext_uncertainty-estimation | 0 | — | — | — | — |
| ask_bt_pos | fake | proj_incontext_probe | 0 | — | — | — | — |
| ask_unc_neg | real | proj_backtracking | 0 | 20.3383 | — | — | — |
| ask_unc_neg | real | proj_uncertainty-estimation | 0 | 8.7282 | — | — | — |
| ask_unc_neg | real | proj_probe | 149 | -2.4006 | -0.1252 | -0.3527 | 0.0709 |
| ask_unc_neg | real | proj_lastprompt_backtracking | 149 | 11.9075 | 0.1848 | 0.1437 | 0.2262 |
| ask_unc_neg | real | proj_lastprompt_uncertainty-estimation | 149 | 9.7791 | -0.0515 | -0.0660 | -0.0369 |
| ask_unc_neg | real | proj_lastprompt_probe | 149 | -1.9825 | -0.2234 | -0.2387 | -0.2072 |
| ask_unc_neg | real | proj_incontext_backtracking | 149 | 18.8387 | -0.1609 | -0.5799 | 0.2217 |
| ask_unc_neg | real | proj_incontext_uncertainty-estimation | 149 | 8.5692 | -0.3533 | -0.8216 | 0.0955 |
| ask_unc_neg | real | proj_incontext_probe | 149 | -2.9817 | -0.0852 | -0.2858 | 0.0985 |
| ask_unc_neg | fake | proj_backtracking | 0 | — | — | — | — |
| ask_unc_neg | fake | proj_uncertainty-estimation | 0 | — | — | — | — |
| ask_unc_neg | fake | proj_probe | 0 | — | — | — | — |
| ask_unc_neg | fake | proj_lastprompt_backtracking | 0 | — | — | — | — |
| ask_unc_neg | fake | proj_lastprompt_uncertainty-estimation | 0 | — | — | — | — |
| ask_unc_neg | fake | proj_lastprompt_probe | 0 | — | — | — | — |
| ask_unc_neg | fake | proj_incontext_backtracking | 0 | — | — | — | — |
| ask_unc_neg | fake | proj_incontext_uncertainty-estimation | 0 | — | — | — | — |
| ask_unc_neg | fake | proj_incontext_probe | 0 | — | — | — | — |
| ask_unc_pos | real | proj_backtracking | 0 | 20.3383 | — | — | — |
| ask_unc_pos | real | proj_uncertainty-estimation | 0 | 8.7282 | — | — | — |
| ask_unc_pos | real | proj_probe | 149 | -2.4006 | 0.1725 | -0.0456 | 0.3735 |
| ask_unc_pos | real | proj_lastprompt_backtracking | 149 | 11.9075 | 0.5426 | 0.4951 | 0.5931 |
| ask_unc_pos | real | proj_lastprompt_uncertainty-estimation | 149 | 9.7791 | -0.0547 | -0.0696 | -0.0396 |
| ask_unc_pos | real | proj_lastprompt_probe | 149 | -1.9825 | -0.5549 | -0.5697 | -0.5405 |
| ask_unc_pos | real | proj_incontext_backtracking | 149 | 18.8387 | 0.3905 | -0.0325 | 0.7946 |
| ask_unc_pos | real | proj_incontext_uncertainty-estimation | 149 | 8.5692 | -0.0600 | -0.5103 | 0.3742 |
| ask_unc_pos | real | proj_incontext_probe | 149 | -2.9817 | 0.1331 | -0.0628 | 0.3251 |
| ask_unc_pos | fake | proj_backtracking | 0 | — | — | — | — |
| ask_unc_pos | fake | proj_uncertainty-estimation | 0 | — | — | — | — |
| ask_unc_pos | fake | proj_probe | 0 | — | — | — | — |
| ask_unc_pos | fake | proj_lastprompt_backtracking | 0 | — | — | — | — |
| ask_unc_pos | fake | proj_lastprompt_uncertainty-estimation | 0 | — | — | — | — |
| ask_unc_pos | fake | proj_lastprompt_probe | 0 | — | — | — | — |
| ask_unc_pos | fake | proj_incontext_backtracking | 0 | — | — | — | — |
| ask_unc_pos | fake | proj_incontext_uncertainty-estimation | 0 | — | — | — | — |
| ask_unc_pos | fake | proj_incontext_probe | 0 | — | — | — | — |


## Awareness-probe score of each arm's reasoning (unedited model, the probe's own input regime and hidden-state index): per arm, and verbalizing vs non-verbalizing traces within the arm

| arm | framing | n | mean | ci_lo | ci_hi | n_verbalizing | mean_verbalizing | mean_not | p_welch |
|---|---|---|---|---|---|---|---|---|---|
| baseline | real | 149 | -2.4006 | -2.5992 | -2.1838 | 28 | -1.1496 | -2.6900 | 0.0000 |
| baseline | fake | 149 | -2.7419 | -2.9424 | -2.5154 | 13 | -1.2038 | -2.8878 | 0.0000 |
| baseline_seed1 | real | 149 | -2.3571 | -2.5585 | -2.1620 | 22 | -1.4930 | -2.5068 | 0.0004 |
| baseline_seed1 | fake | 149 | -2.7615 | -2.9646 | -2.5511 | 9 | -0.5367 | -2.9046 | 0.0010 |
| ask_bt_neg | real | 149 | -2.4970 | -2.6941 | -2.2925 | 20 | -1.4548 | -2.6580 | 0.0001 |
| ask_bt_neg | fake | 0 | — | — | — | 0 | — | — | — |
| ask_bt_pos | real | 149 | -2.2655 | -2.4690 | -2.0442 | 25 | -1.3753 | -2.4450 | 0.0011 |
| ask_bt_pos | fake | 0 | — | — | — | 0 | — | — | — |
| ask_unc_neg | real | 149 | -2.5258 | -2.7366 | -2.3369 | 14 | -1.4400 | -2.6251 | 0.0087 |
| ask_unc_neg | fake | 0 | — | — | — | 0 | — | — | — |
| ask_unc_pos | real | 149 | -2.2281 | -2.4153 | -2.0471 | 23 | -1.7348 | -2.3181 | 0.0245 |
| ask_unc_pos | fake | 0 | — | — | — | 0 | — | — | — |

