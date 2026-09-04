# Mechanistic arm (Qwen3-4B) — first-pass analysis: f1_prompt

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


## f1_prompt: 1192 traces, 1192 with usable morphology annotation

| arm | framing | n | exec_rate_closed | exec_rate_all | lost | gained | p_mcnemar | task_judged_yes | judge_vs_string_agree | verbalization | closure | repetition | words |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | real | 149 | 0.726 | 0.711 | — | — | — | 0.826 | 0.582 | 0.188 | 0.980 | 0.046 | 334.470 |
| baseline | fake | 149 | 0.882 | 0.852 | — | — | — | 0.851 | 0.762 | 0.088 | 0.966 | 0.047 | 345.228 |
| baseline_seed1 | real | 149 | 0.748 | 0.718 | 3.000 | 6.000 | 0.508 | 0.738 | 0.545 | 0.148 | 0.960 | 0.051 | 353.859 |
| baseline_seed1 | fake | 149 | 0.890 | 0.872 | 5.000 | 6.000 | 1.000 | 0.852 | 0.760 | 0.060 | 0.980 | 0.047 | 331.664 |
| aware_strong | real | 149 | 0.766 | 0.745 | 6.000 | 12.000 | 0.238 | 0.777 | 0.562 | 0.182 | 0.973 | 0.043 | 327.289 |
| aware_strong | fake | 149 | 0.873 | 0.832 | 4.000 | 4.000 | 1.000 | 0.797 | 0.745 | 0.135 | 0.953 | 0.065 | 398.463 |
| prompt_test | real | 149 | 0.841 | 0.819 | 3.000 | 20.000 | 0.000 | 0.738 | 0.614 | 0.181 | 0.973 | 0.052 | 354.376 |
| prompt_test | fake | 149 | 0.910 | 0.879 | 6.000 | 10.000 | 0.454 | 0.851 | 0.790 | 0.115 | 0.966 | 0.048 | 344.168 |


## Morphology: Δ spans per 100 words vs baseline, paired by item

| framing | arm | n_pairs | behaviour | metric | base_mean | arm_mean | delta | ci_lo | ci_hi | p_paired_t |
|---|---|---|---|---|---|---|---|---|---|---|
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
| fake | prompt_test | 149 | deduction | density | 3.843 | 3.983 | 0.139 | -0.111 | 0.401 | 0.280 |
| fake | prompt_test | 149 | adding-knowledge | density | 1.706 | 1.608 | -0.097 | -0.277 | 0.084 | 0.304 |
| fake | prompt_test | 149 | uncertainty-estimation | density | 0.606 | 0.623 | 0.017 | -0.105 | 0.134 | 0.787 |
| fake | prompt_test | 149 | backtracking | density | 0.279 | 0.312 | 0.033 | -0.043 | 0.105 | 0.379 |
| fake | prompt_test | 149 | deduction | density_testlex | 0.131 | 0.123 | -0.008 | -0.059 | 0.043 | 0.744 |
| fake | prompt_test | 149 | adding-knowledge | density_testlex | 0.016 | 0.032 | 0.017 | -0.005 | 0.037 | 0.118 |
| fake | prompt_test | 149 | uncertainty-estimation | density_testlex | 0.062 | 0.136 | 0.075 | 0.033 | 0.118 | 0.001 |
| fake | prompt_test | 149 | backtracking | density_testlex | 0.004 | 0.000 | -0.004 | -0.009 | 0.000 | 0.091 |
| fake | prompt_test | 149 | deduction | density_nontest | 3.712 | 3.860 | 0.148 | -0.113 | 0.415 | 0.266 |
| fake | prompt_test | 149 | adding-knowledge | density_nontest | 1.690 | 1.576 | -0.114 | -0.295 | 0.072 | 0.227 |
| fake | prompt_test | 149 | uncertainty-estimation | density_nontest | 0.545 | 0.487 | -0.058 | -0.184 | 0.062 | 0.358 |
| fake | prompt_test | 149 | backtracking | density_nontest | 0.275 | 0.312 | 0.037 | -0.038 | 0.108 | 0.321 |
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
| real | prompt_test | 149 | deduction | density | 4.160 | 3.970 | -0.191 | -0.407 | 0.017 | 0.093 |
| real | prompt_test | 149 | adding-knowledge | density | 1.570 | 1.594 | 0.025 | -0.150 | 0.200 | 0.791 |
| real | prompt_test | 149 | uncertainty-estimation | density | 0.721 | 0.772 | 0.052 | -0.093 | 0.184 | 0.473 |
| real | prompt_test | 149 | backtracking | density | 0.279 | 0.297 | 0.017 | -0.069 | 0.098 | 0.682 |
| real | prompt_test | 149 | deduction | density_testlex | 0.157 | 0.214 | 0.057 | 0.006 | 0.110 | 0.035 |
| real | prompt_test | 149 | adding-knowledge | density_testlex | 0.016 | 0.060 | 0.044 | 0.017 | 0.071 | 0.002 |
| real | prompt_test | 149 | uncertainty-estimation | density_testlex | 0.132 | 0.129 | -0.003 | -0.045 | 0.041 | 0.891 |
| real | prompt_test | 149 | backtracking | density_testlex | 0.004 | 0.001 | -0.003 | -0.008 | 0.002 | 0.343 |
| real | prompt_test | 149 | deduction | density_nontest | 4.003 | 3.755 | -0.248 | -0.467 | -0.035 | 0.034 |
| real | prompt_test | 149 | adding-knowledge | density_nontest | 1.554 | 1.534 | -0.019 | -0.193 | 0.151 | 0.832 |
| real | prompt_test | 149 | uncertainty-estimation | density_nontest | 0.589 | 0.643 | 0.055 | -0.092 | 0.182 | 0.452 |
| real | prompt_test | 149 | backtracking | density_nontest | 0.276 | 0.296 | 0.020 | -0.066 | 0.101 | 0.634 |


## Morphology Δ by compliance class (decision-change confound): `refuse_both` / `comply_both` hold the decision fixed

| framing | arm | flip_class | n | behaviour | metric | delta_density | ci_lo | ci_hi |
|---|---|---|---|---|---|---|---|---|
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
| fake | prompt_test | gained | 10 | deduction | density | -0.045 | -0.958 | 0.854 |
| fake | prompt_test | gained | 10 | deduction | density_nontest | -0.060 | -1.083 | 0.951 |
| fake | prompt_test | gained | 10 | deduction | density_testlex | 0.015 | -0.178 | 0.207 |
| fake | prompt_test | gained | 10 | uncertainty-estimation | density | 0.464 | 0.212 | 0.758 |
| fake | prompt_test | gained | 10 | uncertainty-estimation | density_nontest | 0.217 | -0.123 | 0.608 |
| fake | prompt_test | gained | 10 | uncertainty-estimation | density_testlex | 0.246 | 0.108 | 0.385 |
| fake | prompt_test | gained | 10 | backtracking | density | -0.222 | -0.466 | 0.025 |
| fake | prompt_test | gained | 10 | backtracking | density_nontest | -0.205 | -0.421 | 0.029 |
| fake | prompt_test | gained | 10 | backtracking | density_testlex | -0.017 | -0.050 | 0.000 |
| fake | prompt_test | lost | 6 | deduction | density | -1.075 | -1.984 | 0.211 |
| fake | prompt_test | lost | 6 | deduction | density_nontest | -1.229 | -2.197 | -0.081 |
| fake | prompt_test | lost | 6 | deduction | density_testlex | 0.153 | -0.146 | 0.463 |
| fake | prompt_test | lost | 6 | uncertainty-estimation | density | -0.538 | -1.305 | 0.196 |
| fake | prompt_test | lost | 6 | uncertainty-estimation | density_nontest | -0.710 | -1.328 | -0.120 |
| fake | prompt_test | lost | 6 | uncertainty-estimation | density_testlex | 0.171 | -0.153 | 0.516 |
| fake | prompt_test | lost | 6 | backtracking | density | 0.113 | -0.244 | 0.599 |
| fake | prompt_test | lost | 6 | backtracking | density_nontest | 0.113 | -0.244 | 0.599 |
| fake | prompt_test | lost | 6 | backtracking | density_testlex | 0.000 | 0.000 | 0.000 |
| fake | prompt_test | comply_both | 116 | deduction | density | 0.288 | 0.007 | 0.575 |
| fake | prompt_test | comply_both | 116 | deduction | density_nontest | 0.294 | 0.011 | 0.581 |
| fake | prompt_test | comply_both | 116 | deduction | density_testlex | -0.006 | -0.062 | 0.051 |
| fake | prompt_test | comply_both | 116 | uncertainty-estimation | density | -0.015 | -0.131 | 0.105 |
| fake | prompt_test | comply_both | 116 | uncertainty-estimation | density_nontest | -0.075 | -0.207 | 0.047 |
| fake | prompt_test | comply_both | 116 | uncertainty-estimation | density_testlex | 0.060 | 0.011 | 0.108 |
| fake | prompt_test | comply_both | 116 | backtracking | density | 0.036 | -0.027 | 0.101 |
| fake | prompt_test | comply_both | 116 | backtracking | density_nontest | 0.040 | -0.023 | 0.105 |
| fake | prompt_test | comply_both | 116 | backtracking | density_testlex | -0.004 | -0.010 | 0.000 |
| fake | prompt_test | refuse_both | 7 | deduction | density | -0.412 | -1.027 | 0.387 |
| fake | prompt_test | refuse_both | 7 | deduction | density_nontest | -0.244 | -0.922 | 0.724 |
| fake | prompt_test | refuse_both | 7 | deduction | density_testlex | -0.167 | -0.419 | 0.057 |
| fake | prompt_test | refuse_both | 7 | uncertainty-estimation | density | 0.099 | -0.345 | 0.608 |
| fake | prompt_test | refuse_both | 7 | uncertainty-estimation | density_nontest | -0.024 | -0.314 | 0.249 |
| fake | prompt_test | refuse_both | 7 | uncertainty-estimation | density_testlex | 0.123 | -0.099 | 0.399 |
| fake | prompt_test | refuse_both | 7 | backtracking | density | 0.025 | -0.256 | 0.346 |
| fake | prompt_test | refuse_both | 7 | backtracking | density_nontest | 0.025 | -0.256 | 0.346 |
| fake | prompt_test | refuse_both | 7 | backtracking | density_testlex | 0.000 | 0.000 | 0.000 |
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
| real | prompt_test | gained | 20 | deduction | density | 0.241 | -0.173 | 0.650 |
| real | prompt_test | gained | 20 | deduction | density_nontest | 0.227 | -0.297 | 0.737 |
| real | prompt_test | gained | 20 | deduction | density_testlex | 0.013 | -0.180 | 0.241 |
| real | prompt_test | gained | 20 | uncertainty-estimation | density | 0.366 | -0.060 | 0.791 |
| real | prompt_test | gained | 20 | uncertainty-estimation | density_nontest | 0.437 | 0.040 | 0.833 |
| real | prompt_test | gained | 20 | uncertainty-estimation | density_testlex | -0.071 | -0.226 | 0.083 |
| real | prompt_test | gained | 20 | backtracking | density | -0.025 | -0.259 | 0.178 |
| real | prompt_test | gained | 20 | backtracking | density_nontest | -0.032 | -0.264 | 0.169 |
| real | prompt_test | gained | 20 | backtracking | density_testlex | 0.008 | 0.000 | 0.023 |
| real | prompt_test | comply_both | 99 | deduction | density | -0.341 | -0.590 | -0.083 |
| real | prompt_test | comply_both | 99 | deduction | density_nontest | -0.377 | -0.630 | -0.113 |
| real | prompt_test | comply_both | 99 | deduction | density_testlex | 0.036 | -0.016 | 0.089 |
| real | prompt_test | comply_both | 99 | uncertainty-estimation | density | 0.085 | -0.074 | 0.245 |
| real | prompt_test | comply_both | 99 | uncertainty-estimation | density_nontest | 0.072 | -0.086 | 0.236 |
| real | prompt_test | comply_both | 99 | uncertainty-estimation | density_testlex | 0.013 | -0.035 | 0.062 |
| real | prompt_test | comply_both | 99 | backtracking | density | 0.013 | -0.073 | 0.096 |
| real | prompt_test | comply_both | 99 | backtracking | density_nontest | 0.019 | -0.068 | 0.101 |
| real | prompt_test | comply_both | 99 | backtracking | density_testlex | -0.006 | -0.014 | 0.000 |
| real | prompt_test | refuse_both | 20 | deduction | density | 0.307 | -0.480 | 1.147 |
| real | prompt_test | refuse_both | 20 | deduction | density_nontest | 0.065 | -0.753 | 0.930 |
| real | prompt_test | refuse_both | 20 | deduction | density_testlex | 0.242 | 0.083 | 0.410 |
| real | prompt_test | refuse_both | 20 | uncertainty-estimation | density | -0.071 | -0.396 | 0.227 |
| real | prompt_test | refuse_both | 20 | uncertainty-estimation | density_nontest | -0.027 | -0.338 | 0.266 |
| real | prompt_test | refuse_both | 20 | uncertainty-estimation | density_testlex | -0.044 | -0.172 | 0.090 |
| real | prompt_test | refuse_both | 20 | backtracking | density | -0.110 | -0.263 | 0.048 |
| real | prompt_test | refuse_both | 20 | backtracking | density_nontest | -0.110 | -0.263 | 0.048 |
| real | prompt_test | refuse_both | 20 | backtracking | density_testlex | 0.000 | 0.000 | 0.000 |


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
| aware_strong | real | proj_backtracking | 149 | 20.3383 | -0.1800 | -0.6088 | 0.2246 |
| aware_strong | real | proj_uncertainty-estimation | 149 | 8.7282 | -0.2790 | -0.5283 | -0.0587 |
| aware_strong | real | proj_probe | 149 | -2.4006 | -0.0935 | -0.2892 | 0.1018 |
| aware_strong | real | proj_lastprompt_backtracking | 149 | 11.9075 | -0.0172 | -0.0332 | -0.0021 |
| aware_strong | real | proj_lastprompt_uncertainty-estimation | 149 | 9.7791 | -0.0313 | -0.0419 | -0.0200 |
| aware_strong | real | proj_lastprompt_probe | 149 | -1.9825 | -0.0005 | -0.0042 | 0.0036 |
| aware_strong | real | proj_incontext_backtracking | 149 | 18.8387 | -0.2289 | -0.6630 | 0.1777 |
| aware_strong | real | proj_incontext_uncertainty-estimation | 149 | 8.5692 | -0.5750 | -0.9798 | -0.1971 |
| aware_strong | real | proj_incontext_probe | 149 | -2.9817 | 0.0006 | -0.1856 | 0.1822 |
| aware_strong | fake | proj_backtracking | 149 | 19.4140 | 0.5754 | 0.1815 | 0.9753 |
| aware_strong | fake | proj_uncertainty-estimation | 149 | 8.3488 | -0.1149 | -0.4379 | 0.1666 |
| aware_strong | fake | proj_probe | 149 | -2.7419 | 0.2801 | 0.0832 | 0.4771 |
| aware_strong | fake | proj_lastprompt_backtracking | 149 | 11.7580 | -0.0229 | -0.0387 | -0.0061 |
| aware_strong | fake | proj_lastprompt_uncertainty-estimation | 149 | 9.6792 | -0.0469 | -0.0584 | -0.0358 |
| aware_strong | fake | proj_lastprompt_probe | 149 | -2.0660 | 0.0052 | 0.0008 | 0.0094 |
| aware_strong | fake | proj_incontext_backtracking | 149 | 17.9669 | 0.5941 | 0.2173 | 0.9865 |
| aware_strong | fake | proj_incontext_uncertainty-estimation | 149 | 7.7193 | 0.2816 | -0.0847 | 0.6640 |
| aware_strong | fake | proj_incontext_probe | 149 | -3.2032 | 0.2155 | 0.0508 | 0.3857 |
| prompt_test | real | proj_backtracking | 149 | 20.3383 | -0.0126 | -0.4212 | 0.3773 |
| prompt_test | real | proj_uncertainty-estimation | 149 | 8.7282 | -0.1593 | -0.4809 | 0.1810 |
| prompt_test | real | proj_probe | 149 | -2.4006 | -0.0597 | -0.2870 | 0.1430 |
| prompt_test | real | proj_lastprompt_backtracking | 149 | 11.9075 | 0.0985 | 0.0695 | 0.1288 |
| prompt_test | real | proj_lastprompt_uncertainty-estimation | 149 | 9.7791 | 0.0460 | 0.0322 | 0.0602 |
| prompt_test | real | proj_lastprompt_probe | 149 | -1.9825 | 0.1365 | 0.1246 | 0.1493 |
| prompt_test | real | proj_incontext_backtracking | 149 | 18.8387 | -0.0530 | -0.4866 | 0.3437 |
| prompt_test | real | proj_incontext_uncertainty-estimation | 149 | 8.5692 | -0.4056 | -0.8381 | 0.0400 |
| prompt_test | real | proj_incontext_probe | 149 | -2.9817 | 0.0322 | -0.1734 | 0.2263 |
| prompt_test | fake | proj_backtracking | 149 | 19.4140 | 0.1107 | -0.2972 | 0.5264 |
| prompt_test | fake | proj_uncertainty-estimation | 149 | 8.3488 | -0.2042 | -0.4677 | 0.0577 |
| prompt_test | fake | proj_probe | 149 | -2.7419 | -0.0009 | -0.2037 | 0.1853 |
| prompt_test | fake | proj_lastprompt_backtracking | 149 | 11.7580 | 0.0949 | 0.0614 | 0.1292 |
| prompt_test | fake | proj_lastprompt_uncertainty-estimation | 149 | 9.6792 | 0.0555 | 0.0407 | 0.0695 |
| prompt_test | fake | proj_lastprompt_probe | 149 | -2.0660 | 0.1400 | 0.1291 | 0.1519 |
| prompt_test | fake | proj_incontext_backtracking | 149 | 17.9669 | 0.1232 | -0.2841 | 0.5270 |
| prompt_test | fake | proj_incontext_uncertainty-estimation | 149 | 7.7193 | -0.2345 | -0.5635 | 0.0883 |
| prompt_test | fake | proj_incontext_probe | 149 | -3.2032 | 0.0494 | -0.1212 | 0.2125 |


## Awareness-probe score of each arm's reasoning (unedited model, the probe's own input regime and hidden-state index): per arm, and verbalizing vs non-verbalizing traces within the arm

| arm | framing | n | mean | ci_lo | ci_hi | n_verbalizing | mean_verbalizing | mean_not | p_welch |
|---|---|---|---|---|---|---|---|---|---|
| baseline | real | 149 | -2.4006 | -2.5992 | -2.1838 | 28 | -1.1496 | -2.6900 | 0.0000 |
| baseline | fake | 149 | -2.7419 | -2.9424 | -2.5154 | 13 | -1.2038 | -2.8878 | 0.0000 |
| baseline_seed1 | real | 149 | -2.3571 | -2.5585 | -2.1620 | 22 | -1.4930 | -2.5068 | 0.0004 |
| baseline_seed1 | fake | 149 | -2.7615 | -2.9646 | -2.5511 | 9 | -0.5367 | -2.9046 | 0.0010 |
| aware_strong | real | 149 | -2.4941 | -2.6946 | -2.2860 | 27 | -1.0875 | -2.7973 | 0.0000 |
| aware_strong | fake | 149 | -2.4618 | -2.6434 | -2.2568 | 20 | -1.4412 | -2.6324 | 0.0003 |
| prompt_test | real | 149 | -2.4603 | -2.6642 | -2.2455 | 27 | -1.2021 | -2.7388 | 0.0000 |
| prompt_test | fake | 149 | -2.7428 | -2.9192 | -2.5443 | 17 | -1.6375 | -2.8892 | 0.0000 |

