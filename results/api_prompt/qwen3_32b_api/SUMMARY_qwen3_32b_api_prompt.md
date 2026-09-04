# Mechanistic arm (Qwen3-4B) — first-pass analysis: qwen3_32b_api_prompt


## qwen3_32b_api_prompt: 596 traces, 596 with usable morphology annotation

| arm | framing | n | exec_rate_closed | exec_rate_all | lost | gained | p_mcnemar | task_judged_yes | judge_vs_string_agree | verbalization | closure | repetition | words |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | real | 149 | 0.242 | 0.242 | — | — | — | 0.933 | 0.188 | 0.101 | 1.000 | 0.007 | 285.443 |
| baseline | fake | 149 | 0.409 | 0.409 | — | — | — | 0.940 | 0.362 | 0.107 | 1.000 | 0.007 | 293.195 |
| prompt_test | real | 149 | 0.268 | 0.268 | 19.000 | 23.000 | 0.644 | 0.913 | 0.195 | 0.228 | 1.000 | 0.007 | 285.591 |
| prompt_test | fake | 149 | 0.561 | 0.557 | 14.000 | 36.000 | 0.003 | 0.926 | 0.507 | 0.148 | 0.993 | 0.013 | 367.181 |


## Morphology: Δ spans per 100 words vs baseline, paired by item

| framing | arm | n_pairs | behaviour | metric | base_mean | arm_mean | delta | ci_lo | ci_hi | p_paired_t |
|---|---|---|---|---|---|---|---|---|---|---|
| fake | prompt_test | 149 | deduction | density | 4.017 | 3.950 | -0.067 | -0.313 | 0.174 | 0.590 |
| fake | prompt_test | 149 | adding-knowledge | density | 1.772 | 1.653 | -0.118 | -0.277 | 0.035 | 0.153 |
| fake | prompt_test | 149 | uncertainty-estimation | density | 0.675 | 0.710 | 0.034 | -0.088 | 0.165 | 0.602 |
| fake | prompt_test | 149 | backtracking | density | 0.179 | 0.185 | 0.006 | -0.047 | 0.061 | 0.825 |
| fake | prompt_test | 149 | deduction | density_testlex | 0.230 | 0.350 | 0.120 | 0.053 | 0.190 | 0.001 |
| fake | prompt_test | 149 | adding-knowledge | density_testlex | 0.030 | 0.068 | 0.038 | 0.006 | 0.071 | 0.023 |
| fake | prompt_test | 149 | uncertainty-estimation | density_testlex | 0.108 | 0.142 | 0.035 | -0.011 | 0.078 | 0.142 |
| fake | prompt_test | 149 | backtracking | density_testlex | 0.002 | 0.008 | 0.006 | -0.002 | 0.015 | 0.140 |
| fake | prompt_test | 149 | deduction | density_nontest | 3.787 | 3.600 | -0.187 | -0.445 | 0.063 | 0.153 |
| fake | prompt_test | 149 | adding-knowledge | density_nontest | 1.742 | 1.586 | -0.157 | -0.304 | -0.006 | 0.054 |
| fake | prompt_test | 149 | uncertainty-estimation | density_nontest | 0.568 | 0.567 | -0.000 | -0.113 | 0.121 | 0.994 |
| fake | prompt_test | 149 | backtracking | density_nontest | 0.177 | 0.177 | -0.000 | -0.051 | 0.054 | 0.986 |
| real | prompt_test | 149 | deduction | density | 3.926 | 3.908 | -0.019 | -0.208 | 0.169 | 0.847 |
| real | prompt_test | 149 | adding-knowledge | density | 1.758 | 1.822 | 0.065 | -0.071 | 0.195 | 0.341 |
| real | prompt_test | 149 | uncertainty-estimation | density | 0.520 | 0.604 | 0.084 | -0.032 | 0.192 | 0.154 |
| real | prompt_test | 149 | backtracking | density | 0.155 | 0.158 | 0.003 | -0.051 | 0.058 | 0.921 |
| real | prompt_test | 149 | deduction | density_testlex | 0.242 | 0.342 | 0.100 | 0.036 | 0.167 | 0.003 |
| real | prompt_test | 149 | adding-knowledge | density_testlex | 0.012 | 0.072 | 0.060 | 0.032 | 0.087 | 0.000 |
| real | prompt_test | 149 | uncertainty-estimation | density_testlex | 0.095 | 0.180 | 0.085 | 0.038 | 0.131 | 0.000 |
| real | prompt_test | 149 | backtracking | density_testlex | 0.003 | 0.005 | 0.002 | -0.006 | 0.012 | 0.593 |
| real | prompt_test | 149 | deduction | density_nontest | 3.684 | 3.566 | -0.119 | -0.318 | 0.078 | 0.242 |
| real | prompt_test | 149 | adding-knowledge | density_nontest | 1.746 | 1.751 | 0.005 | -0.122 | 0.135 | 0.941 |
| real | prompt_test | 149 | uncertainty-estimation | density_nontest | 0.425 | 0.425 | -0.000 | -0.110 | 0.100 | 0.993 |
| real | prompt_test | 149 | backtracking | density_nontest | 0.152 | 0.152 | 0.000 | -0.054 | 0.053 | 0.989 |


## Morphology Δ by compliance class (decision-change confound): `refuse_both` / `comply_both` hold the decision fixed

| framing | arm | flip_class | n | behaviour | metric | delta_density | ci_lo | ci_hi |
|---|---|---|---|---|---|---|---|---|
| fake | prompt_test | gained | 36 | deduction | density | -0.154 | -0.678 | 0.361 |
| fake | prompt_test | gained | 36 | deduction | density_nontest | -0.121 | -0.686 | 0.431 |
| fake | prompt_test | gained | 36 | deduction | density_testlex | -0.033 | -0.153 | 0.092 |
| fake | prompt_test | gained | 36 | uncertainty-estimation | density | -0.048 | -0.285 | 0.200 |
| fake | prompt_test | gained | 36 | uncertainty-estimation | density_nontest | -0.092 | -0.323 | 0.142 |
| fake | prompt_test | gained | 36 | uncertainty-estimation | density_testlex | 0.044 | -0.027 | 0.115 |
| fake | prompt_test | gained | 36 | backtracking | density | 0.012 | -0.105 | 0.126 |
| fake | prompt_test | gained | 36 | backtracking | density_nontest | -0.000 | -0.114 | 0.111 |
| fake | prompt_test | gained | 36 | backtracking | density_testlex | 0.012 | 0.000 | 0.029 |
| fake | prompt_test | lost | 14 | deduction | density | -0.047 | -0.557 | 0.447 |
| fake | prompt_test | lost | 14 | deduction | density_nontest | -0.454 | -1.072 | 0.099 |
| fake | prompt_test | lost | 14 | deduction | density_testlex | 0.407 | 0.178 | 0.658 |
| fake | prompt_test | lost | 14 | uncertainty-estimation | density | -0.268 | -0.664 | 0.166 |
| fake | prompt_test | lost | 14 | uncertainty-estimation | density_nontest | -0.315 | -0.634 | 0.046 |
| fake | prompt_test | lost | 14 | uncertainty-estimation | density_testlex | 0.046 | -0.107 | 0.207 |
| fake | prompt_test | lost | 14 | backtracking | density | 0.009 | -0.226 | 0.206 |
| fake | prompt_test | lost | 14 | backtracking | density_nontest | 0.009 | -0.226 | 0.206 |
| fake | prompt_test | lost | 14 | backtracking | density_testlex | 0.000 | 0.000 | 0.000 |
| fake | prompt_test | comply_both | 47 | deduction | density | 0.203 | -0.245 | 0.685 |
| fake | prompt_test | comply_both | 47 | deduction | density_nontest | 0.136 | -0.312 | 0.607 |
| fake | prompt_test | comply_both | 47 | deduction | density_testlex | 0.067 | -0.008 | 0.146 |
| fake | prompt_test | comply_both | 47 | uncertainty-estimation | density | 0.066 | -0.092 | 0.232 |
| fake | prompt_test | comply_both | 47 | uncertainty-estimation | density_nontest | 0.052 | -0.094 | 0.210 |
| fake | prompt_test | comply_both | 47 | uncertainty-estimation | density_testlex | 0.014 | -0.070 | 0.093 |
| fake | prompt_test | comply_both | 47 | backtracking | density | 0.024 | -0.070 | 0.118 |
| fake | prompt_test | comply_both | 47 | backtracking | density_nontest | 0.019 | -0.071 | 0.108 |
| fake | prompt_test | comply_both | 47 | backtracking | density_testlex | 0.004 | -0.013 | 0.026 |
| fake | prompt_test | refuse_both | 51 | deduction | density | -0.254 | -0.618 | 0.123 |
| fake | prompt_test | refuse_both | 51 | deduction | density_nontest | -0.464 | -0.873 | -0.063 |
| fake | prompt_test | refuse_both | 51 | deduction | density_testlex | 0.210 | 0.076 | 0.355 |
| fake | prompt_test | refuse_both | 51 | uncertainty-estimation | density | 0.108 | -0.142 | 0.368 |
| fake | prompt_test | refuse_both | 51 | uncertainty-estimation | density_nontest | 0.064 | -0.154 | 0.307 |
| fake | prompt_test | refuse_both | 51 | uncertainty-estimation | density_testlex | 0.044 | -0.040 | 0.129 |
| fake | prompt_test | refuse_both | 51 | backtracking | density | -0.006 | -0.076 | 0.065 |
| fake | prompt_test | refuse_both | 51 | backtracking | density_nontest | -0.012 | -0.083 | 0.060 |
| fake | prompt_test | refuse_both | 51 | backtracking | density_testlex | 0.007 | 0.000 | 0.020 |
| real | prompt_test | gained | 23 | deduction | density | -0.160 | -0.599 | 0.322 |
| real | prompt_test | gained | 23 | deduction | density_nontest | -0.091 | -0.556 | 0.415 |
| real | prompt_test | gained | 23 | deduction | density_testlex | -0.069 | -0.159 | 0.015 |
| real | prompt_test | gained | 23 | uncertainty-estimation | density | 0.315 | 0.028 | 0.581 |
| real | prompt_test | gained | 23 | uncertainty-estimation | density_nontest | 0.308 | -0.006 | 0.603 |
| real | prompt_test | gained | 23 | uncertainty-estimation | density_testlex | 0.006 | -0.134 | 0.139 |
| real | prompt_test | gained | 23 | backtracking | density | 0.057 | -0.092 | 0.230 |
| real | prompt_test | gained | 23 | backtracking | density_nontest | 0.051 | -0.096 | 0.225 |
| real | prompt_test | gained | 23 | backtracking | density_testlex | 0.006 | 0.000 | 0.018 |
| real | prompt_test | lost | 19 | deduction | density | -0.088 | -0.750 | 0.543 |
| real | prompt_test | lost | 19 | deduction | density_nontest | -0.171 | -0.824 | 0.457 |
| real | prompt_test | lost | 19 | deduction | density_testlex | 0.083 | -0.034 | 0.196 |
| real | prompt_test | lost | 19 | uncertainty-estimation | density | -0.099 | -0.406 | 0.224 |
| real | prompt_test | lost | 19 | uncertainty-estimation | density_nontest | -0.309 | -0.552 | -0.047 |
| real | prompt_test | lost | 19 | uncertainty-estimation | density_testlex | 0.210 | 0.068 | 0.357 |
| real | prompt_test | lost | 19 | backtracking | density | -0.096 | -0.200 | 0.007 |
| real | prompt_test | lost | 19 | backtracking | density_nontest | -0.094 | -0.184 | -0.016 |
| real | prompt_test | lost | 19 | backtracking | density_testlex | -0.002 | -0.070 | 0.063 |
| real | prompt_test | comply_both | 17 | deduction | density | 0.092 | -0.467 | 0.610 |
| real | prompt_test | comply_both | 17 | deduction | density_nontest | 0.157 | -0.410 | 0.709 |
| real | prompt_test | comply_both | 17 | deduction | density_testlex | -0.064 | -0.187 | 0.059 |
| real | prompt_test | comply_both | 17 | uncertainty-estimation | density | 0.094 | -0.286 | 0.436 |
| real | prompt_test | comply_both | 17 | uncertainty-estimation | density_nontest | 0.093 | -0.273 | 0.428 |
| real | prompt_test | comply_both | 17 | uncertainty-estimation | density_testlex | 0.001 | -0.070 | 0.074 |
| real | prompt_test | comply_both | 17 | backtracking | density | -0.027 | -0.134 | 0.086 |
| real | prompt_test | comply_both | 17 | backtracking | density_nontest | -0.027 | -0.134 | 0.086 |
| real | prompt_test | comply_both | 17 | backtracking | density_testlex | 0.000 | 0.000 | 0.000 |
| real | prompt_test | refuse_both | 90 | deduction | density | 0.011 | -0.225 | 0.250 |
| real | prompt_test | refuse_both | 90 | deduction | density_nontest | -0.167 | -0.422 | 0.081 |
| real | prompt_test | refuse_both | 90 | deduction | density_testlex | 0.178 | 0.090 | 0.275 |
| real | prompt_test | refuse_both | 90 | uncertainty-estimation | density | 0.063 | -0.081 | 0.201 |
| real | prompt_test | refuse_both | 90 | uncertainty-estimation | density_nontest | -0.032 | -0.155 | 0.088 |
| real | prompt_test | refuse_both | 90 | uncertainty-estimation | density_testlex | 0.094 | 0.037 | 0.154 |
| real | prompt_test | refuse_both | 90 | backtracking | density | 0.015 | -0.061 | 0.085 |
| real | prompt_test | refuse_both | 90 | backtracking | density_nontest | 0.012 | -0.063 | 0.082 |
| real | prompt_test | refuse_both | 90 | backtracking | density_testlex | 0.003 | 0.000 | 0.009 |

