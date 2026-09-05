# Mechanistic arm (Qwen3-4B) — first-pass analysis: qwen3_32b_api_prompt


## qwen3_32b_api_prompt: 1788 traces, 1788 with usable morphology annotation

| arm | framing | n | exec_rate_closed | exec_rate_all | lost | gained | p_mcnemar | task_judged_yes | judge_vs_string_agree | verbalization | closure | repetition | words |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | real | 149 | 0.242 | 0.242 | — | — | — | 0.933 | 0.188 | 0.101 | 1.000 | 0.007 | 285.443 |
| baseline | fake | 149 | 0.409 | 0.409 | — | — | — | 0.940 | 0.362 | 0.107 | 1.000 | 0.007 | 293.195 |
| ask_bt_neg | real | 149 | 0.257 | 0.255 | 15.000 | 17.000 | 0.860 | 0.913 | 0.216 | 0.215 | 0.993 | 0.011 | 331.020 |
| ask_bt_neg | fake | 149 | 0.459 | 0.456 | 12.000 | 20.000 | 0.215 | 0.906 | 0.412 | 0.101 | 0.993 | 0.011 | 334.933 |
| ask_bt_pos | real | 149 | 0.215 | 0.215 | 15.000 | 11.000 | 0.557 | 0.913 | 0.168 | 0.161 | 1.000 | 0.007 | 298.483 |
| ask_bt_pos | fake | 149 | 0.376 | 0.376 | 24.000 | 19.000 | 0.542 | 0.946 | 0.322 | 0.134 | 1.000 | 0.009 | 320.611 |
| ask_unc_neg | real | 149 | 0.275 | 0.275 | 15.000 | 20.000 | 0.500 | 0.940 | 0.215 | 0.148 | 1.000 | 0.007 | 304.946 |
| ask_unc_neg | fake | 149 | 0.510 | 0.510 | 11.000 | 26.000 | 0.020 | 0.933 | 0.443 | 0.148 | 1.000 | 0.013 | 348.000 |
| ask_unc_pos | real | 149 | 0.242 | 0.242 | 15.000 | 15.000 | 1.000 | 0.933 | 0.215 | 0.174 | 1.000 | 0.010 | 323.329 |
| ask_unc_pos | fake | 149 | 0.450 | 0.450 | 13.000 | 19.000 | 0.377 | 0.899 | 0.349 | 0.148 | 1.000 | 0.015 | 360.081 |
| prompt_test | real | 149 | 0.268 | 0.268 | 19.000 | 23.000 | 0.644 | 0.913 | 0.195 | 0.228 | 1.000 | 0.007 | 285.591 |
| prompt_test | fake | 149 | 0.561 | 0.557 | 14.000 | 36.000 | 0.003 | 0.926 | 0.507 | 0.148 | 0.993 | 0.013 | 367.181 |


## Morphology: Δ spans per 100 words vs baseline, paired by item

| framing | arm | n_pairs | behaviour | metric | base_mean | arm_mean | delta | ci_lo | ci_hi | p_paired_t |
|---|---|---|---|---|---|---|---|---|---|---|
| fake | ask_bt_neg | 149 | deduction | density | 4.017 | 3.991 | -0.026 | -0.240 | 0.191 | 0.817 |
| fake | ask_bt_neg | 149 | adding-knowledge | density | 1.772 | 1.709 | -0.063 | -0.213 | 0.094 | 0.416 |
| fake | ask_bt_neg | 149 | uncertainty-estimation | density | 0.675 | 0.714 | 0.039 | -0.066 | 0.149 | 0.487 |
| fake | ask_bt_neg | 149 | backtracking | density | 0.179 | 0.175 | -0.004 | -0.060 | 0.053 | 0.885 |
| fake | ask_bt_neg | 149 | deduction | density_testlex | 0.230 | 0.174 | -0.056 | -0.106 | -0.004 | 0.045 |
| fake | ask_bt_neg | 149 | adding-knowledge | density_testlex | 0.030 | 0.018 | -0.011 | -0.037 | 0.013 | 0.377 |
| fake | ask_bt_neg | 149 | uncertainty-estimation | density_testlex | 0.108 | 0.118 | 0.011 | -0.033 | 0.055 | 0.650 |
| fake | ask_bt_neg | 149 | backtracking | density_testlex | 0.002 | 0.005 | 0.003 | -0.003 | 0.012 | 0.459 |
| fake | ask_bt_neg | 149 | deduction | density_nontest | 3.787 | 3.817 | 0.030 | -0.189 | 0.254 | 0.791 |
| fake | ask_bt_neg | 149 | adding-knowledge | density_nontest | 1.742 | 1.690 | -0.052 | -0.201 | 0.106 | 0.501 |
| fake | ask_bt_neg | 149 | uncertainty-estimation | density_nontest | 0.568 | 0.596 | 0.028 | -0.080 | 0.139 | 0.607 |
| fake | ask_bt_neg | 149 | backtracking | density_nontest | 0.177 | 0.170 | -0.007 | -0.063 | 0.047 | 0.804 |
| fake | ask_bt_pos | 149 | deduction | density | 4.017 | 3.886 | -0.130 | -0.356 | 0.105 | 0.285 |
| fake | ask_bt_pos | 149 | adding-knowledge | density | 1.772 | 1.695 | -0.077 | -0.239 | 0.083 | 0.353 |
| fake | ask_bt_pos | 149 | uncertainty-estimation | density | 0.675 | 0.765 | 0.089 | -0.028 | 0.199 | 0.131 |
| fake | ask_bt_pos | 149 | backtracking | density | 0.179 | 0.189 | 0.010 | -0.046 | 0.066 | 0.741 |
| fake | ask_bt_pos | 149 | deduction | density_testlex | 0.230 | 0.237 | 0.007 | -0.050 | 0.068 | 0.808 |
| fake | ask_bt_pos | 149 | adding-knowledge | density_testlex | 0.030 | 0.032 | 0.003 | -0.022 | 0.026 | 0.843 |
| fake | ask_bt_pos | 149 | uncertainty-estimation | density_testlex | 0.108 | 0.138 | 0.031 | -0.017 | 0.078 | 0.209 |
| fake | ask_bt_pos | 149 | backtracking | density_testlex | 0.002 | 0.006 | 0.004 | -0.003 | 0.013 | 0.290 |
| fake | ask_bt_pos | 149 | deduction | density_nontest | 3.787 | 3.649 | -0.138 | -0.366 | 0.098 | 0.258 |
| fake | ask_bt_pos | 149 | adding-knowledge | density_nontest | 1.742 | 1.662 | -0.080 | -0.242 | 0.078 | 0.339 |
| fake | ask_bt_pos | 149 | uncertainty-estimation | density_nontest | 0.568 | 0.627 | 0.059 | -0.051 | 0.166 | 0.301 |
| fake | ask_bt_pos | 149 | backtracking | density_nontest | 0.177 | 0.183 | 0.006 | -0.051 | 0.062 | 0.843 |
| fake | ask_unc_neg | 149 | deduction | density | 4.017 | 4.013 | -0.004 | -0.220 | 0.207 | 0.974 |
| fake | ask_unc_neg | 149 | adding-knowledge | density | 1.772 | 1.678 | -0.094 | -0.253 | 0.060 | 0.258 |
| fake | ask_unc_neg | 149 | uncertainty-estimation | density | 0.675 | 0.684 | 0.008 | -0.110 | 0.128 | 0.892 |
| fake | ask_unc_neg | 149 | backtracking | density | 0.179 | 0.150 | -0.029 | -0.088 | 0.031 | 0.319 |
| fake | ask_unc_neg | 149 | deduction | density_testlex | 0.230 | 0.237 | 0.007 | -0.049 | 0.064 | 0.799 |
| fake | ask_unc_neg | 149 | adding-knowledge | density_testlex | 0.030 | 0.010 | -0.019 | -0.042 | -0.000 | 0.071 |
| fake | ask_unc_neg | 149 | uncertainty-estimation | density_testlex | 0.108 | 0.141 | 0.033 | -0.010 | 0.078 | 0.148 |
| fake | ask_unc_neg | 149 | backtracking | density_testlex | 0.002 | 0.004 | 0.002 | -0.003 | 0.009 | 0.592 |
| fake | ask_unc_neg | 149 | deduction | density_nontest | 3.787 | 3.776 | -0.011 | -0.229 | 0.193 | 0.923 |
| fake | ask_unc_neg | 149 | adding-knowledge | density_nontest | 1.742 | 1.668 | -0.074 | -0.229 | 0.085 | 0.366 |
| fake | ask_unc_neg | 149 | uncertainty-estimation | density_nontest | 0.568 | 0.543 | -0.025 | -0.137 | 0.087 | 0.658 |
| fake | ask_unc_neg | 149 | backtracking | density_nontest | 0.177 | 0.147 | -0.031 | -0.090 | 0.029 | 0.288 |
| fake | ask_unc_pos | 149 | deduction | density | 4.017 | 4.089 | 0.072 | -0.158 | 0.303 | 0.547 |
| fake | ask_unc_pos | 149 | adding-knowledge | density | 1.772 | 1.583 | -0.189 | -0.344 | -0.021 | 0.023 |
| fake | ask_unc_pos | 149 | uncertainty-estimation | density | 0.675 | 0.719 | 0.044 | -0.072 | 0.159 | 0.456 |
| fake | ask_unc_pos | 149 | backtracking | density | 0.179 | 0.158 | -0.021 | -0.079 | 0.031 | 0.446 |
| fake | ask_unc_pos | 149 | deduction | density_testlex | 0.230 | 0.254 | 0.024 | -0.035 | 0.086 | 0.437 |
| fake | ask_unc_pos | 149 | adding-knowledge | density_testlex | 0.030 | 0.028 | -0.002 | -0.029 | 0.021 | 0.881 |
| fake | ask_unc_pos | 149 | uncertainty-estimation | density_testlex | 0.108 | 0.121 | 0.013 | -0.030 | 0.054 | 0.554 |
| fake | ask_unc_pos | 149 | backtracking | density_testlex | 0.002 | 0.004 | 0.002 | -0.005 | 0.012 | 0.598 |
| fake | ask_unc_pos | 149 | deduction | density_nontest | 3.787 | 3.835 | 0.048 | -0.182 | 0.281 | 0.692 |
| fake | ask_unc_pos | 149 | adding-knowledge | density_nontest | 1.742 | 1.555 | -0.187 | -0.338 | -0.020 | 0.023 |
| fake | ask_unc_pos | 149 | uncertainty-estimation | density_nontest | 0.568 | 0.598 | 0.031 | -0.076 | 0.138 | 0.581 |
| fake | ask_unc_pos | 149 | backtracking | density_nontest | 0.177 | 0.154 | -0.024 | -0.080 | 0.028 | 0.392 |
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
| real | ask_bt_neg | 149 | deduction | density | 3.926 | 4.027 | 0.101 | -0.116 | 0.312 | 0.343 |
| real | ask_bt_neg | 149 | adding-knowledge | density | 1.758 | 1.740 | -0.018 | -0.155 | 0.141 | 0.813 |
| real | ask_bt_neg | 149 | uncertainty-estimation | density | 0.520 | 0.617 | 0.097 | -0.016 | 0.206 | 0.091 |
| real | ask_bt_neg | 149 | backtracking | density | 0.155 | 0.192 | 0.037 | -0.017 | 0.092 | 0.191 |
| real | ask_bt_neg | 149 | deduction | density_testlex | 0.242 | 0.200 | -0.042 | -0.095 | 0.012 | 0.127 |
| real | ask_bt_neg | 149 | adding-knowledge | density_testlex | 0.012 | 0.011 | -0.001 | -0.015 | 0.012 | 0.870 |
| real | ask_bt_neg | 149 | uncertainty-estimation | density_testlex | 0.095 | 0.142 | 0.047 | 0.002 | 0.092 | 0.040 |
| real | ask_bt_neg | 149 | backtracking | density_testlex | 0.003 | 0.006 | 0.003 | -0.006 | 0.013 | 0.467 |
| real | ask_bt_neg | 149 | deduction | density_nontest | 3.684 | 3.827 | 0.143 | -0.080 | 0.370 | 0.194 |
| real | ask_bt_neg | 149 | adding-knowledge | density_nontest | 1.746 | 1.729 | -0.017 | -0.156 | 0.142 | 0.825 |
| real | ask_bt_neg | 149 | uncertainty-estimation | density_nontest | 0.425 | 0.475 | 0.050 | -0.059 | 0.155 | 0.362 |
| real | ask_bt_neg | 149 | backtracking | density_nontest | 0.152 | 0.186 | 0.034 | -0.021 | 0.088 | 0.232 |
| real | ask_bt_pos | 149 | deduction | density | 3.926 | 3.837 | -0.090 | -0.286 | 0.109 | 0.373 |
| real | ask_bt_pos | 149 | adding-knowledge | density | 1.758 | 1.782 | 0.024 | -0.112 | 0.154 | 0.712 |
| real | ask_bt_pos | 149 | uncertainty-estimation | density | 0.520 | 0.605 | 0.085 | -0.022 | 0.190 | 0.126 |
| real | ask_bt_pos | 149 | backtracking | density | 0.155 | 0.132 | -0.023 | -0.074 | 0.026 | 0.363 |
| real | ask_bt_pos | 149 | deduction | density_testlex | 0.242 | 0.231 | -0.011 | -0.062 | 0.045 | 0.678 |
| real | ask_bt_pos | 149 | adding-knowledge | density_testlex | 0.012 | 0.007 | -0.005 | -0.019 | 0.009 | 0.518 |
| real | ask_bt_pos | 149 | uncertainty-estimation | density_testlex | 0.095 | 0.112 | 0.017 | -0.024 | 0.058 | 0.419 |
| real | ask_bt_pos | 149 | backtracking | density_testlex | 0.003 | 0.006 | 0.003 | -0.006 | 0.013 | 0.507 |
| real | ask_bt_pos | 149 | deduction | density_nontest | 3.684 | 3.606 | -0.079 | -0.289 | 0.131 | 0.450 |
| real | ask_bt_pos | 149 | adding-knowledge | density_nontest | 1.746 | 1.775 | 0.029 | -0.105 | 0.158 | 0.658 |
| real | ask_bt_pos | 149 | uncertainty-estimation | density_nontest | 0.425 | 0.493 | 0.068 | -0.034 | 0.165 | 0.199 |
| real | ask_bt_pos | 149 | backtracking | density_nontest | 0.152 | 0.126 | -0.026 | -0.077 | 0.021 | 0.292 |
| real | ask_unc_neg | 149 | deduction | density | 3.926 | 3.888 | -0.039 | -0.243 | 0.171 | 0.722 |
| real | ask_unc_neg | 149 | adding-knowledge | density | 1.758 | 1.682 | -0.076 | -0.208 | 0.057 | 0.257 |
| real | ask_unc_neg | 149 | uncertainty-estimation | density | 0.520 | 0.646 | 0.126 | 0.009 | 0.234 | 0.026 |
| real | ask_unc_neg | 149 | backtracking | density | 0.155 | 0.152 | -0.003 | -0.058 | 0.052 | 0.903 |
| real | ask_unc_neg | 149 | deduction | density_testlex | 0.242 | 0.228 | -0.014 | -0.065 | 0.035 | 0.596 |
| real | ask_unc_neg | 149 | adding-knowledge | density_testlex | 0.012 | 0.014 | 0.002 | -0.014 | 0.018 | 0.821 |
| real | ask_unc_neg | 149 | uncertainty-estimation | density_testlex | 0.095 | 0.092 | -0.003 | -0.041 | 0.035 | 0.883 |
| real | ask_unc_neg | 149 | backtracking | density_testlex | 0.003 | 0.002 | -0.001 | -0.009 | 0.005 | 0.683 |
| real | ask_unc_neg | 149 | deduction | density_nontest | 3.684 | 3.660 | -0.025 | -0.238 | 0.196 | 0.827 |
| real | ask_unc_neg | 149 | adding-knowledge | density_nontest | 1.746 | 1.668 | -0.078 | -0.208 | 0.052 | 0.249 |
| real | ask_unc_neg | 149 | uncertainty-estimation | density_nontest | 0.425 | 0.554 | 0.129 | 0.018 | 0.237 | 0.021 |
| real | ask_unc_neg | 149 | backtracking | density_nontest | 0.152 | 0.150 | -0.002 | -0.056 | 0.053 | 0.940 |
| real | ask_unc_pos | 149 | deduction | density | 3.926 | 3.958 | 0.032 | -0.154 | 0.213 | 0.737 |
| real | ask_unc_pos | 149 | adding-knowledge | density | 1.758 | 1.668 | -0.089 | -0.226 | 0.051 | 0.215 |
| real | ask_unc_pos | 149 | uncertainty-estimation | density | 0.520 | 0.627 | 0.107 | -0.004 | 0.221 | 0.058 |
| real | ask_unc_pos | 149 | backtracking | density | 0.155 | 0.167 | 0.012 | -0.041 | 0.065 | 0.653 |
| real | ask_unc_pos | 149 | deduction | density_testlex | 0.242 | 0.258 | 0.016 | -0.033 | 0.065 | 0.514 |
| real | ask_unc_pos | 149 | adding-knowledge | density_testlex | 0.012 | 0.009 | -0.003 | -0.018 | 0.012 | 0.670 |
| real | ask_unc_pos | 149 | uncertainty-estimation | density_testlex | 0.095 | 0.081 | -0.014 | -0.053 | 0.023 | 0.471 |
| real | ask_unc_pos | 149 | backtracking | density_testlex | 0.003 | 0.005 | 0.002 | -0.006 | 0.010 | 0.660 |
| real | ask_unc_pos | 149 | deduction | density_nontest | 3.684 | 3.700 | 0.016 | -0.177 | 0.210 | 0.875 |
| real | ask_unc_pos | 149 | adding-knowledge | density_nontest | 1.746 | 1.660 | -0.086 | -0.219 | 0.055 | 0.224 |
| real | ask_unc_pos | 149 | uncertainty-estimation | density_nontest | 0.425 | 0.546 | 0.121 | 0.013 | 0.234 | 0.029 |
| real | ask_unc_pos | 149 | backtracking | density_nontest | 0.152 | 0.162 | 0.010 | -0.044 | 0.061 | 0.705 |
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
| fake | ask_bt_neg | gained | 20 | deduction | density | 0.109 | -0.418 | 0.655 |
| fake | ask_bt_neg | gained | 20 | deduction | density_nontest | 0.304 | -0.196 | 0.823 |
| fake | ask_bt_neg | gained | 20 | deduction | density_testlex | -0.195 | -0.307 | -0.088 |
| fake | ask_bt_neg | gained | 20 | uncertainty-estimation | density | -0.037 | -0.389 | 0.323 |
| fake | ask_bt_neg | gained | 20 | uncertainty-estimation | density_nontest | -0.042 | -0.442 | 0.362 |
| fake | ask_bt_neg | gained | 20 | uncertainty-estimation | density_testlex | 0.005 | -0.107 | 0.115 |
| fake | ask_bt_neg | gained | 20 | backtracking | density | -0.009 | -0.164 | 0.138 |
| fake | ask_bt_neg | gained | 20 | backtracking | density_nontest | -0.009 | -0.164 | 0.138 |
| fake | ask_bt_neg | gained | 20 | backtracking | density_testlex | 0.000 | 0.000 | 0.000 |
| fake | ask_bt_neg | lost | 12 | deduction | density | 0.064 | -0.724 | 0.818 |
| fake | ask_bt_neg | lost | 12 | deduction | density_nontest | 0.080 | -0.800 | 0.929 |
| fake | ask_bt_neg | lost | 12 | deduction | density_testlex | -0.016 | -0.162 | 0.146 |
| fake | ask_bt_neg | lost | 12 | uncertainty-estimation | density | -0.017 | -0.464 | 0.375 |
| fake | ask_bt_neg | lost | 12 | uncertainty-estimation | density_nontest | -0.029 | -0.385 | 0.315 |
| fake | ask_bt_neg | lost | 12 | uncertainty-estimation | density_testlex | 0.012 | -0.119 | 0.147 |
| fake | ask_bt_neg | lost | 12 | backtracking | density | -0.204 | -0.401 | -0.018 |
| fake | ask_bt_neg | lost | 12 | backtracking | density_nontest | -0.204 | -0.401 | -0.018 |
| fake | ask_bt_neg | lost | 12 | backtracking | density_testlex | 0.000 | 0.000 | 0.000 |
| fake | ask_bt_neg | comply_both | 48 | deduction | density | -0.028 | -0.454 | 0.407 |
| fake | ask_bt_neg | comply_both | 48 | deduction | density_nontest | -0.016 | -0.436 | 0.407 |
| fake | ask_bt_neg | comply_both | 48 | deduction | density_testlex | -0.013 | -0.092 | 0.070 |
| fake | ask_bt_neg | comply_both | 48 | uncertainty-estimation | density | 0.042 | -0.142 | 0.231 |
| fake | ask_bt_neg | comply_both | 48 | uncertainty-estimation | density_nontest | 0.064 | -0.119 | 0.248 |
| fake | ask_bt_neg | comply_both | 48 | uncertainty-estimation | density_testlex | -0.022 | -0.099 | 0.048 |
| fake | ask_bt_neg | comply_both | 48 | backtracking | density | 0.016 | -0.083 | 0.113 |
| fake | ask_bt_neg | comply_both | 48 | backtracking | density_nontest | 0.022 | -0.078 | 0.120 |
| fake | ask_bt_neg | comply_both | 48 | backtracking | density_testlex | -0.005 | -0.016 | 0.000 |
| fake | ask_bt_neg | refuse_both | 68 | deduction | density | -0.090 | -0.379 | 0.196 |
| fake | ask_bt_neg | refuse_both | 68 | deduction | density_nontest | -0.037 | -0.348 | 0.278 |
| fake | ask_bt_neg | refuse_both | 68 | deduction | density_testlex | -0.053 | -0.149 | 0.033 |
| fake | ask_bt_neg | refuse_both | 68 | uncertainty-estimation | density | 0.076 | -0.081 | 0.225 |
| fake | ask_bt_neg | refuse_both | 68 | uncertainty-estimation | density_nontest | 0.044 | -0.099 | 0.184 |
| fake | ask_bt_neg | refuse_both | 68 | uncertainty-estimation | density_testlex | 0.032 | -0.043 | 0.110 |
| fake | ask_bt_neg | refuse_both | 68 | backtracking | density | 0.009 | -0.067 | 0.089 |
| fake | ask_bt_neg | refuse_both | 68 | backtracking | density_nontest | 0.002 | -0.076 | 0.083 |
| fake | ask_bt_neg | refuse_both | 68 | backtracking | density_testlex | 0.007 | 0.000 | 0.022 |
| fake | ask_bt_pos | gained | 19 | deduction | density | -0.169 | -0.897 | 0.623 |
| fake | ask_bt_pos | gained | 19 | deduction | density_nontest | -0.102 | -0.863 | 0.688 |
| fake | ask_bt_pos | gained | 19 | deduction | density_testlex | -0.067 | -0.217 | 0.083 |
| fake | ask_bt_pos | gained | 19 | uncertainty-estimation | density | 0.097 | -0.189 | 0.406 |
| fake | ask_bt_pos | gained | 19 | uncertainty-estimation | density_nontest | 0.094 | -0.236 | 0.419 |
| fake | ask_bt_pos | gained | 19 | uncertainty-estimation | density_testlex | 0.003 | -0.133 | 0.132 |
| fake | ask_bt_pos | gained | 19 | backtracking | density | -0.011 | -0.151 | 0.126 |
| fake | ask_bt_pos | gained | 19 | backtracking | density_nontest | -0.027 | -0.161 | 0.104 |
| fake | ask_bt_pos | gained | 19 | backtracking | density_testlex | 0.016 | 0.000 | 0.047 |
| fake | ask_bt_pos | lost | 24 | deduction | density | 0.017 | -0.584 | 0.601 |
| fake | ask_bt_pos | lost | 24 | deduction | density_nontest | -0.132 | -0.728 | 0.505 |
| fake | ask_bt_pos | lost | 24 | deduction | density_testlex | 0.149 | -0.001 | 0.308 |
| fake | ask_bt_pos | lost | 24 | uncertainty-estimation | density | 0.124 | -0.225 | 0.448 |
| fake | ask_bt_pos | lost | 24 | uncertainty-estimation | density_nontest | 0.118 | -0.202 | 0.421 |
| fake | ask_bt_pos | lost | 24 | uncertainty-estimation | density_testlex | 0.006 | -0.122 | 0.149 |
| fake | ask_bt_pos | lost | 24 | backtracking | density | -0.039 | -0.222 | 0.133 |
| fake | ask_bt_pos | lost | 24 | backtracking | density_nontest | -0.052 | -0.232 | 0.120 |
| fake | ask_bt_pos | lost | 24 | backtracking | density_testlex | 0.013 | -0.021 | 0.052 |
| fake | ask_bt_pos | comply_both | 37 | deduction | density | -0.199 | -0.744 | 0.311 |
| fake | ask_bt_pos | comply_both | 37 | deduction | density_nontest | -0.229 | -0.783 | 0.292 |
| fake | ask_bt_pos | comply_both | 37 | deduction | density_testlex | 0.030 | -0.068 | 0.124 |
| fake | ask_bt_pos | comply_both | 37 | uncertainty-estimation | density | 0.106 | -0.068 | 0.270 |
| fake | ask_bt_pos | comply_both | 37 | uncertainty-estimation | density_nontest | 0.060 | -0.123 | 0.230 |
| fake | ask_bt_pos | comply_both | 37 | uncertainty-estimation | density_testlex | 0.046 | -0.031 | 0.123 |
| fake | ask_bt_pos | comply_both | 37 | backtracking | density | 0.013 | -0.098 | 0.130 |
| fake | ask_bt_pos | comply_both | 37 | backtracking | density_nontest | 0.013 | -0.098 | 0.130 |
| fake | ask_bt_pos | comply_both | 37 | backtracking | density_testlex | 0.000 | 0.000 | 0.000 |
| fake | ask_bt_pos | refuse_both | 69 | deduction | density | -0.134 | -0.437 | 0.171 |
| fake | ask_bt_pos | refuse_both | 69 | deduction | density_nontest | -0.101 | -0.400 | 0.199 |
| fake | ask_bt_pos | refuse_both | 69 | deduction | density_testlex | -0.033 | -0.125 | 0.063 |
| fake | ask_bt_pos | refuse_both | 69 | uncertainty-estimation | density | 0.066 | -0.113 | 0.242 |
| fake | ask_bt_pos | refuse_both | 69 | uncertainty-estimation | density_nontest | 0.028 | -0.140 | 0.191 |
| fake | ask_bt_pos | refuse_both | 69 | uncertainty-estimation | density_testlex | 0.038 | -0.027 | 0.110 |
| fake | ask_bt_pos | refuse_both | 69 | backtracking | density | 0.031 | -0.048 | 0.113 |
| fake | ask_bt_pos | refuse_both | 69 | backtracking | density_nontest | 0.031 | -0.048 | 0.113 |
| fake | ask_bt_pos | refuse_both | 69 | backtracking | density_testlex | 0.000 | 0.000 | 0.000 |
| fake | ask_unc_neg | gained | 26 | deduction | density | 0.045 | -0.457 | 0.559 |
| fake | ask_unc_neg | gained | 26 | deduction | density_nontest | 0.070 | -0.468 | 0.625 |
| fake | ask_unc_neg | gained | 26 | deduction | density_testlex | -0.025 | -0.167 | 0.119 |
| fake | ask_unc_neg | gained | 26 | uncertainty-estimation | density | 0.007 | -0.339 | 0.371 |
| fake | ask_unc_neg | gained | 26 | uncertainty-estimation | density_nontest | -0.017 | -0.351 | 0.310 |
| fake | ask_unc_neg | gained | 26 | uncertainty-estimation | density_testlex | 0.024 | -0.089 | 0.136 |
| fake | ask_unc_neg | gained | 26 | backtracking | density | 0.064 | -0.056 | 0.194 |
| fake | ask_unc_neg | gained | 26 | backtracking | density_nontest | 0.064 | -0.056 | 0.194 |
| fake | ask_unc_neg | gained | 26 | backtracking | density_testlex | 0.000 | 0.000 | 0.000 |
| fake | ask_unc_neg | lost | 11 | deduction | density | 0.147 | -0.355 | 0.671 |
| fake | ask_unc_neg | lost | 11 | deduction | density_nontest | 0.035 | -0.507 | 0.585 |
| fake | ask_unc_neg | lost | 11 | deduction | density_testlex | 0.112 | -0.057 | 0.293 |
| fake | ask_unc_neg | lost | 11 | uncertainty-estimation | density | -0.288 | -0.734 | 0.182 |
| fake | ask_unc_neg | lost | 11 | uncertainty-estimation | density_nontest | -0.321 | -0.687 | 0.066 |
| fake | ask_unc_neg | lost | 11 | uncertainty-estimation | density_testlex | 0.034 | -0.202 | 0.267 |
| fake | ask_unc_neg | lost | 11 | backtracking | density | -0.200 | -0.491 | 0.096 |
| fake | ask_unc_neg | lost | 11 | backtracking | density_nontest | -0.200 | -0.491 | 0.096 |
| fake | ask_unc_neg | lost | 11 | backtracking | density_testlex | 0.000 | 0.000 | 0.000 |
| fake | ask_unc_neg | comply_both | 50 | deduction | density | 0.116 | -0.299 | 0.531 |
| fake | ask_unc_neg | comply_both | 50 | deduction | density_nontest | 0.117 | -0.301 | 0.550 |
| fake | ask_unc_neg | comply_both | 50 | deduction | density_testlex | -0.001 | -0.086 | 0.076 |
| fake | ask_unc_neg | comply_both | 50 | uncertainty-estimation | density | -0.020 | -0.175 | 0.127 |
| fake | ask_unc_neg | comply_both | 50 | uncertainty-estimation | density_nontest | -0.034 | -0.202 | 0.116 |
| fake | ask_unc_neg | comply_both | 50 | uncertainty-estimation | density_testlex | 0.014 | -0.055 | 0.080 |
| fake | ask_unc_neg | comply_both | 50 | backtracking | density | -0.054 | -0.144 | 0.037 |
| fake | ask_unc_neg | comply_both | 50 | backtracking | density_nontest | -0.051 | -0.139 | 0.036 |
| fake | ask_unc_neg | comply_both | 50 | backtracking | density_testlex | -0.003 | -0.015 | 0.006 |
| fake | ask_unc_neg | refuse_both | 62 | deduction | density | -0.148 | -0.445 | 0.168 |
| fake | ask_unc_neg | refuse_both | 62 | deduction | density_nontest | -0.157 | -0.455 | 0.159 |
| fake | ask_unc_neg | refuse_both | 62 | deduction | density_testlex | 0.009 | -0.085 | 0.112 |
| fake | ask_unc_neg | refuse_both | 62 | uncertainty-estimation | density | 0.084 | -0.095 | 0.256 |
| fake | ask_unc_neg | refuse_both | 62 | uncertainty-estimation | density_nontest | 0.031 | -0.125 | 0.188 |
| fake | ask_unc_neg | refuse_both | 62 | uncertainty-estimation | density_testlex | 0.053 | -0.016 | 0.115 |
| fake | ask_unc_neg | refuse_both | 62 | backtracking | density | -0.017 | -0.104 | 0.067 |
| fake | ask_unc_neg | refuse_both | 62 | backtracking | density_nontest | -0.024 | -0.111 | 0.061 |
| fake | ask_unc_neg | refuse_both | 62 | backtracking | density_testlex | 0.007 | 0.000 | 0.021 |
| fake | ask_unc_pos | gained | 19 | deduction | density | 0.309 | -0.234 | 0.863 |
| fake | ask_unc_pos | gained | 19 | deduction | density_nontest | 0.372 | -0.162 | 0.943 |
| fake | ask_unc_pos | gained | 19 | deduction | density_testlex | -0.063 | -0.253 | 0.122 |
| fake | ask_unc_pos | gained | 19 | uncertainty-estimation | density | 0.249 | -0.135 | 0.614 |
| fake | ask_unc_pos | gained | 19 | uncertainty-estimation | density_nontest | 0.231 | -0.124 | 0.562 |
| fake | ask_unc_pos | gained | 19 | uncertainty-estimation | density_testlex | 0.018 | -0.086 | 0.107 |
| fake | ask_unc_pos | gained | 19 | backtracking | density | 0.036 | -0.107 | 0.181 |
| fake | ask_unc_pos | gained | 19 | backtracking | density_nontest | 0.036 | -0.107 | 0.181 |
| fake | ask_unc_pos | gained | 19 | backtracking | density_testlex | 0.000 | 0.000 | 0.000 |
| fake | ask_unc_pos | lost | 13 | deduction | density | 0.367 | -0.354 | 1.078 |
| fake | ask_unc_pos | lost | 13 | deduction | density_nontest | 0.126 | -0.593 | 0.814 |
| fake | ask_unc_pos | lost | 13 | deduction | density_testlex | 0.241 | 0.120 | 0.374 |
| fake | ask_unc_pos | lost | 13 | uncertainty-estimation | density | -0.184 | -0.540 | 0.179 |
| fake | ask_unc_pos | lost | 13 | uncertainty-estimation | density_nontest | -0.169 | -0.510 | 0.183 |
| fake | ask_unc_pos | lost | 13 | uncertainty-estimation | density_testlex | -0.015 | -0.185 | 0.151 |
| fake | ask_unc_pos | lost | 13 | backtracking | density | -0.188 | -0.419 | 0.045 |
| fake | ask_unc_pos | lost | 13 | backtracking | density_nontest | -0.188 | -0.419 | 0.045 |
| fake | ask_unc_pos | lost | 13 | backtracking | density_testlex | 0.000 | 0.000 | 0.000 |
| fake | ask_unc_pos | comply_both | 48 | deduction | density | -0.065 | -0.562 | 0.443 |
| fake | ask_unc_pos | comply_both | 48 | deduction | density_nontest | -0.046 | -0.554 | 0.463 |
| fake | ask_unc_pos | comply_both | 48 | deduction | density_testlex | -0.019 | -0.096 | 0.056 |
| fake | ask_unc_pos | comply_both | 48 | uncertainty-estimation | density | 0.154 | 0.005 | 0.317 |
| fake | ask_unc_pos | comply_both | 48 | uncertainty-estimation | density_nontest | 0.129 | -0.034 | 0.300 |
| fake | ask_unc_pos | comply_both | 48 | uncertainty-estimation | density_testlex | 0.026 | -0.035 | 0.080 |
| fake | ask_unc_pos | comply_both | 48 | backtracking | density | 0.028 | -0.069 | 0.128 |
| fake | ask_unc_pos | comply_both | 48 | backtracking | density_nontest | 0.033 | -0.063 | 0.133 |
| fake | ask_unc_pos | comply_both | 48 | backtracking | density_testlex | -0.005 | -0.016 | 0.000 |
| fake | ask_unc_pos | refuse_both | 69 | deduction | density | 0.047 | -0.245 | 0.355 |
| fake | ask_unc_pos | refuse_both | 69 | deduction | density_nontest | 0.009 | -0.288 | 0.313 |
| fake | ask_unc_pos | refuse_both | 69 | deduction | density_testlex | 0.038 | -0.059 | 0.140 |
| fake | ask_unc_pos | refuse_both | 69 | uncertainty-estimation | density | -0.047 | -0.232 | 0.126 |
| fake | ask_unc_pos | refuse_both | 69 | uncertainty-estimation | density_nontest | -0.055 | -0.235 | 0.106 |
| fake | ask_unc_pos | refuse_both | 69 | uncertainty-estimation | density_testlex | 0.008 | -0.061 | 0.080 |
| fake | ask_unc_pos | refuse_both | 69 | backtracking | density | -0.040 | -0.109 | 0.033 |
| fake | ask_unc_pos | refuse_both | 69 | backtracking | density_nontest | -0.049 | -0.115 | 0.022 |
| fake | ask_unc_pos | refuse_both | 69 | backtracking | density_testlex | 0.009 | 0.000 | 0.026 |
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
| real | ask_bt_neg | gained | 17 | deduction | density | 0.809 | 0.308 | 1.320 |
| real | ask_bt_neg | gained | 17 | deduction | density_nontest | 1.021 | 0.484 | 1.577 |
| real | ask_bt_neg | gained | 17 | deduction | density_testlex | -0.212 | -0.316 | -0.116 |
| real | ask_bt_neg | gained | 17 | uncertainty-estimation | density | 0.214 | -0.024 | 0.478 |
| real | ask_bt_neg | gained | 17 | uncertainty-estimation | density_nontest | 0.151 | -0.113 | 0.414 |
| real | ask_bt_neg | gained | 17 | uncertainty-estimation | density_testlex | 0.064 | -0.036 | 0.160 |
| real | ask_bt_neg | gained | 17 | backtracking | density | 0.054 | -0.086 | 0.177 |
| real | ask_bt_neg | gained | 17 | backtracking | density_nontest | 0.054 | -0.086 | 0.177 |
| real | ask_bt_neg | gained | 17 | backtracking | density_testlex | 0.000 | 0.000 | 0.000 |
| real | ask_bt_neg | lost | 15 | deduction | density | -0.322 | -0.975 | 0.355 |
| real | ask_bt_neg | lost | 15 | deduction | density_nontest | -0.380 | -1.022 | 0.312 |
| real | ask_bt_neg | lost | 15 | deduction | density_testlex | 0.058 | -0.106 | 0.220 |
| real | ask_bt_neg | lost | 15 | uncertainty-estimation | density | -0.229 | -0.647 | 0.202 |
| real | ask_bt_neg | lost | 15 | uncertainty-estimation | density_nontest | -0.336 | -0.737 | 0.070 |
| real | ask_bt_neg | lost | 15 | uncertainty-estimation | density_testlex | 0.106 | -0.035 | 0.241 |
| real | ask_bt_neg | lost | 15 | backtracking | density | 0.065 | -0.071 | 0.214 |
| real | ask_bt_neg | lost | 15 | backtracking | density_nontest | 0.053 | -0.072 | 0.183 |
| real | ask_bt_neg | lost | 15 | backtracking | density_testlex | 0.012 | -0.070 | 0.088 |
| real | ask_bt_neg | comply_both | 21 | deduction | density | 0.006 | -0.644 | 0.699 |
| real | ask_bt_neg | comply_both | 21 | deduction | density_nontest | 0.084 | -0.612 | 0.820 |
| real | ask_bt_neg | comply_both | 21 | deduction | density_testlex | -0.078 | -0.200 | 0.053 |
| real | ask_bt_neg | comply_both | 21 | uncertainty-estimation | density | -0.008 | -0.193 | 0.172 |
| real | ask_bt_neg | comply_both | 21 | uncertainty-estimation | density_nontest | -0.027 | -0.231 | 0.179 |
| real | ask_bt_neg | comply_both | 21 | uncertainty-estimation | density_testlex | 0.020 | -0.094 | 0.136 |
| real | ask_bt_neg | comply_both | 21 | backtracking | density | -0.023 | -0.176 | 0.135 |
| real | ask_bt_neg | comply_both | 21 | backtracking | density_nontest | -0.023 | -0.176 | 0.135 |
| real | ask_bt_neg | comply_both | 21 | backtracking | density_testlex | 0.000 | 0.000 | 0.000 |
| real | ask_bt_neg | refuse_both | 95 | deduction | density | 0.060 | -0.194 | 0.313 |
| real | ask_bt_neg | refuse_both | 95 | deduction | density_nontest | 0.075 | -0.183 | 0.323 |
| real | ask_bt_neg | refuse_both | 95 | deduction | density_testlex | -0.015 | -0.084 | 0.052 |
| real | ask_bt_neg | refuse_both | 95 | uncertainty-estimation | density | 0.147 | 0.008 | 0.294 |
| real | ask_bt_neg | refuse_both | 95 | uncertainty-estimation | density_nontest | 0.106 | -0.023 | 0.240 |
| real | ask_bt_neg | refuse_both | 95 | uncertainty-estimation | density_testlex | 0.042 | -0.021 | 0.103 |
| real | ask_bt_neg | refuse_both | 95 | backtracking | density | 0.043 | -0.023 | 0.116 |
| real | ask_bt_neg | refuse_both | 95 | backtracking | density_nontest | 0.040 | -0.027 | 0.112 |
| real | ask_bt_neg | refuse_both | 95 | backtracking | density_testlex | 0.004 | 0.000 | 0.011 |
| real | ask_bt_pos | gained | 11 | deduction | density | 0.262 | -0.399 | 0.938 |
| real | ask_bt_pos | gained | 11 | deduction | density_nontest | 0.456 | -0.230 | 1.162 |
| real | ask_bt_pos | gained | 11 | deduction | density_testlex | -0.194 | -0.313 | -0.068 |
| real | ask_bt_pos | gained | 11 | uncertainty-estimation | density | 0.470 | -0.015 | 0.994 |
| real | ask_bt_pos | gained | 11 | uncertainty-estimation | density_nontest | 0.471 | 0.030 | 0.948 |
| real | ask_bt_pos | gained | 11 | uncertainty-estimation | density_testlex | -0.000 | -0.094 | 0.097 |
| real | ask_bt_pos | gained | 11 | backtracking | density | -0.107 | -0.402 | 0.168 |
| real | ask_bt_pos | gained | 11 | backtracking | density_nontest | -0.107 | -0.402 | 0.168 |
| real | ask_bt_pos | gained | 11 | backtracking | density_testlex | 0.000 | 0.000 | 0.000 |
| real | ask_bt_pos | lost | 15 | deduction | density | -0.493 | -1.153 | 0.152 |
| real | ask_bt_pos | lost | 15 | deduction | density_nontest | -0.401 | -1.092 | 0.265 |
| real | ask_bt_pos | lost | 15 | deduction | density_testlex | -0.092 | -0.202 | 0.029 |
| real | ask_bt_pos | lost | 15 | uncertainty-estimation | density | -0.354 | -0.692 | -0.011 |
| real | ask_bt_pos | lost | 15 | uncertainty-estimation | density_nontest | -0.344 | -0.646 | -0.052 |
| real | ask_bt_pos | lost | 15 | uncertainty-estimation | density_testlex | -0.010 | -0.107 | 0.086 |
| real | ask_bt_pos | lost | 15 | backtracking | density | 0.028 | -0.102 | 0.161 |
| real | ask_bt_pos | lost | 15 | backtracking | density_nontest | 0.058 | -0.063 | 0.180 |
| real | ask_bt_pos | lost | 15 | backtracking | density_testlex | -0.029 | -0.088 | 0.000 |
| real | ask_bt_pos | comply_both | 21 | deduction | density | 0.123 | -0.466 | 0.761 |
| real | ask_bt_pos | comply_both | 21 | deduction | density_nontest | 0.099 | -0.446 | 0.702 |
| real | ask_bt_pos | comply_both | 21 | deduction | density_testlex | 0.025 | -0.091 | 0.147 |
| real | ask_bt_pos | comply_both | 21 | uncertainty-estimation | density | 0.094 | -0.094 | 0.273 |
| real | ask_bt_pos | comply_both | 21 | uncertainty-estimation | density_nontest | 0.163 | -0.067 | 0.366 |
| real | ask_bt_pos | comply_both | 21 | uncertainty-estimation | density_testlex | -0.068 | -0.156 | 0.018 |
| real | ask_bt_pos | comply_both | 21 | backtracking | density | -0.038 | -0.152 | 0.074 |
| real | ask_bt_pos | comply_both | 21 | backtracking | density_nontest | -0.064 | -0.175 | 0.043 |
| real | ask_bt_pos | comply_both | 21 | backtracking | density_testlex | 0.026 | 0.000 | 0.065 |
| real | ask_bt_pos | refuse_both | 102 | deduction | density | -0.113 | -0.338 | 0.117 |
| real | ask_bt_pos | refuse_both | 102 | deduction | density_nontest | -0.125 | -0.365 | 0.114 |
| real | ask_bt_pos | refuse_both | 102 | deduction | density_testlex | 0.013 | -0.057 | 0.078 |
| real | ask_bt_pos | refuse_both | 102 | uncertainty-estimation | density | 0.106 | -0.017 | 0.232 |
| real | ask_bt_pos | refuse_both | 102 | uncertainty-estimation | density_nontest | 0.065 | -0.052 | 0.181 |
| real | ask_bt_pos | refuse_both | 102 | uncertainty-estimation | density_testlex | 0.040 | -0.014 | 0.095 |
| real | ask_bt_pos | refuse_both | 102 | backtracking | density | -0.019 | -0.076 | 0.036 |
| real | ask_bt_pos | refuse_both | 102 | backtracking | density_nontest | -0.022 | -0.079 | 0.033 |
| real | ask_bt_pos | refuse_both | 102 | backtracking | density_testlex | 0.003 | 0.000 | 0.010 |
| real | ask_unc_neg | gained | 20 | deduction | density | -0.090 | -0.611 | 0.478 |
| real | ask_unc_neg | gained | 20 | deduction | density_nontest | 0.065 | -0.519 | 0.692 |
| real | ask_unc_neg | gained | 20 | deduction | density_testlex | -0.155 | -0.290 | -0.012 |
| real | ask_unc_neg | gained | 20 | uncertainty-estimation | density | 0.428 | 0.024 | 0.819 |
| real | ask_unc_neg | gained | 20 | uncertainty-estimation | density_nontest | 0.436 | 0.038 | 0.817 |
| real | ask_unc_neg | gained | 20 | uncertainty-estimation | density_testlex | -0.008 | -0.136 | 0.082 |
| real | ask_unc_neg | gained | 20 | backtracking | density | 0.067 | -0.102 | 0.233 |
| real | ask_unc_neg | gained | 20 | backtracking | density_nontest | 0.067 | -0.102 | 0.233 |
| real | ask_unc_neg | gained | 20 | backtracking | density_testlex | 0.000 | 0.000 | 0.000 |
| real | ask_unc_neg | lost | 15 | deduction | density | 0.438 | -0.373 | 1.300 |
| real | ask_unc_neg | lost | 15 | deduction | density_nontest | 0.457 | -0.294 | 1.277 |
| real | ask_unc_neg | lost | 15 | deduction | density_testlex | -0.018 | -0.153 | 0.113 |
| real | ask_unc_neg | lost | 15 | uncertainty-estimation | density | -0.274 | -0.523 | -0.009 |
| real | ask_unc_neg | lost | 15 | uncertainty-estimation | density_nontest | -0.287 | -0.527 | -0.035 |
| real | ask_unc_neg | lost | 15 | uncertainty-estimation | density_testlex | 0.012 | -0.086 | 0.129 |
| real | ask_unc_neg | lost | 15 | backtracking | density | 0.019 | -0.146 | 0.192 |
| real | ask_unc_neg | lost | 15 | backtracking | density_nontest | 0.048 | -0.104 | 0.207 |
| real | ask_unc_neg | lost | 15 | backtracking | density_testlex | -0.029 | -0.088 | 0.000 |
| real | ask_unc_neg | comply_both | 21 | deduction | density | -0.331 | -0.842 | 0.153 |
| real | ask_unc_neg | comply_both | 21 | deduction | density_nontest | -0.296 | -0.789 | 0.171 |
| real | ask_unc_neg | comply_both | 21 | deduction | density_testlex | -0.035 | -0.120 | 0.046 |
| real | ask_unc_neg | comply_both | 21 | uncertainty-estimation | density | -0.045 | -0.261 | 0.171 |
| real | ask_unc_neg | comply_both | 21 | uncertainty-estimation | density_nontest | -0.021 | -0.268 | 0.210 |
| real | ask_unc_neg | comply_both | 21 | uncertainty-estimation | density_testlex | -0.024 | -0.107 | 0.064 |
| real | ask_unc_neg | comply_both | 21 | backtracking | density | -0.002 | -0.126 | 0.130 |
| real | ask_unc_neg | comply_both | 21 | backtracking | density_nontest | -0.002 | -0.126 | 0.130 |
| real | ask_unc_neg | comply_both | 21 | backtracking | density_testlex | 0.000 | 0.000 | 0.000 |
| real | ask_unc_neg | refuse_both | 93 | deduction | density | -0.038 | -0.306 | 0.228 |
| real | ask_unc_neg | refuse_both | 93 | deduction | density_nontest | -0.060 | -0.339 | 0.229 |
| real | ask_unc_neg | refuse_both | 93 | deduction | density_testlex | 0.022 | -0.041 | 0.085 |
| real | ask_unc_neg | refuse_both | 93 | uncertainty-estimation | density | 0.165 | 0.033 | 0.297 |
| real | ask_unc_neg | refuse_both | 93 | uncertainty-estimation | density_nontest | 0.164 | 0.038 | 0.287 |
| real | ask_unc_neg | refuse_both | 93 | uncertainty-estimation | density_testlex | 0.001 | -0.047 | 0.050 |
| real | ask_unc_neg | refuse_both | 93 | backtracking | density | -0.023 | -0.090 | 0.047 |
| real | ask_unc_neg | refuse_both | 93 | backtracking | density_nontest | -0.025 | -0.092 | 0.044 |
| real | ask_unc_neg | refuse_both | 93 | backtracking | density_testlex | 0.003 | 0.000 | 0.008 |
| real | ask_unc_pos | gained | 15 | deduction | density | -0.148 | -0.688 | 0.347 |
| real | ask_unc_pos | gained | 15 | deduction | density_nontest | 0.049 | -0.510 | 0.578 |
| real | ask_unc_pos | gained | 15 | deduction | density_testlex | -0.197 | -0.319 | -0.056 |
| real | ask_unc_pos | gained | 15 | uncertainty-estimation | density | 0.654 | 0.275 | 1.011 |
| real | ask_unc_pos | gained | 15 | uncertainty-estimation | density_nontest | 0.655 | 0.291 | 0.988 |
| real | ask_unc_pos | gained | 15 | uncertainty-estimation | density_testlex | -0.001 | -0.126 | 0.094 |
| real | ask_unc_pos | gained | 15 | backtracking | density | 0.026 | -0.097 | 0.146 |
| real | ask_unc_pos | gained | 15 | backtracking | density_nontest | 0.026 | -0.097 | 0.146 |
| real | ask_unc_pos | gained | 15 | backtracking | density_testlex | 0.000 | 0.000 | 0.000 |
| real | ask_unc_pos | lost | 15 | deduction | density | 0.011 | -0.621 | 0.638 |
| real | ask_unc_pos | lost | 15 | deduction | density_nontest | -0.039 | -0.638 | 0.536 |
| real | ask_unc_pos | lost | 15 | deduction | density_testlex | 0.051 | -0.070 | 0.194 |
| real | ask_unc_pos | lost | 15 | uncertainty-estimation | density | 0.001 | -0.259 | 0.289 |
| real | ask_unc_pos | lost | 15 | uncertainty-estimation | density_nontest | -0.049 | -0.311 | 0.210 |
| real | ask_unc_pos | lost | 15 | uncertainty-estimation | density_testlex | 0.049 | -0.051 | 0.161 |
| real | ask_unc_pos | lost | 15 | backtracking | density | 0.187 | -0.019 | 0.418 |
| real | ask_unc_pos | lost | 15 | backtracking | density_nontest | 0.187 | -0.019 | 0.418 |
| real | ask_unc_pos | lost | 15 | backtracking | density_testlex | 0.000 | 0.000 | 0.000 |
| real | ask_unc_pos | comply_both | 21 | deduction | density | 0.320 | -0.226 | 0.878 |
| real | ask_unc_pos | comply_both | 21 | deduction | density_nontest | 0.266 | -0.314 | 0.846 |
| real | ask_unc_pos | comply_both | 21 | deduction | density_testlex | 0.054 | -0.036 | 0.148 |
| real | ask_unc_pos | comply_both | 21 | uncertainty-estimation | density | -0.061 | -0.254 | 0.147 |
| real | ask_unc_pos | comply_both | 21 | uncertainty-estimation | density_nontest | -0.019 | -0.216 | 0.189 |
| real | ask_unc_pos | comply_both | 21 | uncertainty-estimation | density_testlex | -0.042 | -0.142 | 0.048 |
| real | ask_unc_pos | comply_both | 21 | backtracking | density | -0.054 | -0.141 | 0.034 |
| real | ask_unc_pos | comply_both | 21 | backtracking | density_nontest | -0.033 | -0.129 | 0.059 |
| real | ask_unc_pos | comply_both | 21 | backtracking | density_testlex | -0.021 | -0.063 | 0.000 |
| real | ask_unc_pos | refuse_both | 98 | deduction | density | 0.001 | -0.216 | 0.206 |
| real | ask_unc_pos | refuse_both | 98 | deduction | density_nontest | -0.035 | -0.258 | 0.184 |
| real | ask_unc_pos | refuse_both | 98 | deduction | density_testlex | 0.036 | -0.030 | 0.097 |
| real | ask_unc_pos | refuse_both | 98 | uncertainty-estimation | density | 0.075 | -0.060 | 0.212 |
| real | ask_unc_pos | refuse_both | 98 | uncertainty-estimation | density_nontest | 0.095 | -0.042 | 0.229 |
| real | ask_unc_pos | refuse_both | 98 | uncertainty-estimation | density_testlex | -0.020 | -0.066 | 0.029 |
| real | ask_unc_pos | refuse_both | 98 | backtracking | density | -0.002 | -0.069 | 0.065 |
| real | ask_unc_pos | refuse_both | 98 | backtracking | density_nontest | -0.010 | -0.075 | 0.056 |
| real | ask_unc_pos | refuse_both | 98 | backtracking | density_testlex | 0.008 | 0.000 | 0.020 |
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

