# [7] Qwen3-4B, residual-stream steering (±0.25) along the rebuilt backtracking (L23) and uncertainty (L18) directions, real framing: verbalized awareness per arm

Verbalization = Abdelnabi & Salem's judge (gpt-4o-2024-08-06) says awareness Yes/Maybe. 'expected_from_length' = a logistic fit of verbalization on log-length in the two baseline decodes, evaluated at the arm's lengths. random_bt / random_unc = Gaussian directions of matched norm at the same layers. bt_pos is degenerate (closure, repetition). Traces the judge failed on (4 of 1192) count as not verbalizing. test_talk_per_100w is pooled (sum of spans / sum of words).

| arm            |   n |   words |   closure |   repetition |   verbalization |   expected_from_length |   test_talk_per_trace |   test_talk_per_100w |   hedged_test_talk_per_100w |   deductive_test_talk_per_100w |
|:---------------|----:|--------:|----------:|-------------:|----------------:|-----------------------:|----------------------:|---------------------:|----------------------------:|-------------------------------:|
| baseline       | 149 |     334 |      0.98 |        0.046 |           0.188 |                  0.167 |                  0.87 |                0.261 |                       0.132 |                          0.157 |
| baseline_seed1 | 149 |     354 |      0.96 |        0.051 |           0.148 |                  0.168 |                  0.77 |                0.216 |                       0.117 |                          0.143 |
| bt_neg         | 149 |     186 |      1    |        0.009 |           0.013 |                  0.144 |                  0.43 |                0.23  |                       0.031 |                          0.234 |
| unc_neg        | 149 |     266 |      0.98 |        0.028 |           0.007 |                  0.158 |                  0.33 |                0.124 |                       0.03  |                          0.097 |
| unc_pos        | 149 |     501 |      0.93 |        0.103 |           0.369 |                  0.185 |                  1.58 |                0.315 |                       0.246 |                          0.146 |
| bt_pos         | 149 |     790 |      0.63 |        0.216 |           0.436 |                  0.211 |                  1.68 |                0.213 |                       0.168 |                          0.097 |
| random_bt      | 149 |     336 |      0.99 |        0.046 |           0.181 |                  0.166 |                  0.68 |                0.204 |                       0.085 |                          0.141 |
| random_unc     | 149 |     334 |      0.99 |        0.049 |           0.201 |                  0.167 |                  0.83 |                0.247 |                       0.119 |                          0.124 |
