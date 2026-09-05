# [10] Robustness checks behind the 'ways this could be wrong' table

- Awareness verdicts over every Qwen3-4B trace judged in this project: Yes 6, Maybe 772, No 4515. The positive class is Yes or Maybe.

## Per condition (real framing): is the judge's evidence test language; verbalization among 200–300-word traces; closed vs unclosed traces

| condition      |   positives | evidence_is_test_language   | verbalization_200_300_words   |   verbalization_closed | verbalization_unclosed   |   n_unclosed |
|:---------------|------------:|:----------------------------|:------------------------------|-----------------------:|:-------------------------|-------------:|
| baseline       |          28 | 28/28                       | 0.19 (n=53)                   |                   0.19 | 0.00                     |            3 |
| baseline_seed1 |          22 | 20/22                       | 0.17 (n=46)                   |                   0.15 | 0.00                     |            6 |
| random_bt      |          27 | 19/27                       | 0.08 (n=49)                   |                   0.18 | 0.00                     |            2 |
| random_unc     |          30 | 26/30                       | 0.15 (n=46)                   |                   0.2  | 0.00                     |            2 |
| shuffled_bt    |          44 | 44/44                       | 0.37 (n=19)                   |                   0.34 | 0.16                     |           37 |
| shuffled_unc   |          41 | 38/41                       | 0.25 (n=28)                   |                   0.28 | 0.24                     |           21 |
| bt_neg         |           2 | 2/2                         | 0.00 (n=42)                   |                   0.01 | —                        |            0 |
| unc_neg        |           1 | 1/1                         | 0.00 (n=65)                   |                   0.01 | 0.00                     |            3 |
| unc_pos        |          55 | 53/55                       | 0.48 (n=31)                   |                   0.37 | 0.30                     |           10 |
| bt_pos         |          65 | 61/65                       | 0.45 (n=11)                   |                   0.53 | 0.27                     |           55 |

## F2: uncertainty direction at ±0.25 on 60 held-out trigger prompts (real / hypothetical), vector built from all spans, non-test spans, or test-language spans

| arm             | framing      |   n |   verbalization |   closure |   words |
|:----------------|:-------------|----:|----------------:|----------:|--------:|
| baseline        | hypothetical |  60 |           0.05  |     0.95  | 513.283 |
| baseline        | real         |  60 |           0.167 |     0.95  | 503.133 |
| unc_all_neg     | hypothetical |  60 |           0.017 |     0.95  | 471.767 |
| unc_all_neg     | real         |  60 |           0.05  |     0.983 | 433.233 |
| unc_all_pos     | hypothetical |  60 |           0.167 |     0.917 | 639.2   |
| unc_all_pos     | real         |  60 |           0.267 |     0.95  | 561.133 |
| unc_nontest_neg | hypothetical |  60 |           0.033 |     0.95  | 471.317 |
| unc_nontest_neg | real         |  60 |           0.05  |     0.983 | 435     |
| unc_nontest_pos | hypothetical |  60 |           0.183 |     0.917 | 604.667 |
| unc_nontest_pos | real         |  60 |           0.3   |     0.933 | 571     |
| unc_testlex_neg | hypothetical |  60 |           0.083 |     0.95  | 394.617 |
| unc_testlex_neg | real         |  60 |           0.033 |     0.983 | 350.4   |
| unc_testlex_pos | hypothetical |  60 |           0.15  |     0.8   | 782.833 |
| unc_testlex_pos | real         |  60 |           0.183 |     0.933 | 721     |

## Residual stream at the last prompt token, projected onto the awareness probe direction (unit norm), change vs no intervention, real framing

| arm            |   n_pairs |   base_mean |   delta |   ci_lo |   ci_hi |
|:---------------|----------:|------------:|--------:|--------:|--------:|
| aware_strong   |       149 |     -1.9825 | -0.0005 | -0.0042 |  0.0036 |
| prompt_test    |       149 |     -1.9825 |  0.1365 |  0.1246 |  0.1493 |
| aware          |       149 |     -1.9825 |  0.0024 | -0.0022 |  0.0071 |
| unaware_strong |       149 |     -1.9825 | -0.0136 | -0.0178 | -0.0096 |
| random_aware   |       149 |     -1.9825 | -0.0073 | -0.0111 | -0.0034 |
| bt_neg         |       149 |     -1.9825 |  0      |  0      |  0      |
| unc_neg        |       149 |     -1.9825 | -0.3083 | -0.3249 | -0.2928 |
| unc_pos        |       149 |     -1.9825 |  0.2358 |  0.2257 |  0.2461 |

