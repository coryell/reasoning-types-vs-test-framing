# [6] Test-talk spans per trace added in items that refuse under both conditions

Test-talk = spans matching d10.parse.TEST_LEXICON (any Venhoff label); split by whether the annotator labelled the span uncertainty-estimation (hedged form) or deduction.

| cell                                                  | framing   |   n_refuse_both |   test_talk_base | test_talk_delta       | hedged_form_delta     | deductive_form_delta   |
|:------------------------------------------------------|:----------|----------------:|-----------------:|:----------------------|:----------------------|:-----------------------|
| Distill-Qwen-32B, aware +0.07                         | real      |             104 |             0.13 | +1.01 [+0.79, +1.24]* | +0.41 [+0.27, +0.57]* | +0.41 [+0.28, +0.56]*  |
| Distill-Qwen-32B, aware +0.07                         | fake      |              85 |             0.33 | +1.71 [+1.35, +2.08]* | +0.53 [+0.36, +0.69]* | +0.72 [+0.48, +0.98]*  |
| QwQ-32B, aware +0.05                                  | real      |              62 |             0.97 | +1.52 [+1.11, +1.95]* | +0.81 [+0.52, +1.11]* | +0.39 [+0.13, +0.66]*  |
| QwQ-32B, aware +0.05                                  | fake      |              19 |             1.37 | +1.11 (n<20, no CI)   | +0.68 (n<20, no CI)   | +0.37 (n<20, no CI)    |
| Qwen3-32B, aware +0.043                               | real      |             106 |             0.99 | +0.98 [+0.75, +1.22]* | +0.53 [+0.37, +0.70]* | +0.35 [+0.18, +0.53]*  |
| Qwen3-32B, aware +0.043                               | fake      |              78 |             1.04 | +1.00 [+0.62, +1.42]* | +0.49 [+0.33, +0.65]* | +0.36 [+0.10, +0.63]*  |
| Qwen3-32B, unaware −0.32                              | real      |              65 |             0.97 | -0.31 [-0.54, -0.08]* | -0.08 [-0.22, +0.06]  | -0.23 [-0.43, -0.03]*  |
| Qwen3-32B, unaware −0.32                              | fake      |              49 |             0.86 | -0.22 [-0.53, +0.10]  | +0.02 [-0.14, +0.18]  | -0.27 [-0.51, -0.02]*  |
| Qwen3-32B, test cue (seed 0)                          | real      |              90 |             0.9  | +0.84 [+0.53, +1.17]* | +0.24 [+0.09, +0.40]* | +0.49 [+0.24, +0.74]*  |
| Qwen3-32B, test cue (seed 0)                          | fake      |              51 |             1.06 | +1.18 [+0.61, +1.82]* | +0.16 [-0.08, +0.39]  | +0.75 [+0.35, +1.18]*  |
| Qwen3-32B, test cue (seed 1)                          | real      |              94 |             0.89 | +0.77 [+0.48, +1.07]* | +0.29 [+0.13, +0.45]* | +0.31 [+0.11, +0.52]*  |
| Qwen3-32B, test cue (seed 1)                          | fake      |              51 |             1.14 | +0.75 [+0.33, +1.20]* | +0.08 [-0.12, +0.27]  | +0.51 [+0.16, +0.90]*  |
| Qwen3-4B, test cue                                    | real      |              20 |             0.9  | +0.70 [+0.15, +1.30]* | -0.05 [-0.40, +0.30]  | +0.55 [+0.25, +0.85]*  |
| Qwen3-4B, test cue                                    | fake      |               7 |             1.29 | +1.00 (n<20, no CI)   | +0.71 (n<20, no CI)   | +0.00 (n<20, no CI)    |
| Distill-Qwen-32B, aware +0.07, sampled vs sampled α=0 | real      |             104 |             0.17 | +1.18 [+0.89, +1.52]* | +0.36 [+0.23, +0.49]* | +0.52 [+0.32, +0.76]*  |
| Distill-Qwen-32B, aware +0.07, sampled vs sampled α=0 | fake      |              84 |             0.43 | +1.54 [+1.20, +1.89]* | +0.60 [+0.42, +0.77]* | +0.57 [+0.35, +0.82]*  |
| floor: Distill-Qwen-32B sampled vs greedy α=0         | real      |             118 |             0.14 | +0.07 [-0.04, +0.18]  | +0.03 [-0.02, +0.08]  | +0.04 [-0.03, +0.12]   |
| floor: Distill-Qwen-32B sampled vs greedy α=0         | fake      |             107 |             0.31 | +0.11 [-0.06, +0.28]  | -0.02 [-0.10, +0.07]  | +0.16 [+0.04, +0.28]*  |
| floor: Qwen3-4B second baseline decode                | real      |              33 |             1.27 | +0.03 [-0.33, +0.42]  | -0.09 [-0.30, +0.12]  | +0.00 [-0.24, +0.21]   |
| floor: Qwen3-4B second baseline decode                | fake      |              11 |             1.18 | +0.09 (n<20, no CI)   | +0.27 (n<20, no CI)   | -0.27 (n<20, no CI)    |
| floor: Qwen3-32B baseline decodes                     | real      |             104 |             0.94 | -0.07 [-0.26, +0.12]  | -0.07 [-0.17, +0.04]  | +0.04 [-0.12, +0.20]   |
| floor: Qwen3-32B baseline decodes                     | fake      |              67 |             1.03 | +0.10 [-0.12, +0.33]  | +0.09 [-0.06, +0.24]  | +0.01 [-0.18, +0.21]   |
