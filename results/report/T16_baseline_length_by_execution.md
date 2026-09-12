# [16] Mean reasoning words of refused vs executed traces with no intervention; item-consistency check for Distill's fake-framing deduction drop

| model         | framing   | source      |   refused_mean_words |   executed_mean_words |   gap |
|:--------------|:----------|:------------|---------------------:|----------------------:|------:|
| deepseek_qwen | fake      | shipped α=0 |                  224 |                   359 |   135 |
| deepseek_qwen | real      | shipped α=0 |                  228 |                   358 |   130 |
| qwen3         | fake      | shipped α=0 |                  216 |                   308 |    92 |
| qwen3         | real      | shipped α=0 |                  218 |                   313 |    95 |
| qwq           | fake      | shipped α=0 |                  190 |                   328 |   138 |
| qwq           | real      | shipped α=0 |                  188 |                   249 |    61 |
| qwen3_4b      | fake      | baseline    |                  244 |                   327 |    83 |
| qwen3_4b      | real      | baseline    |                  269 |                   336 |    67 |
| qwen3_32b_api | fake      | baseline    |                  251 |                   354 |   103 |
| qwen3_32b_api | real      | baseline    |                  257 |                   373 |   116 |

Distill, fake framing, non-test deduction sentences per trace among stable refusers: greedy pair mean -0.84 (n=85), sampled pair mean -0.73 (n=84); per-task changes on the 67 tasks in both pairs, Pearson r = 0.02.
