# [18] Abdelnabi & Salem's Table 4 (execution %, real / hypothetical) against a recount of their released action outputs with the string rule

Paper values transcribed from arXiv:2505.14617v3 Table 4. `recount_closed` divides by traces that closed `</think>`; `recount_all` counts unclosed traces as not executed.

| model         | arm              | framing   |   paper_table4 |   recount_closed |   recount_all |   n |   n_closed |   diff_closed |
|:--------------|:-----------------|:----------|---------------:|-----------------:|--------------:|----:|-----------:|--------------:|
| qwq           | alpha0.0         | real      |          36.2  |             36.2 |          36.2 | 149 |        149 |           0   |
| qwq           | alpha0.0         | fake      |          69.7  |             69.8 |          69.8 | 149 |        149 |           0.1 |
| qwq           | alpha0.3         | real      |          28.8  |             28.9 |          28.9 | 149 |        149 |           0.1 |
| qwq           | alpha0.3         | fake      |          53    |             53   |          53   | 149 |        149 |           0   |
| qwq           | alpha0.05_aware  | real      |          56.3  |             56.4 |          56.4 | 149 |        149 |           0.1 |
| qwq           | alpha0.05_aware  | fake      |          85.2  |             85.2 |          85.2 | 149 |        149 |           0   |
| qwen3         | alpha0.0         | real      |          20.1  |             20.1 |          20.1 | 149 |        149 |           0   |
| qwen3         | alpha0.0         | fake      |          37.5  |             37.6 |          37.6 | 149 |        149 |           0.1 |
| qwen3         | alpha0.3         | real      |          41.8  |             41.9 |          41.6 | 149 |        148 |           0.1 |
| qwen3         | alpha0.3         | fake      |          68    |             68   |          67.1 | 149 |        147 |           0   |
| qwen3         | alpha0.043_aware | real      |          22.2  |             22.3 |          22.1 | 149 |        148 |           0.1 |
| qwen3         | alpha0.043_aware | fake      |          42.2  |             42.6 |          42.3 | 149 |        148 |           0.4 |
| deepseek_qwen | alpha0.0         | real      |          12    |             12.1 |          12.1 | 149 |        149 |           0.1 |
| deepseek_qwen | alpha0.0         | fake      |          20.8  |             20.8 |          20.8 | 149 |        149 |           0   |
| deepseek_qwen | alpha0.35        | real      |           9.39 |              9.4 |           9.4 | 149 |        149 |           0   |
| deepseek_qwen | alpha0.35        | fake      |          15.4  |             15.4 |          15.4 | 149 |        149 |           0   |
| deepseek_qwen | alpha0.07_aware  | real      |          27.5  |             27.5 |          27.5 | 149 |        149 |           0   |
| deepseek_qwen | alpha0.07_aware  | fake      |          40.2  |             40.3 |          40.3 | 149 |        149 |           0.1 |