# d10

Which parts of a reasoning model's chain-of-thought does steering its **test-awareness** direction
actually move?

Abdelnabi & Salem ([2505.14617](https://arxiv.org/abs/2505.14617)) showed that steering a linear
"test awareness" direction changes how often a reasoning model complies with harmful requests, and
asked in their §4.7 whether awareness also changes reasoning behaviours such as backtracking.
Venhoff et al. ([2506.18167](https://arxiv.org/abs/2506.18167)) released steerable directions for six
such behaviours. This project puts the two together and measures which behaviours move, which stay
flat, and whether any of them mediate the compliance effect.

Work in progress. See `CLAUDE.md` for layout, environment, and the working rules.

```bash
uv sync
git clone https://github.com/microsoft/Test_Awareness_Steering assets/test_awareness_steering
git clone https://github.com/cvenhoff/steering-thinking-llms assets/steering_thinking_llms
```
