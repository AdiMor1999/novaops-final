# evals/

The measuring side. Starting code: `lesson-10-context-engineering/code/evals/` and
`lesson-11-observability-and-evals/code/03-score-the-run/`, `04-llm-judge/`.

- `targets.py` is the bar, declared before any result is seen. The hook blocks edits to it.
  If a target looks wrong, stop and say why. Do not move it.
- Deterministic checks run first: required and forbidden facts, forbidden tools, expected and
  forbidden sources, `should_use_tools`. Judges run second: citation quality, and "claims
  access was granted".
- The golden dataset uses illustrative tool names. Map them to our tool names in one place, in
  the scorer, and nowhere else.
- Input is `GOLDEN-DATASETS.json`. `EVALUATION-INPUTS.yaml` is read once, in task #11, by a
  runner that reseeds before each case.
- If a score is low because a golden row or a judge is wrong, report the row and the reason.
  Do not change the agent to fit a bad expectation.
- Scores are written to Langfuse against the trace they measure.
