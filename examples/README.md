# Examples

Self-contained runnable scripts that compose `kge_kernels` surfaces to
reproduce the method tuples described in `framework.tex`. Each script
is standalone — you can copy it into any project that depends on `tkk`.

| Script | What it shows |
|---|---|
| [`01_sbr_exhaustive.py`](01_sbr_exhaustive.py) | SBR-style **exhaustive** scoring: one-shot fuzzy-logic proof aggregation via `KGEScoreAtom` + `TNormStateRepr("min")` + `TNormTrajRepr("min")` + `MaxQueryRepr` + `ExhaustiveSelect` fed through the reference `search_and_score`. |
| [`02_train_kge.py`](02_train_kge.py) | Pure KGE training loop using `kge_kernels.training.train_kge` with `NSSALoss`, `make_cosine_warmup_scheduler`, and `evaluate_filtered_ranking`. Synthesizes a toy KG in-memory so it runs without any dataset files. |
| [`03_filtered_eval.py`](03_filtered_eval.py) | Standalone filtered ranking evaluation: build filter maps via `kge_kernels.data.build_filter_maps`, then call `kge_kernels.eval.evaluate_filtered_ranking`. |

Run any example from the repo root:

```bash
python examples/01_sbr_exhaustive.py
python examples/02_train_kge.py
python examples/03_filtered_eval.py
```

All examples are CPU-friendly; they print a short summary and exit
with status 0 on success. They are exercised as smoke-tests in CI.
