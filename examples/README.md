# Examples

Self-contained runnable scripts that compose `kge_kernels` surfaces to
reproduce the method tuples described in `framework.tex`. Each script
is standalone — you can copy it into any project that depends on `tkk`.

| Script | What it shows |
|---|---|
| [`01_sbr_exhaustive.py`](01_sbr_exhaustive.py) | SBR-style **exhaustive** scoring: one-shot fuzzy-logic proof aggregation via `KGEScoreAtom` + `TNormStateRepr("min")` + `TNormTrajRepr("min")` + `MaxQueryRepr` + `ExhaustiveSelect` composed in a `ProofScorer`. |

Run from the repo root:

```bash
python examples/01_sbr_exhaustive.py
```

All examples are CPU-friendly; they print a short summary and exit
with status 0 on success. They are exercised as smoke-tests in CI.
