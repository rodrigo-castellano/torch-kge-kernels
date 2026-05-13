# Reproducibility — multi-backend reasoner parity

_Generated: 2026-05-13T08:26:52.171990+00:00_


**Paper source**: IJCAI'25 baselines from BASELINES table in tkk:scripts/run_reasoner_parity_sweep.py


**Inflation note**: ablation_d{2,3} and countries_s{2,3} are TAIL-only-corruption datasets; paper baselines are inflated by the keras-ns main KGCEvalDataset bug. paper_real_mrr = max(0, 2*paper_mrr - 100). family + wn18rr are HEAD+TAIL → paper_real_mrr == paper_mrr.


**Source of truth**: `docs/reproducibility.json` — one record per (dataset, reasoner, grounder, backend) keeping the LATEST run on disk. Each record contains the checkpoint path, stdout log, config path, MRR/Hits@k, wall time, and run timestamp.


## Test MRR (%)

- Exhaustive eval unless marked `*` (sampled — 100 or 1000 negs per dataset).
- `Paper` column shows the paper-reported number; for TAIL-only datasets the `(real)` value in parens is the un-inflated baseline (`paper(real) = max(0, 2·paper - 100)`).
- `Δ torch` is `torch.exh_mrr - paper(real)` (signed). Positive = torch beats paper.

| Dataset | Reasoner | Grounder | torch-ns | keras-main | keras-ijcai | Paper (real) | Δ torch |
|---|---|---|---:|---:|---:|---:|---:|
| ablation_d2 | dcr | BC01 | 33.33* | 43.9 | — | 33.8 (0.0) | +33.33 |
| ablation_d2 | dcr | BC12 | 92.48* | 96.0 | — | 94.0 (88.0) | +4.48 |
| ablation_d2 | dcr | BC13 | 90.67* | 95.0 | — | 95.0 (90.0) | +0.67 |
| ablation_d2 | no_reasoner | — | 100.0* | — | — | — | — |
| ablation_d2 | r2n | BC01 | 33.33* | 54.7 | — | — | — |
| ablation_d2 | r2n | BC11 | 48.0* | — | — | 71.0 (42.0) | +6.00 |
| ablation_d2 | r2n | BC12 | 98.0* | 94.3 | — | 97.2 (94.4) | +3.60 |
| ablation_d2 | r2n | BC13 | 96.0* | 93.0 | — | 98.0 (96.0) | +0.00 |
| ablation_d2 | sbr | BC01 | 33.33* | 45.1 | — | 32.2 (0.0) | +33.33 |
| ablation_d2 | sbr | BC12 | 94.48* | 94.0 | — | 96.8 (93.6) | +0.88 |
| ablation_d2 | sbr | BC13 | 95.33* | 97.0 | — | 97.4 (94.8) | +0.53 |
| ablation_d3 | dcr | BC01 | — | 47.4 | — | 34.0 (0.0) | — |
| ablation_d3 | dcr | BC12 | — | 37.9 | — | 50.0 (0.0) | — |
| ablation_d3 | dcr | BC13 | 85.9* | 88.9 | — | 86.7 (73.4) | +12.50 |
| ablation_d3 | no_reasoner | BC01 | — | 93.3 | — | — | — |
| ablation_d3 | no_reasoner | — | 92.0* | — | — | — | — |
| ablation_d3 | r2n | BC01 | — | 43.9 | — | 71.0 (42.0) | — |
| ablation_d3 | r2n | BC12 | — | 47.4 | — | 74.0 (48.0) | — |
| ablation_d3 | r2n | BC13 | 82.57* | 84.6 | — | 96.6 (93.2) | -10.63 |
| ablation_d3 | sbr | BC01 | 33.33* | 40.9 | — | 34.8 (0.0) | +33.33 |
| ablation_d3 | sbr | BC12 | — | 43.9 | — | 32.0 (0.0) | — |
| ablation_d3 | sbr | BC13 | 88.6* | 88.3 | — | 86.8 (73.6) | +15.00 |
| countries_s2 | dcr | BC01 | 95.83* | 92.4 | — | 99.5 (99.0) | -3.17 |
| countries_s2 | dcr | BC12 | 96.53* | 94.6 | — | 99.0 (98.0) | -1.47 |
| countries_s2 | dcr | BC13 | 86.39* | 96.9 | — | 97.0 (94.0) | -7.61 |
| countries_s2 | r2n | BC01 | 100.0* | 89.2 | — | 99.0 (98.0) | +2.00 |
| countries_s2 | r2n | BC12 | 97.92* | 93.8 | — | 99.0 (98.0) | -0.08 |
| countries_s2 | r2n | BC13 | 80.56* | 88.7 | — | 99.0 (98.0) | -17.44 |
| countries_s2 | sbr | BC01 | 95.83* | 95.1 | — | 99.5 (99.0) | -3.17 |
| countries_s2 | sbr | BC12 | 93.75* | 97.9 | — | 99.5 (99.0) | -5.25 |
| countries_s2 | sbr | BC13 | 97.22* | 95.8 | — | 99.5 (99.0) | -1.78 |
| countries_s3 | dcr | BC01 | 93.06* | 93.1 | — | 93.5 (87.0) | +6.06 |
| countries_s3 | dcr | BC12 | 84.72* | 93.8 | — | 96.9 (93.8) | -9.08 |
| countries_s3 | dcr | BC13 | 83.12* | 97.9 | — | 97.6 (95.2) | -12.08 |
| countries_s3 | no_reasoner | BC01 | — | 83.0 | — | — | — |
| countries_s3 | no_reasoner | — | 94.79* | — | — | — | — |
| countries_s3 | r2n | BC01 | 80.9* | 93.8 | — | 90.7 (81.4) | -0.50 |
| countries_s3 | r2n | BC12 | 81.94* | 95.8 | — | 88.9 (77.8) | +4.14 |
| countries_s3 | r2n | BC13 | 71.32* | 86.8 | — | 89.5 (79.0) | -7.68 |
| countries_s3 | sbr | BC01 | 97.92* | 91.8 | — | 95.3 (90.6) | +7.32 |
| countries_s3 | sbr | BC12 | 95.83* | 97.9 | — | 96.8 (93.6) | +2.23 |
| countries_s3 | sbr | BC13 | 89.58* | 97.9 | — | 97.7 (95.4) | -5.82 |
| family | dcr | BC01 | 88.51 | 86.5 | 86.4 | 90.1 | -1.59 |
| family | dcr | BC12 | 81.99 | 97.7* | — | 90.1 | -8.11 |
| family | r2n | BC01 | 94.0 | 91.9 | 91.9 | 94.0 | +0.00 |
| family | r2n | BC12 | 93.39 | 97.6* | — | 91.8 | +1.59 |
| family | sbr | BC01 | 93.5 | 92.8 | 91.3 | 86.9 | +6.60 |
| family | sbr | BC12 | 85.23 | 97.4* | — | 87.7 | -2.47 |
| wn18rr | dcr | BC01 | 44.92* | 45.1* | — | 44.2 | +0.72 |
| wn18rr | dcr | BC12 | 46.15* | — | — | 45.6 | +0.55 |
| wn18rr | r2n | BC01 | 50.2* | 45.4* | — | 44.2 | +6.00 |
| wn18rr | r2n | BC12 | 50.48* | — | — | 44.1 | +6.38 |
| wn18rr | sbr | BC01 | 49.02* | 47.9* | — | 44.0 | +5.02 |
| wn18rr | sbr | BC12 | 50.06* | — | — | 44.7 | +5.36 |

## Wall time (seconds, torch-ns end-to-end vs paper)

| Dataset | Reasoner | Grounder | torch wall | paper wall | speedup |
|---|---|---|---:|---:|---:|
| ablation_d2 | dcr | BC01 | — | — | — |
| ablation_d2 | dcr | BC12 | — | — | — |
| ablation_d2 | dcr | BC13 | — | — | — |
| ablation_d2 | no_reasoner | — | — | — | — |
| ablation_d2 | r2n | BC01 | 38 | — | — |
| ablation_d2 | r2n | BC11 | — | — | — |
| ablation_d2 | r2n | BC12 | 80 | — | — |
| ablation_d2 | r2n | BC13 | 117 | — | — |
| ablation_d2 | sbr | BC01 | 21 | — | — |
| ablation_d2 | sbr | BC12 | 39 | — | — |
| ablation_d2 | sbr | BC13 | 88 | — | — |
| ablation_d3 | dcr | BC01 | — | — | — |
| ablation_d3 | dcr | BC12 | — | — | — |
| ablation_d3 | dcr | BC13 | — | — | — |
| ablation_d3 | no_reasoner | BC01 | — | — | — |
| ablation_d3 | no_reasoner | — | — | — | — |
| ablation_d3 | r2n | BC01 | — | — | — |
| ablation_d3 | r2n | BC12 | — | — | — |
| ablation_d3 | r2n | BC13 | 111 | — | — |
| ablation_d3 | sbr | BC01 | — | — | — |
| ablation_d3 | sbr | BC12 | — | — | — |
| ablation_d3 | sbr | BC13 | 74 | — | — |
| countries_s2 | dcr | BC01 | — | — | — |
| countries_s2 | dcr | BC12 | — | — | — |
| countries_s2 | dcr | BC13 | — | — | — |
| countries_s2 | r2n | BC01 | — | — | — |
| countries_s2 | r2n | BC12 | — | — | — |
| countries_s2 | r2n | BC13 | — | — | — |
| countries_s2 | sbr | BC01 | — | — | — |
| countries_s2 | sbr | BC12 | — | — | — |
| countries_s2 | sbr | BC13 | — | — | — |
| countries_s3 | dcr | BC01 | — | — | — |
| countries_s3 | dcr | BC12 | — | — | — |
| countries_s3 | dcr | BC13 | — | — | — |
| countries_s3 | no_reasoner | BC01 | — | — | — |
| countries_s3 | no_reasoner | — | — | — | — |
| countries_s3 | r2n | BC01 | — | — | — |
| countries_s3 | r2n | BC12 | — | — | — |
| countries_s3 | r2n | BC13 | — | — | — |
| countries_s3 | sbr | BC01 | 55 | — | — |
| countries_s3 | sbr | BC12 | — | — | — |
| countries_s3 | sbr | BC13 | 34 | — | — |
| family | dcr | BC01 | 1287 | 24139 | 18.8× |
| family | dcr | BC12 | 17305 | 23812 | 1.4× |
| family | r2n | BC01 | 1008 | 16189 | 16.1× |
| family | r2n | BC12 | 7220 | 77058 | 10.7× |
| family | sbr | BC01 | 2098 | 15276 | 7.3× |
| family | sbr | BC12 | 17080 | 70803 | 4.1× |
| wn18rr | dcr | BC01 | 1712 | 28471 | 16.6× |
| wn18rr | dcr | BC12 | 8558 | 81571 | 9.5× |
| wn18rr | r2n | BC01 | 2158 | 22797 | 10.6× |
| wn18rr | r2n | BC12 | 6263 | 79566 | 12.7× |
| wn18rr | sbr | BC01 | 2707 | 23851 | 8.8× |
| wn18rr | sbr | BC12 | 8444 | 74518 | 8.8× |
