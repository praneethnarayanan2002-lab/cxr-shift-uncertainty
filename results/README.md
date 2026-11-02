# results

Empty on purpose. Each run writes to `results/<run_name>/`:

- `results_table.md`, `metrics.csv`, `conformal.csv`, `shortcut.csv`, `results.json`
- `predictions_<dataset>.csv` (label, raw probability, temperature-scaled probability)
- `figures/` (reliability diagrams, conformal coverage, Grad-CAM grids, shortcut check)
- `train_history.json`, `config_used.json`

`results/comparison.md` compares the baseline and the ablations. Model weights (`*.pt`)
and per-sample prediction files are git-ignored; tables and figures can be committed.
