"""Compare finished runs (for example the baseline and the strong augmentation ablation).

    python scripts/compare_runs.py results/baseline results/strong_aug --out results/comparison.md
"""

import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", help="run directories containing metrics.csv")
    parser.add_argument("--out", default="results/comparison.md")
    args = parser.parse_args()

    rows = []
    for run in args.runs:
        run = Path(run)
        metrics = pd.read_csv(run / "metrics.csv")
        conformal = pd.read_csv(run / "conformal.csv")
        main_cp = conformal[(conformal["score"] == conformal["score"].iloc[0]) & (~conformal["mondrian"])]
        main_cp = main_cp.set_index("dataset")
        for _, r in metrics.iterrows():
            rows.append({
                "run": run.name,
                "dataset": r["dataset"],
                "AUROC": r["auroc"],
                "AUROC 95% CI": f"[{r['auroc_lo']:.3f}, {r['auroc_hi']:.3f}]",
                "AUPRC": r["auprc"],
                "sensitivity": r["sensitivity"],
                "specificity": r["specificity"],
                "ECE (TS)": r["ece_ts"],
                "coverage": main_cp.loc[r["dataset"], "coverage"] if r["dataset"] in main_cp.index else float("nan"),
                "set size": main_cp.loc[r["dataset"], "avg_set_size"] if r["dataset"] in main_cp.index else float("nan"),
            })
    df = pd.DataFrame(rows)
    text = to_markdown(df)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(text + "\n")
    df.to_csv(Path(args.out).with_suffix(".csv"), index=False)
    print(text)


def to_markdown(df: pd.DataFrame) -> str:
    def fmt(v):
        return f"{v:.3f}" if isinstance(v, float) else str(v)

    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
