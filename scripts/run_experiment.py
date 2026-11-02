"""Train and evaluate one configuration.

Examples:
    python scripts/run_experiment.py --config configs/default.yaml
    python scripts/run_experiment.py --config configs/strong_aug.yaml
    python scripts/run_experiment.py --config configs/default.yaml --checkpoint results/baseline/best_model.pt
    python scripts/run_experiment.py --config configs/default.yaml --set train.epochs=3 data.target.max_samples.test=5000
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cxrshift.pipeline import run  # noqa: E402
from cxrshift.utils import load_config, setup_logging  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None, help="skip training and load these weights")
    parser.add_argument("--skip-train", action="store_true", help="evaluate the untrained model (debugging)")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="config overrides")
    args = parser.parse_args()
    setup_logging()
    cfg = load_config(args.config, args.set)
    results = run(cfg, checkpoint=args.checkpoint, skip_train=args.skip_train)
    out = Path(cfg.get("output_dir", "results")) / cfg.get("run_name", "run")
    print((out / "results_table.md").read_text())
    print(f"Wrote results to {out}")
    return results


if __name__ == "__main__":
    main()
