"""Print medmnist label names and the binary label counts produced by a config.

Run this before training to confirm the pneumonia mapping is what you expect:

    python scripts/check_label_mapping.py --config configs/default.yaml
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cxrshift.data import load_split  # noqa: E402
from cxrshift.utils import load_config, setup_logging  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", nargs="*", default=[])
    args = parser.parse_args()
    setup_logging()
    cfg = load_config(args.config, args.set)
    for role in ("source", "target"):
        ds = cfg["data"][role]
        where = ds.get('name') if ds.get('loader') == 'medmnist' else ds.get('root', '')
        print(f"\n== {role}: {ds.get('loader')} {where}")
        print("label_map:", ds["label_map"])
        if ds.get("loader") == "medmnist":
            import medmnist

            info = medmnist.INFO[ds["name"]]
            print("task:", info["task"])
            print("label names:", info["label"])
        splits = ["train", "val", "test"] if role == "source" else [ds.get("eval_split", "test")]
        for split in splits:
            d = load_split(ds, split)
            dropped = d.meta.get("dropped", 0)
            print(f"  {split}: kept {len(d)} (positives {int(d.labels.sum())}, "
                  f"negatives {int((d.labels == 0).sum())}), dropped {dropped} with other labels")


if __name__ == "__main__":
    main()
