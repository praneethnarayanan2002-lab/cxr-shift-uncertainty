"""Full evaluation of a trained model: discrimination, calibration, conformal prediction,
Grad-CAM and shortcut checks on the source test set and the shifted target test set.

Everything that is fit (temperature, decision threshold, conformal quantile) uses source data
only. The target set is never used for fitting.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from . import calibration as cal  # noqa: E402
from . import conformal as conf  # noqa: E402
from .data import SplitData, match_histograms, reference_histogram  # noqa: E402
from .gradcam import GradCAM, center_mass, plot_gradcam_grid, select_examples  # noqa: E402
from .metrics import binary_metrics, bootstrap_ci, choose_threshold  # noqa: E402
from .shortcut import apply_mask, corner_baseline  # noqa: E402
from .train import predict_logits  # noqa: E402
from .utils import ensure_dir, save_json  # noqa: E402

logger = logging.getLogger("cxrshift.evaluate")


def _fmt(x, nd=3):
    return "nan" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def build_eval_sets(cfg: dict, data: dict[str, SplitData]) -> dict[str, SplitData]:
    """Source test, target test and (optionally) the histogram-matched target test."""
    sets = {"source_test": data["source_test"], "target_test": data["target_test"]}
    if cfg.get("mitigation", {}).get("hist_match", False):
        ref = reference_histogram(data["source_train"].images, seed=int(cfg.get("seed", 0)))
        matched = match_histograms(data["target_test"].images, ref)
        sets["target_test_histmatch"] = data["target_test"].with_images(matched, "target_test_histmatch")
    return sets


def _gradcam_section(cfg, model, eval_sets, probs, threshold, device, fig_dir) -> dict:
    gcfg = cfg.get("gradcam", {})
    if not gcfg.get("enabled", True):
        return {}
    k = int(gcfg.get("per_group", 4))
    n_stats = int(gcfg.get("n_stats", 200))
    image_size = int(cfg["data"]["image_size"])
    cam = GradCAM(model, gcfg.get("target_layer", "backbone.convnext.encoder.stages.-1"))
    rng = np.random.default_rng(int(cfg.get("seed", 0)))
    out = {}
    try:
        for name in ("source_test", "target_test"):
            split = eval_sets[name]
            p = probs[name][:, 1]
            preds = (p >= threshold).astype(int)
            confid = np.where(preds == 1, p, 1 - p)
            groups = select_examples(split.labels, preds, confid, k, seed=int(cfg.get("seed", 0)))

            def cams_for(idx):
                x = torch.from_numpy(split.images[idx]).float().div(255.0)[:, None]
                if x.shape[-1] != image_size:
                    x = torch.nn.functional.interpolate(x, size=(image_size, image_size), mode="bilinear",
                                                        align_corners=False, antialias=True)
                chunks = [cam(xb.to(device), class_idx=1) for xb in torch.split(x, 16)]
                return x[:, 0].numpy(), np.concatenate(chunks)

            # Example grid: rows are outcome groups, heatmaps for the pneumonia logit.
            imgs, cams, titles = [], [], []
            for gname, idx in groups.items():
                if not idx:
                    continue
                gi, gc = cams_for(np.asarray(idx))
                imgs.extend(gi)
                cams.extend(gc)
                titles.extend(f"{gname} p={p[i]:.2f}" for i in idx)
            if imgs:
                ncols = max(1, k)
                nrows = int(np.ceil(len(imgs) / ncols))
                fig = plt.figure(figsize=(2.0 * ncols, 2.1 * nrows))
                plot_gradcam_grid(fig, np.asarray(imgs), np.asarray(cams), titles, ncols)
                fig.suptitle(f"Grad-CAM (pneumonia logit), {name}", fontsize=9)
                fig.tight_layout()
                fig.savefig(fig_dir / f"gradcam_{name}.png", dpi=130)
                plt.close(fig)

            # Heatmap statistics on a random sample: how much mass falls inside the central region.
            sample = rng.choice(len(split), size=min(n_stats, len(split)), replace=False)
            _, sc = cams_for(np.sort(sample))
            mass = center_mass(sc, float(gcfg.get("border_frac", 0.15)))
            correct = preds[np.sort(sample)] == split.labels[np.sort(sample)]
            out[name] = {
                "n_examples": {g: len(v) for g, v in groups.items()},
                "center_mass_mean": float(mass.mean()),
                "center_mass_correct": float(mass[correct].mean()) if correct.any() else float("nan"),
                "center_mass_incorrect": float(mass[~correct].mean()) if (~correct).any() else float("nan"),
                "n_stats": int(len(sample)),
            }
    finally:
        cam.remove()
    return out


def _shortcut_section(cfg, model, data, eval_sets, device, fig_dir) -> list[dict]:
    scfg = cfg.get("shortcut", {})
    if not scfg.get("enabled", True):
        return []
    kinds = scfg.get("masks", ["full", "border_masked", "corners_masked", "corners_only"])
    bf, cf = float(scfg.get("border_frac", 0.1)), float(scfg.get("corner_frac", 0.2))
    rows = []
    examples = {}
    for name in ("source_test", "target_test"):
        split = eval_sets[name]
        for kind in kinds:
            masked = split.with_images(apply_mask(split.images, kind, bf, cf), f"{name}:{kind}")
            p = cal.softmax(predict_logits(model, masked, cfg, device))[:, 1]
            auc = float(roc_auc_score(split.labels, p)) if len(np.unique(split.labels)) == 2 else float("nan")
            rows.append({"dataset": name, "test": f"model on {kind}", "auroc": auc})
            examples.setdefault(kind, masked.images[0])
    base = corner_baseline(
        data["source_train"].images, data["source_train"].labels,
        {n: (eval_sets[n].images, eval_sets[n].labels) for n in ("source_test", "target_test")},
        corner_frac=cf, seed=int(cfg.get("seed", 0)),
    )
    for name, auc in base.items():
        rows.append({"dataset": name, "test": "logistic regression on corners", "auroc": auc})

    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5), gridspec_kw={"width_ratios": [1.6, 1]})
    pivot = df.pivot(index="test", columns="dataset", values="auroc")
    pivot.plot.barh(ax=axes[0])
    axes[0].axvline(0.5, color="k", ls="--", lw=1)
    axes[0].set_xlim(0, 1)
    axes[0].set_xlabel("AUROC")
    axes[0].set_title("Shortcut check", fontsize=9)
    axes[0].legend(fontsize=7, loc="lower left", framealpha=0.9)
    axes[1].axis("off")
    strip = np.concatenate([examples[k] for k in kinds], axis=1)
    axes[1].imshow(strip, cmap="gray")
    axes[1].set_title("  |  ".join(kinds), fontsize=7)
    fig.tight_layout()
    fig.savefig(fig_dir / "shortcut_check.png", dpi=130)
    plt.close(fig)
    return rows


def run_evaluation(cfg: dict, model, data: dict[str, SplitData], device, out_dir) -> dict:
    out_dir = ensure_dir(out_dir)
    fig_dir = ensure_dir(out_dir / "figures")
    seed = int(cfg.get("seed", 0))
    ecfg = cfg.get("eval", {})
    n_bins = int(ecfg.get("ece_bins", 15))
    model.to(device).eval()

    eval_sets = build_eval_sets(cfg, data)
    fit_sets = {"source_val": data["source_val"], "source_cal": data["source_cal"]}
    logits = {name: predict_logits(model, s, cfg, device) for name, s in {**fit_sets, **eval_sets}.items()}

    # 1. Temperature and threshold, both fit on the source validation part only.
    temperature = cal.fit_temperature(logits["source_val"], data["source_val"].labels)
    probs = {name: cal.softmax(z, temperature) for name, z in logits.items()}
    raw_probs = {name: cal.softmax(z) for name, z in logits.items()}
    tcfg = ecfg.get("threshold", {})
    threshold = choose_threshold(
        data["source_val"].labels, probs["source_val"][:, 1],
        tcfg.get("method", "youden"), float(tcfg.get("target_sensitivity", 0.9)),
    )
    logger.info("Temperature %.3f, threshold %.3f", temperature, threshold)

    # 2. Discrimination and calibration metrics.
    n_boot = int(ecfg.get("n_bootstrap", 1000))
    metric_rows = []
    for name, split in eval_sets.items():
        y, p = split.labels, probs[name][:, 1]
        row = {"dataset": name, **binary_metrics(y, p, threshold)}
        row["auroc_lo"], row["auroc_hi"] = bootstrap_ci(y, p, roc_auc_score, n_boot, seed=seed) if n_boot else (np.nan, np.nan)
        row.update(cal.calibration_summary(logits[name], y, temperature, n_bins))
        metric_rows.append(row)
        pd.DataFrame({"label": y, "p_raw": raw_probs[name][:, 1], "p_ts": p}).to_csv(
            out_dir / f"predictions_{name}.csv", index=False)

    # 3. Reliability diagrams, raw and temperature scaled.
    fig, axes = plt.subplots(2, len(eval_sets), figsize=(3.3 * len(eval_sets), 6.4), squeeze=False)
    for j, (name, split) in enumerate(eval_sets.items()):
        cal.plot_reliability(axes[0, j], raw_probs[name], split.labels, n_bins, f"{name} (raw)")
        cal.plot_reliability(axes[1, j], probs[name], split.labels, n_bins, f"{name} (T={temperature:.2f})")
    fig.tight_layout()
    fig.savefig(fig_dir / "reliability.png", dpi=130)
    plt.close(fig)

    # 4. Conformal prediction, calibrated on the held-out source calibration part.
    ccfg = cfg.get("conformal", {})
    alpha = float(ccfg.get("alpha", 0.1))
    cal_p, cal_y = probs["source_cal"], data["source_cal"].labels
    conformal_rows, by_conf = [], {}
    for score in ccfg.get("scores", ["lac"]):
        for mondrian in ccfg.get("mondrian", [False, True]):
            cp = conf.SplitConformal(alpha, score, mondrian).fit(cal_p, cal_y)
            for name, split in eval_sets.items():
                sets = cp.predict_sets(probs[name])
                conformal_rows.append({"dataset": name, "score": score, "mondrian": mondrian, "alpha": alpha,
                                       **conf.set_metrics(sets, split.labels)})
                if score == ccfg.get("scores", ["lac"])[0] and not mondrian:
                    by_conf[name] = conf.coverage_by_confidence(sets, split.labels, probs[name])
    alphas = ccfg.get("curve_alphas", [0.02, 0.05, 0.1, 0.15, 0.2, 0.3])
    curve = conf.coverage_curve(cal_p, cal_y, {n: (probs[n], s.labels) for n, s in eval_sets.items()},
                                alphas, score=ccfg.get("scores", ["lac"])[0])
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    conf.plot_coverage_curve(axes[0], axes[1], curve)
    fig.suptitle("Split conformal (calibrated on source)", fontsize=9)
    fig.tight_layout()
    fig.savefig(fig_dir / "conformal_coverage.png", dpi=130)
    plt.close(fig)

    # 5. Grad-CAM and 6. shortcut check.
    gradcam_stats = _gradcam_section(cfg, model, eval_sets, probs, threshold, device, fig_dir)
    shortcut_rows = _shortcut_section(cfg, model, data, eval_sets, device, fig_dir)

    # Save tables.
    metrics_df = pd.DataFrame(metric_rows)
    conformal_df = pd.DataFrame(conformal_rows)
    metrics_df.to_csv(out_dir / "metrics.csv", index=False)
    conformal_df.to_csv(out_dir / "conformal.csv", index=False)
    if shortcut_rows:
        pd.DataFrame(shortcut_rows).to_csv(out_dir / "shortcut.csv", index=False)
    results = {
        "run_name": cfg.get("run_name"),
        "temperature": temperature,
        "threshold": threshold,
        "metrics": metric_rows,
        "conformal": conformal_rows,
        "conformal_by_confidence": by_conf,
        "conformal_curve": curve,
        "gradcam": gradcam_stats,
        "shortcut": shortcut_rows,
        "split_sizes": {k: len(v) for k, v in data.items()},
    }
    save_json(results, out_dir / "results.json")
    (out_dir / "results_table.md").write_text(results_markdown(results))
    return results


def results_markdown(results: dict) -> str:
    """Markdown summary table of one run."""
    conf_main = {r["dataset"]: r for r in results["conformal"] if not r["mondrian"]}
    lines = [
        f"Run: {results.get('run_name')}  ",
        f"Temperature (fit on source val): {results['temperature']:.3f}  ",
        f"Threshold (chosen on source val): {results['threshold']:.3f}",
        "",
        "| dataset | n | prevalence | AUROC [95% CI] | AUPRC | accuracy | sensitivity | specificity "
        "| ECE raw | ECE TS | conformal coverage | avg set size |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results["metrics"]:
        c = conf_main.get(r["dataset"], {})
        lines.append(
            f"| {r['dataset']} | {r['n']} | {_fmt(r['prevalence'])} | {_fmt(r['auroc'])} "
            f"[{_fmt(r['auroc_lo'])}, {_fmt(r['auroc_hi'])}] | {_fmt(r['auprc'])} | {_fmt(r['accuracy'])} "
            f"| {_fmt(r['sensitivity'])} | {_fmt(r['specificity'])} | {_fmt(r['ece_raw'])} | {_fmt(r['ece_ts'])} "
            f"| {_fmt(c.get('coverage'))} | {_fmt(c.get('avg_set_size'))} |"
        )
    if results.get("shortcut"):
        lines += ["", "| shortcut test | dataset | AUROC |", "|---|---|---|"]
        for r in results["shortcut"]:
            lines.append(f"| {r['test']} | {r['dataset']} | {_fmt(r['auroc'])} |")
    return "\n".join(lines) + "\n"
