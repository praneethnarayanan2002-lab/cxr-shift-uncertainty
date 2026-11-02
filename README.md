# cxr-shift-uncertainty

Pneumonia classification on chest X-rays, evaluated under dataset shift.

## What it is

A pretrained image classifier (`facebook/convnext-tiny-224` from Hugging Face) is fine-tuned to
detect pneumonia on one public chest X-ray dataset and then tested, unchanged, on a second
dataset from a different hospital and patient population. The project measures how much
performance drops, whether the model's probabilities stay calibrated, whether conformal
prediction sets keep their promised coverage, and whether the model looks at the lungs or at
other parts of the image.

The point is careful evaluation, not a high accuracy number.

- Source: PneumoniaMNIST (pediatric chest X-rays from Guangzhou, Kermany et al.), 224px, from the
  `medmnist` package. Labels: normal vs. pneumonia.
- Target: ChestMNIST (adult chest X-rays from the NIH ChestX-ray14 dataset), 224px. Mapped to
  pneumonia (pneumonia label present) vs. no finding (all 14 labels absent). Images with other
  findings but no pneumonia are dropped.

Label names are read from `medmnist.INFO` at runtime and matched by name. A misspelled or missing
label name raises an error instead of silently picking the wrong column.

## Why dataset shift matters

A model trained and tested on splits of the same dataset is tested on images from the same
scanners, the same hospital, the same age group and the same labelling process. A deployed model
sees none of those guarantees. Here the shift is large on purpose: children vs. adults, different
equipment, different label sources (clinician-graded vs. text-mined from radiology reports) and a
very different pneumonia prevalence. Measuring how discrimination, calibration and uncertainty
estimates behave under that shift says more about a model than one in-distribution test score.

## How to run it

### Full run (GPU)

Open `notebooks/colab_run.ipynb` in Google Colab (or Kaggle), select a GPU runtime and run all
cells. The notebook clones this repo, installs `requirements.txt`, downloads both datasets and the
pretrained weights, trains the baseline and the ablation, runs every evaluation, and writes tables
and figures to `results/`.

### Locally

```bash
pip install -r requirements.txt
python scripts/check_label_mapping.py --config configs/default.yaml
python scripts/run_experiment.py --config configs/default.yaml
python scripts/run_experiment.py --config configs/strong_aug.yaml
python scripts/compare_runs.py results/baseline results/strong_aug --out results/comparison.md
```

Any config value can be overridden from the command line:

```bash
python scripts/run_experiment.py --config configs/default.yaml --set train.epochs=5 data.target.max_samples.test=5000
```

To evaluate saved weights without retraining, pass `--checkpoint results/baseline/best_model.pt`.

### Tests

```bash
pytest -q
```

The tests use synthetic images and a tiny randomly initialised ConvNeXt (`model.backbone: tiny`),
so they need no downloads and run on CPU in under a minute. `configs/smoke.yaml` runs the whole
pipeline the same way. Numbers from the smoke config are meaningless.

### Swapping datasets

Datasets are config entries. `loader: medmnist` takes any MedMNIST flag and size; `loader: folder`
reads an ImageFolder-style tree (`root/<split>/<class>/*.png`), for example the Kaggle chest X-ray
pneumonia dataset. See `configs/kaggle_folder_example.yaml`. The label mapping (`positive`,
`negative`, the special `no_finding` negative for multi-label data) is part of each dataset entry.

## Project layout

```
src/cxrshift/
  data.py          loaders (medmnist, folder, synthetic), label mapping, augmentation, histogram matching
  model.py         Hugging Face classifier wrapper with a 2-class head, tiny test model
  train.py         fine-tuning with early stopping, batched inference
  metrics.py       AUROC, AUPRC, sensitivity, specificity, threshold selection, bootstrap CIs
  calibration.py   ECE, Brier, NLL, reliability diagrams, temperature scaling
  conformal.py     split conformal prediction (LAC and APS scores, class-conditional option)
  gradcam.py       Grad-CAM and heatmap statistics
  shortcut.py      border/corner masking and a corner-only baseline
  evaluate.py      runs all evaluations and writes tables and figures
  pipeline.py      load data, train, evaluate
  utils.py         config loading and overrides, seeding, helpers
configs/           default.yaml (baseline), strong_aug.yaml (ablation), smoke.yaml (tests), Kaggle example
scripts/           run_experiment.py, check_label_mapping.py, compare_runs.py
notebooks/         colab_run.ipynb, the single run-all notebook
tests/             pytest suite on synthetic data
results/           run outputs (empty in the repo)
```

## Methods

**Data splits.** Source train is used for training. Source validation is split in two, stratified
by label: one half for early stopping, temperature scaling and threshold selection, the other half
held out as the conformal calibration set. Source test and target test are used only for
evaluation. Nothing is fit on target data.

**1. Baseline fine-tune.** ConvNeXt-Tiny (ImageNet-pretrained) with a new 2-class head. Grayscale
images are repeated to 3 channels and normalised with ImageNet statistics. AdamW, cosine schedule
with warmup, class-weighted cross entropy, light affine augmentation, mixed precision on GPU.
Early stopping on source validation AUROC (patience 3); the best checkpoint is kept.

**2. Metrics.** AUROC with a 95% bootstrap confidence interval, AUPRC, accuracy, sensitivity and
specificity. The decision threshold is chosen on source validation (Youden's J by default, or a
target sensitivity) and then applied unchanged to both test sets. Prevalence is reported next to
AUPRC because AUPRC depends on it, and pneumonia prevalence differs a lot between the two datasets.

**3. Calibration.** Top-label expected calibration error (15 equal-width bins), Brier score and
NLL, before and after temperature scaling. The temperature is fit by minimising NLL on source
validation. Reliability diagrams are drawn for each test set, raw and scaled.

**4. Conformal prediction.** Split conformal with target coverage 90% (alpha = 0.1), calibrated on
the held-out source calibration set. Two scores (LAC: 1 - p(true class); APS: cumulative
probability mass), each with a marginal and a class-conditional (Mondrian) version. Reported for
each test set: empirical coverage, average set size, share of empty, singleton and full sets, and
coverage within each class. To show where coverage breaks, coverage is also reported by
confidence bin and as a curve over several alpha values.

**5. Grad-CAM.** Heatmaps for the pneumonia logit from the last ConvNeXt stage, for the most
confident true positives, true negatives, false positives and false negatives on both test sets.
For a random sample of test images, the share of heatmap mass inside the central region of the
image is reported for correct and incorrect predictions.

**6. Shortcut check.** The trained model is re-evaluated on images whose border is masked, whose
corners are masked, and where only the corners are visible (masked pixels are set to the image's
mean). Separately, a logistic regression is trained on downsampled corner patches of the source
training images. If either corner-only test gives an AUROC well above 0.5, the classes can be
separated from regions that contain text markers, collimation edges and padding rather than lungs.

**7. Shift-mitigation ablation.** Two cheap interventions are compared with the baseline:
- Stronger training augmentation (brightness, contrast, gamma, blur, noise, larger affine range),
  which needs a second training run (`configs/strong_aug.yaml`).
- Test-time histogram matching of target images to the pooled intensity histogram of source
  training images, reported as `target_test_histmatch` in every run.

## Results

Not run yet. The table below will be filled in from `results/baseline/results_table.md` and
`results/comparison.md` after the Colab run. No numbers here are estimated.

| run | test set | n | prevalence | AUROC [95% CI] | AUPRC | sensitivity | specificity | ECE raw | ECE after TS | conformal coverage (target 0.90) | avg set size |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | PneumoniaMNIST test (source) | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| baseline | ChestMNIST test (target) | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| baseline | ChestMNIST test, histogram matched | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| strong aug | PneumoniaMNIST test (source) | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| strong aug | ChestMNIST test (target) | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

| shortcut test | source test AUROC | target test AUROC |
|---|---|---|
| model on full image | TBD | TBD |
| model with border masked | TBD | TBD |
| model with corners masked | TBD | TBD |
| model on corners only | TBD | TBD |
| logistic regression on corners | TBD | TBD |

Figures (reliability diagrams, conformal coverage curves, Grad-CAM grids, shortcut check) will be
in `results/baseline/figures/`.

## Limitations

- This is research code for studying evaluation under dataset shift. It is not a clinical tool and
  must not be used to make decisions about patients.
- MedMNIST images are resized (and for some datasets centre-cropped) versions of the originals. Some findings, and
  some shortcuts, look different at full resolution.
- The two datasets differ in many ways at once (age, hospital, equipment, labelling method,
  prevalence), so a performance drop cannot be attributed to any single factor.
- ChestMNIST labels were text-mined from radiology reports and are noisy. "Pneumonia" in NIH and
  in the pediatric dataset are not defined the same way, so part of the drop is label shift.
- Dropping ChestMNIST images with other findings makes the target task easier than real triage,
  where the negatives include other diseases.
- Conformal guarantees assume calibration and test data are exchangeable. Under shift they do not
  hold, which is what is being measured, not a defect of the method.
- Grad-CAM is a coarse (7x7 at 224px input) and imperfect explanation. A heatmap on the lungs does
  not prove the model uses the right features.
- Results come from one training seed unless rerun with different `seed` values.

## Next steps

- Repeat runs over several seeds and report the spread.
- Add a Kaggle or RSNA pneumonia dataset as a third domain and train on combinations of sources.
- Try conformal methods designed for shift (weighted conformal with estimated density ratios, or
  recalibrating on a small labelled target sample) and compare the coverage.
- Test-time adaptation (for example updating batch norm or layer norm statistics) as a further
  mitigation baseline.
- Evaluate at full resolution with the original datasets.
