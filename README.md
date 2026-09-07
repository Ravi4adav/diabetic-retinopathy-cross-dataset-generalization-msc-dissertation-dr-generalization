# Cross-Dataset Generalization of Deep Learning Models for Diabetic Retinopathy Grading

MSc Dissertation — Data Science, Artificial Intelligence and Digital Business
GISMA University of Applied Sciences
Author: Ravi Yadav | Supervisor: Dr. Ramin Baghaei Mehr

## Overview

This repository contains the full experimental pipeline for the dissertation
*"Evaluating the Generalizability of Deep Learning Models for Multi-Class
Diabetic Retinopathy Detection Using Public Retinal Image Datasets."*

The project trains an EfficientNetB0 baseline for five-class diabetic
retinopathy (DR) severity classification on the APTOS 2019 dataset, evaluates
it zero-shot on two independent external datasets (Messidor-2, IDRiD), runs
two controlled ablation studies, quantifies training-seed instability via a
multi-seed analysis, and tests Test-Time Augmentation (TTA) as a lightweight,
retraining-free mitigation for the observed cross-dataset performance gap.

## Repository Contents

| File | Description |
|---|---|
| `dr-crossdataset-generalization.ipynb` | Full experimental notebook (data loading, baseline training, cross-dataset evaluation, ablations, multi-seed analysis, TTA) |
| `aptos_train.csv`, `aptos_val.csv`, `aptos_test.csv` | Stratified 70/15/15 train/val/test split files for APTOS 2019 (post-deduplication) |
| `IDRiD_test.csv`, `Messidor_test.csv` | Deduplicated file lists for the external test datasets |
| `results_summary.csv`, `results_summary_with_ci.csv` | Consolidated results tables (point estimates and bootstrapped 95% confidence intervals) |

**Note:** trained model checkpoint files (`.keras`) are not included due to file
size. Re-run the notebook with the fixed seed to retrain; see "Reproducibility"
below regarding expected variation.

## Dataset Sources

- **APTOS 2019** — Official Kaggle competition page:
  https://www.kaggle.com/c/aptos2019-blindness-detection

- **Messidor-2** — Accessed via a third-party Kaggle mirror (preprocessed: black
  background cropped by the uploader):
  https://www.kaggle.com/datasets/mariaherrerot/messidor2preprocess
  Original source: https://www.adcis.net/en/third-party/messidor2/
  Labels: "MESSIDOR-2 DR Grades" (Google Brain / Krause et al., 2018):
  https://www.kaggle.com/datasets/google-brain/messidor2-dr-grades

- **IDRiD** — Accessed via a third-party Kaggle mirror (direct redistribution):
  https://www.kaggle.com/datasets/mariaherrerot/idrid-dataset
  Original source (Porwal et al., 2018): https://doi.org/10.21227/H25W98

All three datasets use the same 0–4 ICDR-style severity scale (No DR, Mild,
Moderate, Severe, Proliferative DR). See Section 3.3 (Label Harmonization) of
the dissertation for a discussion of grading-protocol differences across
datasets, since identical numeric scales do not guarantee identical grading
conditions.

Due to licensing terms, the image files themselves are not redistributed here;
they must be downloaded directly from the sources above.

## Environment

Experiments were run on Kaggle's standard GPU-enabled notebook environment
(NVIDIA T4/P100), using:

- TensorFlow 2.20.0
- scikit-learn 1.6.1
- NumPy 2.0.2
- Pandas 2.3.3

## Reproducibility

Random seeds are fixed (`tf.random.set_seed(42)`, `np.random.seed(42)`) and
deterministic GPU execution is enabled. However, residual GPU-level
nondeterminism means exact point values — particularly Severe DR (Class 3)
recall, which has limited test-set support — may vary slightly between runs.
This variability is itself measured and reported: see Section 3.9 (Multi-Seed
Stability Analysis) of the dissertation, which retrains the baseline across
three seeds and reports the resulting mean, standard deviation, and range for
key metrics, rather than treating a single run as definitive.

## How to Reproduce

1. Download the three datasets from the sources listed above and place them
   in the paths referenced at the top of the notebook (adjust paths as needed
   for your environment).
2. Open `dr-crossdataset-generalization.ipynb` in a GPU-enabled environment
   (Kaggle Notebooks or equivalent) with the package versions listed above.
3. Run all cells sequentially from top to bottom. The notebook is organised
   into 16 numbered sections, covering data preparation, baseline training,
   cross-dataset evaluation, ablation studies, multi-seed analysis, and TTA.
4. Expect a full run to take approximately 2–3 hours on a single GPU, due to
   the baseline training, two ablation retrains, and three additional
   multi-seed training runs.

## Dissertation Structure Reference

For full methodological detail, rationale, and discussion of results, see the
accompanying dissertation document, Chapter 3 (Methodology) and Chapter 4
(Results).
