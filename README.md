# Cross-Dataset Generalization of Deep Learning Models for Diabetic Retinopathy Grading

MSc Dissertation — Data Science, Artificial Intelligence and Digital Business
GISMA University of Applied Sciences
Author: Ravi Yadav | Supervisor: Dr. Ramin Baghaei Mehr

**Full title:** *Evaluating the Generalizability of Deep Learning Models for
Multi-Class Diabetic Retinopathy Detection Using Public Retinal Image Datasets*

## Overview

This repository contains the full experimental pipeline for the dissertation
above. The project trains an EfficientNetB0 baseline for five-class diabetic
retinopathy (DR) severity classification on the APTOS 2019 dataset, evaluates
it zero-shot on two independent external datasets (Messidor-2, IDRiD), runs
two controlled ablation studies, quantifies training-seed instability via a
multi-seed analysis, and tests Test-Time Augmentation (TTA) as a lightweight,
retraining-free mitigation for the observed cross-dataset performance gap.

The dissertation is organised around four research questions:

- **RQ1** — How does the baseline model perform on APTOS 2019 under standard evaluation?
- **RQ2** — How does performance change when evaluated on Messidor-2 and IDRiD without fine-tuning?
- **RQ3** — Can TTA improve cross-dataset performance without retraining?
- **RQ4** — Which DR severity classes are most affected by cross-dataset distribution shift?

## Key Results

| Dataset | Accuracy | Macro F1 | QW Kappa | Class 3 (Severe) Recall |
|---|---|---|---|---|
| APTOS (in-distribution) | 0.814 | 0.657 | 0.889 | 0.429 |
| IDRiD (external) | 0.524 | 0.386 | 0.763 | 0.072 |
| Messidor-2 (external) | 0.601 | 0.290 | 0.380 | 0.040 |

Performance degrades substantially on both external datasets relative to
APTOS, and this degradation is not uniform across severity classes: Severe DR
(Class 3) recall is consistently and severely affected, falling to 0.072 on
IDRiD and 0.040 on Messidor-2. Test-time augmentation, tested as a lightweight
mitigation, produced only small and inconsistent effects and did not reliably
close this gap. Two controlled ablations (class weighting vs. none; frozen vs.
fine-tuned layers) and a 3-seed stability analysis were conducted to
strengthen the modelling contribution and to characterise training-run
variability. Full results, tables, and discussion are provided in the
accompanying dissertation document (Chapters 4 and 5).

## Repository Contents

| File | Description |
|---|---|
| `dr-crossdataset-generalization.ipynb` | Full experimental notebook: data loading, baseline training, cross-dataset evaluation, ablation studies, multi-seed analysis, TTA and repeated TTA evaluation |
| `aptos_train.csv`, `aptos_val.csv`, `aptos_test.csv` | Stratified 70/15/15 train/val/test split files for APTOS 2019 (post-deduplication) |
| `IDRiD_test.csv`, `Messidor_test.csv` | Deduplicated file lists for the external test datasets |
| `results_summary.csv`, `results_summary_with_ci.csv` | Consolidated results tables (point estimates and bootstrapped 95% confidence intervals) |

**Note:** trained model checkpoint files (`.keras`) are not included due to
file size. Re-run the notebook with the fixed seed to retrain; see
"Reproducibility" below regarding expected variation.

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
This variability is itself measured and reported: Section 3.9 of the
dissertation retrains the baseline across three seeds and reports the
resulting mean, standard deviation, and range for key metrics, rather than
treating a single run as definitive. Section 3.8.2 similarly reports repeated
TTA evaluations to quantify uncertainty in TTA's own effect.

## How to Reproduce

1. Download the three datasets from the sources listed above and place them
   in the paths referenced at the top of the notebook (adjust paths as needed
   for your environment).
2. Open `dr-crossdataset-generalization.ipynb` in a GPU-enabled environment
   (Kaggle Notebooks or equivalent) with the package versions listed above.
3. Run all cells sequentially from top to bottom. The notebook is organised
   into numbered sections covering data preparation, baseline training,
   cross-dataset evaluation, ablation studies, multi-seed analysis, and TTA.
4. Expect a full run to take approximately 2–3 hours on a single GPU, due to
   the baseline training, two ablation retrains, and three additional
   multi-seed training runs.

## Dissertation Structure

| Chapter | Contents |
|---|---|
| 1. Introduction | Clinical background, literature-motivated gap, novelty statement, research questions |
| 2. Literature Review | Evaluation gap, architecture progression, methodological considerations, domain shift/TTA/evaluation metrics/class imbalance, comparison table |
| 3. Methodology | Datasets, provenance, label harmonization, data cleaning, split, model architecture, ablations, TTA, multi-seed analysis, reproducibility |
| 4. Results | RQ1–RQ4 findings |
| 5. Discussion | Interpretation of findings and clinical implications |
| 6. Limitations & Future Work | Scope boundaries and concrete next steps |
| 7. Conclusion | Summary of contributions |

For full methodological detail, rationale, and discussion of results, see the
accompanying dissertation document.
