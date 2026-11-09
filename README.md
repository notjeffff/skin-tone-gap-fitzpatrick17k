# Hidden at the Darkest Skin Type

Code for *Hidden at the Darkest Skin Type: Fitzpatrick Groupings Dilute a Malignancy-Detection Gap at FST VI*.

Fairness audits of dermatology classifiers usually compare light skin (Fitzpatrick skin types I–III) with dark skin (IV–VI), or merge types V and VI. This repository evaluates malignancy detection on Fitzpatrick17k for each skin type separately. It shows that the gap is concentrated at FST VI and that both standard groupings dilute it.

## Main results (ResNet-50, original Fitzpatrick17k images)

| FST | Images | Malignant | Sensitivity, trained on FST I–III | Sensitivity, trained on all types (5-fold CV) |
|---|---|---|---|---|
| I   | 2,947 | 453 | 0.73 [0.68, 0.78] | 0.73 [0.68, 0.77] |
| II  | 4,808 | 742 | 0.75 [0.71, 0.79] | 0.74 [0.71, 0.77] |
| III | 3,308 | 456 | 0.78 [0.73, 0.82] | 0.77 [0.73, 0.80] |
| IV  | 2,781 | 301 | 0.69 [0.64, 0.74] | 0.77 [0.72, 0.82] |
| V   | 1,533 | 147 | 0.59 [0.51, 0.67] | 0.65 [0.57, 0.73] |
| VI  |   635 |  61 | **0.41 [0.30, 0.54]** | **0.48 [0.36, 0.60]** |

Sensitivity is at threshold 0.5 with 95% Wilson intervals. Light-only results use scores averaged over 5 seeds, on the FST I–III images tested in at least one seed. The image and malignant counts are dataset totals.

- **Gap at FST VI.** FST I–III minus VI sensitivity is +0.34 [0.21, 0.47] under light-only training and +0.27 [0.14, 0.40] under all-type training.
- **Grouping dilutes it.** Under all-type training, a light/dark (I–III vs IV–VI) audit reports a sensitivity gap of only 0.04.
- **Robustness checks.** The gap holds under a second skin-type annotation, after removing near-duplicates (Fitzpatrick17k-C), and after adjusting for source website and cancer subtype.

## Repository layout

```
notebooks/
  01_light_only_resnet50.ipynb              ResNet-50 trained on FST I–III, 5 seeds
  02_efficientnet_and_interventions.ipynb   EfficientNet-B0; +FST IV–V and size-matched training
  03_ddi_external_check.ipynb               all models applied to DDI (no fine-tuning)
  04_posthoc_analyses.ipynb                 duplicates, leak-free interventions, pooled statistics, figures
  05_all_skin_type_cv.ipynb                 all six types, duplicate-aware 5-fold CV
  06_example_images_figure.ipynb            example-image figure
  preprocessing_comparison/                 same pipeline on the background-removed Kaggle copy
src/                                        latest version of every script
requirements.txt
```

Each notebook writes the scripts it runs with `%%writefile`, so it is self-contained on Kaggle. `src/` holds the latest version of each script for reading or running locally.

## Running

The notebooks were run on Kaggle (GPU T4). Run them in order. To let later notebooks attach an earlier notebook's output, save that earlier notebook with *Save & Run All*.

| Notebook | Inputs | Hardware | Time |
|---|---|---|---|
| 01 | `fitzpatrick17k_original` (images) + `fitzpatrick17k` (label CSV) | GPU | ~4–5 h |
| 02 | same datasets + output of 01 | GPU | several hours (11 training runs) |
| 03 | DDI (private dataset) + outputs of 01 and 02 | GPU | inference only |
| 04 | `fitzpatrick17k` + outputs of 02 and 03; internet on | CPU | ~15–25 min |
| 05 | same datasets as 01; internet on | GPU | ~4–5 h |
| 06 | same datasets as 01 | CPU | a few minutes |

## Data

- **Fitzpatrick17k** (Groh et al., CVPRW 2021). Labels come from the official release. Original, unprocessed images come from the Kaggle mirror [`mobaswiralfarabi/fitzpatrick17k-original`](https://www.kaggle.com/datasets/mobaswiralfarabi/fitzpatrick17k-original); `verify_orig.py` checks every file against its MD5 identifier. The CSV from [`nazmusresan/fitzpatrick17k`](https://www.kaggle.com/datasets/nazmusresan/fitzpatrick17k) is used only for labels. That copy's images are background-removed and are used only in `preprocessing_comparison/`.
- **Fitzpatrick17k-C** duplicate clusters (Abhishek, Jain and Hamarneh, *Scientific Data* 2025). Notebooks 04 and 05 download these from the authors' GitHub repository.
- **DDI** (Daneshjou et al., *Science Advances* 2022). DDI is distributed by Stanford AIMI under a research use agreement. It is **not included here and must not be redistributed**. Request access through the dataset's official page and upload it to Kaggle as a **private** dataset.

No images or model weights are stored in this repository.

## Requirements

`torch`, `torchvision` (ResNet-50 and EfficientNet-B0 with ImageNet weights), `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `pillow`, `statsmodels`. Everything is preinstalled on Kaggle; `requirements.txt` lists the packages for a local setup.

## Intended use

This is research code for auditing classifier performance across skin types. The models are not clinical tools and must not be used for diagnosis.
