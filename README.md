# Protein Mutation Impact Prediction

## Overview
This project predicts the functional and stability effects of protein mutations using machine learning and protein sequence representations. The main goal is to compare classical feature-based models with pretrained protein embeddings (ESM-2) and evaluate whether embeddings improve predictitions.

The project includes:
- Physicochemical feature engineering
- Character n-gram baselines
- Pretrained protein embeddings (ESM-2)
- Hybrid models combining both feature types
- Statistical evaluation and ablation studies
- Model interpretability using SHAP

---

## Dataset
The dataset is sourced from Hugging Face:

- `proteinglm/stability_prediction`

It contains:
- Protein amino acid sequences (`seq`)
- Stability change / mutation effect labels (`label`)

A local snapshot is stored in `/data` for reproducibility:
- `train.csv`
- `test.csv`

---

## Project Structure

protein-mutation-prediction/
data/ # train/test CSV snapshots

notebooks/ # step-by-step experiments
experiments/ # full reproducibility pipeline
src/ # feature extraction + evaluation modules
results/ # model outputs and metrics
figures/ # generated plots
regression_metrics.csv

pipeline.py # full end-to-end pipeline
requirements.txt
README.md


---

## Modeling Approach

### 1. Physicochemical Baseline
- Amino acid composition
- Hydrophobicity features
- Net charge
- Sequence length

### 2. N-gram Models
- Character-level 1–2 gram representation
- Models:
  - Ridge Regression
  - Random Forest
  - Gradient Boosting

### 3. Protein Embeddings (ESM-2)
- Pretrained transformer model (`facebook/esm2_t6_8M_UR50D`)
- Sequence-level embeddings via mean pooling

### 4. Hybrid Model
- Concatenation of:
  - Physicochemical features
  - ESM embeddings
- Ridge regression model

---

## Evaluation Strategy

Models are evaluated using:
- MAE (Mean Absolute Error)
- R² Score

Ablation studies include:
- Physicochemical only
- ESM only
- Hybrid model

Statistical significance testing:
- Paired t-test across evaluation splits

---

## Interpretability

Model interpretability is performed using SHAP:

- SHAP bar plots (global importance)
- SHAP beeswarm plots (feature distribution effects)

Outputs saved to:
results/figures/


---

## Key Outputs

After running `pipeline.py`, the following are generated:

- `results/regression_metrics.csv`
- `results/figures/shap_bar.png`
- `results/figures/shap_beeswarm.png`

---

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt

Run full pipeline:

python pipeline.py
