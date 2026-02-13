# Protein Mutation Impact Prediction

## Overview
This project focuses on predicting the functional and stability effects of protein mutations
using machine learning models. The goal is to compare baseline models with embedding-based
approaches and evaluate their ability to generalize across proteins.

## Dataset
The dataset contains:
- Wild-type protein sequence
- Mutant sequence
- Mutation position
- Experimental outcome (stability change or functional impact)

## Project Structure
```
protein-mutation-prediction/
│
├── data/               # Raw and processed datasets
├── notebooks/          # Jupyter notebooks for EDA and experiments
├── src/                # Source code for preprocessing and modeling
├── figures/            # Generated plots and visualizations
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

## Modeling Approach
- Baseline Models:
  - Logistic Regression
  - Random Forest
- Advanced Models:
  - Gradient Boosting
  - Embedding-based Neural Networks

## Validation Strategy
- Stratified cross-validation by protein ID
- Metrics:
  - AUC (classification)
  - RMSE (regression)

## Future Work
- Integrate pretrained protein embeddings
- SHAP-based feature interpretation
- Hyperparameter optimization

## Author
s k
