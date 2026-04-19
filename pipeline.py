import os
import numpy as np
import pandas as pd
import torch
import shap
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from transformers import AutoTokenizer, AutoModel

from src.features import extract_physicochemical_features
from src.evaluation import evaluate_regression_model

# load data

train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

X_train_text = train_df["seq"].values
X_test_text = test_df["seq"].values

y_train = train_df["label"].values
y_test = test_df["label"].values


# physiochemical features

X_train_phys = np.array([
    extract_physicochemical_features(seq) for seq in X_train_text
])

X_test_phys = np.array([
    extract_physicochemical_features(seq) for seq in X_test_text
])


# esm embedding

model_name = "facebook/esm2_t6_8M_UR50D"

tokenizer = AutoTokenizer.from_pretrained(model_name)
esm_model = AutoModel.from_pretrained(model_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
esm_model = esm_model.to(device)
esm_model.eval()


def get_embedding(seq):
    inputs = tokenizer(seq, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = esm_model(**inputs)

    return outputs.last_hidden_state.mean(dim=1).cpu().numpy().flatten()


print("Generating ESM embeddings...")

X_train_esm = np.array([get_embedding(seq) for seq in X_train_text])
X_test_esm = np.array([get_embedding(seq) for seq in X_test_text])


# models

phys_model = Ridge()
esm_model_ridge = Ridge()
hybrid_model = Ridge()

# train models
phys_model.fit(X_train_phys, y_train)
esm_model_ridge.fit(X_train_esm, y_train)

X_train_hybrid = np.hstack([X_train_phys, X_train_esm])
X_test_hybrid = np.hstack([X_test_phys, X_test_esm])

hybrid_model.fit(X_train_hybrid, y_train)


# predictions

phys_preds = phys_model.predict(X_test_phys)
esm_preds = esm_model_ridge.predict(X_test_esm)
hybrid_preds = hybrid_model.predict(X_test_hybrid)


# evaluate

results = pd.DataFrame([
    {
        "Model": "Physicochemical",
        "MAE": mean_absolute_error(y_test, phys_preds),
        "R2": r2_score(y_test, phys_preds)
    },
    {
        "Model": "ESM",
        "MAE": mean_absolute_error(y_test, esm_preds),
        "R2": r2_score(y_test, esm_preds)
    },
    {
        "Model": "Hybrid",
        "MAE": mean_absolute_error(y_test, hybrid_preds),
        "R2": r2_score(y_test, hybrid_preds)
    }
])

os.makedirs("results", exist_ok=True)

results.to_csv("results/regression_metrics.csv", index=False)
print(results)


# hybrid shap explanation

print("Running SHAP analysis...")

explainer = shap.Explainer(hybrid_model, X_train_hybrid)
shap_values = explainer(X_test_hybrid[:200])

os.makedirs("results/figures", exist_ok=True)

# shap bar
plt.figure()
shap.plots.bar(shap_values, show=False)
plt.savefig("results/figures/shap_bar.png", bbox_inches="tight")
plt.close()

# shap beeswarm
plt.figure()
shap.plots.beeswarm(shap_values, show=False)
plt.savefig("results/figures/shap_beeswarm.png", bbox_inches="tight")
plt.close()

print("SHAP plots saved.")


# final summary

print("\nFINAL MODEL SUMMARY")
print(results)

print("\nPipeline complete. All outputs saved to /results")
