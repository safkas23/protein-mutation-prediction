import os
import numpy as np
import pandas as pd
import torch
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score
from sklearn.linear_model import Ridge
from difflib import SequenceMatcher
from src.features import extract_physicochemical_features
from src.evaluation import evaluate_regression_model, paired_t_test, compute_cv_summary
from src.embedding_loader import load_esm_model, get_embeddings_batch, combine_features

# load data
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

X_train_text = train_df["seq"].values
X_test_text = test_df["seq"].values

y_train = train_df["label"].values
y_test = test_df["label"].values

def assign_protein_group(seq):
    """Assign protein family label based on sequence length."""
    length = len(seq)
    if length == 43:
        return 0  # GB1
    elif length == 50:
        return 1  # GFP
    else:
        return -1
 
train_groups = np.array([assign_protein_group(s) for s in X_train_text])
 
logo = LeaveOneGroupOut()
valid_group_mask = train_groups != -1
X_train_text_grouped = X_train_text[valid_group_mask]
y_train_grouped = y_train[valid_group_mask]
groups_grouped = train_groups[valid_group_mask]
 
print(f"Protein group distribution (train): "
      f"GB1={np.sum(train_groups==0)}, GFP={np.sum(train_groups==1)}")

# physiochemical features
X_train_phys = np.array([
    extract_physicochemical_features(seq) for seq in X_train_text
])
X_test_phys = np.array([
    extract_physicochemical_features(seq) for seq in X_test_text
])
X_train_phys_grouped = X_train_phys[valid_group_mask]

# esm embedding
tokenizer, esm_encoder, device = load_esm_model()
 
X_train_esm = get_embeddings_batch(X_train_text, tokenizer, esm_encoder, device)
X_test_esm  = get_embeddings_batch(X_test_text,  tokenizer, esm_encoder, device)
X_train_esm_grouped = X_train_esm[valid_group_mask]

# hybrid feature matrices
X_train_hybrid = combine_features(X_train_phys, X_train_esm)
X_test_hybrid  = combine_features(X_test_phys,  X_test_esm)
X_train_hybrid_grouped = X_train_hybrid[valid_group_mask]

# num of features
n_phys = X_train_phys.shape[1]

# train models
phys_model = Ridge()
esm_model_ridge = Ridge()
hybrid_model = Ridge()

phys_model.fit(X_train_phys, y_train)
esm_model_ridge.fit(X_train_esm, y_train)
hybrid_model.fit(X_train_hybrid, y_train)

# predictions on held-out test
phys_preds = phys_model.predict(X_test_phys)
esm_preds = esm_model_ridge.predict(X_test_esm)
hybrid_preds = hybrid_model.predict(X_test_hybrid)

# evaluate
os.makedirs("results", exist_ok=True)

results = pd.DataFrame([
    {
        "Model": "Physicochemical (Ridge)",
        **evaluate_regression_model(y_test, phys_preds)
    },
    {
        "Model": "ESM (Ridge)",
        **evaluate_regression_model(y_test, esm_preds)
    },
    {
        "Model": "Hybrid (Ridge)",
        **evaluate_regression_model(y_test, hybrid_preds)
    }
])

results.to_csv("results/regression_metrics.csv", index=False)
print("\nRegression Metrics")
print(results.to_string(index=False))

# ablated model
print("\nComputing SHAP values for ablation...")
explainer = shap.Explainer(hybrid_model, X_train_hybrid)
shap_values = explainer(X_test_hybrid[:200])
 
mean_shap = np.abs(shap_values.values).mean(axis=0)
n_phys = X_train_phys.shape[1]

embedding_shap = mean_shap[n_phys:]
top_embedding_indices = np.argsort(embedding_shap)[-10:] + n_phys

# remove top 10 features
X_train_ablated = np.delete(X_train_hybrid, top_embedding_indices, axis=1)
X_test_ablated = np.delete(X_test_hybrid, top_embedding_indices, axis=1)
 
ablated_model = Ridge()
ablated_model.fit(X_train_ablated, y_train)
ablated_preds = ablated_model.predict(X_test_ablated)

ablation_results = pd.DataFrame([
    {"Model": "Physicochemical only",                
     **evaluate_regression_model(y_test, phys_preds)
     },
    {"Model": "ESM only",                            
     **evaluate_regression_model(y_test, esm_preds)
     },
    {"Model": "Hybrid (Phys + ESM)",                 
     **evaluate_regression_model(y_test, hybrid_preds)
     },
    {"Model": "Ablated (top-10 ESM dims removed)",   
     **evaluate_regression_model(y_test, ablated_preds)
    },
])

ablation_results.to_csv("results/ablation_results.csv", index=False)
print("\nAblation Study")
print(ablation_results.to_string(index=False))

# protein-stratified validation
print("\nRunning protein-stratified cross-validation...")
 
phys_cv = cross_val_score(
    Ridge(), X_train_phys_grouped, y_train_grouped,
    cv=logo, groups=groups_grouped,
    scoring="neg_mean_absolute_error"
)
esm_cv = cross_val_score(
    Ridge(), X_train_esm_grouped, y_train_grouped,
    cv=logo, groups=groups_grouped,
    scoring="neg_mean_absolute_error"
)
hybrid_cv = cross_val_score(
    Ridge(), X_train_hybrid_grouped, y_train_grouped,
    cv=logo, groups=groups_grouped,
    scoring="neg_mean_absolute_error"
)
 
phys_cv_mae   = -phys_cv
esm_cv_mae    = -esm_cv
hybrid_cv_mae = -hybrid_cv
 
print(f"  Physicochemical CV MAE per fold: {phys_cv_mae}")
print(f"  ESM CV MAE per fold:             {esm_cv_mae}")
print(f"  Hybrid CV MAE per fold:          {hybrid_cv_mae}")

phys_summary   = compute_cv_summary(phys_cv_mae)
esm_summary    = compute_cv_summary(esm_cv_mae)
hybrid_summary = compute_cv_summary(hybrid_cv_mae)
 
# paired t-test 
stat_h_vs_p = paired_t_test(hybrid_cv_mae, phys_cv_mae)
stat_h_vs_e = paired_t_test(hybrid_cv_mae, esm_cv_mae)

stats_results = pd.DataFrame([
    {
        "Comparison":      "Hybrid vs Physicochemical",
        "CV_Strategy":     "Protein-stratified",
        "n_folds":         len(phys_cv_mae),
        "Phys_MAE_mean":   phys_summary["mean"],
        "Phys_MAE_std":    phys_summary["std"],
        "ESM_MAE_mean":    esm_summary["mean"],
        "ESM_MAE_std":     esm_summary["std"],
        "Hybrid_MAE_mean": hybrid_summary["mean"],
        "Hybrid_MAE_std":  hybrid_summary["std"],
        "t_stat":          stat_h_vs_p["t_stat"],
        "p_value":         stat_h_vs_p["p_value"],
    },
    {
        "Comparison":      "Hybrid vs ESM",
        "CV_Strategy":     "Protein-stratified",
        "n_folds":         len(esm_cv_mae),
        "Phys_MAE_mean":   phys_summary["mean"],
        "Phys_MAE_std":    phys_summary["std"],
        "ESM_MAE_mean":    esm_summary["mean"],
        "ESM_MAE_std":     esm_summary["std"],
        "Hybrid_MAE_mean": hybrid_summary["mean"],
        "Hybrid_MAE_std":  hybrid_summary["std"],
        "t_stat":          stat_h_vs_e["t_stat"],
        "p_value":         stat_h_vs_e["p_value"],
    }
])
 
stats_results.to_csv("results/statistical_tests.csv", index=False)
print("\nStatistical Tests")
print(stats_results[["Comparison", "Hybrid_MAE_mean", "Hybrid_MAE_std",
                      "t_stat", "p_value"]].to_string(index=False))

# negative control
np.random.seed(42)
y_shuffled = np.random.permutation(y_train)
nc_model = Ridge()
nc_model.fit(X_train_hybrid, y_shuffled)
nc_preds = nc_model.predict(X_test_hybrid)

negative_control = pd.DataFrame([{
    "Description": "Hybrid model trained on shuffled labels",
    **evaluate_regression_model(y_test, nc_preds)
}])
negative_control.to_csv(
    "results/negative_control_results.csv", 
    index=False
    )

# shap plots
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

print("\nSHAP plots saved.")

# shap feature removal
shap_removal_rows = []
for n_remove in [5, 10, 20, 50]:
    top_idx = np.argsort(mean_shap[n_phys:])[-n_remove:] + n_phys
    Xtr_r = np.delete(X_train_hybrid, top_idx, axis=1)
    Xte_r = np.delete(X_test_hybrid, top_idx, axis=1)
    m = Ridge().fit(Xtr_r, y_train)
    shap_removal_rows.append({
        "Features_Removed": n_remove,
        **evaluate_regression_model(y_test, m.predict(Xte_r))
    })
pd.DataFrame(shap_removal_rows).to_csv("results/shap_feature_removal.csv", index=False)

# multi-model comparisons
model_zoo = {
    "Ridge":            Ridge(),
    "RandomForest":     RandomForestRegressor(n_estimators=100, random_state=42),
    "GradientBoosting": GradientBoostingRegressor(random_state=42)
}
rows = []
for mname, model in model_zoo.items():
    for feature_set, Xtr, Xte in [
        ("Physicochemical", X_train_phys,   X_test_phys),
        ("ESM",             X_train_esm,    X_test_esm),
        ("Hybrid",          X_train_hybrid, X_test_hybrid),
    ]:
        model.fit(Xtr, y_train)
        preds = model.predict(Xte)
        rows.append({"Model": mname, "Features": feature_set,
                     **evaluate_regression_model(y_test, preds)})

pd.DataFrame(rows).to_csv("results/model_comparison.csv", index=False)

# feature importance consistency
rf_fi = RandomForestRegressor(random_state=42).fit(X_train_hybrid, y_train).feature_importances_
gb_fi = GradientBoostingRegressor(random_state=42).fit(X_train_hybrid, y_train).feature_importances_
rg_fi = np.abs(Ridge().fit(X_train_hybrid, y_train).coef_)
 
pd.DataFrame({
    "Feature":       range(len(rg_fi)),
    "Ridge":         rg_fi,
    "RandomForest":  rf_fi,
    "GradientBoost": gb_fi
}).to_csv("results/feature_importance_consistency.csv", index=False)

# train/test sequence leakage check
train_seqs = set(X_train_text)
test_seqs  = set(X_test_text)
exact_overlap = len(train_seqs.intersection(test_seqs))
 
similar_pairs = 0
for test_seq in list(test_seqs)[:100]:
    for train_seq in list(train_seqs)[:500]:
        if SequenceMatcher(None, test_seq, train_seq).ratio() > 0.90:
            similar_pairs += 1
            break
 
pd.DataFrame([{
    "Exact_Overlap":        exact_overlap,
    "Near_Duplicate_Count": similar_pairs,
    "ESM_Pretraining_Note": (
        "ESM-2 was pretrained on UniRef50. Overlap between test sequences "
        "and UniRef50 was not checked programmatically. This is a known "
        "limitation: embedding model performance may be inflated if test "
        "proteins appeared in ESM-2 pretraining data."
    )
}]).to_csv("results/leakage_analysis.csv", index=False)
 
print("\nLeakage analysis complete.")
print("\nPipeline Complete. All outputs saved in /results")
