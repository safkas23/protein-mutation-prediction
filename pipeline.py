import os
import numpy as np
import pandas as pd
import torch
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_score
from scipy.stats import ttest_rel
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from transformers import AutoTokenizer, AutoModel
from src.features import extract_physicochemical_features
from src.evaluation import regression_metrics
from difflib import SequenceMatcher


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

ablated_model = Ridge()
ablated_model.fit(X_train_phys, y_train)

ablated_preds = ablated_model.predict(X_test_phys)

ablation_results = pd.DataFrame([
    {"Model":"Physicochemical",
     "MAE":mean_absolute_error(y_test,phys_preds)},

    {"Model":"ESM",
     "MAE":mean_absolute_error(y_test,esm_preds)},

    {"Model":"Hybrid",
     "MAE":mean_absolute_error(y_test,hybrid_preds)},

    {"Model":"Ablated",
     "MAE":mean_absolute_error(y_test,ablated_preds)}
])

ablation_results.to_csv(
    "results/ablation_results.csv",
    index=False
)

y_shuffled=np.random.permutation(y_train)

nc_model=Ridge()
nc_model.fit(X_train_hybrid,y_shuffled)
nc_preds=nc_model.predict(X_test_hybrid)

negative_control=pd.DataFrame([
{
"MAE":
mean_absolute_error(y_test,nc_preds),

"R2":
r2_score(y_test,nc_preds)
}
])

negative_control.to_csv(
"results/negative_control_results.csv",
index=False
)

kf=KFold(
n_splits=5,
shuffle=True,
random_state=42
)

phys_cv=cross_val_score(
Ridge(),
X_train_phys,
y_train,
cv=kf,
scoring="neg_mean_absolute_error"
)

hybrid_cv=cross_val_score(
Ridge(),
X_train_hybrid,
y_train,
cv=kf,
scoring="neg_mean_absolute_error"
)

p=ttest_rel(
-hybrid_cv,
-phys_cv
)

stats_results=pd.DataFrame([
{
"p_value":p.pvalue
}
])

stats_results.to_csv(
"results/statistical_tests.csv",
index=False
)

# hybrid shap explanation
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

# shap removal
top_features=np.argsort(
np.abs(
shap_values.values
).mean(axis=0)
)[-10:]

X_train_removed=np.delete(
X_train_hybrid,
top_features,
axis=1
)

X_test_removed=np.delete(
X_test_hybrid,
top_features,
axis=1
)

removed_model=Ridge()
removed_model.fit(
X_train_removed,
y_train
)

removed_preds=removed_model.predict(
X_test_removed
)

shap_removal=pd.DataFrame([
{
"MAE":
mean_absolute_error(
y_test,
removed_preds
)
}
])

shap_removal.to_csv(
"results/shap_feature_removal.csv",
index=False
)

# model comparisons
models={
"Ridge":
Ridge(),

"RandomForest":
RandomForestRegressor(),

"GradientBoosting":
GradientBoostingRegressor()
}

rows=[]
for model_name,model in models.items():

    for feature_set,Xtr,Xte in [

    ("Physicochemical",
     X_train_phys,
     X_test_phys),

    ("ESM",
     X_train_esm,
     X_test_esm),

    ("Hybrid",
     X_train_hybrid,
     X_test_hybrid)

    ]:

        model.fit(Xtr,y_train)

        preds=model.predict(Xte)

        rows.append({
        "Model":
        model_name,

        "Features":
        feature_set,

        "MAE":
        mean_absolute_error(
            y_test,
            preds
        ),

        "R2":
        r2_score(
            y_test,
            preds
        )
        })

pd.DataFrame(rows).to_csv(
"results/model_comparison.csv",
index=False
)

# protein leakage analysis
def seq_similarity(a,b):
    return SequenceMatcher(
        None,
        a,
        b
    ).ratio()

train_seqs=set(X_train_text)
test_seqs=set(X_test_text)

# exact overlap
exact_overlap=len(
train_seqs.intersection(
test_seqs
))

# near duplicates
similar_pairs=0

for test_seq in list(test_seqs)[:100]:

    for train_seq in list(train_seqs)[:500]:

        if seq_similarity(
            test_seq,
            train_seq
        ) > .90:
            similar_pairs += 1
            break

leakage_results=pd.DataFrame([{

"Exact_Overlap":
exact_overlap,

"Near_Duplicate_Count":
similar_pairs

}])

leakage_results.to_csv(
"results/leakage_analysis.csv",
index=False
)

print("Leakage analysis complete.")

# feature importance consistency
rf=RandomForestRegressor()

rf.fit(
X_train_hybrid,
y_train
)

rf_importance=rf.feature_importances_
gb=GradientBoostingRegressor()

gb.fit(
X_train_hybrid,
y_train
)

gb_importance=gb.feature_importances_

ridge=Ridge()
ridge.fit(
X_train_hybrid,
y_train
)

ridge_importance=np.abs(
ridge.coef_
)

importance_df=pd.DataFrame({

"Feature":

range(
len(ridge_importance)
),

"Ridge":
ridge_importance,

"RandomForest":
rf_importance,

"GradientBoost":
gb_importance

})

importance_df.to_csv(
"results/feature_importance_consistency.csv",
index=False
)

print(
"Feature importance consistency saved."
)

print(results)
print("Pipeline Complete.")
print("All outputs saved in /results")
