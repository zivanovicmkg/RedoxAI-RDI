# -*- coding: utf-8 -*-
"""
RedoxAI V5 — Context-Aware Elastic Model (stabilna verzija bez problematične paralelizacije)

Ulaz:
  redox_data_normalized.csv  (u folderu: E:\OneDrive\Desktop\Redox B2)
Izlaz (u istom folderu):
  ai_learned_weights_v5.csv
  RDI_AI_v5_vs_Viability.png
  feature_importance_v5_rf.png
  ai_weight_distribution_v5.png
  v5_cv_scores.txt
"""

import os
os.chdir(r"E:\OneDrive\Desktop\Redox B2")

# Ograniči BLAS niti radi stabilnosti na Windows-u/IDLE
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations

from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ---------- 1) UČITAVANJE ----------
df = pd.read_csv("redox_data_normalized.csv")

# Target
y = df["viability_pct"].astype(float).values

# Numeričke kolone (uzmi samo one koje zaista postoje)
num_candidates = [
    "norm_superoxide_uM_per_viable_cell",
    "norm_nitrites_uM_per_viable_cell",
    "norm_h2o2_uM_per_viable_cell",
    "norm_gsh_uM_per_viable_cell",
    "norm_gssg_uM_per_viable_cell",
    "norm_GSH_GSSG_ratio",
    "norm_ROS_total",
    "norm_RNS_contribution",
    "norm_Antioxidant_capacity",
    "norm_GSH_efficiency",
    "norm_time_h",
    "norm_concentration_uM",
]
num_cols = [c for c in num_candidates if c in df.columns]

# Kategorijske (ako postoje)
cat_candidates = ["cell_type", "treatment", "batch", "experiment_id"]
cat_cols = [c for c in cat_candidates if c in df.columns]

# Grupe za GroupKFold
if "experiment_id" in df.columns:
    groups = df["experiment_id"].astype(str).values
elif "batch" in df.columns:
    groups = df["batch"].astype(str).values
else:
    # fallback: svaka tačka je posebna grupa (radi, ali nije idealno)
    groups = np.arange(len(df))

# ---------- 2) DIZAJN OSOBINA ----------
ct = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ],
    remainder="drop",
)

# Interakcije (samo interakcije, bez kvadrata) — možeš staviti interaction_only=False ako želiš i kvadrate
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
interaction_pairs = list(combinations(num_cols, 2))  # samo zbog imenovanja kasnije

# ---------- 3) MODEL ----------
enet = ElasticNetCV(
    l1_ratio=[0.05, 0.1, 0.3, 0.5, 0.7, 0.9],
    alphas=np.logspace(-3, 2, 30),
    cv=5,
    max_iter=20000,
    n_jobs=1,  # bez paralelizacije
    random_state=RANDOM_STATE,
)

pipe = Pipeline(steps=[
    ("prep", ct),
    ("poly", poly),
    ("model", enet),
])

# ---------- 4) VALIDACIJA (GroupKFold) ----------
n_splits = max(3, min(5, len(np.unique(groups))))
cv = GroupKFold(n_splits=n_splits)

X_base = df[num_cols + cat_cols].copy()

# KLJUČNA IZMJENA: prosleđujemo groups=groups
cv_scores = cross_val_score(
    pipe, X_base, y,
    scoring="r2",
    cv=cv,
    groups=groups,
    n_jobs=1,
)

cv_mean = float(np.mean(cv_scores))
cv_std = float(np.std(cv_scores))

# ---------- 5) FIT NA CELOM SKUPU I RDI ----------
pipe.fit(X_base, y)
rdi_v5 = pipe.predict(X_base)

# ---------- 6) KOEFICIJENTI ----------
prep = pipe.named_steps["prep"]
model = pipe.named_steps["model"]

num_feat_names = num_cols.copy()
cat_feat_names = []
if len(cat_cols) > 0:
    ohe = [tr for tr in prep.transformers_ if tr[0] == "cat"][0][1]
    cat_feat_names = list(ohe.get_feature_names_out(cat_cols))

prepped_names = num_feat_names + cat_feat_names

# Imena posle PolynomialFeatures: bazne + sve interakcije (prepped × prepped)
poly_names = []
for i, name in enumerate(prepped_names):
    poly_names.append(name)
for i in range(len(prepped_names)):
    for j in range(i+1, len(prepped_names)):
        poly_names.append(f"{prepped_names[i]} {prepped_names[j]}")

coefs = model.coef_.ravel()
intercept = float(model.intercept_)

coef_df = pd.DataFrame({
    "feature": poly_names[:len(coefs)],
    "coef": coefs
})

def is_numeric_or_num_interaction(fname: str) -> bool:
    if fname in num_cols:
        return True
    if " " in fname:
        a, b = fname.split(" ", 1)
        return (a in num_cols) and (b in num_cols)
    return False

coef_df_readable = coef_df[coef_df["feature"].apply(is_numeric_or_num_interaction)].copy()
coef_df_readable.sort_values("coef", ascending=False, inplace=True)

# ---------- 7) RANDOM FOREST importance (samo bazne numeričke) ----------
rf = RandomForestRegressor(
    n_estimators=600,
    random_state=RANDOM_STATE,
    n_jobs=1,
)
rf.fit(df[num_cols], y)
rf_imp = pd.Series(rf.feature_importances_, index=num_cols).sort_values(ascending=False)

# ---------- 8) SNIMANJE ----------
out_weights = "ai_learned_weights_v5.csv"
out_scatter = "RDI_AI_v5_vs_Viability.png"
out_rf = "feature_importance_v5_rf.png"
out_hist = "ai_weight_distribution_v5.png"
out_cv = "v5_cv_scores.txt"

coef_df_readable.to_csv(out_weights, index=False, encoding="utf-8-sig")

with open(out_cv, "w", encoding="utf-8") as f:
    f.write(f"GroupKFold(n_splits={n_splits}) CV R²: {cv_mean:.3f} ± {cv_std:.3f}\n")
    f.write(f"Train R²: {r2_score(y, rdi_v5):.3f}\n")
    f.write(f"Intercept: {intercept:.6f}\n\n")
    f.write("TOP koeficijenti (num & num×num):\n")
    f.write(coef_df_readable.head(30).to_string(index=False))

plt.figure(figsize=(8, 6))
plt.scatter(rdi_v5, y, alpha=0.8)
plt.xlabel("RDI_AI_v5 (predikcija modela)")
plt.ylabel("Viability (%)")
plt.title(f"RDI_AI_v5 vs Viability | Train R²={r2_score(y, rdi_v5):.3f} | CV R²={cv_mean:.3f}±{cv_std:.3f}")
plt.grid(alpha=0.2)
plt.tight_layout()
plt.savefig(out_scatter, dpi=150)
plt.close()

plt.figure(figsize=(8, 6))
rf_imp.iloc[::-1].plot(kind="barh")
plt.xlabel("RF importance")
plt.title("Random Forest – najvažnije osobine (V5)")
plt.tight_layout()
plt.savefig(out_rf, dpi=150)
plt.close()

top10 = coef_df_readable.reindex(coef_df_readable["coef"].abs().sort_values(ascending=False).index).head(10)
plt.figure(figsize=(12, 6))
plt.bar(range(len(top10)), top10["coef"])
plt.xticks(range(len(top10)), [t.replace("_uM_per_viable_cell","").replace("norm_","") for t in top10["feature"]], rotation=45, ha="right")
plt.ylabel("Koeficijent (ElasticNet)")
plt.title("Top koeficijenti (V5) – bazne i num×num interakcije")
plt.tight_layout()
plt.savefig(out_hist, dpi=150)
plt.close()

print("V5 završeno.")
print(f"CV R²: {cv_mean:.3f} ± {cv_std:.3f}")
print(f"Train R²: {r2_score(y, rdi_v5):.3f}")
print(f"Sačuvano: {out_weights}, {out_scatter}, {out_rf}, {out_hist}, {out_cv}")
