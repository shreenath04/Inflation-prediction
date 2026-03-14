import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import LeaveOneOut

# ── 1. Load data ──────────────────────────────────────────────────────────────
df = pd.read_csv("final_model_data.csv")

# ── 2. Build regime3 feature ──────────────────────────────────────────────────
NEG_DEV_EPS = -0.03
regime = np.zeros(len(df), dtype=int)
regime[df["is_crisis"] == 1] = +1
regime[(df["is_crisis"] == 1) & (df["Deviation"] <= NEG_DEV_EPS)] = -1
df["regime3"] = regime

# ── 3. Feature sets ───────────────────────────────────────────────────────────
POS_FEATS = ["Synthetic_Target_Rate", "Real GDP (Percent Change)", "Deviation", "is_post_2008", "regime3"]
NEG_FEATS = ["Unemployment Rate", "Deviation", "is_post_2008", "regime3"]
TARGET    = "Inflation Rate"

y = df[TARGET].values

# ── 4. LOOCV to compute ensemble weights ──────────────────────────────────────
print("Running LOOCV to compute ensemble weights...")
loo = LeaveOneOut()
pred_pos, pred_neg, true_y = [], [], []

for train_idx, test_idx in loo.split(df):
    X_train_pos = df[POS_FEATS].iloc[train_idx]
    X_test_pos  = df[POS_FEATS].iloc[test_idx]
    X_train_neg = df[NEG_FEATS].iloc[train_idx]
    X_test_neg  = df[NEG_FEATS].iloc[test_idx]
    y_train     = y[train_idx]
    y_test      = y[test_idx]

    rf_p = RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42)
    rf_n = RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42)
    rf_p.fit(X_train_pos, y_train)
    rf_n.fit(X_train_neg, y_train)

    pred_pos.append(rf_p.predict(X_test_pos)[0])
    pred_neg.append(rf_n.predict(X_test_neg)[0])
    true_y.append(y_test[0])

# ── 5. Compute weights from LOOCV performance ─────────────────────────────────
r2p  = max(r2_score(true_y, pred_pos), 1e-6)
r2n  = max(r2_score(true_y, pred_neg), 1e-6)
msep = max(mean_squared_error(true_y, pred_pos), 1e-9)
msen = max(mean_squared_error(true_y, pred_neg), 1e-9)

sp   = r2p / msep
sn   = r2n / msen
w_pos = sp / (sp + sn)
w_neg = 1.0 - w_pos

pred_final = w_pos * np.array(pred_pos) + w_neg * np.array(pred_neg)
r2_final   = r2_score(true_y, pred_final)

print(f"R² (pos): {r2p:.3f} | R² (neg): {r2n:.3f} | R² (ensemble): {r2_final:.3f}")
print(f"Weights  → pos: {w_pos:.2f} | neg: {w_neg:.2f}")

# ── 6. Train final models on FULL dataset ─────────────────────────────────────
print("Training final models on full dataset...")
rf_pos_full = RandomForestRegressor(n_estimators=400, max_depth=5, random_state=42)
rf_neg_full = RandomForestRegressor(n_estimators=400, max_depth=5, random_state=42)
rf_pos_full.fit(df[POS_FEATS], y)
rf_neg_full.fit(df[NEG_FEATS], y)

# ── 7. Save everything ────────────────────────────────────────────────────────
joblib.dump(rf_pos_full, "model_pos.pkl")
joblib.dump(rf_neg_full, "model_neg.pkl")
joblib.dump({"w_pos": w_pos, "w_neg": w_neg}, "weights.pkl")
joblib.dump({"pos_feats": POS_FEATS, "neg_feats": NEG_FEATS}, "features.pkl")

print("✅ Saved: model_pos.pkl, model_neg.pkl, weights.pkl, features.pkl")
