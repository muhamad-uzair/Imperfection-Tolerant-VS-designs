import re
import json
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ExpSineSquared, ConstantKernel as C
from sklearn.model_selection import cross_val_score, cross_val_predict, KFold, RepeatedKFold
from sklearn.metrics import root_mean_squared_error
from pyDOE import lhs
import imageio.v2 as imageio
import matplotlib
import matplotlib.pyplot as plt

# === PLOTTING SETUP ===
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams.update({'font.size': 22, 'axes.labelsize': 22, 'xtick.labelsize': 22, 'ytick.labelsize': 22})

# === CONFIG ===
CSV_PATH = r"Link to sythetic imperfection dataset (.csv file)" #provide dataset link here please
MASS_MODEL_OUT = "mass_model.pkl"
PCR_MODEL_OUT  = "pcr_model.pkl"
RLP_MODEL_OUT  = "rlp_model.pkl"
FEATURES_JSON  = "model_features.json"

USE_KRIGING_PRETRAIN = True
SEED_SIZE, BATCH, ITERS = 150, 30, 30
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# === LOAD DATA ===
df = pd.read_csv(CSV_PATH)

# === Inputs: winding angles ===
angle_cols = sorted([c for c in df.columns if re.match(r'^[Ss]\d+_a\d+$', c)])
if not angle_cols:
    angle_cols = [c for c in df.columns if c.lower().startswith("alpha") and "mean" not in c.lower()]

if not angle_cols:
    raise ValueError("No angle columns found. Expected columns like 'S8_a1'..'S8_a8' or 'alpha*' (excluding mean).")

df[angle_cols] = df[angle_cols].fillna(0)


# === Targets ===
def pick(colnames, *cands):
    for c in cands:
        if c in colnames: return c
    return None

mass_col = pick(df.columns, "Mass", "mass")
pcr_col  = pick(df.columns, "Pcr", "buckling", "Buckling")
rlp_col  = pick(df.columns, "RL_Perfect", "Perfect Linear buckling", "RL perfect", "RLperfect")

for need, name in [(mass_col, "Mass"), (pcr_col, "Pcr"), (rlp_col, "RL_Perfect")]:
    if need is None:
        raise ValueError(f"Missing required target column (e.g., '{name}').")
df = df.dropna(subset=[mass_col, pcr_col, rlp_col])
print(f"Total Samples: {len(df)}")

# Inputs/outputs
input_cols = angle_cols
X      = df[input_cols]
y_mass = df[mass_col]
y_pcr  = df[pcr_col]
y_rlp  = df[rlp_col]   
X_full = X.copy()

# === SURROGATE MODEL ===
def model():
    kernel = C(1.0) * (RBF() + ExpSineSquared(length_scale=1.0, periodicity=3.0))
    return make_pipeline(
        StandardScaler(),
        GaussianProcessRegressor(kernel=kernel, alpha=1e-5, normalize_y=True,
                                 random_state=42, n_restarts_optimizer=10)
    )

# === Kriging pretrain on Pcr uncertainty ===
if USE_KRIGING_PRETRAIN:
    rng = np.random.RandomState(RANDOM_SEED)
    all_idx = np.arange(len(X))

    seed_pick = rng.choice(all_idx, size=min(SEED_SIZE, len(X)), replace=False)
    labeled = set(seed_pick.tolist())
    selection_log = [(int(i), 0) for i in seed_pick]   
    iter_sigma_stats = []  
    gp_pcr = model()

    for t in range(1, ITERS + 1):  # iterations labeled 1..ITERS
        labeled_idx = np.fromiter(labeled, dtype=int)
        pool_idx = np.setdiff1d(all_idx, labeled_idx)
        if pool_idx.size == 0:
            print("[Kriging pretrain] Pool exhausted; stopping early.")
            break

        gp_pcr.fit(X.iloc[labeled_idx], y_pcr.iloc[labeled_idx])
        _, sigma = gp_pcr.predict(X.iloc[pool_idx], return_std=True)

        take = min(BATCH, len(pool_idx))
        order = np.argsort(-sigma)  
        top = pool_idx[order[:take]]
        top_sigmas = sigma[order[:take]]
        for i in top:
            selection_log.append((int(i), t))
        iter_sigma_stats.append({
            "iter": t,
            "max_sigma_selected": float(np.max(top_sigmas)),
            "mean_sigma_selected": float(np.mean(top_sigmas)),
        })

        labeled.update(top.tolist())
        print(f"[Kriging pretrain] Iter {t}/{ITERS}: added {take} points (labeled={len(labeled)})")
    selected_idx = np.array(sorted(labeled))
    X      = X.iloc[selected_idx].reset_index(drop=True)
    y_mass = y_mass.iloc[selected_idx].reset_index(drop=True)
    y_pcr  = y_pcr.iloc[selected_idx].reset_index(drop=True)
    y_rlp  = y_rlp.iloc[selected_idx].reset_index(drop=True)
    print(f"[Kriging pretrain] Final selected size: {len(X)} rows (from {len(all_idx)} total).")

    # === Save selection order for inspection ===
    sel_df = pd.DataFrame(selection_log, columns=["row_index", "iter_added"]) \
             .sort_values(["iter_added", "row_index"])
    sel_df.to_csv("kriging_selection_order, K.csv", index=False)

    # === 2D PCA map: color by iteration when point was added ===
    scaler = StandardScaler()
    Z = PCA(n_components=2, random_state=RANDOM_SEED).fit_transform(scaler.fit_transform(X_full.values))

    # base scatter
    plt.figure(figsize=(8,6))
    plt.scatter(Z[:,0], Z[:,1], s=10, c="#cccccc", alpha=0.5, label="Pool")

    # color selected by iteration (0 = seeds)
    cmap = plt.get_cmap("viridis")
    iters_present = sorted(sel_df["iter_added"].unique())
    it_max = max(iters_present) if iters_present else 0
    for it in iters_present:
        if it % 10 != 0 and it != 0:
            continue
        idx_this_it = sel_df.loc[sel_df["iter_added"] == it, "row_index"].to_numpy()
        if idx_this_it.size == 0:
            continue
        color = cmap(0.0) if it_max == 0 else cmap(it / (it_max + 1e-9))
        lab = "Seeds" if it == 0 else (f"Iter {it}")
        plt.scatter(Z[idx_this_it,0], Z[idx_this_it,1],
                    s=24 if it==0 else 22,
                    edgecolors="black" if it==0 else "none",
                    linewidths=0.6 if it==0 else 0.0,
                    c=[color], alpha=0.9 if it==0 else 0.85, label=lab)

    plt.xlabel("Design-space direction 1")
    plt.ylabel("Design-space direction 2")
    plt.legend(loc="upper center", fontsize=20, bbox_to_anchor=(0.5, -0.20), ncol=5, frameon=False, handletextpad=0.0, columnspacing=1.0)
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig("kriging_selection_map.pdf")
    #plt.savefig("kriging_selection_map.png", dpi=1200)
    plt.close()

    """# === Animation (GIF) of cumulative selections === #uncomment for GIF visulas
    frames_dir = "kriging_frames"
    os.makedirs(frames_dir, exist_ok=True)
    gif_path = "kriging_selection_evolution.gif"

    # build frames: cumulative from seeds (0) up to last iteration
    frame_iters = [it for it in iters_present if (it == 0 or it % 3 == 0 or it == iters_present[-1])]
    for it in frame_iters:
        plt.figure(figsize=(8,6))
        plt.scatter(Z[:,0], Z[:,1], s=10, c="#cccccc", alpha=0.5,  label="Pool")

        # cumulative selections up to current iter
        for i_sub in range(it + 1):
            idx_this = sel_df.loc[sel_df["iter_added"] == i_sub, "row_index"].to_numpy()
            if idx_this.size == 0:
                continue
            color = cmap(0.0) if it_max == 0 else cmap(i_sub / (it_max + 1e-9))
            lab = "Seeds" if i_sub == 0 else f"Iter {i_sub}"
            plt.scatter(Z[idx_this,0], Z[idx_this,1], s=20, c=[color], label=lab, alpha=0.9)

        plt.xlabel("Design-space direction 1")
        plt.ylabel("Design-space direction 2")
        #plt.title(f"Kriging pretraining — up to iteration {it}")
        plt.legend(loc="upper right", fontsize=8)
        plt.grid(True, alpha=0.2)
        plt.tight_layout()
        frame_path = os.path.join(frames_dir, f"frame_{it:02d}.png")
        plt.savefig(frame_path, dpi=300)
        plt.close()

    # stitch frames into GIF
    frames = [imageio.imread(os.path.join(frames_dir, f"frame_{it:02d}.png")) for it in frame_iters]
    imageio.mimsave(gif_path, frames, duration=1.0)  # seconds per frame
    print(f"[GIF saved] {gif_path}")
    """

# === TRAIN MODELS ===
mass_model = model()
pcr_model  = model()
rlp_model  = model()

mass_model.fit(X, y_mass)
pcr_model.fit(X, y_pcr)
rlp_model.fit(X, y_rlp)

# === CROSS-VAL METRICS ===
K = 7  # number of folds
kf = KFold(n_splits=K, shuffle=True, random_state=RANDOM_SEED)

print(f"Cross-validated R² scores ({K}-fold):")
cv_results = {}
for name, mdl, y in [
    ("Mass",       mass_model, y_mass),
    ("Pcr",        pcr_model,  y_pcr),
    ("RL_Perfect", rlp_model,  y_rlp),
    #("RL_Imperfect", rli_model,  y_rli),
    ]:
    scores = cross_val_score(mdl, X, y, cv=kf, scoring='r2')
    preds  = cross_val_predict(mdl, X, y, cv=kf)
    rmse   = root_mean_squared_error(y, preds)
    nrmse  = 100 * rmse / (y.max() - y.min())
    cv_results[name] = {"scores": scores, "preds": preds, "rmse": rmse,
                        "r2": scores.mean(), "nrmse": nrmse}
    print(f"- {name} R²: {scores.mean():.4f} (± {scores.std():.4f})")
    print(f"  {name} RMSE: {rmse:.4f} ({nrmse:.1f}%)")

# === SAVE MODELS + FEATURES ===
joblib.dump(mass_model, MASS_MODEL_OUT)
joblib.dump(pcr_model,  PCR_MODEL_OUT)
joblib.dump(rlp_model,  RLP_MODEL_OUT)

with open(FEATURES_JSON, "w") as f:
    json.dump({
        "input_cols": input_cols,
        "angle_cols": angle_cols,
        "targets": {
            "mass": mass_col, "pcr": pcr_col,
            "rl_perfect": rlp_col,
        }
    }, f, indent=2)
print("\nFinal models and feature metadata saved.")

# === CONFIDENCE INTERVALS VIA LHS ===
def normal_ci(mdl, Xq):
    mu, std = mdl.predict(Xq, return_std=True)
    return mu, mu - 1.96*std, mu + 1.96*std

X_min, X_max   = X.min(), X.max()
num_samples    = 100
lhs_unit       = lhs(len(input_cols), samples=num_samples)
lhs_scaled     = lhs_unit * (X_max.values - X_min.values) + X_min.values
X_uncertain    = pd.DataFrame(lhs_scaled, columns=input_cols)

pcr_mean, pcr_lo, pcr_hi = normal_ci(pcr_model, X_uncertain)
pd.DataFrame({"Pcr_mean": pcr_mean, "Pcr_lower_95": pcr_lo, "Pcr_upper_95": pcr_hi}).to_csv(
    "pcr_confidence_bounds, K.csv", index=False)

mass_mean, mass_lo, mass_hi = normal_ci(mass_model, X_uncertain)
pd.DataFrame({"Mass_mean": mass_mean, "Mass_lower_95": mass_lo, "Mass_upper_95": mass_hi}).to_csv(
    "mass_confidence_bounds, K.csv", index=False)

rlp_mean, rlp_lo, rlp_hi = normal_ci(rlp_model, X_uncertain)
pd.DataFrame({"RL_Perfect_mean": rlp_mean, "RL_Perfect_lower_95": rlp_lo, "RL_Perfect_upper_95": rlp_hi}).to_csv(
    "rl_perfect_confidence_bounds, K.csv", index=False)

print("Saved CI CSVs for Pcr, Mass, RL_Perfect.")

# === SENSITIVITY ===
def safe_corr(x, y):
    if np.allclose(np.std(x.values), 0.0): return 0.0
    return float(np.corrcoef(x, y)[0, 1])

sens_angles = pd.DataFrame({
    "Angle": angle_cols,
    "Mass Corr":         [safe_corr(X[a], y_mass) for a in angle_cols],
    "Pcr Corr":          [safe_corr(X[a], y_pcr)  for a in angle_cols],
    "RL_Perfect Corr":   [safe_corr(X[a], y_rlp)  for a in angle_cols],
})
sens_angles.to_csv("angle_sensitivity_correlation, K.csv", index=False)
print("Sensitivity analysis saved.")

# === DIAGNOSTIC SCATTER PLOTS ===
def diag_plot(y_true, y_pred, xlab, ylab, out_png, cv_key, scale=1.0):
    r2    = cv_results[cv_key]["r2"]
    rmse  = cv_results[cv_key]["rmse"] * scale   
    nrmse = cv_results[cv_key]["nrmse"]          

    yt = np.asarray(y_true) * scale
    yp = np.asarray(y_pred) * scale

    plt.figure(figsize=(8, 6))
    plt.scatter(yt, yp, alpha=0.8, label='Predicted')
    lo, hi = np.min(yt), np.max(yt)
    plt.plot([lo, hi], [lo, hi], 'r--', label='Ideal (y=x)')
    plt.xlabel(xlab); plt.ylabel(ylab)
    plt.grid(False); plt.legend(loc='upper left')
    plt.text(0.54, 0.18, f"R²: {r2:.4f}\nRMSE: {rmse:.2f} ({nrmse:.1f}%)",
             transform=plt.gca().transAxes, va='top',
             bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray"))
    plt.tight_layout(); #plt.savefig(out_png, dpi=1200); 
    plt.savefig(out_png); plt.close()


diag_plot(
    y_pcr,  cv_results["Pcr"]["preds"],
    r"Actual $P_\mathrm{cr}$ (kN)", r"Predicted $P_\mathrm{cr}$ (kN)",
    "pcr_fit_scatter, K.pdf", "Pcr", scale=1e-3
)

diag_plot(
    y_mass, cv_results["Mass"]["preds"],
    "Actual Mass (g)", "Predicted Mass (g)",
    "mass_fit_scatter, K.pdf", "Mass", scale=1.0
)

diag_plot(
    y_rlp,  cv_results["RL_Perfect"]["preds"],
    r"Actual $RL_\mathrm{perfect}$ (kN)", r"Predicted $RL_\mathrm{perfect}$ (kN)",
    "rl_perfect_fit_scatter, K.pdf", "RL_Perfect", scale=1e-3
)

print("Saved fit scatter plots.")
