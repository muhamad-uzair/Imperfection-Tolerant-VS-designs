import os, torch
import re
import numpy as np
import pandas as pd
import joblib
from ax.modelbridge.generation_strategy import GenerationStrategy, GenerationStep
from ax.modelbridge.registry import Models
from ax.service.ax_client import AxClient, ObjectiveProperties
from botorch.utils.multi_objective.hypervolume import Hypervolume
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import time
from time import perf_counter as now

# === Plot styling ===
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams.update({
    'font.size': 18,
    'axes.labelsize': 18,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
})

np.random.seed(42)
os.environ["OMP_NUM_THREADS"] = "1"   
os.environ["MKL_NUM_THREADS"] = "1"   
torch.set_num_threads(24) # let PyTorch use your 24 physical cores, set according to your system

# === Load models & feature schema ===
MODEL_DIR = r"Link to the folder having surrogates & dataset"
MASS_MODEL = os.path.join(MODEL_DIR, "mass_model.pkl")
PCR_MODEL  = os.path.join(MODEL_DIR, "pcr_model.pkl")
RLP_MODEL  = os.path.join(MODEL_DIR, "rlp_model.pkl") 
FEATURES_JSON = os.path.join(MODEL_DIR, "model_features.json")

mass_model = joblib.load(MASS_MODEL)
pcr_model  = joblib.load(PCR_MODEL)
rlp_model  = joblib.load(RLP_MODEL)


# === Feature columns & bounds ===
input_cols = None
if os.path.exists(FEATURES_JSON):
    import json
    with open(FEATURES_JSON, "r") as f:
        meta = json.load(f)
        input_cols = meta.get("input_cols") or meta.get("angle_cols")

if input_cols is None:
    raise RuntimeError("model_features.json missing or does not contain 'input_cols'.")

DATASET_CSV = os.path.join(MODEL_DIR, "Dataset.csv")
df_bounds_src = pd.read_csv(DATASET_CSV)
missing_in_ds = [c for c in input_cols if c not in df_bounds_src.columns]
if missing_in_ds:
    raise ValueError(f"Dataset used for bounds is missing columns: {missing_in_ds}")

angle_min = df_bounds_src[input_cols].min()
angle_max = df_bounds_src[input_cols].max()
bounds = [(float(angle_min[c]), float(angle_max[c])) for c in input_cols]
dim = len(bounds)

def _pack_df_from_x(x_vec):
    """Pack a 1xN dataframe with input_cols in the trained order."""
    return pd.DataFrame([x_vec], columns=input_cols)


def _predict_mean_std(model, df_features):
    mean, std = model.predict(df_features, return_std=True)
    return float(mean[0]), float(std[0])

def predict_mass(x):
    df_features = _pack_df_from_x(x)
    return _predict_mean_std(mass_model, df_features)

def predict_pcr(x):
    df_features = _pack_df_from_x(x)
    return _predict_mean_std(pcr_model, df_features)

def predict_rl(x):
    df_features = _pack_df_from_x(x)
    return _predict_mean_std(rlp_model, df_features)

def nondominated_mask(P):
    n = len(P)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # j dominates i if j no-worse in all and strictly better in at least one
            if (P[j,0] <= P[i,0] and P[j,1] >= P[i,1] and P[j,2] >= P[i,2] and
                ((P[j,0] < P[i,0]) or (P[j,1] > P[i,1]) or (P[j,2] > P[i,2]))):
                keep[i] = False
                break
    return keep

# Config
N_INIT   = 50       # Sobol warmup points
N_BATCHES = 65     # qNEHVI iterations
Q        = 6        # points per iteration, set a number considering cores of the system
MASS_LIMIT = 164.0  # set None to disable hard mass constraint

parameters = [
    {"name": f"x{i}", "type": "range", "bounds": [float(a), float(b)]}
    for i, (a, b) in enumerate(bounds)
]


# Objective thresholds. Adjust once you see scales.
objectives = {
    "pcr_lb": ObjectiveProperties(minimize=False, threshold=2e4), 
    "kdf":    ObjectiveProperties(minimize=False, threshold=0.50), 
    "mass":   ObjectiveProperties(minimize=True, threshold=164.0),  
}


def eval_surrogates(param_dict):
    x = np.array([param_dict[f"x{i}"] for i in range(dim)], dtype=float)
    m_mean, m_std = predict_mass(x)
    p_mean, p_std = predict_pcr(x)
    r_mean, r_std = predict_rl(x)

    # conservative lower bounds
    p_lb = float(p_mean - 1.96 * p_std)
    r_lb = float(r_mean - 1.96 * r_std)
    kdf_lb = float(p_lb / r_lb)

    return {
        "pcr_lb": (p_lb, max(1e-6, p_std)),
        "kdf":    (kdf_lb, 0.02),
        "mass":   (m_mean, max(1e-6, m_std)),

        "kdf_constr":  (kdf_lb - 0.9999, 1e-6),
        "mass_constr": (m_mean - MASS_LIMIT, 1e-6) if MASS_LIMIT is not None else (0.0, 1e-6),

        # optional logs:
        "rl_lb": (r_lb, max(1e-6, r_std)),
    }
moo_kwargs = {
    "use_saas": False,
    "acquisition_options": {"num_outcome_mc_samples": 64},
    "optimizer_kwargs": {"num_restarts": 4, "raw_samples": 96, "batch_limit": 4},
}

gs = GenerationStrategy(steps=[
    GenerationStep(model=Models.SOBOL, num_trials=N_INIT, max_parallelism=N_INIT),
    GenerationStep(model=Models.MOO,   num_trials=N_BATCHES * Q, max_parallelism=Q),
])

ax = AxClient(generation_strategy=gs, enforce_sequential_optimization=False, random_seed=42)
ax.create_experiment(
    name="mobo_qnehvi_new_design_vars",
    parameters=parameters,
    objectives=objectives,
    outcome_constraints=(["kdf_constr <= 0.0"] + (["mass_constr <= 0.0"] if MASS_LIMIT is not None else [])),
    overwrite_existing_experiment=True,
)

# === Run Optimisation ===
sobol_time_total   = 0.0
propose_time_total = 0.0
eval_time_total    = 0.0
batches_timing     = [] 

T0_all = now()
# Sobol warmup
t0 = now()
for _ in range(N_INIT):
    params, trial_index = ax.get_next_trial()
    metrics = eval_surrogates(params)
    ax.complete_trial(trial_index=trial_index, raw_data=metrics)
sobol_time_total = now() - t0
print(f"[TIMING] Sobol warmup: {sobol_time_total:.2f}s for {N_INIT} trials")

# qNEHVI loop
for it in range(N_BATCHES):
    t0 = now()
    batch = [ax.get_next_trial() for _ in range(Q)]
    t1 = now()
    for params, tid in batch:
        metrics = eval_surrogates(params)
        ax.complete_trial(trial_index=tid, raw_data=metrics)
    t2 = now()

    propose = t1 - t0
    evalc   = t2 - t1
    total   = t2 - t0

    propose_time_total += propose
    eval_time_total    += evalc

    batches_timing.append({
        "batch": it+1,
        "q": Q,
        "propose_s": propose,
        "eval_complete_s": evalc,
        "total_s": total,
    })
    print(f"[qNEHVI] Batch {it+1}/{N_BATCHES} (q={Q}) | "
          f"propose={propose:.2f}s | eval+complete={evalc:.2f}s | total={total:.2f}s")
    
    """# === Optional: Check BO selections for certain batches ===
    SAVE_FRAMES = [1, 16, 32, N_BATCHES]  # pick which batches to visualize
    if (it + 1) in SAVE_FRAMES:
        b = it + 1
        frames_dir = "bo_frames"
        os.makedirs(frames_dir, exist_ok=True)

        df_sofar = ax.get_trials_data_frame()
        if "trial_index" not in df_sofar.columns and "trial" in df_sofar.columns:
            df_sofar = df_sofar.rename(columns={"trial": "trial_index"})
        df_plot = df_sofar.copy()

        from sklearn.decomposition import PCA
        x_cols = [c for c in df_plot.columns if c.startswith("x")]
        if len(x_cols) < 2:
            print(f"[FRAME] Skipping batch {b}: not enough x-columns.")
            continue

        X_all = df_plot[x_cols].to_numpy(float)
        Z = PCA(n_components=2).fit_transform(X_all)

        df_chron = df_plot.sort_values("trial_index").reset_index(drop=True)
        sobol_mask = df_chron["trial_index"] < N_INIT
        ti = df_chron["trial_index"].to_numpy()

        plt.figure(figsize=(8, 6))
        plt.scatter(Z[sobol_mask, 0], Z[sobol_mask, 1],
                s=18, c="#4a4a4a", alpha=0.9,
                label=f"Sobol (n={sobol_mask.sum()})", marker="o")

        markers = ['^', 's', 'D', 'P', 'X', 'v', 'o']
        cmap = plt.cm.plasma
        colors = [cmap(v) for v in np.linspace(0.05, 0.95, len(SAVE_FRAMES))]
        color_map_dict = {bnum: col for bnum, col in zip(SAVE_FRAMES, colors)}
        selected_up_to_now = [k for k in SAVE_FRAMES if k <= b]   # e.g., at b=4 → [1,2,4]

        for idx, i in enumerate(selected_up_to_now, start=1):
            start_i = N_INIT + (i - 1) * Q
            end_i   = N_INIT + i * Q - 1
            m_i = (ti >= start_i) & (ti <= end_i)
            if not m_i.any():
                continue
            color = color_map_dict[i]
            plt.scatter(
                Z[m_i, 0], Z[m_i, 1],
                s=26, c=[color], alpha=0.95,
                label=f"Batch {i}",
                marker=markers[(i-1) % len(markers)]
            )

        plt.xlabel("Design-space direction 1")
        plt.ylabel("Design-space direction 2")
        plt.legend(
            loc="upper center", bbox_to_anchor=(0.5, -0.20),
            ncol=5, frameon=False, fontsize=12,
            handletextpad=0.0, columnspacing=1.0)
        plt.tight_layout()

        out = os.path.join(frames_dir, f"bo_batch_{b:02d}.png")
        plt.savefig(out, dpi=1200)
        plt.close()
        print(f"[FRAME] Saved visualization for batch {b} → {out}")
        """

T1_all = now()
total_runtime = T1_all - T0_all

print("\n=== RUNTIME SUMMARY ===")
print(f"Sobol:          {sobol_time_total:.2f}s")
print(f"Propose total:  {propose_time_total:.2f}s")
print(f"Eval+complete:  {eval_time_total:.2f}s")
print(f"End-to-end BO:  {total_runtime:.2f}s "
      f"(includes Sobol + all batches, excludes plotting below)")

# Optional: write batch timings to CSV
pd.DataFrame(batches_timing).to_csv("bo_batch_timings.csv", index=False)
print("[SAVED] bo_batch_timings.csv")

# === Collect results ===
df_all_ax = ax.get_trials_data_frame()
lower = {c.lower(): c for c in df_all_ax.columns}
def has_all(*cols): return all(c in lower for c in cols)

if has_all("metric_name", "mean"): 
    name_map = {lower["mean"]: "mean", lower["metric_name"]: "metric_name"}
    if "trial_index" in lower: name_map[lower["trial_index"]] = "trial_index"
    elif "trial" in lower:     name_map[lower["trial"]]       = "trial_index"
    for alt in ("sem", "se", "stderr", "standard_error"):
        if alt in lower: name_map[lower[alt]] = "sem"; break
    df_all_ax = df_all_ax.rename(columns=name_map)
    df_all = df_all_ax.pivot_table(index="trial_index", columns="metric_name", values="mean").reset_index()
    param_rows = []
    for t in ax.experiment.trials.values():
        if not t.status.is_completed: continue
        xs = dict(t.arm.parameters); xs["trial_index"] = t.index
        param_rows.append(xs)
    df_params = pd.DataFrame(param_rows)
    df_all = df_all.merge(df_params, on="trial_index", how="left")
else: 
    df_all = df_all_ax.copy()
    if "trial_index" not in df_all.columns and "trial" in lower:
        df_all = df_all.rename(columns={lower["trial"]: "trial_index"})
    if not any(c.startswith("x") for c in df_all.columns):
        param_rows = []
        for t in ax.experiment.trials.values():
            if not t.status.is_completed: continue
            xs = dict(t.arm.parameters); xs["trial_index"] = t.index
            param_rows.append(xs)
        df_params = pd.DataFrame(param_rows)
        df_all = df_all.merge(df_params, on="trial_index", how="left")

# === Pareto set for the three objectives ===
pf = ax.get_pareto_optimal_parameters()
rows_pf = []
for tid, (xs, _vals) in pf.items():
    rec = df_all.loc[df_all["trial_index"] == tid]
    if rec.empty: 
        continue
    rec = rec.iloc[0]
    row = {"trial_index": int(tid)}
    row.update({k: float(v) for k, v in xs.items()})
    row["pcr_lb"] = float(rec.get("pcr_lb", np.nan))
    row["rl_lb"]  = float(rec.get("rl_lb",  np.nan))
    row["kdf"]    = float(rec.get("kdf",    np.nan))
    row["mass"]   = float(rec.get("mass",   np.nan))
    
    rows_pf.append(row)
df_pf = pd.DataFrame(rows_pf).dropna(subset=["pcr_lb","rl_lb","kdf","mass"])

# Save CSVs
df_all.to_csv("bo_all_trials.csv", index=False)
df_pf.to_csv("bo_pareto_front.csv", index=False)
print(f"[SAVED] bo_all_trials.csv  ({len(df_all)} trials)")
print(f"[SAVED] bo_pareto_front.csv ({len(df_pf)} non-dominated)")

# === Plots ===
def ensure_cols(df, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing expected columns: {missing}. Available: {list(df.columns)}")

""" # uncomment if you need to visualize single pareto front & not combined
ensure_cols(df_all, ["mass", "pcr_lb"])
plt.figure(figsize=(8,6))
plt.scatter(df_all["mass"], df_all["pcr_lb"], s=10, alpha=0.3, label="All")
if not df_pf.empty and "pcr_lb" in df_pf.columns:
    plt.scatter(df_pf["mass"], df_pf["pcr_lb"], s=30, label="Pareto")
plt.xlabel("Mass (g)"); plt.ylabel("Pcr (N)")
plt.grid(True); plt.legend(); plt.tight_layout()
plt.savefig("bo_pareto_mass_vs_pcrLB.png", dpi=1200)

ensure_cols(df_all, ["mass", "kdf"])
plt.figure(figsize=(8,6))
plt.scatter(df_all["mass"], df_all["kdf"], s=10, alpha=0.3, label="All")
if not df_pf.empty and "kdf" in df_pf.columns:
    plt.scatter(df_pf["mass"], df_pf["kdf"], s=30, label="Pareto")
plt.xlabel("Mass (g)"); plt.ylabel("KDF")
plt.ylim(0, 1.02)
plt.grid(True); plt.legend(); plt.tight_layout()
plt.savefig("bo_pareto_mass_vs_KDFLB.png", dpi=1200)

# === Mass vs (PCR & KDF): ALL trials ===
ensure_cols(df_all, ["mass", "pcr_lb", "kdf"])
fig, ax1 = plt.subplots(figsize=(8,6))
ax2 = ax1.twinx()

ax1.scatter(df_all["mass"], df_all["pcr_lb"], s=14, alpha=0.35, label="All - PCR [N]", marker="o", color="black")
ax2.scatter(df_all["mass"], df_all["kdf"],    s=14, alpha=0.35, label="All - KDF", marker="x", color="black")

ax1.set_xlabel("Mass (g)")
ax1.set_ylabel("PCR (N)")
ax2.set_ylabel("KDF")
ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

h1,l1 = ax1.get_legend_handles_labels()
h2,l2 = ax2.get_legend_handles_labels()
ax1.legend(h1+h2, l1+l2, loc="best")

ax1.grid(True); fig.tight_layout()
plt.savefig("bo_all_trials.png", dpi=1200)
print("[SAVED] all_trials.png")
"""
# === Mass vs (PCR & KDF): PARETO only ===
if not df_pf.empty:
    ensure_cols(df_pf, ["mass", "pcr_lb", "kdf"])
    fig, ax1 = plt.subplots(figsize=(8,6))
    ax2 = ax1.twinx()

    ax1.scatter(df_pf["mass"], df_pf["pcr_lb"]/1000, s=30, label="Pareto - PCR (kN)", marker="o", color="black")
    ax2.scatter(df_pf["mass"], df_pf["kdf"],    s=30, label="Pareto - KDF", marker="x", color="black")

    ax1.set_xlabel("Mass (g)")
    ax1.set_ylabel("PCR (kN)")
    ax2.set_ylabel("KDF")
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    h1,l1 = ax1.get_legend_handles_labels()
    h2,l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, loc="best")

    ax1.grid(True); fig.tight_layout()
    plt.savefig("bo_pareto.pdf")
    print("[SAVED] pareto.pdf")


# === Convergence plot: best-so-far PCR lower bound ===
ensure_cols(df_all, ["trial_index", "pcr_lb"])
df_conv = df_all[["trial_index", "pcr_lb"]].dropna().sort_values("trial_index")
df_conv["best_pcr_lb_so_far"] = df_conv["pcr_lb"].cummax()

plt.figure(figsize=(8,6))
plt.plot(df_conv["trial_index"], df_conv["best_pcr_lb_so_far"]/1000, label="Pcr (kN)", color="black")
plt.axvline(N_INIT - 0.5, linestyle="--", label="End of Sobol warmup", color="black")
plt.xlabel("Trial index")
plt.ylabel("PCR (kN)")
plt.grid(True); plt.legend(); plt.tight_layout()
plt.savefig("bo_pcr.pdf")
print("[SAVED] bo_pcr.pdf")
"""
# === Feasible-only convergence ===
feas = ( (df_all.get("mass", np.inf) <= MASS_LIMIT) &
         (df_all.get("kdf", -np.inf) <= 0.9999 if "kdf" in df_all else True) )
df_conv_feas = df_all.loc[feas, ["trial_index", "pcr_lb"]].dropna().sort_values("trial_index")
if not df_conv_feas.empty:
    df_conv_feas["best_pcr_lb_so_far"] = df_conv_feas["pcr_lb"].cummax()
    plt.figure(figsize=(8,6))
    plt.plot(df_conv["trial_index"], df_conv["best_pcr_lb_so_far"], linewidth=2.5, label="All trials")
    plt.plot(df_conv_feas["trial_index"], df_conv_feas["best_pcr_lb_so_far"], linewidth=1.5, label="Feasible only", color="black")
    plt.axvline(N_INIT - 0.5, linestyle="--", linewidth=1, label="End of Sobol warmup")
    plt.xlabel("Trial index"); plt.ylabel("PCR (N)")
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig("bo_convergence_feasible.png", dpi=1200)
    print("[SAVED] convergence_feasible.png")
"""
# === Hypervolume vs trials (BoTorch) ===
ref_pt = torch.tensor([
    float(objectives["pcr_lb"].threshold),
    float(objectives["kdf"].threshold),
    float(-objectives["mass"].threshold),
])

def hv_of_front(front_df):
    if front_df.empty:
        return 0.0
    P = torch.tensor(np.column_stack([
        front_df["pcr_lb"].to_numpy(dtype=float),
        front_df["kdf"].to_numpy(dtype=float),
        (-front_df["mass"]).to_numpy(dtype=float),
    ]))
    hv = Hypervolume(ref_point=ref_pt)
    return float(hv.compute(P))

hv_progress = []
for t in sorted(df_all["trial_index"].unique()):
    df_t = df_all[df_all["trial_index"] <= t][["mass","pcr_lb","kdf"]].dropna()
    
    P = df_t[["mass","pcr_lb","kdf"]].to_numpy(dtype=float)
    mask = nondominated_mask(P)
    df_pf_t = df_t.loc[mask]
    hv_progress.append((t, hv_of_front(df_pf_t)))

hv_df = pd.DataFrame(hv_progress, columns=["trial", "hv"])
hv_df["hv"] = hv_df["hv"].cummax()
plt.figure(figsize=(8,6))
plt.plot(hv_df["trial"], hv_df["hv"]/1000, label="HV", color="black")
plt.axvline(N_INIT - 0.5, linestyle="--", label="End of Sobol warmup", color="black")
plt.xlabel("Trial index"); plt.ylabel("Hypervolume")
plt.grid(True); plt.legend(); plt.tight_layout()
plt.savefig("bo_hypervolume.pdf")
print("[SAVED] bo_hypervolume.pdf")

# === Representative designs from Pareto set ===
if not df_pf.empty:
    best_pcr = df_pf.loc[df_pf["pcr_lb"].idxmax()]
    best_mass = df_pf.loc[df_pf["mass"].idxmin()]
    best_kdf = df_pf.loc[df_pf["kdf"].idxmax()]
    reps = pd.DataFrame([best_pcr, best_mass, best_kdf])
    reps.insert(0, "which", ["best_pcr","best_mass","best_kdf"])
    reps.to_csv("bo_best_designs.csv", index=False)
    print("\n[RESULTS] Representative optimized designs saved.csv")
    print(reps[["which","mass","pcr_lb","rl_lb","kdf"]].to_string(index=False))
