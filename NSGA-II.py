import os, json, warnings
import numpy as np
import pandas as pd
import joblib
import time
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

warnings.filterwarnings("ignore", category=FutureWarning)
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams.update({
    'font.size': 18,
    'axes.labelsize': 18,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
})

np.random.seed(42)

# === Load models & feature schema ===
MODEL_DIR = r"Link to the folder having surrogates & dataset"
MASS_MODEL = os.path.join(MODEL_DIR, "mass_model.pkl")
PCR_MODEL  = os.path.join(MODEL_DIR, "pcr_model.pkl")
RLP_MODEL  = os.path.join(MODEL_DIR, "rlp_model.pkl")
FEATURES_JSON = os.path.join(MODEL_DIR, "model_features.json")
DATASET_CSV = os.path.join(MODEL_DIR, "Dataset.csv")

mass_model = joblib.load(MASS_MODEL)
pcr_model  = joblib.load(PCR_MODEL)
rlp_model  = joblib.load(RLP_MODEL)

with open(FEATURES_JSON, "r") as f:
    meta = json.load(f)
input_cols = meta.get("input_cols") or meta.get("angle_cols")
if input_cols is None:
    raise RuntimeError("model_features.json missing or does not contain 'input_cols'.")

df_bounds_src = pd.read_csv(DATASET_CSV)
missing_in_ds = [c for c in input_cols if c not in df_bounds_src.columns]
if missing_in_ds:
    raise ValueError(f"Dataset used for bounds is missing columns: {missing_in_ds}")

angle_min = df_bounds_src[input_cols].min()
angle_max = df_bounds_src[input_cols].max()
bounds = [(float(angle_min[c]), float(angle_max[c])) for c in input_cols]
dim = len(bounds)

# === Constraints ===
MASS_LIMIT = 164.0        # set None to disable
KDF_LIMIT  = 0.9999

def _pack_df_from_x(x_vec):
    return pd.DataFrame([x_vec], columns=input_cols)

def _predict_mean_std(model, df_features):
    mean, std = model.predict(df_features, return_std=True)
    return float(mean[0]), float(std[0])

def eval_surrogates(x):
    df_features = _pack_df_from_x(x)
    m_mean, m_std = _predict_mean_std(mass_model, df_features)
    p_mean, p_std = _predict_mean_std(pcr_model,  df_features)
    r_mean, r_std = _predict_mean_std(rlp_model,  df_features)

    p_lb = float(p_mean - 1.96 * p_std)
    r_lb = float(r_mean - 1.96 * r_std)
    kdf_lb = float(p_lb / max(1e-12, r_lb))

    # objectives: minimize [mass, -pcr_lb, -kdf_lb]
    f_mass = m_mean
    f_pcr  = -p_lb
    f_kdf  = -kdf_lb

    # constraints (≤ 0 is feasible)
    c_mass = (m_mean - MASS_LIMIT) if MASS_LIMIT is not None else 0.0
    c_kdf  = (kdf_lb - KDF_LIMIT)

    return {
        "mass": m_mean,
        "pcr_lb": p_lb,
        "rl_lb": r_lb,
        "kdf": kdf_lb,
        "f": (f_mass, f_pcr, f_kdf),
        "c": (c_mass, c_kdf),
    }

def nondominated_mask(P):  # P: [n,3] with [mass, pcr_lb, kdf] (note: pcr,kdf maximized)
    n = len(P)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # j dominates i if mass lower AND (pcr,kdf) higher-or-equal with at least one strict
            if (P[j,0] <= P[i,0] and P[j,1] >= P[i,1] and P[j,2] >= P[i,2] and
                ((P[j,0] < P[i,0]) or (P[j,1] > P[i,1]) or (P[j,2] > P[i,2]))):
                keep[i] = False
                break
    return keep

def is_feasible(mass, kdf):
        return ((MASS_LIMIT is None or mass <= MASS_LIMIT) and (kdf <= KDF_LIMIT))
# === DEAP NSGA-II ===
def run_deap_nsga2(pop_size=160, n_gen=250, seed=42, cxpb=0.9, mutpb=0.1, eta_cx=15.0, eta_mut=20.0):
    from deap import base, creator, tools
    rng = np.random.default_rng(seed)

    # Fitness: 3 objectives, all minimized (mass, -pcr, -kdf)
    if not hasattr(creator, "FitnessMin3"):
        creator.create("FitnessMin3", base.Fitness, weights=(-1.0, -1.0, -1.0))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMin3)

    toolbox = base.Toolbox()
    for i, (a, b) in enumerate(bounds):
        toolbox.register(f"attr_{i}", rng.uniform, a, b)
    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (tuple(getattr(toolbox, f"attr_{i}") for i in range(dim))), n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Evaluation with feasibility-first selection
    def eval_ind(ind):
        out = eval_surrogates(ind)
        # return objectives
        return out["f"]

    toolbox.register("evaluate", eval_ind)
    toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=[a for a,_ in bounds],
                     up=[b for _,b in bounds], eta=eta_cx)
    toolbox.register("mutate", tools.mutPolynomialBounded, low=[a for a,_ in bounds],
                     up=[b for _,b in bounds], eta=eta_mut, indpb=1.0/dim)
    toolbox.register("select", tools.selNSGA2)

    pop = toolbox.population(n=pop_size)
    # Initialize & assign crowding distances
    invalid = [ind for ind in pop if not ind.fitness.valid]
    fits = list(map(toolbox.evaluate, invalid))
    for ind, fit in zip(invalid, fits):
        ind.fitness.values = fit
    pop = toolbox.select(pop, len(pop))

    log = []
    hof = tools.ParetoFront()

    for gen in range(1, n_gen+1):
        offspring = tools.selTournamentDCD(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if rng.random() < cxpb:
                toolbox.mate(c1, c2)
                del c1.fitness.values, c2.fitness.values

        for mut in offspring:
            if rng.random() < mutpb:
                toolbox.mutate(mut)
                del mut.fitness.values

        invalid = [ind for ind in offspring if not ind.fitness.valid]
        fits = list(map(toolbox.evaluate, invalid))
        for ind, fit in zip(invalid, fits):
            ind.fitness.values = fit

        pop = toolbox.select(pop + offspring, pop_size)
        hof.update(pop)

        # Log generation best feasible PCR (for quick convergence figure)
        records = []
        for ind in pop:
            met = eval_surrogates(ind)
            records.append(met)
        df_gen = pd.DataFrame([{
            "mass": r["mass"], "pcr_lb": r["pcr_lb"], "rl_lb": r["rl_lb"],
            "kdf": r["kdf"], "c_mass": r["c"][0], "c_kdf": r["c"][1]
        } for r in records])
        log.append({
            "gen": gen,
            "best_pcr_feasible": df_gen.loc[(df_gen["c_mass"]<=0.0)&(df_gen["c_kdf"]<=0.0), "pcr_lb"].max()
            if not df_gen.empty else np.nan
        })
        print(f"[NSGA-II] Gen {gen}/{n_gen}")

    # Collect all final pop evaluations
    all_rows = []
    for ind in pop:
        met = eval_surrogates(ind)
        row = {"mass": met["mass"], "pcr_lb": met["pcr_lb"], "rl_lb": met["rl_lb"],
               "kdf": met["kdf"]}#, "c_mass": met["c"][0], "c_kdf": met["c"][1]
        for i, v in enumerate(ind):
            row[f"x{i}"] = float(v)
        all_rows.append(row)
    df_all = pd.DataFrame(all_rows)

    # Pareto: take feasible, then nondominated in (mass min, pcr/kdf max)
    feas = [is_feasible(m, k) for m, k in zip(df_all["mass"], df_all["kdf"])]
    df_feas = df_all.loc[feas].copy()
    if not df_feas.empty:
        P = df_feas[["mass","pcr_lb","kdf"]].to_numpy(float)
        mask = nondominated_mask(P)
        df_pf = df_feas.loc[mask].copy()
    else:
        df_pf = pd.DataFrame(columns=df_all.columns)

    return {"all": df_all, "pf": df_pf, "log": pd.DataFrame(log)}

# === Run Optimisation ===
start = time.time()
res = run_deap_nsga2()
end = time.time()
print(f"\n[TIME] Optimisation took {end - start:.2f} seconds ({(end-start)/60:.2f} minutes)\n")

df_all = res["all"].copy()
df_pf  = res["pf"].copy()
df_all.to_csv("ga_all_evals.csv", index=False)
df_pf.to_csv("ga_pareto_front.csv", index=False)
print(f"[SAVED] ga_all_evals.csv  ({len(df_all)} evals)")
print(f"[SAVED] ga_pareto_front.csv ({len(df_pf)} non-dominated)")

# === Plots ===
def ensure_cols(df, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing expected columns: {missing}. Available: {list(df.columns)}")

""" # uncomment to check all trials
# Mass vs (PCR & KDF)
ensure_cols(df_all, ["mass","pcr_lb","kdf"])
fig, ax1 = plt.subplots(figsize=(8,6))
ax2 = ax1.twinx()
ax1.scatter(df_all["mass"], df_all["pcr_lb"], s=14, alpha=0.35, label="All - PCR (N)", marker="o", color="black")
ax2.scatter(df_all["mass"], df_all["kdf"],    s=14, alpha=0.35, label="All - KDF", marker="x", color="black")
ax1.set_xlabel("Mass (g)")
ax1.set_ylabel("PCR (N)")
ax2.set_ylabel("KDF")
ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
h1,l1 = ax1.get_legend_handles_labels()
h2,l2 = ax2.get_legend_handles_labels()
ax1.legend(h1+h2, l1+l2, loc="best")
ax1.grid(True); fig.tight_layout()
plt.savefig("ga_all.png", dpi=1200)
print("[SAVED] ga_all.png")
"""
if not df_pf.empty:
    ensure_cols(df_pf, ["mass","pcr_lb","kdf"])
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
    plt.savefig("ga_pareto.pdf")
    print("[SAVED] ga_pareto.pdf")

# === Pcr Convergence ===
if "log" in res and not res["log"].empty:
    df_conv = res["log"].copy()
    plt.figure(figsize=(8,6))
    plt.plot(df_conv["gen"], df_conv["best_pcr_feasible"]/1000, label="PCR (kN)", color="black")
    plt.xlabel("Generation"); plt.ylabel("PCR (kN)")
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig("ga_pcr.pdf")
    print("[SAVED] ga_pcr.pdf")

# === Hypervolume vs evals ===
import torch
from botorch.utils.multi_objective.hypervolume import Hypervolume
pcr_thr = 2e4
kdf_thr = 0.50
mass_thr = 164.0
ref_pt = torch.tensor([float(pcr_thr), float(kdf_thr), float(-mass_thr)])

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

# === HV progress ===
df_eval = df_all.copy()
perm = np.random.permutation(len(df_eval))
hv_vals, best = [], 0.0
for k in range(1, len(perm)+1):
    sub = df_eval.iloc[perm[:k]]
    feas = [is_feasible(m, k) for m, k in zip(sub["mass"], sub["kdf"])]
    subf = sub.loc[feas]
    if not subf.empty:
        P = subf[["mass","pcr_lb","kdf"]].to_numpy(float)
        mask = nondominated_mask(P)
        hv = hv_of_front(subf.loc[mask])
        best = max(best, hv)
    hv_vals.append(best)

plt.figure(figsize=(8,6))
plt.plot(np.arange(1,len(hv_vals)+1), np.array(hv_vals)/1000, label="HV", color="black")
plt.xlabel("Evaluations"); plt.ylabel("Hypervolume")
plt.grid(True); plt.legend(); plt.tight_layout()
plt.savefig("ga_hypervolume.pdf")
print("[SAVED] ga_hypervolume.pdf")

# === Representative designs from Pareto set ===
if not df_pf.empty:
    best_pcr = df_pf.loc[df_pf["pcr_lb"].idxmax()]
    best_mass = df_pf.loc[df_pf["mass"].idxmin()]
    best_kdf = df_pf.loc[df_pf["kdf"].idxmax()]
    reps = pd.DataFrame([best_pcr, best_mass, best_kdf])
    reps.insert(0, "which", ["best_pcr","best_mass","best_kdf"])
    reps.to_csv("ga_best_designs.csv", index=False)
    print("\n[RESULTS] Representative GA designs saved.")
    print(reps[["which","mass","pcr_lb","rl_lb","kdf"]].to_string(index=False))
