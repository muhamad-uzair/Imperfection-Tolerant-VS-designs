import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (kept for 3D plots)
from scipy.interpolate import griddata
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter
from scipy.stats import truncnorm
from pyDOE import lhs
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.colorbar import ColorbarBase

# === GLOBAL FONT SETTINGS ===
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams.update({
    'font.size': 22,
    'axes.labelsize': 22,
    'xtick.labelsize': 22,
    'ytick.labelsize': 22,
})

# === Sampling choice ===
sampling_method = "lhs"
sampling_bound_sigma = 3

# === CONFIG ===
data_dir = r"Link to downloaded stiched dataset folder" #provide folder link here please
file_list = sorted([f for f in os.listdir(data_dir) if f.endswith("msi_theta_z_imp.txt") and f != "S8-1-measurement-3_msi_theta_z_imp.txt"])
output_dir = os.path.join(data_dir, f"pca_results_{sampling_method}")
os.makedirs(output_dir, exist_ok=True)

# === GRID SETTINGS ===
theta_grid = np.linspace(0, 2 * np.pi, 100)
z_grid = np.linspace(0, 1, 100)
TH, ZH = np.meshgrid(theta_grid, z_grid)

# === Global output paths ===
global_score_log = os.path.join(output_dir, "All_pca_scores.csv")

# === Winding-angle helpers ===
def mean_angle_from_filename(fname):
    """Map dataset naming (S1/S2/S4/S8) to a mean winding angle in degrees."""
    base = os.path.basename(fname)
    if base.startswith("S1"):  
        return 50.0
    if base.startswith("S2"):  
        return 58.0
    if base.startswith("S4"):  
        return 59.0
    if base.startswith("S8"):  
        return float(np.mean([55, 57, 61, 57, 57, 61, 57, 55]))  
    return np.nan  

def sample_s8_angles(lo=55.0, hi=61.0):
    """Simple per-segment random S8 winding angles (you can swap for a smoother sampler later)."""
    return np.random.uniform(lo, hi, size=8)

# === LHS sampler, returns residual draws centered ===
def sample_lhs(num_samples, num_modes, sigma_array, bound=3.0):
    lhs_unit = lhs(num_modes, samples=num_samples)
    sigma_array = sigma_array.reshape(1, -1)

    # === DAMPING FACTOR ===
    damping = 1.0 / (np.arange(1, num_modes + 1) ** 0.5)  # square-root decay
    damping = damping.reshape(1, -1)
    scaled = (lhs_unit * 2 * bound - bound) * sigma_array * damping
    return scaled

# === Colorbar Image ===
vmin, vmax = 0, 1
fig_cb = plt.figure(figsize=(6, 0.4))
ax_cb  = fig_cb.add_axes([0.1, 0.4, 0.8, 0.3])  
cb = ColorbarBase(
    ax_cb,
    cmap=matplotlib.colormaps["viridis"],
    norm=Normalize(vmin, vmax),
    orientation='horizontal'
)
cb.set_ticks([vmin, 0.5*(vmin+vmax), vmax])
cb.set_ticklabels([f"{v:.1f}" for v in [vmin, (vmin+vmax)/2, vmax]])
cb.set_label("Imperfection (mm)", labelpad=4)
cb.ax.tick_params(labelsize=10, width=0.8, length=3)
cb.ax.xaxis.label.set_fontsize(11)
fig_cb.savefig(os.path.join(output_dir, "test_colorbar.pdf"),
               bbox_inches='tight', transparent=True)
plt.close(fig_cb)

# === Per-cylinder processing ===
def process_cylinder(cyl_id, file_list, num_samples=3):
    imp_samples = []
    mean_IMP = None

    # === keep-rows logic preserved ===
    cut_rows = int(0.05 * TH.shape[0])
    keep_rows = TH.shape[0] - cut_rows
    TH_trimmed = TH[:keep_rows, :]
    ZH_trimmed = ZH[:keep_rows, :]
    z_mask = ZH[:, 0] < (1 - 0.05)
    flat_mask = np.tile(z_mask[:, np.newaxis], (1, TH.shape[1])).flatten()

    print(f"\n--- Processing files ---")
    for i, fname in enumerate(sorted(file_list), 1):
        print(f"Loading: {fname}")
        data = np.loadtxt(os.path.join(data_dir, fname))
        theta = np.mod(data[:, 0], 2 * np.pi)
        z = (data[:, 1] - data[:, 1].min()) / (data[:, 1].max() - data[:, 1].min())
        imp = data[:, 2]

        IMP = griddata((theta, z), imp, (TH, ZH), method="cubic")
        IMP[np.isnan(IMP)] = 0
        IMP = gaussian_filter(IMP, sigma=0)
        imp_samples.append(IMP.reshape(-1))

        real_sample = IMP[:keep_rows, :]  
        sample_id = os.path.splitext(fname)[0]
        #np.save(os.path.join(output_dir, f"{sample_id}_real.npy"), real_sample)
        mean_IMP = IMP if mean_IMP is None else mean_IMP + IMP

    print(f"[INFO] Total samples loaded: {len(imp_samples)}")
    mean_IMP /= len(file_list)

    # === build masked matrix X, center ===
    X_raw = np.stack(imp_samples)
    X_masked = X_raw[:, flat_mask]
    mean_raw_masked = np.mean(X_masked, axis=0)
    X = X_masked - mean_raw_masked

    # === PCA selection logic preserved: ≥1% per-mode or cum ≤95% ===
    print("[INFO] Performing PCA...")
    full_pca = PCA().fit(X)
    """  # uncomment to check individual & commulative scores of all modes
    print("[DEBUG] Variance ratio for each PCA mode:")
    for i, var in enumerate(full_pca.explained_variance_ratio_, 1):
        print(f"Mode {i}: {var:.5f}")
    print("[DEBUG] Cumulative variance:")
    print(np.cumsum(full_pca.explained_variance_ratio_))
    """
    cumulative = np.cumsum(full_pca.explained_variance_ratio_)
    mask_modes = np.logical_or(full_pca.explained_variance_ratio_ >= 0.01, cumulative <= 0.95)
    pca = PCA()
    pca.components_ = full_pca.components_[mask_modes]
    pca.explained_variance_ratio_ = full_pca.explained_variance_ratio_[mask_modes]
    pca.mean_ = full_pca.mean_
    num_modes = pca.components_.shape[0]
    print(f"[INFO] Retained {num_modes} PCA modes")

    # === Save PCA data ===
    print("[INFO] Saving PCA data...")
    mean_full = np.zeros(TH.size)
    mean_full[flat_mask] = pca.mean_
    np.save(os.path.join(output_dir, "ALL_mean.npy"), mean_full.reshape(TH.shape)[:keep_rows, :])

    components_full = []
    for comp in pca.components_:
        full = np.zeros(TH.size)
        full[flat_mask] = comp
        components_full.append(full.reshape(TH.shape))
    np.save(os.path.join(output_dir, "ALL_pca_components.npy"), np.array(components_full))
    np.save(os.path.join(output_dir, "ALL_pca_variance_ratios.npy"), pca.explained_variance_ratio_)

    # === Variance Spectrum ===
    plt.figure()
    plt.bar(range(1, len(full_pca.explained_variance_ratio_) + 1), full_pca.explained_variance_ratio_)
    plt.xlabel("PCA Mode")
    plt.ylabel("Explained Variance Ratio")
    plt.yticks([0, 1])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ALL_variance_spectrum.pdf"))
    plt.close()

    # === Cumulative Variance ===
    plt.figure()
    yvals = np.cumsum(full_pca.explained_variance_ratio_)
    plt.plot(np.cumsum(full_pca.explained_variance_ratio_), marker='o')
    plt.axhline(0.95, color='r', linestyle='--', label='95% threshold')
    plt.xlabel("Number of PCA Modes")
    plt.ylabel("Cumulative Variance")
    plt.grid(True)
    ax = plt.gca()
    ymin, ymax = yvals.min(), yvals.max()
    yticks = np.round(np.linspace(ymin, ymax, 3), 2)
    ax.set_yticks(yticks)
    plt.subplots_adjust(left=0.18, bottom=0.18, right=0.97, top=0.9)
    plt.savefig(os.path.join(output_dir, "ALL_cumulative_variance.pdf"))
    plt.close()

    # === 2D & 3D PCA mode plots ===
    plot_3D = True
    for i in range(num_modes):
        mode_full = np.zeros(TH.size)
        mode_full[flat_mask] = pca.components_[i]
        mode_2d = mode_full.reshape(TH.shape)

        plt.figure()
        contour = plt.contourf(TH_trimmed, ZH_trimmed, mode_2d[:keep_rows, :], 300, cmap="viridis", antialiased=False)
        plt.xticks(
            ticks=[0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi],
            labels=[r'$0$', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'],
            fontsize=20
        )
        plt.yticks(
            ticks=[0, 0.5, 0.95],
            labels=[r'$0$', r'$H/2$', r'$H$'],
            fontsize=20
        )
        ax = plt.gca()
        plt.savefig(os.path.join(output_dir, f"Mode_{i+1}.png"), dpi=1200)
        plt.close()

        if plot_3D:
            radius_nominal = 68      # mm
            height_nominal = 300     # mm

            mode_2d = mode_2d[:keep_rows, :]
            R = (radius_nominal) + mode_2d
            Zz = ZH_trimmed * height_nominal
            Xx = R * np.cos(TH_trimmed)
            Yy = R * np.sin(TH_trimmed)

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            surf = ax.plot_surface(
                Xx, Yy, Zz,
                facecolors=plt.cm.viridis((mode_2d - mode_2d.min()) / (mode_2d.max() - mode_2d.min())),
                rstride=1, cstride=1, antialiased=False, shade=False
            )
            mappable = plt.cm.ScalarMappable(cmap='viridis')
            mappable.set_array(mode_2d)
            vmin, vmax = mode_2d.min(), mode_2d.max()
            tick_vals = [vmin, (vmin + vmax)/2, vmax]
            tick_labels = [f"{v*1:.2f}" for v in tick_vals]
            #cbar.set_ticks(tick_vals)
            #cbar.set_ticklabels(tick_labels)
            #cbar.set_label("Imperfection [mm]", fontsize=20)
            #cbar.ax.tick_params(labelsize=20)

            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_zlabel("")
            ax.set_xticks([-radius_nominal, 0, radius_nominal])
            ax.set_xticklabels([r'$0$', r'$\pi$', r'$2\pi$'], fontsize=20)
            ax.set_yticks([radius_nominal, 0, -radius_nominal])
            ax.set_yticklabels([r'$0$', r'$\pi$', r'$2\pi$'], fontsize=20)
            ax.set_zticks([0, height_nominal/2, height_nominal])
            ax.set_zticklabels([r'$0$', r'$H/2$', r'$H$'], fontsize=20)
            ax.view_init(elev=30, azim=135)
            plt.subplots_adjust(left=0.05, right=0.7, top=0.99, bottom=0.02)
            plt.savefig(os.path.join(output_dir, f"Mode_{i+1}_3Dcylinder.png"), dpi=1200)
            plt.close()

    # === Log PCA alpha scores for real samples ===
    print("[INFO] Logging PCA alpha scores for real samples...")
    scores = X @ pca.components_.T  # [n_real_samples, num_modes]
    mode_stds = np.std(scores, axis=0)

    result = {"Scope": cyl_id}
    for i, (var, amp) in enumerate(zip(pca.explained_variance_ratio_, mode_stds)):
        result[f"Var{i+1}"] = round(var, 5)
        result[f"ModeAmp{i+1}"] = round(amp, 5)
    summary = [result]
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(os.path.join(output_dir, "pca_summary.csv"), index=False)

    angle_cols = [f"S8_a{k}" for k in range(1, 9)] + ["S8_mean"]
    with open(global_score_log, "w") as f:
        header = ["Cylinder", "Sample"] + [f"Alpha{i+1}" for i in range(num_modes)] + ["Amplitude_mm", "RMS_mm"] + angle_cols
        f.write(",".join(header) + "\n")

    for idx, fname in enumerate(sorted(file_list)):
        sample_id = os.path.splitext(fname)[0]
        alpha_str = ",".join([f"{v:.6f}" for v in scores[idx]])
        amp = real_sample.max() - real_sample.min()
        rms = np.sqrt(np.mean((real_sample - real_sample.mean()) ** 2))
        empty_angles = ",".join([""] * len(angle_cols))
        with open(global_score_log, "a") as f:
            f.write(f"{cyl_id},{sample_id},{alpha_str},{amp:.6f},{rms:.6f},{empty_angles}\n")

    # === Learn score ~ a + b * mean_angle from experimental files ===
    print("[INFO] Fitting per-mode linear trend vs mean winding angle ...")
    mean_angles_all = np.array([mean_angle_from_filename(f) for f in sorted(file_list)])
    valid_mask = ~np.isnan(mean_angles_all)
    if np.count_nonzero(valid_mask) < 2:
        # fallback: no conditioning possible
        print("[WARN] Not enough labeled files for angle-conditioning; reverting to unconditioned LHS.")
        a_vec = np.zeros(num_modes)
        b_vec = np.zeros(num_modes)
        resid_std = np.std(scores, axis=0) + 1e-12
    else:
        A = np.c_[np.ones(valid_mask.sum()), mean_angles_all[valid_mask]]  # [1, mean_angle]
        Y = scores[valid_mask, :]                                         # [n_valid, num_modes]
        AB, *_ = np.linalg.lstsq(A, Y, rcond=None)                        # shape (2, num_modes)
        a_vec = AB[0, :]
        b_vec = AB[1, :]

        # residual std per mode (scatter around the angle trend)
        Y_hat = A @ AB
        resid = Y - Y_hat
        resid_std = np.std(resid, axis=0, ddof=1) + 1e-12
    
    # === Save regression coefficients for later reconstruction ===
    np.save(os.path.join(output_dir, "angle_trend_a.npy"), a_vec)       # shape [num_modes]
    np.save(os.path.join(output_dir, "angle_trend_b.npy"), b_vec)       # shape [num_modes]
    np.save(os.path.join(output_dir, "residual_std.npy"),  resid_std)   # shape [num_modes]
    print("[SAVED] angle_trend_a.npy, angle_trend_b.npy, residual_std.npy")

    # === Generate synthetic samples with LHS residuals centered at μ(angle) ===
    print(f"[INFO] Generating {num_samples} synthetic samples...")
    if sampling_method == "lhs":

        residual_draws = sample_lhs(num_samples, num_modes, resid_std, bound=sampling_bound_sigma)
        all_scores = []
        all_angles = []
        for n in range(num_samples):
            angles8 = sample_s8_angles(55, 61)        
            mean_a = float(np.mean(angles8))
            mu_vec = a_vec + b_vec * mean_a          
            scores_i = mu_vec + residual_draws[n, :]  
            all_scores.append(scores_i)
            all_angles.append(angles8)

    # === Reconstruct, save .npy, and append to CSV (unchanged + angles logged) ===
    for i, scores_i in enumerate(all_scores, 1):
        recon_masked = mean_raw_masked + np.dot(scores_i, pca.components_)
        full_sample = np.zeros(TH.size)
        full_sample[flat_mask] = recon_masked
        sample_2d = gaussian_filter(full_sample.reshape(TH.shape), sigma=1)[:keep_rows, :]
        np.save(os.path.join(output_dir, f"ALL_sample_{i}.npy"), sample_2d)
        amp = sample_2d.max() - sample_2d.min()
        rms = np.sqrt(np.mean((sample_2d - sample_2d.mean()) ** 2))
        alpha_str = ",".join([f"{v:.6f}" for v in scores_i])

        # log angles used for this synthetic
        angles8 = all_angles[i-1]
        angles_str = ",".join([f"{a:.3f}" for a in angles8] + [f"{np.mean(angles8):.3f}"])

        with open(global_score_log, "a") as f:
            f.write(f"{cyl_id},ALL_sample_{i},{alpha_str},{amp:.6f},{rms:.6f},{angles_str}\n")

    print(f"[DONE] Saved synthetic samples and PCA scores to: {output_dir}")

# === Run Global PCA ===
process_cylinder("ALL", file_list)
