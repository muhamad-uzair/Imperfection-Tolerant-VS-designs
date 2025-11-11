import os
import re
import math
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# === PLOTTING SETUP ===
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams.update({'font.size': 14, 'axes.labelsize': 14, 'xtick.labelsize': 14, 'ytick.labelsize': 14})

# === USER CONFIG ===
# PCA output folder containing All_pca_scores.csv and ALL_sample_*.npy
pca_output_dir = r"Link to the folder"

# Base Abaqus model (template INP) to modify per sample (NONLINEAR baseline)
template_inp_nl = r"Link to the file"

# Base Abaqus model (template INP) for LINEAR eigenvalue case (no imperfection)
template_inp_lin = r"Link to the file"

# Output subfolders (created inside pca_output_dir)
out_subdir_nl  = "modified_inp_nonlinear"
out_subdir_lin = "modified_inp_linear"

# Skip first N rows (experimental scans) in PCA CSV
skip_first_rows = 35

# Limit number of synthetic samples to process (None = all)
max_samples = None

# If your node Z direction is not min..max over height, adjust here (False means auto-detect)
force_z_min = None  # e.g., 0.0
force_z_max = None  # e.g., 300.0 (mm)

# --- Physical constants for mass calc (units consistent with model) ---
DENSITY = 1.6e-3   # g/mm^3
# ===============================================================

CSV_NAME = "All_pca_scores.csv"
ELSET_SEGMENT_REGEX = r'^[cC](\d+)-1$'  # captures 1..8 from elset=c3-1

# === CSV: read S8 angles ===
def load_angles(csv_path: str, skip_rows: int = 35):
    df = pd.read_csv(csv_path)
    if skip_rows:
        df = df.iloc[skip_rows:].reset_index(drop=True)

    angle_cols = [f"S8_a{k}" for k in range(1, 9)]
    needed = ["Sample"] + angle_cols
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise RuntimeError(f"CSV missing columns: {missing}")

    df = df[needed].copy()
    for c in angle_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=angle_cols, how="any").reset_index(drop=True)
    if df.empty:
        raise RuntimeError("No synthetic rows with complete S8_a1..S8_a8 after skipping.")
    return df, angle_cols

# === INP: utility parsers ===
def parse_shell_sections(lines):
    out = []
    cur_start, cur_header = None, None
    for i, line in enumerate(lines):
        if line.lstrip().startswith("*Shell Section"):
            if cur_start is not None:
                out.append((cur_start, i, cur_header))
            cur_start = i
            cur_header = line.rstrip("\n")
        elif line.lstrip().startswith("*") and cur_start is not None:
            out.append((cur_start, i, cur_header))
            cur_start, cur_header = None, None
    if cur_start is not None:
        out.append((cur_start, len(lines), cur_header))
    return out

def get_segment_index_from_header(header_line: str):
    m = re.search(r"elset\s*=\s*([^,\s]+)", header_line, flags=re.IGNORECASE)
    if not m:
        return None
    elset = m.group(1).strip()
    m2 = re.match(ELSET_SEGMENT_REGEX, elset, flags=re.IGNORECASE)
    return int(m2.group(1)) if m2 else None

def is_ply_line(s: str) -> bool:
    if s.lstrip().startswith("*"): return False
    if "Ply" not in s:            return False
    return (s.count(",") >= 4)

def rewrite_ply_line(line: str, theta_deg: float, keep_sign: bool = True) -> str:
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 5:
        return line
    try:
        base_thk = float(parts[0])
    except Exception:
        return line

    old_angle = None
    try:
        old_angle = float(parts[3])
    except Exception:
        pass

    mag = float(theta_deg)
    angle_out = (1.0 if (keep_sign and old_angle is not None and old_angle >= 0) else
                 (-1.0 if (keep_sign and old_angle is not None and old_angle < 0) else 1.0)) * mag

    c = abs(math.cos(math.radians(angle_out)))
    if c < 1e-12: c = 1e-12
    new_thk = base_thk / c  # laminate-equivalent thickness adjustment

    parts[0] = f"{new_thk:.6f}"   # thickness
    parts[3] = f"{angle_out:.3f}" # angle (3 decimals)
    return ", ".join(parts)

def apply_angles_to_sections(lines, row, angle_cols):
    lines = list(lines)
    sections = parse_shell_sections(lines)
    seg_angle = {i + 1: float(row[angle_cols[i]]) for i in range(8)}
    audit = []
    for (s, e, header) in sections:
        seg_idx = get_segment_index_from_header(header)
        if seg_idx is None or seg_idx not in seg_angle:
            continue
        theta = seg_angle[seg_idx]
        for i in range(s + 1, e):
            line = lines[i]
            if is_ply_line(line):
                old = line.rstrip("\n")
                new = rewrite_ply_line(old, theta_deg=theta, keep_sign=True)
                lines[i] = new + "\n"
                audit.append({
                    "Segment": seg_idx, "LineIndex": i,
                    "Theta_deg": theta, "OldLine": old, "NewLine": new
                })
    return lines, audit

# === Imperfection field -> nodes ===
def parse_node_blocks(lines):
    blocks = []
    i = 0
    n = len(lines)
    while i < n:
        if lines[i].lstrip().startswith("*Node"):
            j = i + 1
            while j < n and not lines[j].lstrip().startswith("*"):
                j += 1
            blocks.append((i, j))
            i = j
        else:
            i += 1
    return blocks

def read_nodes_from_block(lines, start, end):
    ids, coords = [], []
    for k in range(start + 1, end):
        s = lines[k].strip()
        if not s or s.startswith("*"):
            continue
        parts = [p.strip() for p in s.split(",")]
        if len(parts) < 4:
            continue
        try:
            nid = int(float(parts[0]))
            x = float(parts[1]); y = float(parts[2]); z = float(parts[3])
            ids.append(nid); coords.append([x, y, z])
        except Exception:
            pass
    return np.array(ids, dtype=int), np.array(coords, dtype=float).reshape(-1, 3)

def write_nodes_to_block(lines, start, end, ids, coords):
    i = start + 1
    fmt = "{:d}, {: .6f}, {: .6f}, {: .6f}\n"
    for nid, (x, y, z) in zip(ids, coords):
        lines[i] = fmt.format(nid, x, y, z)
        i += 1

def ring_from_z(z_val, z_min, height, inc_with_z=True):
    """Map a z-value to axial ring index 1..8."""
    z_norm = min(max((z_val - z_min) / max(height,1e-12), 0.0), 1.0 - 1e-12)
    idx0 = int(np.floor(z_norm * 8.0))  # 0..7
    seg = idx0 + 1                      # 1..8
    if not inc_with_z: 
        seg = 9 - seg
    return seg

def avg_radius_from_nodes(lines):
    """Compute average radius across all nodes."""
    blocks = parse_node_blocks(lines)
    rs = []
    for (s, e) in blocks:
        _, coords = read_nodes_from_block(lines, s, e)
        if coords.size:
            r = np.sqrt(coords[:,0]**2 + coords[:,1]**2)
            rs.append(r)
    if not rs:
        return 1.0
    return float(np.concatenate(rs).mean())

def bilinear_sample(field2d, theta, z_norm):
    NZ, NTH = field2d.shape
    twopi = 2.0 * math.pi
    t = theta % twopi
    t_norm = t / twopi
    zc = min(max(z_norm, 0.0), 0.95)
    zn = (zc / 0.95) * (NZ - 1)
    tn = t_norm * (NTH - 1)
    i0 = int(math.floor(zn)); j0 = int(math.floor(tn))
    i1 = min(i0 + 1, NZ - 1); j1 = (j0 + 1) % NTH
    dz = zn - i0; dt = tn - j0
    v00 = field2d[i0, j0]; v01 = field2d[i0, j1]
    v10 = field2d[i1, j0]; v11 = field2d[i1, j1]
    v0 = v00 * (1 - dt) + v01 * dt
    v1 = v10 * (1 - dt) + v11 * dt
    return v0 * (1 - dz) + v1 * dz

def inject_geometric_imperfection(base_lines, field2d, seg_angles, inc_with_z=True):
    lines = list(base_lines)
    blocks = parse_node_blocks(lines)
    if not blocks:
        return base_lines, {"num_nodes": 0, "num_blocks": 0}

    # gather z extents
    all_coords = []
    for (s, e) in blocks:
        _, coords = read_nodes_from_block(lines, s, e)
        all_coords.append(coords)
    all_coords_arr = np.vstack(all_coords) if all_coords else np.zeros((0, 3))
    if all_coords_arr.shape[0] == 0:
        return base_lines, {"num_nodes": 0, "num_blocks": len(blocks)}

    z_min = force_z_min if force_z_min is not None else float(all_coords_arr[:, 2].min())
    z_max = force_z_max if force_z_max is not None else float(all_coords_arr[:, 2].max())
    z_span = max(1e-12, (z_max - z_min))
    R_avg = avg_radius_from_nodes(lines)
    k_by_ring = {
        seg: math.tan(math.radians(abs(phi))) / max(R_avg, 1e-12)
        for seg, phi in seg_angles.items()
    }


    total_nodes = 0
    for (s, e) in blocks:
        ids, coords = read_nodes_from_block(lines, s, e)
        if coords.shape[0] == 0:
            continue
        x = coords[:, 0]; y = coords[:, 1]; z = coords[:, 2]
        theta = np.arctan2(y, x); theta = np.where(theta < 0, theta + 2*np.pi, theta)
        r = np.sqrt(x*x + y*y)
        z_norm = (z - z_min) / z_span
        imp_vals = []
        for th, zn, zval in zip(theta, z_norm, z):
            ring = ring_from_z(zval, z_min, z_span, inc_with_z=inc_with_z)
            k = k_by_ring.get(ring, 0.0)
            theta_p = th + k*(zval - z_min)
            theta_m = th - k*(zval - z_min)
            imp_p = bilinear_sample(field2d, theta_p, zn)
            imp_m = bilinear_sample(field2d, theta_m, zn)
            imp_vals.append(0.5*(imp_p + imp_m))
        imp = np.array(imp_vals, dtype=float)

        r_new = r + imp
        x_new = r_new * np.cos(theta); y_new = r_new * np.sin(theta)
        new_coords = np.column_stack([x_new, y_new, z])
        write_nodes_to_block(lines, s, e, ids, new_coords)
        total_nodes += ids.size

    return lines, {"num_nodes": total_nodes, "num_blocks": len(blocks)}

# === Thickness parsing & mass ===
def parse_segment_thicknesses(lines):
    seg_thk = {}
    for (s, e, header) in parse_shell_sections(lines):
        seg_idx = get_segment_index_from_header(header)
        if not seg_idx:
            continue
        t_sum = 0.0
        for i in range(s + 1, e):
            line = lines[i]
            if is_ply_line(line):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 1:
                    try:
                        t_sum += float(parts[0])
                    except Exception:
                        pass
        if t_sum > 0:
            seg_thk[seg_idx] = t_sum
    return seg_thk

def compute_mass_quick(lines, density_g_per_mm3, seg_index_increases_with_z: bool = True):
    """
    Approximate shell mass by integrating over a (z,θ) grid with thickness varying by z:
      - The cylinder is cut into 8 axial segments (rings): c1-1..c8-1.
      - Each ring's total laminate thickness (sum of plies after rewrite) is applied
        to the corresponding axial zone across the full 0..2π.
      - Area cell ~ r_avg * dθ * dz, volume = area * thickness(z), mass = volume * density.
    """
    # === Nodes -> radius grid ===
    blocks = parse_node_blocks(lines)
    if not blocks:
        return 0.0

    all_xyz = []
    for (s, e) in blocks:
        _, coords = read_nodes_from_block(lines, s, e)
        if coords.shape[0]:
            all_xyz.append(coords)
    if not all_xyz:
        return 0.0
    nodes = np.vstack(all_xyz)

    z_min = float(nodes[:, 2].min())
    z_max = float(nodes[:, 2].max())
    height = max(1e-12, z_max - z_min)

    # Grid resolution (keep consistent with your PCA field)
    N_THETA = 100
    N_Z = 95
    dtheta = 2 * math.pi / N_THETA
    dz = height / N_Z

    theta = np.arctan2(nodes[:, 1], nodes[:, 0])
    theta = np.where(theta < 0, theta + 2*np.pi, theta)
    r = np.sqrt(nodes[:, 0]**2 + nodes[:, 1]**2)
    z = nodes[:, 2]

    theta_bins = np.linspace(0, 2*np.pi, N_THETA + 1)
    z_bins = np.linspace(z_min, z_max, N_Z + 1)

    grid_r_sum = np.zeros((N_Z, N_THETA))
    grid_count = np.zeros((N_Z, N_THETA), dtype=int)

    i_t = np.clip(np.digitize(theta, theta_bins) - 1, 0, N_THETA - 1)
    i_z = np.clip(np.digitize(z, z_bins) - 1, 0, N_Z - 1)
    for rr, it, iz in zip(r, i_t, i_z):
        grid_r_sum[iz, it] += rr
        grid_count[iz, it] += 1

    with np.errstate(invalid="ignore"):
        grid_r_avg = np.where(grid_count > 0, grid_r_sum / grid_count, np.nan)

    # Fill empties along θ then z (simple 1D interpolation)
    for iz in range(N_Z):
        row = grid_r_avg[iz]
        if np.all(np.isnan(row)): 
            continue
        mask = ~np.isnan(row)
        if not mask.all():
            row[~mask] = np.interp(np.flatnonzero(~mask), np.flatnonzero(mask), row[mask])
        grid_r_avg[iz] = row

    for it in range(N_THETA):
        col = grid_r_avg[:, it]
        if np.all(np.isnan(col)):
            continue
        mask = ~np.isnan(col)
        if not mask.all():
            col[~mask] = np.interp(np.flatnonzero(~mask), np.flatnonzero(mask), col[mask])
        grid_r_avg[:, it] = col

    # === Thickness by axial segment (rings) ===
    seg_thk = parse_segment_thicknesses(lines)  # {1: t1, ..., 8: t8}

    # Map z to segment index 1..8 (equal-height rings).
    def seg_from_z(z_val: float) -> int:
        # normalize to [0, 1); clamp top edge to last segment
        z_norm = min(max((z_val - z_min) / height, 0.0), 1.0 - 1e-12)
        idx0 = int(math.floor(z_norm * 8.0))  # 0..7
        seg = idx0 + 1                          # 1..8
        if not seg_index_increases_with_z:
            seg = 9 - seg
        return seg

    # Precompute thickness for each z-row
    thk_by_z_idx = np.zeros(N_Z)
    for iz in range(N_Z):
        z_center = z_min + (iz + 0.5) * dz
        seg = seg_from_z(z_center)
        thk_by_z_idx[iz] = seg_thk.get(seg, 0.0)

    # === Integrate mass over grid ===
    mass = 0.0
    for iz in range(N_Z):
        r_row = grid_r_avg[iz]                  # shape (N_THETA,)
        dA_row = r_row * dtheta * dz            # area strip per θ-bin
        mass += np.nansum(dA_row) * thk_by_z_idx[iz] * density_g_per_mm3

    return float(mass)

def delta_r_colorbar(cmap_name="viridis", vmin=0.0, vmax=1.0, out_dir=None):
    """
    Save a standalone HORIZONTAL colorbar image for Δr with fixed limits [vmin, vmax].
    If the file already exists, do nothing.
    """
    if out_dir is None:
        out_dir = os.getcwd()
    legend_path = os.path.join(out_dir, "legend_delta_r_0-1mm.pdf")
    if os.path.exists(legend_path):
        return legend_path

    norm = Normalize(vmin=vmin, vmax=vmax)
    sm = ScalarMappable(norm=norm, cmap=cmap_name)
    sm.set_array([])

    # Wide, short figure for horizontal bar
    fig, ax = plt.subplots(figsize=(8, 0.2))
    cbar = fig.colorbar(sm, cax=ax, orientation="horizontal")
    cbar.set_label("Δr (mm)")
    fig.savefig(legend_path, bbox_inches="tight", transparent=True)
    plt.close(fig)
    return legend_path

# === Job runner ===
def run_job(job_name, template_inp_path, out_subdir, df_angles, angle_cols,
            use_imperfection: bool):
    out_dir = os.path.join(pca_output_dir, out_subdir)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    if not os.path.exists(template_inp_path):
        raise FileNotFoundError(f"{job_name}: Template INP not found: {template_inp_path}")

    with open(template_inp_path, "r") as f:
        base_lines = f.readlines()

    results = []
    for idx, row in df_angles.iterrows():
        sample_tag = str(row["Sample"])  # e.g., "ALL_sample_17"
        # optional PCA field (only needed if we inject imperfection)
        npy_path = os.path.join(pca_output_dir, f"{sample_tag}.npy")

        lines_geo = list(base_lines)
        if use_imperfection:
            if not os.path.exists(npy_path):
                print(f"[{job_name}] [WARN] Field not found for {sample_tag}: {npy_path} (skipping)")
                continue
            field2d = np.load(npy_path).astype(float)
            if field2d.ndim != 2:
                print(f"[{job_name}] [WARN] Field is not 2D for {sample_tag} ({field2d.shape}); skipping.")
                continue

            seg_angles = {i + 1: float(row[angle_cols[i]]) for i in range(8)}

            # If your c1-1 is bottom and c8-1 is top, leave inc_with_z=True
            lines_geo, _ = inject_geometric_imperfection(
                base_lines, field2d, seg_angles, inc_with_z=True
            )
        else:
            lines_geo = list(base_lines)


        # Apply ply angle/thickness
        lines_final, _ = apply_angles_to_sections(lines_geo, row, angle_cols)

        # Write
        out_inp = os.path.join(out_dir, f"Modified_{sample_tag}.inp")
        with open(out_inp, "w") as f:
            f.writelines(lines_final)
            
        """# === Uncomment if: optional visualization of injected imperfection ===
        try:
            # Only visualize when we injected imperfections (nonlinear job)
            if use_imperfection:
                base_inp_for_plot = template_inp_path       # before
                mod_inp_for_plot  = out_inp                 # after

                def _read_nodes(path):
                    ids, xyz = [], []
                    with open(path, 'r') as fh:
                        L = fh.readlines()
                    i = 0
                    while i < len(L):
                        if L[i].lstrip().startswith("*Node"):
                            i += 1
                            while i < len(L) and not L[i].lstrip().startswith("*"):
                                s = L[i].strip()
                                if s:
                                    parts = [p.strip() for p in s.split(",")]
                                    if len(parts) >= 4:
                                        try:
                                            nid = int(float(parts[0]))
                                            x, y, z = map(float, parts[1:4])
                                            ids.append(nid); xyz.append([x,y,z])
                                        except:
                                            pass
                                i += 1
                        else:
                            i += 1
                    return np.array(ids, int), np.array(xyz, float).reshape(-1,3)

                id_b, xyz_b = _read_nodes(base_inp_for_plot)
                id_m, xyz_m = _read_nodes(mod_inp_for_plot)
                ob = np.argsort(id_b); om = np.argsort(id_m)
                id_b, xyz_b = id_b[ob], xyz_b[ob]; id_m, xyz_m = id_m[om], xyz_m[om]
                if np.array_equal(id_b, id_m):
                    x0,y0,z0 = xyz_b[:,0], xyz_b[:,1], xyz_b[:,2]
                    x1,y1,z1 = xyz_m[:,0], xyz_m[:,1], xyz_m[:,2]
                    theta = np.arctan2(y0, x0); theta = np.where(theta<0, theta+2*np.pi, theta)
                    r0 = np.sqrt(x0*x0 + y0*y0)
                    r1 = np.sqrt(x1*x1 + y1*y1)
                    dr = r1 - r0
                    zmin, zmax = z0.min(), z0.max()
                    NTH, NZ = 180, 120
                    th_bins = np.linspace(0, 2*np.pi, NTH+1)
                    z_bins  = np.linspace(zmin, zmax, NZ+1)
                    grid_sum = np.zeros((NZ, NTH)); grid_n = np.zeros((NZ, NTH), int)
                    ti = np.clip(np.digitize(theta, th_bins)-1, 0, NTH-1)
                    zi = np.clip(np.digitize(z0,  z_bins)-1,  0, NZ-1)
                    for d, i, j in zip(dr, zi, ti):
                        grid_sum[i,j] += d; grid_n[i,j] += 1
                    grid = np.where(grid_n>0, grid_sum/np.maximum(grid_n,1), np.nan)
                    delta_r_colorbar(out_dir=os.path.dirname(mod_inp_for_plot))

                    # plot
                    fig, ax = plt.subplots()
                    im = ax.imshow(grid, extent=[0, 360, zmin, zmax], aspect='auto')

                    ax.set_xlabel("θ (deg)", fontsize=24)
                    ax.set_ylabel("z (mm)", fontsize=24)

                    # Force y-axis ticks to 0, 100, 200, 300
                    ax.set_yticks([0, 100, 200, 300])
                    ax.set_ylim(zmin, zmax)
                    ax.minorticks_off()
                    ax.tick_params(axis='both', labelsize=24)

                    # Ring edges
                    z_edges = np.linspace(zmin, zmax, 9)

                    # Draw ring boundaries (use ax.*, not plt.*)
                    for ze in z_edges:
                        ax.hlines(ze, 0, 360, colors='white', linewidth=2, alpha=0.9)

                    R_avg_plot = float(r0.mean())
                    def k_deg_per_mm(phi_deg, R):
                        return (180.0/np.pi) * math.tan(math.radians(abs(phi_deg))) / max(R, 1e-12)
                    theta0_seeds = list(range(0, 360, 60))
                    for i_ring in range(8):
                        z0, z1 = z_edges[i_ring], z_edges[i_ring+1]
                        phi = float(seg_angles.get(i_ring+1, 0.0))
                        kdeg = k_deg_per_mm(phi, R_avg_plot)
                        z_line = np.linspace(z0, z1, 100)
                        for sign in (+1, -1):
                            for theta0 in theta0_seeds:
                                theta_line = (theta0 + sign * kdeg * (z_line - z0)) % 360.0
                                ax.plot(theta_line, z_line, color='white', linewidth=1.2, alpha=0.9)

                    png_path = os.path.splitext(mod_inp_for_plot)[0] + "_delta_r.pdf"
                    fig.savefig(png_path, dpi=1200, bbox_inches="tight")
                    plt.close(fig)
                    print(f"[{job_name}] [PLOT] Saved mesh Δr heatmap → {png_path}")
                else:
                    print(f"[{job_name}] [PLOT] Node sets differ; skip Δr plot.")
        except Exception as e:
            print(f"[{job_name}] [PLOT] Visualization failed: {e}")
        # === END: optional visualization ==="""

        # Mass is optional; harmless for linear too, so keep it
        mass_g = compute_mass_quick(lines_final, DENSITY)
        results.append((os.path.basename(out_inp), mass_g))

        print(f"[{job_name}] [OK] {sample_tag}: wrote {out_inp} | Mass ≈ {mass_g:.6g} g | "
              f"angles = " + ", ".join(f"{row[c]:.3f}" for c in angle_cols))

    # Save mass summary
    if results:
        df = pd.DataFrame(results, columns=["File", "Mass_g"])
        df.to_csv(os.path.join(out_dir, "mass_summary.csv"), index=False)
        print(f"[{job_name}] [INFO] Mass summary written: {os.path.join(out_dir, 'mass_summary.csv')}")
    print(f"[{job_name}] [DONE] Modified INPs in: {out_dir}")

# === Main flow ===
def main():
    csv_path = os.path.join(pca_output_dir, CSV_NAME)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df_angles, angle_cols = load_angles(csv_path, skip_rows=skip_first_rows)
    if isinstance(max_samples, int) and max_samples > 0:
        df_angles = df_angles.iloc[:max_samples].reset_index(drop=True)

    # === Job A: NONLINEAR (imperfection + angle/thickness) ===
    run_job(
        job_name="NONLINEAR",
        template_inp_path=template_inp_nl,
        out_subdir=out_subdir_nl,
        df_angles=df_angles,
        angle_cols=angle_cols,
        use_imperfection=True
    )

    # === Job B: LINEAR eigenvalue (angle/thickness only, NO imperfection) ===
    run_job(
        job_name="LINEAR",
        template_inp_path=template_inp_lin,
        out_subdir=out_subdir_lin,
        df_angles=df_angles,
        angle_cols=angle_cols,
        use_imperfection=False
    )

    run_job(
        job_name="NONLINEAR_from_RL",
        template_inp_path=template_inp_lin,
        out_subdir=out_subdir_nl + "_RL",
        df_angles=df_angles,
        angle_cols=angle_cols,
        use_imperfection=True
    )

if __name__ == "__main__":
    main()
