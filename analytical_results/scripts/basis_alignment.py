import argparse
from pathlib import Path
import yaml
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import svd
from scipy.optimize import linear_sum_assignment

from src.data.data_processing import whiten_data, compute_teacher_cov, get_Wprod_hist
from src.optim.eg_sgd_two_layer import simulate_GD_two_layer, simulate_EG_two_layer
from src.inits.init_two_layer import build_init

def make_rng(seed):
    return np.random.RandomState(seed)

def pretty_norm(A):
    return float(np.linalg.norm(A))

def teacher_from_config(N3, N1, rng, designed=False, spectrum=None):
    if not designed:
        return rng.randn(N3, N1)
    K = min(N1, N3)
    if not spectrum or len(spectrum) == 0:
        spectrum = [1.0, 0.8, 0.55, 0.35, 0.22, 0.14]
    if len(spectrum) < K:
        last = spectrum[-1]
        spectrum = spectrum + [last*(0.75**(i+1)) for i in range(K-len(spectrum))]
    s_vals = np.array(spectrum[:K], dtype=float)
    U_rand, _ = np.linalg.qr(rng.randn(N3, N3))
    V_rand, _ = np.linalg.qr(rng.randn(N1, N1))
    return U_rand[:, :K] @ np.diag(s_vals) @ V_rand[:, :K].T

def teacher_svd(Sigma_yx):
    U, S, Vt = svd(Sigma_yx, full_matrices=False)
    return U, S, Vt.T

def compute_svd_over_time(Wprod_hist, R_keep=None):
    T, N3, N1 = Wprod_hist.shape
    K = min(N3, N1) if R_keep is None else min(R_keep, N3, N1)
    U = np.zeros((T, N3, K))
    S = np.zeros((T, K))
    V = np.zeros((T, N1, K))
    for t in range(T):
        u, s, vt = svd(Wprod_hist[t], full_matrices=False)
        U[t], S[t], V[t] = u[:, :K], s[:K], vt.T[:, :K]
    return U, S, V

def overlap_matrices_over_time(U_series, U_tchr, use_abs=True):
    T, N, K = U_series.shape
    Ms = np.zeros((T, K, K))
    for t in range(T):
        M = U_series[t].T @ U_tchr
        Ms[t] = np.abs(M) if use_abs else M
    return Ms

def final_assignment(M_abs_seq, M_abs_seq_V=None, weight_uv=0.5):
    M_U = M_abs_seq[-1]
    if M_abs_seq_V is not None:
        M_V = M_abs_seq_V[-1]
        M = weight_uv * M_U + (1.0 - weight_uv) * M_V
    else:
        M = M_U
    row_ind, col_ind = linear_sum_assignment(-M)
    perm = np.empty_like(col_ind)
    perm[row_ind] = col_ind
    return perm

def apply_perm(series, perm):
    out = series.copy()
    out = out[:, :, perm]
    return out

def align_sign_to_teacher(series, teacher, tol=1e-8):
    out = series.copy()
    T, N, K = out.shape
    K_use = min(K, teacher.shape[1])
    for i in range(K_use):
        ref = teacher[:, i]
        dots = out[:, :, i] @ ref
        flips = (dots < -tol)[:, None]
        out[:, :, i] = np.where(flips, -out[:, :, i], out[:, :, i])
    return out

def cosine_alignment_series(U_series, V_series, U_tchr, V_tchr):
    K = min(U_series.shape[2], U_tchr.shape[1])
    T = U_series.shape[0]
    cos_u = np.zeros((T, K)); cos_v = np.zeros((T, K))
    for t in range(T):
        for i in range(K):
            cos_u[t, i] = abs(float(U_series[t, :, i].T @ U_tchr[:, i]))
            cos_v[t, i] = abs(float(V_series[t, :, i].T @ V_tchr[:, i]))
    return cos_u, cos_v


def set_rc():
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
    })

def mode_color(i):
    cmap = plt.get_cmap("tab10")
    return cmap(i % 10)

def plot_alignment_with_bands(ax, t_gd, cos_gd_mu, cos_gd_sd, t_eg, cos_eg_mu, cos_eg_sd,
                              max_modes=3, ylabel="Abs. cosine to teacher U"):
    K = 0
    if cos_gd_mu is not None: K = cos_gd_mu.shape[1]
    if cos_eg_mu is not None: K = max(K, cos_eg_mu.shape[1])
    K = min(K, max_modes)

    for i in range(K):
        color = mode_color(i)
        if t_gd is not None and cos_gd_mu is not None:
            ax.plot(t_gd, cos_gd_mu[:, i], color=color, lw=2.2, ls="-", label=None if i else "GD")
            if cos_gd_sd is not None:
                ax.fill_between(t_gd,
                                cos_gd_mu[:, i]-cos_gd_sd[:, i],
                                cos_gd_mu[:, i]+cos_gd_sd[:, i],
                                alpha=0.15, linewidth=0, color=color)
        if t_eg is not None and cos_eg_mu is not None:
            ax.plot(t_eg, cos_eg_mu[:, i], color=color, lw=2.2, ls="--", label=None if i else "EG")
            if cos_eg_sd is not None:
                ax.fill_between(t_eg,
                                cos_eg_mu[:, i]-cos_eg_sd[:, i],
                                cos_eg_mu[:, i]+cos_eg_sd[:, i],
                                alpha=0.15, linewidth=0, color=color)
    ax.axhline(1.0, color="k", lw=1, ls=":")
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Training step")
    ax.set_ylabel(ylabel)

def single_seed_run(cfg, seed,
                    whiten_data,
                    compute_teacher_cov,
                    simulate_GD_two_layer,
                    simulate_EG_two_layer, 
                    build_init):
    rng = make_rng(seed)
    P, N1, N2, N3 = cfg["P"], cfg["N1"], cfg["N2"], cfg["N3"]

    X = rng.randn(P, N1)
    Xw = whiten_data(X)
    Sigx_check = (Xw.T @ Xw) / P
    whiten_error = pretty_norm(Sigx_check - np.eye(N1))

    M_true = teacher_from_config(N3, N1, rng, designed=cfg["design_teacher"], spectrum=cfg["spectrum"])

    Y_clean = Xw @ M_true.T
    Sigma_x, Sigma_yx = compute_teacher_cov(Xw, Y_clean)
    U_tchr, S_tchr, V_tchr = teacher_svd(Sigma_yx)

    W21_0, W32_0 = build_init(
        name=cfg.get("init_name", "gauss-small"),
        Sigma_yx=Sigma_yx, N1=N1, N2=N2, N3=N3,
        seed=seed,
        scale=cfg.get("init_scale", 1e-2)
    )

    dims = (N1, N2, N3)

    t_gd = U_gd = S_gd = V_gd = None
    if cfg["run_gd"]:
        t_gd, W_hist_gd = simulate_GD_two_layer(
            W21_0, W32_0, Sigma_x, Sigma_yx,
            eta=cfg["eta_gd"], n_steps=cfg["steps"], record_every=cfg["record_every"]
        )
        Wprod_gd = get_Wprod_hist(W_hist_gd, dims)
        U_gd, S_gd, V_gd = compute_svd_over_time(Wprod_gd, R_keep=cfg.get("R_keep"))

    t_eg = U_eg = S_eg = V_eg = None
    if cfg["run_eg"]:
        t_eg, W_hist_eg = simulate_EG_two_layer(
            W21_0, W32_0, Sigma_x, Sigma_yx,
            eta=cfg["eta_eg"], n_steps=cfg["steps"], record_every=cfg["record_every"]
        )
        Wprod_eg = get_Wprod_hist(W_hist_eg, dims)
        U_eg, S_eg, V_eg = compute_svd_over_time(Wprod_eg, R_keep=cfg.get("R_keep"))

    cos_u_gd = cos_v_gd = cos_u_eg = cos_v_eg = None

    if cfg["run_gd"]:
        MsU_gd = overlap_matrices_over_time(U_gd, U_tchr, use_abs=True)
        MsV_gd = overlap_matrices_over_time(V_gd, V_tchr, use_abs=True)
        permU_gd = final_assignment(MsU_gd, MsV_gd, weight_uv=0.5)
        U_gd_p = apply_perm(U_gd, permU_gd)
        V_gd_p = apply_perm(V_gd, permU_gd) if cfg.get("shared_perm_uv", True) else apply_perm(V_gd, final_assignment(MsV_gd))
        U_gd_a = align_sign_to_teacher(U_gd_p, U_tchr)
        V_gd_a = align_sign_to_teacher(V_gd_p, V_tchr)
        cos_u_gd, cos_v_gd = cosine_alignment_series(U_gd_a, V_gd_a, U_tchr, V_tchr)

    if cfg["run_eg"]:
        MsU_eg = overlap_matrices_over_time(U_eg, U_tchr, use_abs=True)
        MsV_eg = overlap_matrices_over_time(V_eg, V_tchr, use_abs=True)
        permU_eg = final_assignment(MsU_eg, MsV_eg, weight_uv=0.5)
        U_eg_p = apply_perm(U_eg, permU_eg)
        V_eg_p = apply_perm(V_eg, permU_eg) if cfg.get("shared_perm_uv", True) else apply_perm(V_eg, final_assignment(MsV_eg))
        U_eg_a = align_sign_to_teacher(U_eg_p, U_tchr)
        V_eg_a = align_sign_to_teacher(V_eg_p, V_tchr)
        cos_u_eg, cos_v_eg = cosine_alignment_series(U_eg_a, V_eg_a, U_tchr, V_tchr)

    return {
        "t_gd": t_gd, "t_eg": t_eg,
        "cos_u_gd": cos_u_gd, "cos_v_gd": cos_v_gd,
        "cos_u_eg": cos_u_eg, "cos_v_eg": cos_v_eg,
        "U_tchr": U_tchr, "V_tchr": V_tchr,
        "whiten_error": whiten_error,
    }

def stack_and_stats(arr_list):
    A = np.stack(arr_list, axis=0) 
    return A.mean(axis=0), A.std(axis=0)

def parse_args():
    ap = argparse.ArgumentParser(description="Basis alignment (GD vs EG) with YAML and outdir.")
    ap.add_argument("--config", type=str, default=None, help="YAML config with defaults (CLI overrides)")
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--record-every", type=int, default=None)
    ap.add_argument("--eta-gd", type=float, default=None)
    ap.add_argument("--eta-eg", type=float, default=None)
    ap.add_argument("--design-teacher", action="store_true")
    ap.add_argument("--spectrum", type=float, nargs="*", default=None)
    ap.add_argument("--N1", type=int, default=None)
    ap.add_argument("--N2", type=int, default=None)
    ap.add_argument("--N3", type=int, default=None)
    ap.add_argument("--P", type=int, default=None)
    ap.add_argument("--noise-std", type=float, default=None)
    ap.add_argument("--out", type=str, choices=["pdf","svg","png"], default=None)
    ap.add_argument("--outfile", type=str, default=None)
    ap.add_argument("--outdir", type=str, default=None)

    ap.add_argument("--run-gd", action="store_true")
    ap.add_argument("--no-run-gd", dest="run_gd", action="store_false")
    ap.set_defaults(run_gd=True)
    ap.add_argument("--run-eg", action="store_true")
    ap.add_argument("--no-run-eg", dest="run_eg", action="store_false")
    ap.set_defaults(run_eg=True)
    ap.add_argument("--shared-perm-uv", action="store_true")
    ap.add_argument("--no-shared-perm-uv", dest="shared_perm_uv", action="store_false")
    ap.set_defaults(shared_perm_uv=True)

    args = ap.parse_args()

    ycfg = {}
    if args.config:
        with open(args.config, "r") as f:
            ycfg = yaml.safe_load(f) or {}
        if not isinstance(ycfg, dict):
            raise ValueError("YAML config must be a mapping/dict.")

    cfg = dict(ycfg)
    for k, v in vars(args).items():
        if k == "config":
            continue
        if v is not None:
            cfg[k] = v

    defaults = {
        "seeds": 20,
        "steps": 4000,
        "record_every": 20,
        "eta_gd": 1e-2,
        "eta_eg": 1e-1,
        "design_teacher": False,
        "spectrum": None,
        "N1": 8, "N2": 16, "N3": 8, "P": 800,
        "noise_std": 0.02,
        "out": "pdf",
        "outfile": "alignment.pdf",
        "outdir": "results",
        "R_keep": None,
        "init_name": "gauss-small",
        "init_scale": 1e-2,
        "run_gd": True,
        "run_eg": True,
        "shared_perm_uv": True,
    }
    for k, dv in defaults.items():
        cfg.setdefault(k, dv)

    if isinstance(cfg.get("spectrum", None), list) and len(cfg["spectrum"]) == 0:
        cfg["spectrum"] = None

    outdir = Path(cfg["outdir"])
    outdir.mkdir(parents=True, exist_ok=True)
    cfg["outdir"] = str(outdir)

    return cfg


def main():
    cfg = parse_args()
    set_rc()

    gd_U_runs, gd_V_runs = [], []
    eg_U_runs, eg_V_runs = [], []
    t_gd_ref = None
    t_eg_ref = None
    whiten_errors = []

    for si in range(cfg["seeds"]):
        seed = si
        res = single_seed_run(
            cfg, seed,
            whiten_data, compute_teacher_cov,
            simulate_GD_two_layer, simulate_EG_two_layer,
            build_init
        )
        whiten_errors.append(res["whiten_error"])

        if cfg["run_gd"]:
            if t_gd_ref is None:
                t_gd_ref = res["t_gd"]
            gd_U_runs.append(res["cos_u_gd"])
            gd_V_runs.append(res["cos_v_gd"])
        if cfg["run_eg"]:
            if t_eg_ref is None:
                t_eg_ref = res["t_eg"]
            eg_U_runs.append(res["cos_u_eg"])
            eg_V_runs.append(res["cos_v_eg"])

    cos_u_gd_mu = cos_u_gd_sd = cos_u_eg_mu = cos_u_eg_sd = None
    cos_v_gd_mu = cos_v_gd_sd = cos_v_eg_mu = cos_v_eg_sd = None

    if cfg["run_gd"] and len(gd_U_runs) > 0:
        cos_u_gd_mu, cos_u_gd_sd = stack_and_stats(gd_U_runs)
        cos_v_gd_mu, cos_v_gd_sd = stack_and_stats(gd_V_runs)

    if cfg["run_eg"] and len(eg_U_runs) > 0:
        cos_u_eg_mu, cos_u_eg_sd = stack_and_stats(eg_U_runs)
        cos_v_eg_mu, cos_v_eg_sd = stack_and_stats(eg_V_runs)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharex=False)
    plot_alignment_with_bands(
        axes[0],
        t_gd_ref, cos_u_gd_mu, cos_u_gd_sd,
        t_eg_ref, cos_u_eg_mu, cos_u_eg_sd,
        max_modes=3,
        ylabel="Abs. cosine to teacher U",
    )
    axes[0].set_title("Left singular vectors (U)")

    plot_alignment_with_bands(
        axes[1],
        t_gd_ref, cos_v_gd_mu, cos_v_gd_sd,
        t_eg_ref, cos_v_eg_mu, cos_v_eg_sd,
        max_modes=3,
        ylabel="Abs. cosine to teacher V",
    )
    axes[1].set_title("Right singular vectors (V)")

    handles = [
        plt.Line2D([0],[0], color="k", ls="-", lw=2.2, label="GD"),
        plt.Line2D([0],[0], color="k", ls="--", lw=2.2, label="EG"),
    ]
    axes[1].legend(handles=handles, loc="lower right", frameon=False)

    #fig.suptitle("Alignment to teacher singular vectors (mean ± s.d. over seeds)", y=1.02)
    fig.tight_layout()

    outpath = Path(cfg["outdir"]) / cfg["outfile"]
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, bbox_inches="tight")

    print(f"Saved figure to: {outpath}")
if __name__ == "__main__":
    main()
