import argparse, os
from pathlib import Path
import yaml
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import svd
import matplotlib.animation as animation

from src.utils.vis_style import EG_COLOR, GD_COLOR, TARGET_COLOR, _set_pub_style
from src.data.data_processing import whiten_data, compute_teacher_cov
from src.optim.eg_sgd_shallow import simulate_EG_shallow, simulate_GD_shallow
from src.inits.init_shallow import build_init as build_init_shallow

def finite_diff_series(W_series, delta_t=1.0):
    W_series = np.asarray(W_series)
    T = W_series.shape[0]
    Wdot = np.empty_like(W_series)
    if T == 1:
        Wdot[0] = 0.0
        return Wdot
    Wdot[0]  = (W_series[1] - W_series[0]) / delta_t
    Wdot[-1] = (W_series[-1] - W_series[-2]) / delta_t
    if T > 2:
        Wdot[1:-1] = (W_series[2:] - W_series[:-2]) / (2.0 * delta_t)
    return Wdot

def omega_from_W_and_Wdot(W, Wdot, tiny=1e-10, degeneracy_safe=True):
    U, s, Vt = svd(W, full_matrices=False)
    V = Vt.T
    K = len(s)
    if np.allclose(Wdot, 0):
        return U, s, V, np.zeros((K, K))
    Gamma = U.T @ Wdot @ V
    Omega = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            if i == j:
                continue
            num = s[i] * Gamma[j, i] + s[j] * Gamma[i, j]
            denom = s[j]**2 - s[i]**2
            if degeneracy_safe and abs(denom) < tiny:
                Omega[i, j] = Gamma[i, j] / max(s[i], s[j], tiny)
            else:
                Omega[i, j] = num / denom
    return U, s, V, Omega

def svd_and_omega_history(W_hist_unflat, degeneracy_safe=True):
    W_series = np.asarray(W_hist_unflat)
    T, N3, N1 = W_series.shape
    Wdot = finite_diff_series(W_series)
    K = min(N3, N1)
    S = np.zeros((T, K))
    Omega_hist = np.zeros((T, K, K))
    U_series = np.zeros((T, N3, K))
    V_series = np.zeros((T, N1, K))
    for t in range(T):
        U, s, V, Omega = omega_from_W_and_Wdot(W_series[t], Wdot[t], degeneracy_safe=degeneracy_safe)
        Kt = min(K, len(s))
        S[t, :Kt] = s[:Kt]
        U_series[t, :, :Kt] = U[:, :Kt]
        V_series[t, :, :Kt] = V[:, :Kt]
        Omega_hist[t, :Kt, :Kt] = Omega[:Kt, :Kt]
    return {"S": S, "Omega": Omega_hist, "U": U_series, "V": V_series}


def set_rc():
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.size": 12, "axes.labelsize": 12, "axes.titlesize": 13,
        "legend.fontsize": 11, "xtick.labelsize": 11, "ytick.labelsize": 11,
        "figure.dpi": 120, "savefig.dpi": 300, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": False
    })

def plot_singulars(ax, t, S, s_targets=None):
    from src.utils.vis_style import EG_COLOR, GD_COLOR, TARGET_COLOR, _set_pub_style
    _set_pub_style(ax)
    K = S.shape[1]
    for k in range(K):
        ax.plot(t, S[:, k], lw=1.8, color=EG_COLOR)
    if s_targets is not None:
        for k in range(min(K, len(s_targets))):
            ax.axhline(s_targets[k], ls=":", color=TARGET_COLOR, alpha=0.6, lw=1.2)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Singular values")
    ax.set_title("Singular-value trajectories")

def plot_coupling_and_heatmap(ax_curves, ax_heat, t, Omega_hist, pairs=None, aggregate=False):
    from src.utils.vis_style import EG_COLOR, GD_COLOR, TARGET_COLOR, _set_pub_style
    _set_pub_style(ax_curves); _set_pub_style(ax_heat)
    T, K, _ = Omega_hist.shape
    if aggregate:
        off = Omega_hist.copy()
        off[:, range(K), range(K)] = 0.0
        fro = np.linalg.norm(off.reshape(T, -1), axis=1)
        ax_curves.plot(t, fro, lw=2.0, color=EG_COLOR)
        ax_curves.set_ylabel(r"$\|\Omega_{\mathrm{off}}\|_F$")
        ax_curves.set_title("Coupling (aggregate)")
    else:
        pairs = pairs or [(0,1)]
        for (i,j) in pairs:
            if i < K and j < K and i != j:
                ax_curves.plot(t, np.abs(Omega_hist[:, i, j]), lw=1.8, label=f"|Ω$_{{{i+1},{j+1}}}$|")
        ax_curves.legend(frameon=False, loc="upper right", ncol=2)
        ax_curves.set_ylabel(r"|Ω$_{ij}$|")
        ax_curves.set_title("Coupling (pairwise)")
    ax_curves.set_xlabel("Training step")

    im = ax_heat.imshow(Omega_hist[-1], cmap="coolwarm", vmin=-1.0, vmax=1.0, origin="lower")
    ax_heat.set_title("Final Ω (U-space)")
    ax_heat.set_xlabel("j"); ax_heat.set_ylabel("i")
    return im

def parse_args():
    ap = argparse.ArgumentParser(description="Mode coupling (one-layer).")
    ap.add_argument("--config", "--configs", dest="config", type=str, default=None,
                    help="Path to YAML config; CLI overrides YAML values.")
    ap.add_argument("--algo", type=str, choices=["eg","gd"], default=None)
    ap.add_argument("--P", type=int, default=None)
    ap.add_argument("--N1", type=int, default=None)
    ap.add_argument("--N3", type=int, default=None)
    ap.add_argument("--noise_std", type=float, default=None)
    ap.add_argument("--eta", type=float, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--record_every", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--pairs", type=str, default=None, help='e.g. "1,2 1,3" (1-indexed)')
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--degeneracy_safe", action="store_true")
    ap.add_argument("--no_degeneracy_safe", dest="degeneracy_safe", action="store_false")
    ap.set_defaults(degeneracy_safe=True)

    ap.add_argument("--outdir", type=str, default=None)
    ap.add_argument("--outfile", type=str, default=None)
    ap.add_argument("--no_show", action="store_true")
    ap.add_argument("--save_npz", type=str, default=None)
    ap.add_argument("--animate", action="store_true",
                    help="Save a GIF of the evolving coupling.")
    ap.add_argument("--gif", type=str, default=None,
                    help="Filename (relative to outdir) for the GIF. Defaults to <outfile stem>_omega.gif")
    ap.add_argument("--gif-mode", type=str, choices=["matrix","offdiag"], default="matrix",
                    help="Animate KxK matrix or off-diagonal time–index strip.")
    ap.add_argument("--gif-fps", type=int, default=8, help="Animation FPS.")

    pre, _ = ap.parse_known_args()
    ycfg = {}
    if pre.config:
        with open(pre.config, "r") as f:
            ycfg = yaml.safe_load(f) or {}
        valid = {a.dest for a in ap._actions if a.dest != "help"}
        unknown = set(ycfg.keys()) - valid
        if unknown:
            raise ValueError(f"Unknown YAML key(s): {sorted(unknown)}")
        ap.set_defaults(**ycfg)

    args = ap.parse_args()
    cfg = vars(args).copy()
            
    defaults = dict(
        algo="eg", P=600, N1=8, N3=8, noise_std=0.01,
        eta=0.1, steps=1200, record_every=20, seed=0,
        pairs=None, aggregate=False, degeneracy_safe=True,
        outdir="results/mode_coupling_one", outfile="one_layer_mode_coupling.pdf",
        save_npz=None,
    )
    for k,v in defaults.items():
        if cfg.get(k) is None:
            cfg[k] = v

    if isinstance(cfg["pairs"], str) and cfg["pairs"].strip():
        pairs = []
        for tok in cfg["pairs"].split():
            i,j = tok.split(",")
            pairs.append((int(i)-1, int(j)-1))
        cfg["pairs"] = pairs

    outdir = Path(cfg["outdir"]); outdir.mkdir(parents=True, exist_ok=True)
    if not os.path.isabs(cfg["outfile"]):
        cfg["outfile"] = str(outdir / cfg["outfile"])
    if cfg["save_npz"] and not os.path.isabs(cfg["save_npz"]):
        cfg["save_npz"] = str(outdir / cfg["save_npz"])

    if cfg.get("animate"):
        if not cfg.get("gif"):
            cfg["gif"] = str(Path(cfg["outfile"]).with_suffix("").parent /
                            (Path(cfg["outfile"]).with_suffix("").name + "_omega.gif"))
        elif not os.path.isabs(cfg["gif"]):
            cfg["gif"] = str(Path(cfg["outdir"]) / cfg["gif"])

    return cfg

def animate_mode_coupling(
    Omega_hist,
    t,
    outfile,
    mode="matrix",
    fps=8,
    cmap="magma_r",
    abs_values=True,
    vmin=None,
    vmax=None,
):
    Path(outfile).parent.mkdir(parents=True, exist_ok=True)
    M = np.abs(Omega_hist) if abs_values else Omega_hist
    T, K, _ = M.shape

    if vmin is None: vmin = float(np.nanmin(M))
    if vmax is None: vmax = float(np.nanmax(M))
    if vmin == vmax: vmax = vmin + 1e-9

    if mode == "matrix":
        fig, ax = plt.subplots(figsize=(4.8, 4.2))
        im = ax.imshow(M[0], vmin=vmin, vmax=vmax, cmap=cmap, origin="lower")
        cbar = fig.colorbar(im, ax=ax)
        ax.set_xlabel("j"); ax.set_ylabel("i")
        title_txt = ax.text(0.02, 0.98, f"t={t[0]}", transform=ax.transAxes,
                            ha="left", va="top")

        def update(k):
            im.set_data(M[k])
            title_txt.set_text(f"t={t[k]}")
            return (im, title_txt)

    elif mode == "offdiag":
        mask = ~np.eye(K, dtype=bool)
        W = K * (K - 1)
        H0 = np.zeros((1, W), dtype=float)
        fig, ax = plt.subplots(figsize=(6.0, 4.2))
        im = ax.imshow(H0, vmin=vmin, vmax=vmax, cmap=cmap, aspect="auto",
                       origin="lower", extent=[0.5, W + 0.5, t[0], t[0]])
        cbar = fig.colorbar(im, ax=ax, label=r"|Ω$_{ij}$|")
        ax.set_xlabel("Off-diagonal index (flattened)")
        ax.set_ylabel("Time step")
        title_txt = ax.text(0.02, 0.98, f"t={t[0]}", transform=ax.transAxes,
                            ha="left", va="top")

        col_index = []
        for i in range(K):
            for j in range(K):
                if i != j:
                    col_index.append((i, j))
        col_index = np.array(col_index)

        def frame_row(k):
            row = np.empty(W, float)
            row[:] = M[k][mask]
            return row

        def update(k):
            im.set_data(frame_row(k)[None, :])
            im.set_extent([0.5, W + 0.5, t[k], t[k]]) 
            title_txt.set_text(f"t={t[k]}")
            return (im, title_txt)
    else:
        raise ValueError("mode must be 'matrix' or 'offdiag'")

    ani = animation.FuncAnimation(fig, update, frames=T, interval=int(1000 / fps), blit=False)
    try:
        ani.save(outfile, writer="pillow", fps=fps)
    finally:
        plt.close(fig)


def main():
    cfg = parse_args()
    set_rc()

    rng = np.random.RandomState(cfg["seed"])
    X = rng.randn(cfg["P"], cfg["N1"])
    Xw = whiten_data(X)
    M_true = rng.randn(cfg["N3"], cfg["N1"])
    Y = Xw @ M_true.T + cfg["noise_std"] * rng.randn(cfg["P"], cfg["N3"])
    Sigma_x, Sigma_yx = compute_teacher_cov(Xw, Y)
    s_targets = svd(Sigma_yx, compute_uv=False)

    W0 = build_init_shallow(1, Sigma_yx, rng, cfg["N3"], cfg["N1"])

    if cfg["algo"] == "eg":
        t, W_hist = simulate_EG_shallow(W0, Sigma_x, Sigma_yx, eta=cfg["eta"],
                                        n_steps=cfg["steps"], record_every=cfg["record_every"])
    else:
        t, W_hist = simulate_GD_shallow(W0, Sigma_x, Sigma_yx, eta=cfg["eta"],
                                        n_steps=cfg["steps"], record_every=cfg["record_every"])

    T = len(t); N3, N1 = cfg["N3"], cfg["N1"]
    W_series = np.zeros((T, N3, N1))
    for i in range(T):
        W_series[i] = W_hist[:, i].reshape(N3, N1)

    hist = svd_and_omega_history(W_series, degeneracy_safe=cfg["degeneracy_safe"])
    S, Omega = hist["S"], hist["Omega"]

    fig_pw, (axc_pw, axh_pw) = plt.subplots(1, 2, figsize=(12.5, 5.0))
    _ = plot_coupling_and_heatmap(axc_pw, axh_pw, t, Omega, pairs=cfg.get("pairs"), aggregate=False)
    fig_pw.tight_layout()
    out_pw = str(Path(cfg["outfile"]).with_suffix("")) + "_pairwise.pdf"
    fig_pw.savefig(out_pw, bbox_inches="tight")
    if not cfg.get("no_show"):
        plt.show()
    plt.close(fig_pw)

    fig_fro, (axc_fro, axh_fro) = plt.subplots(1, 2, figsize=(12.5, 5.0))
    _ = plot_coupling_and_heatmap(axc_fro, axh_fro, t, Omega, aggregate=True)
    fig_fro.tight_layout()
    out_fro = str(Path(cfg["outfile"]).with_suffix("")) + "_fro.pdf"
    fig_fro.savefig(out_fro, bbox_inches="tight")
    if not cfg.get("no_show"):
        plt.show()
    plt.close(fig_fro)

    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    plot_singulars(ax, t, S, s_targets)
    fig.tight_layout()
    fig.savefig(cfg["outfile"], bbox_inches="tight")
    if not cfg["no_show"]:
        plt.show()
    plt.close(fig)

    fig2, (axc, axh) = plt.subplots(1, 2, figsize=(12.5, 5.0))
    im = plot_coupling_and_heatmap(axc, axh, t, Omega, pairs=cfg["pairs"], aggregate=cfg["aggregate"])
    fig2.tight_layout()
    stem = Path(cfg["outfile"]).with_suffix("")
    out2 = str(stem) + "_coupling.pdf"
    fig2.savefig(out2, bbox_inches="tight")
    if not cfg["no_show"]:
        plt.show()
    plt.close(fig2)

    if cfg["save_npz"]:
        np.savez(cfg["save_npz"], t=t, S=S, Omega=Omega, s_targets=s_targets)

    if cfg.get("animate"):
        animate_mode_coupling(
            Omega, t,
            outfile=cfg["gif"],
            mode=cfg.get("gif_mode", "matrix"),
            fps=cfg.get("gif_fps", 8),
            cmap="magma_r",
            abs_values=True
        )
        print(f"Saved animation to: {cfg['gif']}")

    print(f"Saved figures to: {cfg['outfile']} and {out2}")

if __name__ == "__main__":
    main()
