import argparse
import yaml
import json
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import svd
import itertools
import os

from src.optim.eg_sgd_two_layer import simulate_EG_two_layer, simulate_GD_two_layer
from src.inits.init_two_layer import build_init
from src.utils.vis_style import EG_COLOR, GD_COLOR, TARGET_COLOR, _set_pub_style
from src.data.data_processing import whiten_data, compute_teacher_cov, get_Wprod_hist

def compute_svd_over_time(Wprod_hist):
    T, N3, N1 = Wprod_hist.shape
    K = min(N3, N1)
    S = np.zeros((T, K))
    for t in range(T):
        S[t] = svd(Wprod_hist[t], full_matrices=False, compute_uv=False)
    return S

def _welford_init(T, K):
    return np.zeros((T, K)), np.zeros((T, K)), 0

def _welford_update(mean, M2, n, X):
    n1 = n + 1
    delta = X - mean
    mean  = mean + delta / n1
    M2    = M2 + delta * (X - mean)
    return mean, M2, n1

def _welford_finalize(mean, M2, n):
    std = np.sqrt(M2 / max(n - 1, 1))
    return mean, std, n

def seed_average_modes_two_layer(
    W_init_fn, Sigma_x, Sigma_yx,
    eta_eg, eta_gd, n_steps, record_every,
    num_runs, num_modes=None
):
    W21_0, W32_0 = W_init_fn(0)
    t_ref, W_hist_eg0 = simulate_EG_two_layer(W21_0, W32_0, Sigma_x, Sigma_yx,
                                              eta=eta_eg, n_steps=n_steps, record_every=record_every)
    _,    W_hist_gd0 = simulate_GD_two_layer(W21_0, W32_0, Sigma_x, Sigma_yx,
                                             eta=eta_gd, n_steps=n_steps, record_every=record_every)

    dims = (W21_0.shape[1], W21_0.shape[0], W32_0.shape[0])
    S_eg0 = compute_svd_over_time(get_Wprod_hist(W_hist_eg0, dims))
    S_gd0 = compute_svd_over_time(get_Wprod_hist(W_hist_gd0, dims))
    T, K = S_eg0.shape
    if S_gd0.shape != (T, K):
        raise RuntimeError("EG and GD shapes/time grids must match")

    if num_modes is None:
        num_modes = K
    modes = np.arange(min(num_modes, K))

    mean_eg, M2_eg, n_eg = _welford_init(T, K)
    mean_gd, M2_gd, n_gd = _welford_init(T, K)

    mean_eg, M2_eg, n_eg = _welford_update(mean_eg, M2_eg, n_eg, S_eg0)
    mean_gd, M2_gd, n_gd = _welford_update(mean_gd, M2_gd, n_gd, S_gd0)

    for r in range(1, num_runs):
        W21_r, W32_r = W_init_fn(r)
        t_eg_r, W_hist_eg_r = simulate_EG_two_layer(W21_r, W32_r, Sigma_x, Sigma_yx,
                                                    eta=eta_eg, n_steps=n_steps, record_every=record_every)
        t_gd_r, W_hist_gd_r = simulate_GD_two_layer(W21_r, W32_r, Sigma_x, Sigma_yx,
                                                    eta=eta_gd, n_steps=n_steps, record_every=record_every)
        if not (np.array_equal(t_eg_r, t_ref) and np.array_equal(t_gd_r, t_ref)):
            raise RuntimeError("Time grids differ across runs; check steps/record_every.")

        S_eg_r = compute_svd_over_time(get_Wprod_hist(W_hist_eg_r, dims))
        S_gd_r = compute_svd_over_time(get_Wprod_hist(W_hist_gd_r, dims))

        mean_eg, M2_eg, n_eg = _welford_update(mean_eg, M2_eg, n_eg, S_eg_r)
        mean_gd, M2_gd, n_gd = _welford_update(mean_gd, M2_gd, n_gd, S_gd_r)

    mean_eg, std_eg, _ = _welford_finalize(mean_eg, M2_eg, n_eg)
    mean_gd, std_gd, _ = _welford_finalize(mean_gd, M2_gd, n_gd)

    return t_ref, mean_eg[:, modes], std_eg[:, modes], mean_gd[:, modes], std_gd[:, modes], modes

def plot_two_layer_mean_sd(
    t, mean_eg, std_eg, mean_gd, std_gd, true_s, modes,
    plot_curves="both", show=True, out_path=None
):
    assert plot_curves in {"eg", "gd", "both"}
    do_eg = plot_curves in {"eg", "both"} and mean_eg is not None
    do_gd = plot_curves in {"gd", "both"} and mean_gd is not None

    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    _set_pub_style(ax)

    lw_eg, lw_gd = 1.5, 1.5
    if do_eg:
        for j, k in enumerate(modes):
            ax.plot(t, mean_eg[:, j], color=EG_COLOR, linewidth=lw_eg)
            ax.fill_between(t, mean_eg[:, j]-std_eg[:, j], mean_eg[:, j]+std_eg[:, j],
                            color=EG_COLOR, alpha=0.15, linewidth=0)
    if do_gd:
        for j, k in enumerate(modes):
            ax.plot(t, mean_gd[:, j], color=GD_COLOR, linestyle="--", linewidth=lw_gd)
            ax.fill_between(t, mean_gd[:, j]-std_gd[:, j], mean_gd[:, j]+std_gd[:, j],
                            color=GD_COLOR, alpha=0.15, linewidth=0)

    for k in modes:
        if k < len(true_s):
            ax.axhline(y=true_s[k], linestyle=":", color=TARGET_COLOR, alpha=0.5, linewidth=1.2)

    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("Mode Strength", fontsize=12)

    from matplotlib.lines import Line2D
    legend = []
    if do_eg:
        legend.append(Line2D([0],[0], color=EG_COLOR, lw=lw_eg, label="EG (mean ± 1 s.d.)"))
    if do_gd:
        legend.append(Line2D([0],[0], color=GD_COLOR, lw=lw_gd, ls="--", label="GD (mean ± 1 s.d.)"))
    ax.legend(handles=legend, frameon=False, loc="lower right")

    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

def overlay_plot_two_layer(
    W21_0, W32_0, Sigma_x, Sigma_yx, true_s,
    eta_eg=1e-3, eta_gd=1e-3,
    n_steps=2000, record_every=10,
    num_modes=None, seed=None,
    title=None, out_path=None, show=True,
    plot_curves="both",
    annotate_subset=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
):
    assert plot_curves in {"eg", "gd", "both"}, "plot_curves must be 'eg', 'gd', or 'both'"
    do_eg = plot_curves in {"eg", "both"}
    do_gd = plot_curves in {"gd", "both"}

    K = min(W32_0.shape[0], W21_0.shape[1])
    if num_modes is not None and num_modes < K:
        rng = np.random.RandomState(seed)
        modes = np.sort(rng.choice(np.arange(K), size=num_modes, replace=False))
    else:
        modes = np.arange(K)

    t_eg = S_hist_eg = None
    if do_eg:
        t_eg, W_hist_eg = simulate_EG_two_layer(
            W21_0, W32_0, Sigma_x, Sigma_yx, eta=eta_eg,
            n_steps=n_steps, record_every=record_every
        )
        Wprod_eg = get_Wprod_hist(W_hist_eg, (W21_0.shape[1], W21_0.shape[0], W32_0.shape[0]))
        S_hist_eg = compute_svd_over_time(Wprod_eg)

    t_gd = S_hist_gd = None
    if do_gd:
        t_gd, W_hist_gd = simulate_GD_two_layer(
            W21_0, W32_0, Sigma_x, Sigma_yx, eta=eta_gd,
            n_steps=n_steps, record_every=record_every
        )
        Wprod_gd = get_Wprod_hist(W_hist_gd, (W21_0.shape[1], W21_0.shape[0], W32_0.shape[0]))
        S_hist_gd = compute_svd_over_time(Wprod_gd)

    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    _set_pub_style(ax)

    lw_eg, lw_gd = 1.5, 1.5

    for k in modes:
        if do_eg:
            ax.plot(t_eg, S_hist_eg[:, k], color=EG_COLOR, linewidth=lw_eg)
        if do_gd:
            ax.plot(t_gd, S_hist_gd[:, k], color=GD_COLOR, linestyle="--", linewidth=lw_gd)

    for k in modes:
        ax.axhline(y=true_s[k], linestyle=":", color=TARGET_COLOR, alpha=0.5, linewidth=1.2)

    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("Mode Strength", fontsize=12)

    from matplotlib.lines import Line2D
    legend_elems = []
    if do_eg:
        legend_elems.append(Line2D([0], [0], color=EG_COLOR, lw=lw_eg, label="EG (modes 1–K)"))
    if do_gd:
        legend_elems.append(Line2D([0], [0], color=GD_COLOR, lw=lw_gd, ls="--", label="GD (modes 1–K)"))
    ax.legend(handles=legend_elems, frameon=False, loc="lower right")

    if annotate_subset:
        if do_eg:
            for m in annotate_subset:
                k = m - 1
                if k in modes:
                    y = S_hist_eg[-1, k]
                    ax.text(t_eg[-1], y, f"  {m}", va="center", ha="left", fontsize=10, color=EG_COLOR)
        elif do_gd:
            for m in annotate_subset:
                k = m - 1
                if k in modes:
                    y = S_hist_gd[-1, k]
                    ax.text(t_gd[-1], y, f"  {m}", va="center", ha="left", fontsize=10, color=GD_COLOR)

    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser(description="EG vs GD for two-layer linear network (whitened inputs).")
    ap.add_argument("--P", type=int, default=800, help="Number of samples")
    ap.add_argument("--N1", type=int, default=10, help="Input dimension")
    ap.add_argument("--N2", type=int, default=12, help="Hidden width")
    ap.add_argument("--N3", type=int, default=10, help="Output dimension")
    ap.add_argument("--noise_std", type=float, default=0.01, help="Noise std for teacher outputs")
    ap.add_argument("--steps", type=int, default=4000, help="Number of optimization steps")
    ap.add_argument("--record_every", type=int, default=10, help="Record every t steps")
    ap.add_argument("--eta_eg", type=float, default=0.05, help="EG step size")
    ap.add_argument("--eta_gd", type=float, default=0.05, help="GD step size")
    ap.add_argument("--init_name", type=str, default="gauss",choices=["gauss","boost","imbalanced","offdiag"],help="Initialization scheme")
    ap.add_argument("--seed", type=int, default=0, help="Random seed")
    ap.add_argument("--init_kwargs", type=str, default="{}",help="JSON for init kwargs (if given on CLI); YAML can pass a dict")
    ap.add_argument("--num_modes", type=int, default=10, help="How many modes to display")
    ap.add_argument("--out", type=str, default="eg_vs_gd_two_layer.png", help="Output figure path")
    ap.add_argument("--no_show", action="store_true", help="Do not call plt.show()")
    ap.add_argument("--config", type=str, default=None, help="Path to a YAML config file")
    ap.add_argument("--plot_curves", type=str, default="both",choices=["eg", "gd", "both"], help="Which curves to plot.")
    ap.add_argument("--break_order", action="store_true",help="If true, run the contained break-order block (tracked modes, crafted teacher/init).")
    ap.add_argument("--outdir", type=str, default="outputs", help="Directory to save outputs")
    ap.add_argument("--mean", action="store_true",
                    help="If set, plot mean ± 1 s.d. over seeds for EG and GD.")
    ap.add_argument("--num_runs", type=int, default=50,
                    help="Number of random initialisations when --mean is set.")
    
    pre_args, _ = ap.parse_known_args()
    if pre_args.config is not None:
        with open(pre_args.config, "r") as f:
            cfg = yaml.safe_load(f) or {}
        valid_dests = {a.dest for a in ap._actions if a.dest != "help"}
        unknown = set(cfg.keys()) - valid_dests
        if unknown:
            raise ValueError(f"Unknown config key(s) in {pre_args.config}: {sorted(unknown)}")
        ap.set_defaults(**cfg)

    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    if not os.path.isabs(args.out):
        args.out = os.path.join(args.outdir, args.out)

    if isinstance(args.init_kwargs, dict):
        init_kwargs = args.init_kwargs
    else:
        try:
            init_kwargs = json.loads(args.init_kwargs)
        except Exception:
            init_kwargs = {}

    rng = np.random.RandomState(args.seed)
    X = rng.randn(args.P, args.N1)
    Xw = whiten_data(X)

    if args.break_order:
        run_break_order(args, Xw, rng)
        return

    M_true = rng.randn(args.N3, args.N1)
    Y = Xw @ M_true.T + args.noise_std * rng.randn(args.P, args.N3)
    Sigma_x, Sigma_yx = compute_teacher_cov(Xw, Y)
    W21_0, W32_0 = build_init(args.init_name, Sigma_yx, args.N1, args.N2, args.N3,
                              seed=args.seed, **init_kwargs)
    true_s = svd(M_true, compute_uv=False)

    overlay_plot_two_layer(
        W21_0, W32_0, Sigma_x, Sigma_yx, true_s,
        eta_eg=args.eta_eg, eta_gd=args.eta_gd,
        n_steps=args.steps, record_every=args.record_every,
        num_modes=min(args.num_modes, min(args.N1, args.N3)),
        seed=args.seed, out_path=args.out, show=not args.no_show,
        title=f"Two-layer linear: EG vs GD (init={args.init_name})",
        plot_curves=args.plot_curves,
    )

    if args.mean:
        from numpy.random import SeedSequence
        from pathlib import Path

        ss = SeedSequence(args.seed)
        child_ss = ss.spawn(args.num_runs)
        run_seeds = [int(cs.generate_state(1)[0]) for cs in child_ss]

        def W_init_fn(run_idx: int):
            rseed = run_seeds[int(run_idx) % len(run_seeds)]
            W21_i, W32_i = build_init(args.init_name, Sigma_yx, args.N1, args.N2, args.N3,
                                      seed=rseed, **init_kwargs)
            return W21_i, W32_i

        t_ref, mean_eg, std_eg, mean_gd, std_gd, modes = seed_average_modes_two_layer(
            W_init_fn, Sigma_x, Sigma_yx,
            eta_eg=args.eta_eg, eta_gd=args.eta_gd,
            n_steps=args.steps, record_every=args.record_every,
            num_runs=args.num_runs, num_modes=min(args.num_modes, min(args.N1, args.N3))
        )

        p = Path(args.out)
        out_mean = str(p.with_name(p.stem + "_mean" + p.suffix))

        plot_two_layer_mean_sd(
            t_ref, mean_eg, std_eg, mean_gd, std_gd, true_s, modes,
            plot_curves=args.plot_curves, show=not args.no_show, out_path=out_mean
        )

if __name__ == "__main__":
    main()
