import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import svd
from numpy.random import SeedSequence
import os


from src.optim.eg_sgd_shallow import simulate_EG_shallow, simulate_GD_shallow
from src.inits.init_shallow import build_init
from src.utils.vis_style import EG_COLOR, GD_COLOR, TARGET_COLOR, _set_pub_style
from src.data.data_processing import whiten_data, compute_teacher_cov

def compute_svd_over_time(W_hist, dims):
    N3, N1 = dims
    T = W_hist.shape[1]
    K = min(N3, N1)
    s_list = np.zeros((T, K))
    for t in range(T):
        W_t = W_hist[:, t].reshape(N3, N1)
        s_list[t] = svd(W_t, full_matrices=False, compute_uv=False)
    return s_list

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

def seed_average_modes_shallow(
    W_init_fn, Sigma_x, Sigma_yx,
    eta_eg, eta_gd, n_steps, record_every,
    num_runs, num_modes=None
):
    W0 = W_init_fn(0)
    t_ref, W_hist_eg0 = simulate_EG_shallow(W0, Sigma_x, Sigma_yx, eta=eta_eg,
                                            n_steps=n_steps, record_every=record_every)
    _,    W_hist_gd0 = simulate_GD_shallow(W0, Sigma_x, Sigma_yx, eta=eta_gd,
                                           n_steps=n_steps, record_every=record_every)
    S_eg0 = compute_svd_over_time(W_hist_eg0, (W0.shape[0], W0.shape[1]))
    S_gd0 = compute_svd_over_time(W_hist_gd0, (W0.shape[0], W0.shape[1]))
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
        Wr = W_init_fn(r)
        t_eg_r, W_hist_eg_r = simulate_EG_shallow(Wr, Sigma_x, Sigma_yx, eta=eta_eg,
                                                  n_steps=n_steps, record_every=record_every)
        t_gd_r, W_hist_gd_r = simulate_GD_shallow(Wr, Sigma_x, Sigma_yx, eta=eta_gd,
                                                  n_steps=n_steps, record_every=record_every)
        if not (np.array_equal(t_eg_r, t_ref) and np.array_equal(t_gd_r, t_ref)):
            raise RuntimeError("Time grids differ across runs; check steps/record_every.")

        S_eg_r = compute_svd_over_time(W_hist_eg_r, (Wr.shape[0], Wr.shape[1]))
        S_gd_r = compute_svd_over_time(W_hist_gd_r, (Wr.shape[0], Wr.shape[1]))

        mean_eg, M2_eg, n_eg = _welford_update(mean_eg, M2_eg, n_eg, S_eg_r)
        mean_gd, M2_gd, n_gd = _welford_update(mean_gd, M2_gd, n_gd, S_gd_r)

    mean_eg, std_eg, _ = _welford_finalize(mean_eg, M2_eg, n_eg)
    mean_gd, std_gd, _ = _welford_finalize(mean_gd, M2_gd, n_gd)

    return t_ref, mean_eg[:, modes], std_eg[:, modes], mean_gd[:, modes], std_gd[:, modes], modes

def plot_compare_shallow_mean_sd(
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


def plot_compare_shallow(
    W_init, Sigma_x, Sigma_yx, true_s,
    eta_eg=1e-3, eta_gd=1e-2, n_steps=2000, record_every=10,
    num_modes=None, seed=None, title=None, show=True, out_path=None,
    plot_curves="both", annotate_subset=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
):
    assert plot_curves in {"eg", "gd", "both"}, "plot_curves must be 'eg', 'gd', or 'both'"
    do_eg = plot_curves in {"eg", "both"}
    do_gd = plot_curves in {"gd", "both"}

    K = min(W_init.shape[0], W_init.shape[1])
    if num_modes is not None and num_modes < K:
        rng = np.random.RandomState(seed)
        modes = np.sort(rng.choice(np.arange(K), size=num_modes, replace=False))
    else:
        modes = np.arange(K)

    t_eg = S_hist_eg = None
    if do_eg:
        t_eg, W_hist_eg = simulate_EG_shallow(
            W_init, Sigma_x, Sigma_yx, eta=eta_eg, n_steps=n_steps, record_every=record_every
        )
        S_hist_eg = compute_svd_over_time(W_hist_eg, (W_init.shape[0], W_init.shape[1]))

    t_gd = S_hist_gd = None
    if do_gd:
        t_gd, W_hist_gd = simulate_GD_shallow(
            W_init, Sigma_x, Sigma_yx, eta=eta_gd, n_steps=n_steps, record_every=record_every
        )
        S_hist_gd = compute_svd_over_time(W_hist_gd, (W_init.shape[0], W_init.shape[1]))

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
        tmax = None
        if do_eg:
            tmax = t_eg[-1]
            for m in annotate_subset:
                k = m - 1
                if k in modes:
                    y = S_hist_eg[-1, k]
                    ax.text(t_eg[-1], y, f"  {m}", va="center", ha="left", fontsize=10, color=EG_COLOR)
        elif do_gd:
            tmax = t_gd[-1]
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


def plot_mode_mean_eg_network(W_init_fn, Sigma_x, Sigma_yx,eta=1e-3, n_steps=1000, record_every=10,num_runs=5, num_modes=None, seed=None):
    W0 = W_init_fn(0)
    t_list, W_hist0 = simulate_EG_shallow(
        W0, Sigma_x, Sigma_yx, eta=eta, n_steps=n_steps, record_every=record_every
    )
    S0 = compute_svd_over_time(W_hist0, (W0.shape[0], W0.shape[1]))
    T, K = S0.shape

    all_runs = np.zeros((num_runs, T, K))
    all_runs[0] = S0
    for run in range(1, num_runs):
        W0_i = W_init_fn(run)
        _, W_hist_i = simulate_EG_shallow(
            W0_i, Sigma_x, Sigma_yx, eta=eta, n_steps=n_steps, record_every=record_every
        )
        all_runs[run] = compute_svd_over_time(W_hist_i, (W0_i.shape[0], W0_i.shape[1]))

    rng = np.random.RandomState(seed)
    if num_modes is None:
        num_modes = K
    modes = rng.choice(np.arange(K), size=min(num_modes, K), replace=False)

    mean_S = all_runs[:, :, modes].mean(axis=0)  
    std_S  = all_runs[:, :, modes].std(axis=0)

    plt.figure(figsize=(8, 5))
    for idx, k in enumerate(modes):
        plt.plot(t_list, mean_S[:, idx], label=f"mode {k+1}")
        plt.fill_between(
            t_list,
            mean_S[:, idx] - std_S[:, idx],
            mean_S[:, idx] + std_S[:, idx],
            alpha=0.3
        )
    plt.xlabel("EG step")
    plt.ylabel("Singular value of W(t)")
    plt.title(f"EG: mean ± 1 std over {num_runs} runs")
    plt.legend(loc="best", ncol=2)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_weight_entries_over_time(
    W_init, Sigma_x, Sigma_yx,
    eta_eg=1e-3, eta_gd=1e-2,
    n_steps=1000, record_every=1,
    indices=None,
    seed=None,
    show=True, out_path=None
):
    rng = np.random.RandomState(seed)
    N3, N1 = W_init.shape

    W_star = Sigma_yx @ np.linalg.inv(Sigma_x)

    t_eg, W_hist_eg = simulate_EG_shallow(
        W_init, Sigma_x, Sigma_yx, eta=eta_eg, n_steps=n_steps, record_every=record_every
    )
    t_gd, W_hist_gd = simulate_GD_shallow(
        W_init, Sigma_x, Sigma_yx, eta=eta_gd, n_steps=n_steps, record_every=record_every
    )

    if indices is None:
        flat = np.abs(Sigma_yx).ravel()
        top_idx = np.argsort(flat)[::-1]
        chosen = []
        seen_rows, seen_cols = set(), set()
        for idx in top_idx:
            i, j = divmod(idx, N1)
            if i not in seen_rows or j not in seen_cols or len(chosen) < 2:
                chosen.append((i, j))
                seen_rows.add(i)
                seen_cols.add(j)
            if len(chosen) >= 6:
                break
        indices = chosen

    n_plots = 6
    nrows, ncols = 2, 3

    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 6), squeeze=False)

    lw_eg, lw_gd, lw_target = 2.2, 1.8, 1.2

    for ax, (i, j) in zip(axes.ravel(), indices[:n_plots]):
        _set_pub_style(ax)

        y_eg = W_hist_eg[:, :].reshape(N3, N1, -1)[i, j, :]
        y_gd = W_hist_gd[:, :].reshape(N3, N1, -1)[i, j, :]

        ax.plot(t_eg, y_eg, color=EG_COLOR, linewidth=lw_eg)
        ax.plot(t_gd, y_gd, color=GD_COLOR, linestyle="--", linewidth=lw_gd)
        ax.axhline(W_star[i, j], color=TARGET_COLOR, linestyle=":", alpha=0.6, linewidth=lw_target)

        ax.set_title(f"$(i={i},\\ j={j})$", fontsize=12, pad=6)
        ax.set_xlabel("Training step", fontsize=12)
        ax.set_ylabel("$W_{ij}(t)$", fontsize=12)

    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], color=EG_COLOR, lw=lw_eg, label="EG"),
        Line2D([0], [0], color=GD_COLOR, lw=lw_gd, ls="--", label="GD"),
        Line2D([0], [0], color=TARGET_COLOR, lw=lw_target, ls=":", label="$W^*$"),
    ]
    fig.legend(handles=legend_elems, frameon=False, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.02), fontsize=11)

    fig.tight_layout()

    if out_path is not None:
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(
        description="Compare EG vs GD in shallow (one-layer) linear regression with whitened inputs."
    )
    parser.add_argument("--P", type=int, default=500, help="Number of samples")
    parser.add_argument("--N1", type=int, default=10, help="Input dimension")
    parser.add_argument("--N3", type=int, default=10, help="Output dimension")
    parser.add_argument("--noise_std", type=float, default=0.01, help="Noise std for teacher outputs")
    parser.add_argument("--steps", type=int, default=1000, help="Number of optimization steps")
    parser.add_argument("--record_every", type=int, default=10, help="Record every t steps")
    parser.add_argument("--eta_eg", type=float, default=0.1, help="EG step size")
    parser.add_argument("--eta_gd", type=float, default=0.1, help="GD step size")
    parser.add_argument("--init", type=int, default=1, help="Initialization scheme {1..6}")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--num_modes", type=int, default=10, help="How many modes to plot (<= min(N1,N3))")
    parser.add_argument("--out", type=str, default="eg_vs_gd_shallow.png", help="Path to save the plot PNG")
    parser.add_argument("--csv_out", type=str, default=None, help="Optional path to save SVD trajectories (npz)")
    parser.add_argument("--no_show", action="store_true", help="Do not call plt.show()")
    parser.add_argument("--config", type=str, default=None, help="Path to a YAML config file")
    parser.add_argument("--plot_curves",type=str,default="both",choices=["eg", "gd", "both"],help="Which curves to plot: 'eg', 'gd', or 'both'.",)
    parser.add_argument("--break_order", action="store_true",help="If true, use a contained 'diffuse+spike' Σ_yx and tracked SVD to flip EG order.")
    parser.add_argument("--mean", action="store_true",help="If true, run multiple EG initialisations and plot mean ± std of mode trajectories.")
    parser.add_argument("--num_runs", type=int, default=50,help="Number of random initialisations when --mean is set.")
    parser.add_argument("--outdir", type=str, default="outputs", help="Directory to save outputs")

    pre_args, _ = parser.parse_known_args()

    if pre_args.config is not None:
        with open(pre_args.config, "r") as f:
            cfg = yaml.safe_load(f) or {}

        valid_dests = {a.dest for a in parser._actions if a.dest != "help"}
        unknown = set(cfg.keys()) - valid_dests
        if unknown:
            raise ValueError(f"Unknown config key(s) in {pre_args.config}: {sorted(unknown)}")

        parser.set_defaults(**cfg)

    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    if not os.path.isabs(args.out):
        args.out = os.path.join(args.outdir, args.out)

    if args.csv_out is not None and not os.path.isabs(args.csv_out):
        args.csv_out = os.path.join(args.outdir, args.csv_out)

    rng = np.random.RandomState(args.seed)
    X = rng.randn(args.P, args.N1)
    Xw = whiten_data(X)

    M_true = rng.randn(args.N3, args.N1)
    Y = Xw @ M_true.T + args.noise_std * rng.randn(args.P, args.N3)

    Sigma_x, Sigma_yx = compute_teacher_cov(Xw, Y)

    W0 = build_init(args.init, Sigma_yx, rng, args.N3, args.N1)

    true_s = svd(M_true, compute_uv=False)

    plot_compare_shallow(
        W0, Sigma_x, Sigma_yx, true_s,
        eta_eg=args.eta_eg, eta_gd=args.eta_gd,
        n_steps=args.steps, record_every=args.record_every,
        num_modes=min(args.num_modes, min(args.N1, args.N3)),
        seed=args.seed, title=f"Shallow linear network: EG vs GD",
        show=not args.no_show, out_path=args.out,
        plot_curves=args.plot_curves,
    )

    if args.csv_out is not None:
        t_eg, W_hist_eg = simulate_EG_shallow(
            W0, Sigma_x, Sigma_yx, eta=args.eta_eg,
            n_steps=args.steps, record_every=args.record_every
        )
        t_gd, W_hist_gd = simulate_GD_shallow(
            W0, Sigma_x, Sigma_yx, eta=args.eta_gd,
            n_steps=args.steps, record_every=args.record_every
        )
        S_hist_eg = compute_svd_over_time(W_hist_eg, (args.N3, args.N1))
        S_hist_gd = compute_svd_over_time(W_hist_gd, (args.N3, args.N1))
        np.savez(args.csv_out, t_eg=t_eg, t_gd=t_gd,
                 S_hist_eg=S_hist_eg, S_hist_gd=S_hist_gd, true_s=true_s)
    
    if args.mean:
            from numpy.random import SeedSequence
            from pathlib import Path

            ss = SeedSequence(args.seed)
            child_ss = ss.spawn(args.num_runs)
            run_seeds = [int(cs.generate_state(1)[0]) for cs in child_ss]

            def W_init_fn(run_idx: int):
                rng_i = np.random.RandomState(run_seeds[int(run_idx) % len(run_seeds)])
                W0_i = build_init(args.init, Sigma_yx, rng_i, args.N3, args.N1)
                if args.init in (3, 4, 5, 6):
                    W0_i = W0_i + rng_i.randn(args.N3, args.N1) * 1e-6
                return W0_i

            t_ref, mean_eg, std_eg, mean_gd, std_gd, modes = seed_average_modes_shallow(
                W_init_fn, Sigma_x, Sigma_yx,
                eta_eg=args.eta_eg, eta_gd=args.eta_gd,
                n_steps=args.steps, record_every=args.record_every,
                num_runs=args.num_runs, num_modes=min(args.num_modes, min(args.N1, args.N3))
            )

            p = Path(args.out)
            out_mean = str(p.with_name(p.stem + "_mean" + p.suffix))

            plot_compare_shallow_mean_sd(
                t_ref, mean_eg, std_eg, mean_gd, std_gd, true_s, modes,
                plot_curves=args.plot_curves, show=not args.no_show, out_path=out_mean
            )

    plot_weight_entries_over_time(
        W0, Sigma_x, Sigma_yx,
        eta_eg=args.eta_eg, eta_gd=args.eta_gd,
        n_steps=args.steps, record_every=args.record_every,
        seed=args.seed,
        out_path="weights_elementwise.png"
    )




if __name__ == "__main__":
    main()