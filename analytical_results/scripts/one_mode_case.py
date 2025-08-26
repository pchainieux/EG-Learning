import argparse
import os
from pathlib import Path
import yaml
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


from src.utils.vis_style import EG_COLOR, GD_COLOR, TARGET_COLOR, _set_pub_style

def gd_ode(a, t, s, lr):
    return 2.0 * lr * a * (s - a)

def eg_ode(a, t, s, eta):
    return 2.0 * eta * (a ** 1.5) * (s - a)

def gd_step(c, d, s, lr):
    a = c * d
    grad_c = -d * (s - a)
    grad_d = -c * (s - a)
    return c - lr * grad_c, d - lr * grad_d

def eg_step(c, d, s, eta):
    a = c * d
    grad_c = -d * (s - a)
    grad_d = -c * (s - a)
    return c * np.exp(-eta * grad_c), d * np.exp(-eta * grad_d)

def simulate_discrete(step_fn, c0, d0, s, rate, n_steps):
    c, d = float(c0), float(d0)
    t = np.arange(n_steps + 1)
    a_hist = np.empty_like(t, dtype=float)
    a_hist[0] = c * d
    for i in range(1, n_steps + 1):
        c, d = step_fn(c, d, s, rate)
        a_hist[i] = c * d
    return t, a_hist

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
        "text.usetex": False,
    })

def make_figure(cfg, curves):
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    _set_pub_style(ax)

    if curves["t_gd"] is not None:
        ax.plot(curves["t_gd"], curves["a_gd"], linestyle="--", linewidth=1.8,
                color=GD_COLOR, label="GD (discrete)")
    if curves["t_eg"] is not None:
        ax.plot(curves["t_eg"], curves["a_eg"], linestyle="-", linewidth=2.0,
                color=EG_COLOR, label="EG (discrete)")

    if curves["t_ode"] is not None and curves["a_gd_ode"] is not None and cfg["show_continuous"]:
        ax.plot(curves["t_ode"], curves["a_gd_ode"], linewidth=1.2,
                color=GD_COLOR, alpha=0.7, label="GD (continuous)")
    if curves["t_ode"] is not None and curves["a_eg_ode"] is not None and cfg["show_continuous"]:
        ax.plot(curves["t_ode"], curves["a_eg_ode"], linewidth=1.2,
                color=EG_COLOR, alpha=0.7, label="EG (continuous)")

    ax.axhline(y=cfg["s_true"], linestyle=":", color=TARGET_COLOR, alpha=0.8, linewidth=1.5)

    ax.set_xlabel("Number of gradient updates")
    ax.set_ylabel(r"$a(t) \;=\; c(t)\,d(t)$")
    ttl = "Single-mode dynamics  |  GD vs EG"
    if cfg.get("title_suffix"):
        ttl += f"  ({cfg['title_suffix']})"
    ax.set_title(ttl)

    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    return fig

def parse_args():
    ap = argparse.ArgumentParser(description="One-mode GD vs EG with YAML config and outdir.")
    ap.add_argument("--config", "--configs", dest="config", type=str, default=None,
                    help="Path to YAML config; CLI overrides YAML values.")
    ap.add_argument("--s_true", type=float, default=None, help="Teacher singular value (target).")
    ap.add_argument("--a0", type=float, default=None, help="Initial a(0) = c0*d0 (used if c0/d0 not given).")
    ap.add_argument("--c0", type=float, default=None, help="Initial c(0) (overrides a0 if both c0 and d0 given).")
    ap.add_argument("--d0", type=float, default=None, help="Initial d(0) (overrides a0 if both c0 and d0 given).")
    ap.add_argument("--eta_gd", type=float, default=None, help="GD step size.")
    ap.add_argument("--eta_eg", type=float, default=None, help="EG step size.")
    ap.add_argument("--steps", type=int, default=None, help="Number of discrete steps.")
    ap.add_argument("--ode_pts", type=int, default=None, help="Number of points for the ODE curve.")
    ap.add_argument("--show_continuous", action="store_true", help="Plot continuous ODE overlays.")
    ap.add_argument("--no_show_continuous", dest="show_continuous", action="store_false")
    ap.set_defaults(show_continuous=True)

    ap.add_argument("--out", type=str, default=None, help="Output file name (e.g., one_mode.pdf)")
    ap.add_argument("--outdir", type=str, default=None, help="Directory for outputs (created if needed).")
    ap.add_argument("--no_show", action="store_true", help="Skip plt.show().")
    ap.add_argument("--csv_out", type=str, default=None, help="Optional CSV/NPZ path for trajectories (relative to outdir if not absolute).")
    ap.add_argument("--title_suffix", type=str, default=None, help="Optional text appended in the plot title.")

    pre_args, _ = ap.parse_known_args()
    ycfg = {}
    if pre_args.config:
        with open(pre_args.config, "r") as f:
            ycfg = yaml.safe_load(f) or {}
        valid = {a.dest for a in ap._actions if a.dest != "help"}
        unknown = set(ycfg.keys()) - valid
        if unknown:
            raise ValueError(f"Unknown YAML key(s): {sorted(unknown)}")
        ap.set_defaults(**ycfg)
    args = ap.parse_args()

    cfg = vars(args).copy()

    defaults = {
        "s_true": 1.0,
        "a0": 1e-2,
        "eta_gd": 5e-3,
        "eta_eg": 1e-1,
        "steps": 700,
        "ode_pts": 800,
        "out": "one_mode.pdf",
        "outdir": "results",
        "show_continuous": True,
    }
    for k, v in defaults.items():
        if cfg.get(k) is None:
            cfg[k] = v

    if cfg["c0"] is not None and cfg["d0"] is not None:
        pass  
    else:
        z0 = np.sqrt(max(cfg["a0"], 0.0))
        cfg["c0"] = z0
        cfg["d0"] = z0

    outdir = Path(cfg["outdir"])
    outdir.mkdir(parents=True, exist_ok=True)

    if not os.path.isabs(cfg["out"]):
        cfg["out"] = str(outdir / cfg["out"])
    if cfg["csv_out"] is not None and not os.path.isabs(cfg["csv_out"]):
        cfg["csv_out"] = str(outdir / cfg["csv_out"])

    return cfg

def main():
    cfg = parse_args()
    set_rc()

    t_gd, a_gd = simulate_discrete(gd_step, cfg["c0"], cfg["d0"], cfg["s_true"], cfg["eta_gd"], cfg["steps"])
    t_eg, a_eg = simulate_discrete(eg_step, cfg["c0"], cfg["d0"], cfg["s_true"], cfg["eta_eg"], cfg["steps"])

    t_span = np.linspace(0, cfg["steps"], cfg["ode_pts"]) if cfg["show_continuous"] else None
    a_gd_th = odeint(lambda a, t: gd_ode(a, t, cfg["s_true"], cfg["eta_gd"]),
                     y0=[cfg["c0"] * cfg["d0"]], t=t_span).flatten() if t_span is not None else None
    a_eg_th = odeint(lambda a, t: eg_ode(a, t, cfg["s_true"], cfg["eta_eg"]),
                     y0=[cfg["c0"] * cfg["d0"]], t=t_span).flatten() if t_span is not None else None

    fig = make_figure(cfg, dict(
        t_gd=t_gd, a_gd=a_gd,
        t_eg=t_eg, a_eg=a_eg,
        t_ode=t_span, a_gd_ode=a_gd_th, a_eg_ode=a_eg_th,
    ))

    fig.savefig(cfg["out"], bbox_inches="tight")
    if not cfg.get("no_show", False):
        plt.show()
    plt.close(fig)

    if cfg["csv_out"]:
        np.savez(cfg["csv_out"],
                 t_gd=t_gd, a_gd=a_gd,
                 t_eg=t_eg, a_eg=a_eg,
                 t_ode=(t_span if t_span is not None else np.array([])),
                 a_gd_ode=(a_gd_th if a_gd_th is not None else np.array([])),
                 a_eg_ode=(a_eg_th if a_eg_th is not None else np.array([])),
                 s_true=cfg["s_true"], c0=cfg["c0"], d0=cfg["d0"],
                 eta_gd=cfg["eta_gd"], eta_eg=cfg["eta_eg"])
    print(f"Saved figure to: {cfg['out']}")

if __name__ == "__main__":
    main()
