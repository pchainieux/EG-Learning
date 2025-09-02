from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

EG_COLOR = "#1f77b4"
GD_COLOR = "#d62728"
TARGET_COLOR = "#6c757d"

def _set_pub_style(ax):
    ax.grid(True, alpha=0.3, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=12)

try:
    from src.analysis.viz_training import _ema, _rolling
except Exception:
    def _ema(y: np.ndarray, beta: float) -> np.ndarray:
        y = np.asarray(y, dtype=float).reshape(-1)
        out = np.empty_like(y)
        m = 0.0
        for i, v in enumerate(y):
            m = beta * m + (1.0 - beta) * v
            out[i] = m
        return out

    def _rolling(y: np.ndarray, win: int) -> np.ndarray:
        y = np.asarray(y, dtype=float).reshape(-1)
        k = int(max(1, win))
        if k == 1 or y.size == 0:
            return y
        if k % 2 == 0:
            k += 1
        pad = k // 2
        yp = np.pad(y, (pad, pad), mode="edge")
        ker = np.ones(k, dtype=float) / float(k)
        out = np.convolve(yp, ker, mode="valid")
        return out


def _smooth(y, *, ema_beta: Optional[float] = None, window: Optional[int] = None):
    if ema_beta is not None:
        return _ema(np.asarray(y, dtype=float), float(ema_beta))
    if window is not None and int(window) > 1:
        return _rolling(np.asarray(y, dtype=float), int(window))
    return np.asarray(y, dtype=float)

def _load_steps(npz_path: str):
    d = np.load(npz_path, allow_pickle=True)
    step = d.get("step_idx")
    if step is None:
        n = len(d["acc_train_step"])
        step = np.arange(n)
    return {
        "x": np.asarray(step, dtype=float),
        "acc": np.asarray(d["acc_train_step"], dtype=float),
        "acc_dec": np.asarray(d["acc_dec_train_step"], dtype=float),
        "loss": np.asarray(d["loss_train_step"], dtype=float),
    }

def _load_epochs(npz_path: str):
    d = np.load(npz_path, allow_pickle=True)
    x = d.get("epoch_idx", d.get("epoch", None))
    if x is None:
        # fallback to length of val_acc
        v = d.get("val_acc_epoch_mean", d.get("val_acc_epoch"))
        x = np.arange(1, len(v) + 1)
    v = d.get("val_acc_epoch_mean", d.get("val_acc_epoch"))
    return {"x": np.asarray(x, dtype=float), "val": np.asarray(v, dtype=float)}



def plot_accuracy_steps(eg_steps_npz: str, gd_steps_npz: str, outpath: str,
                        *, ema_beta: float = 0.98, target: float | None = None):
    eg = _load_steps(eg_steps_npz)
    gd = _load_steps(gd_steps_npz)

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    _set_pub_style(ax)

    ax.plot(eg["x"][:len(eg["acc"])], _smooth(eg["acc"], ema_beta=ema_beta),
            lw=2.2, color=EG_COLOR, label="EG — overall")
    ax.plot(eg["x"][:len(eg["acc_dec"])], _smooth(eg["acc_dec"], ema_beta=ema_beta),
            lw=2.2, ls="--", color=EG_COLOR, label="EG — decision")

    ax.plot(gd["x"][:len(gd["acc"])], _smooth(gd["acc"], ema_beta=ema_beta),
            lw=2.2, color=GD_COLOR, label="GD — overall")
    ax.plot(gd["x"][:len(gd["acc_dec"])], _smooth(gd["acc_dec"], ema_beta=ema_beta),
            lw=2.2, ls="--", color=GD_COLOR, label="GD — decision")

    if target is not None:
        ax.axhline(target, color=TARGET_COLOR, lw=1.6, ls=":")

    ax.set_xlabel("training steps")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    fig.savefig(outpath.replace(".png", ".pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_loss_steps(eg_steps_npz: str, gd_steps_npz: str, outpath: str,
                    *, ema_beta: float = 0.98, target: float | None = None):
    eg = _load_steps(eg_steps_npz)
    gd = _load_steps(gd_steps_npz)

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    _set_pub_style(ax)

    ax.plot(eg["x"][:len(eg["loss"])], _smooth(eg["loss"], ema_beta=ema_beta),
            lw=2.2, color=EG_COLOR, label="EG")
    ax.plot(gd["x"][:len(gd["loss"])], _smooth(gd["loss"], ema_beta=ema_beta),
            lw=2.2, color=GD_COLOR, label="GD")

    if target is not None:
        ax.axhline(target, color=TARGET_COLOR, lw=1.6, ls=":")

    ax.set_xlabel("training steps")
    ax.set_ylabel("loss")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    fig.savefig(outpath.replace(".png", ".pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_val_accuracy_epochs(eg_epoch_npz: str, gd_epoch_npz: str, outpath: str,
                             *, window: int = 5, target: float | None = None):
    eg = _load_epochs(eg_epoch_npz)
    gd = _load_epochs(gd_epoch_npz)

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    _set_pub_style(ax)

    ax.plot(eg["x"][:len(eg["val"])], _smooth(eg["val"], window=window),
            lw=2.2, color=EG_COLOR, label="EG")
    ax.plot(gd["x"][:len(gd["val"])], _smooth(gd["val"], window=window),
            lw=2.2, color=GD_COLOR, label="GD")

    if target is not None:
        ax.axhline(target, color=TARGET_COLOR, lw=1.6, ls=":")

    ax.set_xlabel("epoch")
    ax.set_ylabel("validation accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    fig.savefig(outpath.replace(".png", ".pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)
