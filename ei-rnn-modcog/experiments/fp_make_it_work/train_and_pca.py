import os, math, json, yaml, random, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from src.models.ei_rnn import EIRNN, EIConfig

# ---------------------------
# Utils
# ---------------------------
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def mkdir_p(p):
    Path(p).mkdir(parents=True, exist_ok=True)

# ---------------------------
# DlyGo-style synthetic dataset
# ---------------------------
class DlyGoDataset:
    """
    Synthetic DlyGo generator (single modality).
    Inputs (per step): [fixation (1), ring_stim (dim_ring), go (1)]  => input_dim = dim_ring + 2
    Outputs (per step): logits over ring bins (dim_ring)
      - We train only on decision steps: target = stimulus angle bin.
    Epoch schedule per trial:
      fixation -- stimulus -- delay -- decision
    """
    def __init__(self, dim_ring=16, fixation=5, stimulus=5, delay=10, decision=5, noise_std=0.0):
        self.dim_ring = dim_ring
        self.fix = fixation
        self.stim = stimulus
        self.delay = delay
        self.decision = decision
        self.T = fixation + stimulus + delay + decision
        self.noise_std = float(noise_std)

        self.input_dim  = 1 + dim_ring + 1     # fix, ring, go
        self.output_dim = dim_ring

    def sample_batch(self, batch_size: int):
        B, T, R = batch_size, self.T, self.dim_ring
        x = np.zeros((B, T, 1 + R + 1), dtype=np.float32)
        y = np.full((B, T), fill_value=-100, dtype=np.int64)  # ignore index by default
        mask = np.zeros((B, T), dtype=np.float32)             # 1.0 at decision steps

        # indices into x last dim
        FIX = 0
        RING = slice(1, 1+R)
        GO = 1 + R

        for b in range(B):
            theta = np.random.randint(0, R)   # stimulus bin
            # fixation + stimulus + delay epochs: keep fixation on
            x[b, :self.fix + self.stim + self.delay, FIX] = 1.0
            # stimulus epoch: present one-hot ring
            x[b, self.fix:self.fix+self.stim, RING][..., theta] = 1.0
            # delay: (no stimulus)
            # decision: fixation off, go on
            decision_start = self.fix + self.stim + self.delay
            x[b, decision_start:, GO] = 1.0

            # optional input noise on ring channels (stimulus steps)
            if self.noise_std > 0:
                noise = np.random.randn(self.stim, R).astype(np.float32) * self.noise_std
                x[b, self.fix:self.fix+self.stim, RING] += noise
                x[b, self.fix:self.fix+self.stim, RING] = np.clip(x[b, self.fix:self.fix+self.stim, RING], 0.0, None)

            # targets only in decision steps: the chosen angle bin
            y[b, decision_start:,] = theta
            mask[b, decision_start:,] = 1.0

        return x, y, mask

# ---------------------------
# Training loop
# ---------------------------
def train_and_eval(cfg_path: str):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    # --- seeds & device
    seed = int(cfg["train"]["seed"])
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- dataset
    ds = DlyGoDataset(
        dim_ring=cfg["task"]["dim_ring"],
        fixation=cfg["task"]["fixation_steps"],
        stimulus=cfg["task"]["stimulus_steps"],
        delay=cfg["task"]["delay_steps"],
        decision=cfg["task"]["decision_steps"],
        noise_std=cfg["task"]["input_noise_std"],
    )
    input_dim, output_dim = ds.input_dim, ds.output_dim
    T = ds.T

    # --- model
    mcfg = cfg["model"]
    eic = EIConfig(
        hidden_size=mcfg["hidden_size"],
        exc_frac=mcfg["exc_frac"],
        spectral_radius=mcfg["spectral_radius"],
        input_scale=mcfg["input_scale"],
        leak=mcfg["leak"],
        nonlinearity=mcfg["nonlinearity"],
        readout=mcfg["readout"],
    )
    model = EIRNN(input_size=input_dim, output_size=output_dim, cfg=eic).to(device)
    model.train()

    # --- optim
    opt = optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    crit = nn.CrossEntropyLoss(reduction="none")  # we'll mask over time

    steps_per_epoch = int(cfg["train"]["steps_per_epoch"])
    epochs = int(cfg["train"]["epochs"])
    bs = int(cfg["train"]["batch_size"])
    clip = float(cfg["train"]["grad_clip"])
    log_every = int(cfg["train"]["log_every"])

    # --- training
    for ep in range(1, epochs+1):
        running_loss, running_acc, seen = 0.0, 0.0, 0
        for step in range(1, steps_per_epoch+1):
            x_np, y_np, m_np = ds.sample_batch(bs)
            x = torch.from_numpy(x_np).to(device)                 # (B,T,inp)
            y = torch.from_numpy(y_np).to(device)                 # (B,T) class ids or -100
            m = torch.from_numpy(m_np).to(device)                 # (B,T) 0/1

            logits = model(x)                                     # (B,T,R)
            # compute masked CE on decision steps
            loss_t = crit(logits.view(-1, output_dim), y.view(-1))
            loss_t = loss_t.view(bs, T)
            loss = (loss_t * m).sum() / (m.sum() + 1e-8)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()

            # metrics (decision steps only)
            with torch.no_grad():
                pred = logits.argmax(dim=-1)                      # (B,T)
                correct = ((pred == y) * (m > 0)).sum().item()
                total = (m > 0).sum().item()
                running_loss += float(loss.item()) * bs
                running_acc  += correct
                seen         += total

            if step % log_every == 0:
                print(f"[ep {ep}/{epochs} | step {step}/{steps_per_epoch}] "
                      f"loss={running_loss/(log_every*bs):.4f} "
                      f"acc_decision={running_acc/seen:.3f}")
                running_loss, running_acc, seen = 0.0, 0.0, 0

        # optional gentle spectral rescale each epoch
        with torch.no_grad():
            model.rescale_spectral_radius_(tol=0.10)

    # ---------------------------
    # Evaluation rollouts + PCA
    # ---------------------------
    model.eval()
    out_dir = Path(cfg["eval"]["save_dir"])
    mkdir_p(out_dir)

    N = int(cfg["eval"]["n_trials"])
    all_X, all_H, all_PRE, all_ACT, all_Y, angle_labels = [], [], [], [], [], []

    with torch.no_grad():
        for _ in range(N):
            # one trial per batch for clean per-trial logging
            x_np, y_np, m_np = ds.sample_batch(1)
            x = torch.from_numpy(x_np).to(device)
            logits, H, PRE, ACT, X = model.forward_with_states(x)
            all_X.append(X.squeeze(0).cpu().numpy())
            all_H.append(H.squeeze(0).cpu().numpy())
            all_PRE.append(PRE.squeeze(0).cpu().numpy())
            all_ACT.append(ACT.squeeze(0).cpu().numpy())
            all_Y.append(logits.squeeze(0).cpu().numpy())

            # label = target angle (constant over decision steps)
            theta = int(y_np[0, -1])
            angle_labels.append(theta)

    # stack for saving
    X_arr   = np.stack(all_X, axis=0)    # (N, T, in)
    H_arr   = np.stack(all_H, axis=0)    # (N, T, H)
    PRE_arr = np.stack(all_PRE, axis=0)  # (N, T, H)
    ACT_arr = np.stack(all_ACT, axis=0)  # (N, T, H)
    Y_arr   = np.stack(all_Y, axis=0)    # (N, T, R)
    labels  = np.array(angle_labels, dtype=np.int64)

    # save everything
    np.savez_compressed(
        out_dir / "trajectories.npz",
        X=X_arr, H=H_arr, PRE=PRE_arr, ACT=ACT_arr, Y=Y_arr, labels=labels,
        meta=json.dumps({"T": ds.T, "dim_ring": ds.dim_ring,
                         "epochs": epochs, "seed": seed}))

    # ---------------------------
    # PCA on delay states (should form a ring)
    # ---------------------------
    fix, stim, delay = ds.fix, ds.stim, ds.delay
    delay_slice = slice(fix + stim, fix + stim + delay)

    H_delay = H_arr[:, delay_slice, :]                 # (N, delay, H)
    H_delay_flat = H_delay.reshape(-1, H_delay.shape[-1])  # (N*delay, H)
    # replicate labels per delay step
    labels_delay = np.repeat(labels, delay)

    pca = PCA(n_components=3, svd_solver="full")
    Z = pca.fit_transform(H_delay_flat)                # (N*delay, 3)
    print("Explained variance ratios (PC1..3):", pca.explained_variance_ratio_[:3])

    # scatter: PC1 vs PC2 colored by angle bin
    plt.figure(figsize=(6,5))
    sc = plt.scatter(Z[:,0], Z[:,1], c=labels_delay, s=6, cmap="hsv")
    plt.xlabel("PC1"); plt.ylabel("PC2"); plt.title("DlyGo delay dynamics (hidden state PCA)")
    cbar = plt.colorbar(sc)
    cbar.set_label("stimulus angle bin")
    plt.tight_layout()
    plt.savefig(out_dir / "pca_delay_ring.png", dpi=160)

    # bonus: plot per-trial trajectory across epochs (PC1–PC2 over time)
    # fit PCA on the whole trial to visualize transitions onto/off the ring
    H_all_flat = H_arr.reshape(-1, H_arr.shape[-1])
    pca_traj = PCA(n_components=3, svd_solver="full").fit(H_all_flat)

    plt.figure(figsize=(7,6))
    for i in range(min(24, N)):
        Zi = pca_traj.transform(H_arr[i])            # (T, 3)
        t = np.arange(ds.T)
        # color code: stim=green, delay=blue, decision=red
        plt.plot(Zi[:fix,0], Zi[:fix,1], alpha=0.3)                     # fixation (grey)
        plt.plot(Zi[fix:fix+stim,0], Zi[fix:fix+stim,1], alpha=0.6)     # stimulus
        plt.plot(Zi[fix+stim:fix+stim+delay,0], Zi[fix+stim:fix+stim+delay,1], alpha=0.8)  # delay
        plt.plot(Zi[fix+stim+delay:,0], Zi[fix+stim+delay:,1], alpha=0.8) # decision
    plt.xlabel("PC1"); plt.ylabel("PC2"); plt.title("Trajectories (subset of trials)")
    plt.tight_layout()
    plt.savefig(out_dir / "pca_full_trajectories.png", dpi=160)

    print(f"Saved arrays + figures to: {out_dir.resolve()}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/dlygo_pca.yaml")
    args = ap.parse_args()
    train_and_eval(args.config)
