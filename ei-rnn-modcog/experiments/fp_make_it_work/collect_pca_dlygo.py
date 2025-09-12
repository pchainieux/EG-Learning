# collect_pca_dlygo.py
import os, json, math
import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

from src.models.ei_rnn import EIRNN, EIConfig  
import src.data.mod_cog_tasks as tasks       

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = tasks.dlygo(dt=100, dim_ring=16) 

input_dim  = env.observation_space.shape[0]
output_dim = env.action_space.n 

ckpt_path = "outputs/dlygo_test_4/singlehead_epoch020.pt"
ckpt = torch.load(ckpt_path, map_location="cpu")

cfg_dict = ckpt.get("cfg", {})
cfg = EIConfig(**cfg_dict) if cfg_dict else EIConfig(hidden_size=256, nonlinearity="softplus")

model = EIRNN(input_size=input_dim, output_size=output_dim, cfg=cfg).to(device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()


def run_n_trials(n_trials=200, max_T=40):

    X_all, H_all, Y_all, meta = [], [], [], []

    for _ in range(n_trials):
        ob, _ = env.reset()

        xs, hs, ys = [], [], []
        H = model.cfg.hidden_size
        h = torch.zeros(1, H, device=device)
        alpha = model._alpha

        W_xh = model.W_xh
        W_hh = model.W_hh
        b_h  = model.b_h
        W_out = model.W_out

        steps = 0
        done = False
        while not done and steps < max_T:
            x_np = ob.astype(np.float32)
            x = torch.from_numpy(x_np)[None, :].to(device) 
            pre = torch.nn.functional.linear(h, W_hh, b_h) + torch.nn.functional.linear(x, W_xh, None)
            act = model._phi(pre)
            h   = (1.0 - alpha) * h + alpha * act

            h_ro = h * model.e_mask if model._readout_mode == "e_only" else h
            y    = W_out(h_ro)

            xs.append(x.detach().cpu().numpy().squeeze(0))
            hs.append(h.detach().cpu().numpy().squeeze(0))
            ys.append(y.detach().cpu().numpy().squeeze(0))

            action = int(y.argmax(dim=-1).item())
            ob, r, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1

        X_all.append(np.stack(xs, axis=0))
        H_all.append(np.stack(hs, axis=0))
        Y_all.append(np.stack(ys, axis=0))

        meta.append({
            "fixation_idx": (0, 4),
            "stimulus_idx": (5, 9),
            "delay_idx":    (10, 14),
            "decision_idx": (15, 19),
        })

    return X_all, H_all, Y_all, meta

X_all, H_all, Y_all, meta = run_n_trials(n_trials=400, max_T=40)


dim_ring = 16
fix_idx = 0
stim_block1 = slice(1, 1+dim_ring)            # modality A
stim_block2 = slice(1+dim_ring, 1+2*dim_ring) # modality B

H_delay = []   # list of (steps_in_delay, H)
labels  = []   # angle class 0..dim_ring-1

for Xi, Hi, Mi in zip(X_all, H_all, meta):
    a,b = Mi["delay_idx"]   # inclusive endpoints as we defined above
    # clamp to available length
    T = Hi.shape[0]
    lo, hi = max(0, a), min(T-1, b)
    if lo > hi: 
        continue

    # determine which modality was active during stimulus
    s_lo, s_hi = Mi["stimulus_idx"]
    s_lo, s_hi = max(0, s_lo), min(T-1, s_hi)
    stim_slice = Xi[s_lo:s_hi+1]
    # average stimulus over stimulus period
    stim_mean = stim_slice.mean(axis=0)
    stim_a = stim_mean[stim_block1].max()
    stim_b = stim_mean[stim_block2].max()
    if stim_a >= stim_b:
        stim_vec = stim_mean[stim_block1]
    else:
        stim_vec = stim_mean[stim_block2]
    theta_bin = int(np.argmax(stim_vec))  # 0..dim_ring-1

    H_delay.append(Hi[lo:hi+1])  # (steps, H)
    labels.extend([theta_bin] * (hi - lo + 1))

H_delay = np.concatenate(H_delay, axis=0)  # (N_delay_steps_total, H)
labels  = np.array(labels)

# ---------------------------
# 4) PCA on hidden states (delay only)
# ---------------------------
pca = PCA(n_components=3, svd_solver="full", whiten=False, random_state=0)
Z = pca.fit_transform(H_delay)  # (N, 3)
print("Explained variance (PC1..3):", pca.explained_variance_ratio_[:3])

# ---------------------------
# 5) Plot PC1 vs PC2 colored by stimulus angle
# ---------------------------
plt.figure(figsize=(6,5))
sc = plt.scatter(Z[:,0], Z[:,1], c=labels, s=10, cmap="hsv")
plt.xlabel("PC1"); plt.ylabel("PC2"); plt.title("DlyGo delay dynamics (hidden state PCA)")
plt.colorbar(sc, label="stimulus angle bin")
plt.tight_layout()
plt.show()
