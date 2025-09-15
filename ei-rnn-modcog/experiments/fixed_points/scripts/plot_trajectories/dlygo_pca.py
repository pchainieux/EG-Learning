import argparse, time, math, random
from pathlib import Path
import yaml
import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from src.data import mod_cog_tasks as mct
from neurogym import Dataset
from src.analysis import viz_training as viz
from src.models.ei_rnn import EIRNN, EIConfig
from src.optim.sgd_eg import SGD_EG
from src.utils.seeding import set_seed_all, pick_device, device_name
from src.training.losses import ModCogLossCombined, decision_mask_from_inputs, row_sum_penalty
from src.training.metrics import accuracy_with_fixation

def ensure_tensor(x, dtype, device):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(device=device, dtype=dtype)
    return x.to(device=device, dtype=dtype)

def mkdir_p(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def build_task_datasets(task_names, batch_size, seq_len, data_cfg, device=None):
    source = (data_cfg.get("source", "online")).lower()
    out = {}
    
    if source == "online":
        from neurogym import Dataset
        for t in task_names:
            env_fn = getattr(mct, t)
            tr_env, va_env = env_fn(), env_fn()
            tr = Dataset(tr_env, batch_size=batch_size, seq_len=seq_len, batch_first=True)
            va = Dataset(va_env, batch_size=batch_size, seq_len=seq_len, batch_first=True)
            out[t] = (tr, va, tr_env.action_space.n, tr_env.observation_space.shape[-1])
        return out
    
    raise NotImplementedError("Use online data source for now")

def extract_batch_data(ds):
    batch = ds()
    if hasattr(batch, 'inputs') and hasattr(batch, 'target'):
        return batch.inputs, batch.target
    elif isinstance(batch, tuple) and len(batch) >= 2:
        return batch[0], batch[1]
    elif isinstance(batch, dict):
        return batch['inputs'], batch['target']
    else:
        return batch.observation, batch.action

@torch.no_grad()
def evaluate_with_activations(model, ds, device, mask_thresh, batches=20):
    model.eval()
    crit = ModCogLossCombined()
    
    tot_l = tot_fix = tot_dec = 0.0
    acc_all = acc_dec = 0.0
    n_dec = 0
    
    all_X, all_H, all_PRE, all_ACT, all_Y, all_labels = [], [], [], [], [], []
    
    for _ in range(batches):
        try:
            X_np, Y_np = extract_batch_data(ds)
            X = ensure_tensor(X_np, torch.float32, device)
            Y = ensure_tensor(Y_np, torch.long, device)
        except Exception as e:
            print(f"Batch extraction error: {e}")
            continue
            
        dec_mask = decision_mask_from_inputs(X, thresh=mask_thresh)
        
        logits, H, PRE, ACT, X_states = model.forward_with_states(X)
        
        l, lf, ld = crit(logits, Y, dec_mask)
        tot_l += l.item(); tot_fix += lf.item(); tot_dec += ld.item()
        
        a_all, a_dec = accuracy_with_fixation(logits, Y, dec_mask)
        acc_all += a_all
        if not math.isnan(a_dec):
            acc_dec += a_dec
            n_dec += 1
        
        all_X.append(X.cpu().numpy())
        all_H.append(H.cpu().numpy())
        all_PRE.append(PRE.cpu().numpy())
        all_ACT.append(ACT.cpu().numpy())
        all_Y.append(logits.cpu().numpy())
        
        batch_size = X.shape[0]
        for b in range(batch_size):
            decision_times = torch.where(dec_mask[b] > 0.5)[0]
            if len(decision_times) > 0:
                label = Y[b, decision_times[-1]].item()
                all_labels.append(label)
            else:
                all_labels.append(0)
    
    model.train()
    
    if not all_H: 
        return None, None
    
    activations = {
        "X": np.concatenate(all_X, axis=0),
        "H": np.concatenate(all_H, axis=0), 
        "PRE": np.concatenate(all_PRE, axis=0),
        "ACT": np.concatenate(all_ACT, axis=0),
        "Y": np.concatenate(all_Y, axis=0),
        "labels": np.array(all_labels)
    }
    
    metrics = {
        "loss": tot_l/batches,
        "loss_fix": tot_fix/batches, 
        "loss_dec": tot_dec/batches,
        "acc": acc_all/batches,
        "acc_dec": acc_dec/max(1, n_dec)
    }
    
    return metrics, activations

def analyze_delay_dynamics(activations, timing_cfg, outdir, epoch=None):
    H_arr = activations["H"]
    labels = activations["labels"]
    
    fix_steps = int(timing_cfg.get("fixation_steps", 5))
    stim_steps = int(timing_cfg.get("stimulus_steps", 5))
    delay_steps = int(timing_cfg.get("delay_steps", 10))
    
    delay_start = fix_steps + stim_steps
    delay_end = delay_start + delay_steps
    
    if delay_end > H_arr.shape[1]:
        print(f"Warning: Not enough timesteps for delay analysis. Have {H_arr.shape[1]}, need {delay_end}")
        delay_end = min(delay_end, H_arr.shape[1])
        delay_steps = delay_end - delay_start
    
    delay_slice = slice(delay_start, delay_end)
    
    H_delay = H_arr[:, delay_slice, :] 
    H_delay_flat = H_delay.reshape(-1, H_delay.shape[-1])
    
    n_trials = H_arr.shape[0]
    if len(labels) != n_trials:
        print(f"Warning: labels size {len(labels)} doesn't match trials {n_trials}")
        if len(labels) > n_trials:
            labels = labels[:n_trials]
        else:
            labels = np.pad(labels, (0, n_trials - len(labels)), mode='constant', 
                          constant_values=labels[-1] if len(labels) > 0 else 0)
    
    labels_delay = np.repeat(labels, delay_steps)
    
    if len(labels_delay) != H_delay_flat.shape[0]:
        print(f"ERROR: Size mismatch! labels_delay: {len(labels_delay)}, H_delay_flat: {H_delay_flat.shape[0]}")
        min_size = min(len(labels_delay), H_delay_flat.shape[0])
        labels_delay = labels_delay[:min_size]
        H_delay_flat = H_delay_flat[:min_size]
    
    pca = PCA(n_components=3, svd_solver="full")
    Z = pca.fit_transform(H_delay_flat)
    
    print(f"PCA shapes: Z={Z.shape}, labels_delay={labels_delay.shape}")
    print(f"PCA explained variance: {pca.explained_variance_ratio_[:3]}")
    
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(Z[:, 0], Z[:, 1], c=labels_delay, s=4, cmap="hsv", alpha=0.7)
    plt.colorbar(scatter, label="Target Action")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    title = f"Delay Period Dynamics (Real ModCog Task)"
    if epoch:
        title += f" - Epoch {epoch}"
    plt.title(title)
    plt.tight_layout()
    
    filename = f"pca_delay_epoch{epoch:03d}.png" if epoch else "pca_delay_final.png"
    plt.savefig(outdir / filename, dpi=200, bbox_inches='tight')
    plt.close()
    
    pca_full = PCA(n_components=3).fit(H_arr.reshape(-1, H_arr.shape[-1]))
    plt.figure(figsize=(8, 6))
    
    n_plot = min(24, H_arr.shape[0])
    for i in range(n_plot):
        Zi = pca_full.transform(H_arr[i])
        
        plt.plot(Zi[:fix_steps, 0], Zi[:fix_steps, 1], alpha=0.3, color='gray', linewidth=0.5)
        plt.plot(Zi[fix_steps:fix_steps+stim_steps, 0], Zi[fix_steps:fix_steps+stim_steps, 1], 
                alpha=0.6, color='green', linewidth=0.7)
        plt.plot(Zi[delay_start:delay_end, 0], Zi[delay_start:delay_end, 1], 
                alpha=0.8, color='blue', linewidth=1.0)
        plt.plot(Zi[delay_end:, 0], Zi[delay_end:, 1], alpha=1.0, color='red', linewidth=0.8)
    
    plt.xlabel("PC1")
    plt.ylabel("PC2") 
    title = f"Trial Trajectories (Real ModCog)"
    if epoch:
        title += f" - Epoch {epoch}"
    plt.title(title)
    plt.tight_layout()
    
    filename = f"trajectories_epoch{epoch:03d}.png" if epoch else "trajectories_final.png"
    plt.savefig(outdir / filename, dpi=200, bbox_inches='tight')
    plt.close()
    
    filename = f"pca_data_epoch{epoch:03d}.npz" if epoch else "pca_data_final.npz"
    np.savez_compressed(outdir / filename,
                        Z_delay=Z, labels_delay=labels_delay,
                        explained_variance_ratio=pca.explained_variance_ratio_,
                        components=pca.components_)
    
    return pca.explained_variance_ratio_

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/dlygo_analysis.yaml")
    args = ap.parse_args()
    
    cfg = yaml.safe_load(open(args.config, "r"))
    
    seed = int(cfg.get("seed", 7))
    set_seed_all(seed, deterministic=False)
    
    dev_str = cfg.get("device", "auto")
    device = pick_device(dev_str)
    print(f"Using device: {device_name(device)}")
    
    outdir = Path(cfg.get("outdir", "runs/exp"))
    outdir.mkdir(parents=True, exist_ok=True)
    
    tasks = cfg.get("tasks", ["dlygo"])
    if isinstance(tasks, str):
        tasks = [tasks]
    
    print(f"Training on ModCog tasks: {tasks}")
    
    data_cfg = cfg.get("data", {})
    seq_len = int(data_cfg.get("seq_len", 400))
    batch_sz = int(data_cfg.get("batch_size", 64))
    
    dsets = build_task_datasets(tasks, batch_sz, seq_len, data_cfg, device=device.type)
    
    model_cfg = cfg.get("model", {})
    
    X0_np, _ = extract_batch_data(dsets[tasks[0]][0])
    input_dim = X0_np.shape[-1]
    output_dim = dsets[tasks[0]][2]
    
    print(f"Task: {tasks[0]} | Input: {input_dim}, Output: {output_dim}")
    
    ei_cfg = EIConfig(
        hidden_size=int(model_cfg.get("hidden_size", 256)),
        exc_frac=float(model_cfg.get("exc_frac", 0.8)),
        spectral_radius=float(model_cfg.get("spectral_radius", 1.2)),
        input_scale=float(model_cfg.get("input_scale", 1.0)),
        leak=float(model_cfg.get("leak", 0.2)),
        nonlinearity=model_cfg.get("nonlinearity", "softplus"),
        readout=model_cfg.get("readout", "e_only"),
        softplus_beta=float(model_cfg.get("softplus_beta", 8.0)),
        softplus_threshold=float(model_cfg.get("softplus_threshold", 20.0))
    )
    
    model = EIRNN(input_size=input_dim, output_size=output_dim, cfg=ei_cfg).to(device)
    
    optim_cfg = cfg.get("optim", {})
    algo = optim_cfg.get("algorithm", "gd").lower()
    
    if algo == "eg":
        eg_params = [model.W_hh]
        gd_params = [p for _, p in model.named_parameters() if p is not model.W_hh]
        eg = optim_cfg.get("eg", {})
        gd = optim_cfg.get("gd", {})
        
        opt = SGD_EG([
            {"params": eg_params, "update_alg": "eg", "lr": float(eg.get("lr", 1.5))},
            {"params": gd_params, "update_alg": "gd", "lr": float(gd.get("lr", 0.01))}
        ])
    else:
        gd = optim_cfg.get("gd", {})
        opt = SGD_EG([{"params": list(model.parameters()), "update_alg": "gd", 
                       "lr": float(gd.get("lr", 0.01))}])
    
    crit = ModCogLossCombined()
    
    train_cfg = cfg.get("train", {})
    epochs = int(train_cfg.get("num_epochs", 20))
    steps_per_epoch = int(train_cfg.get("steps_per_epoch", 300))
    eval_every = int(cfg.get("eval", {}).get("eval_every", 5))
    
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        
        for step in range(steps_per_epoch):
            train_ds = dsets[tasks[0]][0]
            
            try:
                X_np, Y_np = extract_batch_data(train_ds)
                X = ensure_tensor(X_np, torch.float32, device)
                Y = ensure_tensor(Y_np, torch.long, device)
            except Exception as e:
                print(f"Training batch error: {e}")
                continue
            
            dec_mask = decision_mask_from_inputs(X, thresh=0.5)
            logits = model(X)
            loss, _, _ = crit(logits, Y, dec_mask)
            
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            
            if algo == "gd":
                model.project_EI_()
            
            epoch_loss += loss.item()
        
        print(f"Epoch {epoch:03d}: Loss {epoch_loss/steps_per_epoch:.4f}")
        
        if epoch % eval_every == 0 or epoch == epochs:
            print(f"Analyzing dynamics at epoch {epoch}...")
            
            val_ds = dsets[tasks[0]][1]
            result = evaluate_with_activations(
                model, val_ds, device, mask_thresh=0.5, batches=20
            )
            
            if result[0] is not None:
                metrics, activations = result
                print(f"  Val Acc: {metrics['acc']:.3f}, Dec Acc: {metrics['acc_dec']:.3f}")
                
                np.savez_compressed(outdir / f"activations_epoch{epoch:03d}.npz", 
                                   **activations, epoch=epoch)
                
                explained_var = analyze_delay_dynamics(
                    activations, cfg.get("timing", {}), outdir, epoch
                )
                
                print(f"  PCA explained variance: {explained_var[:3]}")
        
        if epoch % 10 == 0 or epoch == epochs:
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'config': cfg
            }, outdir / f"model_epoch{epoch:03d}.pt")
    
    print(f"Training complete! Results in: {outdir}")

if __name__ == "__main__":
    main()
