from __future__ import annotations
if cand.exists():
ckpt.parent.mkdir(parents=True, exist_ok=True)
if ckpt != cand:
import shutil
shutil.copy2(cand, ckpt)
print(f"[orchestrator] normalized checkpoint → {ckpt}")
break
else:
raise FileNotFoundError(
"Checkpoint not found after training.\n"
f"Expected: {ckpt}\n"
"Verify the trainer saves to <outdir>/run/ckpt.pt or one of the common names."
)


fp_cfg = cfg.get("fixed_points", {})
eval_subdir = fp_cfg.get("eval", {}).get("outdir", "eval/fixed_points")
eval_dir = run_dir / eval_subdir
seeds_file = eval_dir / "rollout_seeds.npz"
eval_dir.mkdir(parents=True, exist_ok=True)


device = _device(cfg.get("device", "auto"))
if not seeds_file.exists():
model, _saved = rebuild_model_from_ckpt(ckpt, device=device)
task = cfg.get("tasks", ["dm1"])[0]
env_fn = getattr(mct, task)
X_np, H0_np = _collect_rollout(
model=model.to(device),
env_fn=env_fn,
rollout_cfg=fp_cfg.get("rollout", {}),
device=device,
)
_save_npz(seeds_file, X=X_np, H0=H0_np)
print(f"[orchestrator] wrote seeds → {seeds_file}")


# Run analysis
res = subprocess.run(
[sys.executable, "-u", str(Path(__file__).parent / "unified_fixed_points.py"),
"--run", str(run_dir), "--config", cfg_path_for_trainer],
cwd=str(Path(__file__).parent),
)
if res.returncode != 0:
raise RuntimeError(f"Analysis failed with exit code {res.returncode}")


# Plot
res = subprocess.run(
[sys.executable, "-u", str(Path(__file__).parent / "plot_fixed_points.py"),
"--run", str(run_dir), "--config", cfg_path_for_trainer],
cwd=str(Path(__file__).parent),
)
if res.returncode != 0:
raise RuntimeError(f"Plotting failed with exit code {res.returncode}")




if __name__ == "__main__":
main()