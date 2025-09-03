from __future__ import annotations
ax.set_xlabel('PC1(mem)'); ax.set_ylabel('PC2(mem)'); ax.set_zlabel(f'logit[{z_logit}]')
ax.set_title('PC1–PC2 vs output (z)')
ax.legend(loc='best')


# Panel B: ring plane with angles & circular variance
ax2 = fig.add_subplot(2, 2, 2)
if P is not None:
proj = (H - H.mean(axis=0)) @ P
ang = np.arctan2(proj[:, 1], proj[:, 0])
sc = ax2.scatter(proj[:, 0], proj[:, 1], c=ang, s=14, cmap='twilight')
z = np.exp(1j * ang); V_circ = 1 - np.abs(z.mean())
ax2.set_title(f'Ring plane (V_circ={V_circ:.2f})')
plt.colorbar(sc, ax=ax2, label='angle θ')
else:
ax2.set_title('Ring plane (insufficient points)')
ax2.axhline(0, lw=0.5, c='k'); ax2.axvline(0, lw=0.5, c='k')
ax2.set_xlabel('PC1'); ax2.set_ylabel('PC2')


# Panel C: histogram of margins
ax3 = fig.add_subplot(2, 2, 3)
ax3.hist(summary['margin'].values, bins=30)
ax3.set_xlabel('1 − ρ(J)'); ax3.set_ylabel('count'); ax3.set_title('Margins')


# Panel D: motif counts
ax4 = fig.add_subplot(2, 2, 4)
order = ["stable", "saddle", "rotational", "continuous", "unstable"]
counts = summary['label'].value_counts().reindex(order).fillna(0)
ax4.bar(np.arange(len(order)), counts.values)
ax4.set_xticks(np.arange(len(order))); ax4.set_xticklabels(order, rotation=20)
ax4.set_ylabel('count'); ax4.set_title('Motifs')


fig.tight_layout()
_ensure_dir(out_png)
fig.savefig(out_png, dpi=200)
plt.close(fig)




def main():
args = build_parser().parse_args()
run_dir = Path(args.run)
cfg = load_config(args.config)
eval_dir = run_dir / cfg.get("fixed_points", {}).get("eval", {}).get("outdir", "eval/fixed_points")


df = pd.read_csv(eval_dir / "summary.csv")
fp = np.load(str(eval_dir / "fixed_points.npz"), allow_pickle=True)


plot_hist_rho(df, eval_dir / "rho_hist.png")
plot_stability_bar(df, eval_dir / "stability_bar.png")
plot_margin(df, eval_dir / "margin_hist.png")
plot_rotational(df, eval_dir / "rotational_scatter.png")
plot_residuals(df, eval_dir / "residual_hist.png")
plot_unit_proximity(fp["eigvals"], eval_dir / "unit_proximity.png")


# Four-panel context figure
four_panel_figure(run_dir, cfg, eval_dir / "four_panel.png")
print(f"[plot_fixed_points] wrote figures to {eval_dir}")




if __name__ == "__main__":
main()