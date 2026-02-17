from rl.utils.plotter import MetricsPlotter

json_path = "runs/20260216_113921_single_dual_eval_abcdefg_3(2)/metrics/metrics.json"
save_dir = "model_evaluation/plots/outputs/final_12k"

plotter = MetricsPlotter.from_json(json_path)
plotter.save_all(save_dir, show="both")

print("Plots saved to:", save_dir)
