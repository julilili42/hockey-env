from rl.utils.plotter import MetricsPlotter

json_path = "pretrained/weak_strong_25k/metrics/metrics.json"
save_dir = "model_evaluation/plots/outputs/plot_weak_strong_25k"

plotter = MetricsPlotter.from_json(json_path)
plotter.save_all(save_dir, show="both")

print("Plots saved to:", save_dir)
