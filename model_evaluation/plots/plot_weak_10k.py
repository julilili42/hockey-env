from rl.utils.plotter import MetricsPlotter

json_path = "pretrained/weak_10k/metrics/metrics.json"
save_dir = "model_evaluation/plots/outputs/final_12k"

plotter = MetricsPlotter.from_json(json_path)
plotter.save_all(save_dir, show="weak")

print("Plots saved to:", save_dir)
