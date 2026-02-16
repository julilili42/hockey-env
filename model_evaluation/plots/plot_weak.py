from rl.utils.plotter import MetricsPlotter

json_path = "pretrained/weak_10k/metrics/metrics.json"
save_dir = "plots/outputs/weak_10k"

plotter = MetricsPlotter.from_json(json_path)
plotter.save_all(save_dir)

print("Plots saved to:", save_dir)
