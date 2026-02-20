from rl.utils.plotter import MetricsPlotter

json_path = "pretrained/stage_1/metrics/metrics.json"
save_dir = "model_evaluation/plots/outputs/final_12k1"

plotter = MetricsPlotter.from_json(json_path)
plotter.save_all(save_dir, show="both")

print("Plots saved to:", save_dir)
