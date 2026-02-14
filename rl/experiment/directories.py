import os
import datetime

def create_cluster_run_dirs(run_name, base_dir):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    run_dir = os.path.join(base_dir, "cluster_runs", f"{timestamp}_{run_name}")

    subdirs = {
        "logs": os.path.join(run_dir, "logs"),
        "models": os.path.join(run_dir, "models"),
        "metrics": os.path.join(run_dir, "metrics"),
        "plots": os.path.join(run_dir, "plots"),
        "config": os.path.join(run_dir, "config"),
    }

    for d in subdirs.values():
        os.makedirs(d, exist_ok=True)

    return subdirs
