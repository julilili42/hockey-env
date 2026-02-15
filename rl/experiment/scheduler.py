from dataclasses import dataclass, field
from typing import Optional, Dict, List
import json
from rl.td3.config import TD3Config

@dataclass
class Experiment:
    mode: str
    eval_vs_weak: bool
    episodes: int
    hidden_size: int = 256
    resume_from: Optional[str] = None
    seed: int = 42
    overrides: Dict = field(default_factory=dict)


class ExperimentScheduler:
    def __init__(self):
        self.experiments: List[Experiment] = []

    def add(self, experiment: Experiment):
        self.experiments.append(experiment)

    def run_all(self):
        for i, exp in enumerate(self.experiments, start=1):
            print("\n" + "=" * 60)
            print(f"Running experiment {i}/{len(self.experiments)}")
            print("=" * 60)

            self._run_single(exp)

    def _run_single(self, exp: Experiment):
        from rl.main import run_experiment  

        print("\n" + "-" * 60)
        print("Experiment configuration:")
        print(json.dumps(exp.__dict__, indent=4))
        print("-" * 60)

        if exp.mode == "single":
            config = TD3Config.single()
        elif exp.mode == "joint":
            config = TD3Config.joint()
        else:
            raise ValueError("Unknown mode")

        for key, value in exp.overrides.items():
            if not hasattr(config, key):
                raise ValueError(f"Invalid config override: {key}")
            setattr(config, key, value)

        run_experiment(
            mode=exp.mode,
            eval_vs_weak=exp.eval_vs_weak,
            episodes=exp.episodes,
            hidden_size=exp.hidden_size,
            resume_from=exp.resume_from,
            seed=exp.seed,
            external_config=config,
        )

