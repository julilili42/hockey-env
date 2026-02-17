from rl.experiment.scheduler import Experiment
import os


def get_pretrained_path(name):
    base = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    return os.path.join(base, "pretrained", name)


def noise_study(seed):
    noises = ["gaussian", "pink", "ornstein-uhlenbeck", "uniform"]
    exps = []

    for noise in noises:
        exps.append(
            Experiment(
                mode="single",
                episodes=10_000,
                resume_from=None,     
                seed=seed,
                overrides=dict(
                    curriculum_name="noise_study",
                    noise_mode=noise,
                    prioritized_replay=False,
                    use_self_play=False,
                    use_noise_annealing=True,
                ),
            )
        )
    return exps



def prioritized_selfplay_study(seed):
    pretrained = get_pretrained_path("weak_10k/models/td3_best.pt")
    exps = []

    common = dict(
        curriculum_name="ablation",
        noise_mode="ornstein-uhlenbeck",
        use_noise_annealing=True,
    )

    configs = [
        (False, False),  # Baseline
        (True,  False),  # PER
        (False, True),   # SelfPlay
        (True,  True),   # PER + SelfPlay
    ]

    for prio, sp in configs:
        exps.append(
            Experiment(
                mode="single",
                episodes=10_000,
                resume_from=pretrained,
                seed=seed,
                overrides=dict(
                    **common,
                    prioritized_replay=prio,
                    use_self_play=sp,
                ),
            )
        )

    return exps

