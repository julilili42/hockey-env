from rl.experiment.scheduler import Experiment
import os


def per_experiments():
    return [
        Experiment(
            mode="single",
            episodes=10_000,
            overrides=dict(prioritized_replay=True),
            seed=s,
        )
        for s in [1, 2, 3]
    ] + [
        Experiment(
            mode="single",
            episodes=10_000,
            overrides=dict(prioritized_replay=False),
            seed=s,
        )
        for s in [1, 2, 3]
    ]


def noise_experiments():
    noises = ["gaussian", "pink", "ornstein-uhlenbeck"]
    exps = []

    for noise in noises:
        for seed in [1, 2, 3]:
            exps.append(
                Experiment(
                    mode="single",
                    episodes=10_000,
                    overrides=dict(noise_mode=noise),
                    seed=seed,
                )
            )
    return exps


def pretrained_vs_scratch():
    pretrained = get_pretrained_path("weak/td3_weak_best.pt")

    return [
        Experiment(
            mode="single",
            episodes=10_000,
            resume_from=pretrained,
            seed=1,
        ),
        Experiment(
            mode="single",
            episodes=10_000,
            resume_from=None,
            seed=1,
        )
    ]


def noise_annealing_experiments():
    exps = []

    # --- Baseline (kein Annealing) ---
    for seed in [1, 2, 3]:
        exps.append(
            Experiment(
                mode="single",
                episodes=10_000,
                overrides=dict(
                    use_noise_annealing=False
                ),
                seed=seed,
            )
        )

    # --- Mit Annealing ---
    for seed in [1, 2, 3]:
        exps.append(
            Experiment(
                mode="single",
                episodes=10_000,
                overrides=dict(
                    use_noise_annealing=True,
                    noise_anneal_mode="linear",
                    noise_min_scale=0.05,
                ),
                seed=seed,
            )
        )

    return exps



def get_pretrained_path(name):
    base = os.path.dirname(os.path.dirname(__file__))
    return os.path.join(base, "pretrained", name)
