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



def stage1(seed):
    return [
        Experiment(
            mode="single",
            episodes=10_000,
            resume_from=None,
            seed=seed,
            overrides=dict(
                curriculum_name="stage1",

                use_self_play=False,
                prioritized_replay=False,

                noise_mode="gaussian",
                use_noise_annealing=True,

                lr_q=4e-4,
                lr_pol=4e-4,
            ),
        )
    ]


def stage2(seed):
    pretrained = get_pretrained_path("stage_1/models/td3_best.pt")

    return [
        Experiment(
            mode="single",
            episodes=15_000,
            resume_from=pretrained,
            seed=seed,
            overrides=dict(
                curriculum_name="stage2",

                use_self_play=False,
                prioritized_replay=False,

                lr_q=3e-4,
                lr_pol=3e-4,

                noise_min_scale=0.06,
            ),
        )
    ]


def stage3(seed):
    pretrained = get_pretrained_path("stage_2/models/td3_best.pt")

    return [
        Experiment(
            mode="single",
            episodes=20_000,
            resume_from=pretrained,
            seed=seed,
            overrides=dict(
                curriculum_name="stage3",

                use_self_play=True,
                self_play_interval=150,
                self_play_pool_size=25,

                prioritized_replay=False,

                lr_q=2.5e-4,
                lr_pol=2.5e-4,

                noise_min_scale=0.05,
            ),
        )
    ]


def stage4(seed):
    pretrained = get_pretrained_path("stage_3/models/td3_best.pt")

    return [
        Experiment(
            mode="single",
            episodes=20_000,
            resume_from=pretrained,
            seed=seed,
            overrides=dict(
                curriculum_name="stage4",

                use_self_play=True,
                self_play_interval=100,
                self_play_pool_size=40,

                prioritized_replay=False,

                lr_q=2e-4,
                lr_pol=2e-4,

                noise_min_scale=0.05,
            ),
        )
    ]
