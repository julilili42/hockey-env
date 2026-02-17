STAGE_1 = [
    (1.00, 0.00, 1.00, 0.00),
]

STAGE_2 = [
    (0.33, 0.55, 0.45, 0.00),
    (0.66, 0.45, 0.45, 0.10),
    (1.00, 0.50, 0.40, 0.10),
]

STAGE_3 = [
    (0.15, 0.30, 0.70, 0.00),
    (0.70, 0.60, 0.30, 0.10),
    (1.00, 0.35, 0.35, 0.30),
]

STAGE_4 = [
    (1.0, 0.2, 0.2, 0.6),
]


NOISE_STUDY = [
    (1.0, 0.5, 0.5, 0.0),
]



ABLATION = STAGE_2 


CURRICULA = {
    "stage1": STAGE_1,
    "stage2": STAGE_2,
    "stage3": STAGE_3,
    "stage4": STAGE_4,
    "ablation": ABLATION,
    "noise_study": NOISE_STUDY
}
