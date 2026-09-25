from pathlib import Path

import numpy as np

CONFIG = dict(
    n_spins=60, ratio=0.6, n_test=2048,
    teacher_seed=7, data_seed=17, init_seed=27, training_seed=7,
    a_mean=0.10, a_std=0.02, b_std=0.0,
    acquisition_noise=0.0,
    regularized_batch_size=1,
    parameterized_batch_size=1,
    vanilla_batch_size=1,
    lambdas=[0.01, 0.1, 0.3, 0.6, 1],
    regularized_lrs=[0.1, 0.9, 1.2],
    regularized_fresh_noise=0.0,
    regularized_steps=300_000,
    parameterized_lrs=[0.1, 0.6, 0.9],
    parameterized_noises=[0.6, 0.7, 0.8],
    parameterized_steps=300_000,

    vanilla_lrs=[0.1, 0.6, 0.9, 1.2],
    vanilla_fresh_noise=0.0,
    vanilla_steps=300_000,

    n_repeats=10,

    log_every=1000,
    ratio_window=1000,
    ratio_mode="mean_step",
    ratio_epsilon=1e-12,
    ratio_mask_small=True,
    plot_clean_train=True,
    latent_limit=20.0, weight_limit=100.0,
    hist_bins=61,
)


def _validate(cfg):
    assert cfg["n_spins"] >= 3 and 0 < cfg["ratio"] < 1
    for key in ("regularized_batch_size", "parameterized_batch_size", "vanilla_batch_size"):
        assert isinstance(cfg[key], int) and cfg[key] >= 0
    assert cfg["n_test"] > 0 and cfg["log_every"] > 0
    for key in ("regularized_lrs", "parameterized_lrs", "vanilla_lrs"):
        assert cfg[key] and all(np.isfinite(v) and v > 0 for v in cfg[key])
    for key in ("lambdas", "parameterized_noises"):
        assert cfg[key] and all(np.isfinite(v) and v >= 0 for v in cfg[key])
    assert min(cfg["acquisition_noise"], cfg["regularized_fresh_noise"],
               cfg["vanilla_fresh_noise"], cfg["a_std"], cfg["b_std"]) >= 0
    assert min(cfg["regularized_steps"], cfg["parameterized_steps"], cfg["vanilla_steps"]) > 0
    assert isinstance(cfg["ratio_window"], int) and cfg["ratio_window"] >= 1
    assert cfg["ratio_mode"] in ("mean_step", "endpoint")
    assert np.isfinite(cfg["ratio_epsilon"]) and cfg["ratio_epsilon"] > 0
    assert isinstance(cfg["n_repeats"], int) and cfg["n_repeats"] >= 1


_validate(CONFIG)

LOG_HISTORY_KEYS = (
    "step", "train_mse", "train_observed_mse", "test_mse", "D1",
    "predictor_rms", "charge_rms", "mean_abs_u", "mean_abs_v",
    "sample_u", "sample_v",
)
RATIO_HISTORY_KEYS = (
    "ratio_step", "ratio_count", "delta_charge", "delta_predictor",
    "change_ratio", "ratio_reliable",
)
HISTORY_KEYS = LOG_HISTORY_KEYS + RATIO_HISTORY_KEYS

REPO_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

BEST_SETTINGS_PATH = RESULTS_DIR / "best_settings.json"
RESULTS_JSON_PATH = RESULTS_DIR / "results.json"
WEIGHTS_PATH = RESULTS_DIR / "weights.safetensors"
HISTORIES_PATH = RESULTS_DIR / "histories.safetensors"

PLOT = dict(
    dpi=300,
    combined_size=(20, 23),
    separate_size=(7, 6),
    font_size=15,
    label_size=18,
    title_size=19,
    legend_size=12,
    line_width=3.0,
    tick_count=4,
    mse_floor=1e-16,
    charge_floor=1e-16,
    field_limit=2.0,
    field_grid=501,
    histogram_bins=CONFIG["hist_bins"],
)

MODEL_COLORS = dict(
    parameterized="#0072B2",
    regularized="#D55E00",
    vanilla="#009E73",
)
CHARGE_COLOR = "#7B3294"
MODEL_LABELS = dict(parameterized="Parameterized", regularized="Regularized", vanilla="Vanilla")
MODEL_ORDER = ("parameterized", "regularized", "vanilla")
