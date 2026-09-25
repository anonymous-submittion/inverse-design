import json

import numpy as np
from safetensors.numpy import save_file

from config import CONFIG, RESULTS_DIR, BEST_SETTINGS_PATH, RESULTS_JSON_PATH, WEIGHTS_PATH, HISTORIES_PATH
from common import build_problem, train


def run_repeats(problem, cfg, parameterized, settings, n_rep):
    weights, histories = [], []
    base_seed = cfg["training_seed"]
    for r in range(n_rep):
        w, hist = train(problem, cfg, parameterized, **settings, record=True, training_seed=base_seed + r)
        weights.append(w)
        histories.append(hist)
        print(f"  repeat {r + 1}/{n_rep} done (final D1={np.abs(w - problem.w_star).sum():.6g})")
    return weights, histories


def flatten_histories(name, histories):
    flat = {}
    for r, hist in enumerate(histories):
        for key, values in hist.items():
            flat[f"{name}_{key}_r{r}"] = np.asarray(values, dtype=np.float64)
    return flat


def summarize(name, weights, histories, w_star, n_rep):
    D1s = np.array([np.abs(w - w_star).sum() for w in weights])
    MAEs = np.array([np.abs(w - w_star).mean() for w in weights])
    clean_train = np.array([hist["train_mse"][-1] for hist in histories])
    observed_train = np.array([hist["train_observed_mse"][-1] for hist in histories])
    test = np.array([hist["test_mse"][-1] for hist in histories])
    return dict(
        n_repeats=n_rep,
        D1_mean=float(D1s.mean()), D1_std=float(D1s.std()),
        MAE_mean=float(MAEs.mean()), MAE_std=float(MAEs.std()),
        clean_train_mse_mean=float(clean_train.mean()), clean_train_mse_std=float(clean_train.std()),
        observed_train_mse_mean=float(observed_train.mean()), observed_train_mse_std=float(observed_train.std()),
        test_mse_mean=float(test.mean()), test_mse_std=float(test.std()),
    )


def main():
    cfg = CONFIG
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    problem = build_problem(cfg)
    best_settings = json.loads(BEST_SETTINGS_PATH.read_text())
    n_rep = cfg["n_repeats"]

    print(f"Retraining each winning configuration {n_rep} times...")
    print("Parameterized:")
    W_bin, H_bin = run_repeats(problem, cfg, True, best_settings["parameterized"]["settings"], n_rep)
    print("Regularized:")
    W_reg, H_reg = run_repeats(problem, cfg, False, best_settings["regularized"]["settings"], n_rep)
    print("Vanilla:")
    W_van, H_van = run_repeats(problem, cfg, False, best_settings["vanilla"]["settings"], n_rep)

    # Repeat 0 uses training_seed = cfg["training_seed"], exactly like the
    # search stage, so it must reproduce the score that selected these settings.
    assert np.isclose(np.abs(W_reg[0] - problem.w_star).sum(),
                       best_settings["regularized"]["D1"], rtol=1e-10, atol=1e-10)
    assert np.isclose(np.abs(W_bin[0] - problem.w_star).sum(),
                       best_settings["parameterized"]["D1"], rtol=1e-10, atol=1e-10)
    assert np.isclose(np.abs(W_van[0] - problem.w_star).sum(),
                       best_settings["vanilla"]["D1"], rtol=1e-10, atol=1e-10)

    metrics = {
        "parameterized": summarize("parameterized", W_bin, H_bin, problem.w_star, n_rep),
        "regularized": summarize("regularized", W_reg, H_reg, problem.w_star, n_rep),
        "vanilla": summarize("vanilla", W_van, H_van, problem.w_star, n_rep),
    }
    for name, m in metrics.items():
        print(name, m)

    results = dict(
        config=cfg, d=problem.d, N=problem.N, actual_ratio=problem.N / problem.d,
        selected=best_settings, selection="final raw oracle D1 (repeat 0)", metrics=metrics,
    )
    RESULTS_JSON_PATH.write_text(json.dumps(results, indent=2))

    weight_tensors = dict(
        teacher=problem.w_star.astype(np.float64),
        parameterized=np.stack(W_bin).astype(np.float64),   # (n_repeats, d)
        regularized=np.stack(W_reg).astype(np.float64),
        vanilla=np.stack(W_van).astype(np.float64),
    )
    save_file(weight_tensors, WEIGHTS_PATH)

    history_tensors = {}
    history_tensors.update(flatten_histories("parameterized", H_bin))
    history_tensors.update(flatten_histories("regularized", H_reg))
    history_tensors.update(flatten_histories("vanilla", H_van))
    save_file(history_tensors, HISTORIES_PATH)

    print(f"\nSaved {RESULTS_JSON_PATH.name}, {WEIGHTS_PATH.name} and {HISTORIES_PATH.name} to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
