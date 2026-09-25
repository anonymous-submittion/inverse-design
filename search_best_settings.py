import json

import numpy as np

from config import CONFIG, RESULTS_DIR
from common import build_problem, train


def search(problem, cfg, parameterized, candidates, filename):
    table, best, best_D1 = [], None, np.inf
    for settings in candidates:
        row = dict(settings)
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                w, _ = train(problem, cfg, parameterized, **settings)
            score = float(np.abs(w - problem.w_star).sum())
            row.update(D1=score, status="ok")
            if score < best_D1:
                best, best_D1 = dict(settings), score
        except FloatingPointError as error:
            row.update(D1=None, status=str(error))
        table.append(row)
        print(row, flush=True)
        (RESULTS_DIR / filename).write_text(json.dumps(table, indent=2))
    if best is None:
        raise RuntimeError(f"All candidates in {filename} failed. Revise the learning-rate grid.")
    print("Best settings:", best, "raw D1:", best_D1)
    return best, best_D1


def main():
    cfg = CONFIG
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    problem = build_problem(cfg)
    print(f"n={problem.n}, d={problem.d}, N={problem.N}, actual N/d={problem.N / problem.d:.6f}, "
          f"test size={len(problem.y_test)}")

    regularized_candidates = [
        dict(batch_size=cfg["regularized_batch_size"], lam=lam, lr=lr,
             fresh_noise=cfg["regularized_fresh_noise"], steps=cfg["regularized_steps"])
        for lam in sorted(set(cfg["lambdas"]))
        for lr in sorted(set(cfg["regularized_lrs"]))
    ]
    print(f"\n=== Part 1: regularized model ({len(regularized_candidates)} candidates) ===")
    best_reg, score_reg = search(problem, cfg, False, regularized_candidates, "regularized_search.json")

    parameterized_candidates = [
        dict(batch_size=cfg["parameterized_batch_size"], lr=lr, fresh_noise=noise,
             steps=cfg["parameterized_steps"])
        for lr in sorted(set(cfg["parameterized_lrs"]))
        for noise in sorted(set(cfg["parameterized_noises"]))
    ]
    print(f"\n=== Part 2: parameterized model ({len(parameterized_candidates)} candidates) ===")
    best_bin, score_bin = search(problem, cfg, True, parameterized_candidates, "parameterized_search.json")

    vanilla_candidates = [
        dict(lam=0.0, lr=lr, fresh_noise=cfg["vanilla_fresh_noise"],
             steps=cfg["vanilla_steps"], batch_size=cfg["vanilla_batch_size"])
        for lr in sorted(set(cfg["vanilla_lrs"]))
    ]
    print(f"\n=== Part 3: vanilla regression ({len(vanilla_candidates)} candidates) ===")
    best_van, score_van = search(problem, cfg, False, vanilla_candidates, "vanilla_search.json")

    best_settings = dict(
        regularized=dict(settings=best_reg, D1=score_reg),
        parameterized=dict(settings=best_bin, D1=score_bin),
        vanilla=dict(settings=best_van, D1=score_van),
    )
    (RESULTS_DIR / "best_settings.json").write_text(json.dumps(best_settings, indent=2))
    print(f"\nSaved winning settings for all three models to {RESULTS_DIR / 'best_settings.json'}")


if __name__ == "__main__":
    main()
