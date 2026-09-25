from dataclasses import dataclass

import numpy as np
import torch

torch.set_default_dtype(torch.float64) 


def h(a, b):
    return (3 * a * a * b - b ** 3 + 3 * b) / 2


def charge(a, b):
    return a ** 3 - 3 * a * b * b + 3 * a


@dataclass
class Problem:
    n: int
    d: int
    N: int
    ii: np.ndarray
    jj: np.ndarray
    w_star: np.ndarray
    X: np.ndarray
    y: np.ndarray
    y_clean: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    a0: np.ndarray
    b0: np.ndarray
    w0: np.ndarray
    X_t: torch.Tensor
    y_t: torch.Tensor
    y_clean_t: torch.Tensor
    X_test_t: torch.Tensor
    y_test_t: torch.Tensor
    w_star_t: torch.Tensor

    def matrix(self, w):
        J = np.zeros((self.n, self.n))
        J[self.ii, self.jj] = J[self.jj, self.ii] = w
        return J


def build_problem(cfg: dict) -> Problem:
    n = cfg["n_spins"]
    ii, jj = np.triu_indices(n, k=1)  # count each interaction once
    d = len(ii)
    N = int(np.floor(cfg["ratio"] * d))
    assert 1 <= N < d

    teacher_rng = np.random.default_rng(cfg["teacher_seed"])
    w_star = np.where(teacher_rng.random(d) < 0.5, 1.0, -1.0)
    data_rng = np.random.default_rng(cfg["data_seed"])

    def design(count):
        s = data_rng.choice([-1.0, 1.0], size=(count, n))
        return s[:, ii] * s[:, jj] / np.sqrt(d)

    X = design(N)
    X_test = design(cfg["n_test"])
    y_clean = X @ w_star
    y = y_clean + cfg["acquisition_noise"] * data_rng.standard_normal(N)
    y_test = X_test @ w_star

    init_rng = np.random.default_rng(cfg["init_seed"])
    a0 = init_rng.normal(cfg["a_mean"], cfg["a_std"], d)
    b0 = init_rng.normal(0, cfg["b_std"], d)
    w0 = h(a0, b0)  

    return Problem(
        n=n, d=d, N=N, ii=ii, jj=jj, w_star=w_star,
        X=X, y=y, y_clean=y_clean, X_test=X_test, y_test=y_test,
        a0=a0, b0=b0, w0=w0,
        X_t=torch.as_tensor(X), y_t=torch.as_tensor(y),
        y_clean_t=torch.as_tensor(y_clean), X_test_t=torch.as_tensor(X_test),
        y_test_t=torch.as_tensor(y_test), w_star_t=torch.as_tensor(w_star),
    )


def train(problem: Problem, cfg: dict, parameterized, lr, fresh_noise, steps, batch_size,
          lam=0.0, record=False, training_seed=None):
    ts = cfg["training_seed"] if training_seed is None else training_seed
    batches = np.random.default_rng(ts)
    noises = np.random.default_rng(ts + 1)

    if parameterized:
        a = torch.tensor(problem.a0, requires_grad=True)
        b = torch.tensor(problem.b0, requires_grad=True)
        optimizer = torch.optim.SGD([a, b], lr=lr)
        current_w = lambda: h(a, b)
    else:
        w_param = torch.tensor(problem.w0, requires_grad=True)
        optimizer = torch.optim.SGD([w_param], lr=lr)
        current_w = lambda: w_param

    history = {key: [] for key in ("step", "train_mse", "train_observed_mse", "test_mse", "D1",
                                    "predictor_rms", "charge_rms", "mean_abs_u", "mean_abs_v",
                                    "sample_u", "sample_v")}
    for key in ("ratio_step", "ratio_count", "delta_charge", "delta_predictor",
                "change_ratio", "ratio_reliable"):
        history[key] = []
    measure_ratio = record and parameterized
    if measure_ratio:
        with torch.no_grad():
            previous_charge, previous_w = charge(a, b).clone(), current_w().clone()
            start_charge, start_w = previous_charge.clone(), previous_w.clone()
            charge_sum, predictor_sum, window_count = 0.0, 0.0, 0

    def log(step):
        with torch.no_grad():
            w_now = current_w()
            history["step"].append(step)
            history["train_mse"].append(torch.mean((problem.X_t @ w_now - problem.y_clean_t) ** 2).item())
            history["train_observed_mse"].append(torch.mean((problem.X_t @ w_now - problem.y_t) ** 2).item())
            history["test_mse"].append(torch.mean((problem.X_test_t @ w_now - problem.y_test_t) ** 2).item())
            history["D1"].append(torch.abs(w_now - problem.w_star_t).sum().item())
            history["predictor_rms"].append(torch.sqrt(torch.mean(w_now * w_now)).item())
            history["charge_rms"].append(
                torch.sqrt(torch.mean(charge(a, b) ** 2)).item() if parameterized else np.nan)
            if parameterized:
                history["mean_abs_u"].append(torch.mean(torch.abs(a)).item())
                history["mean_abs_v"].append(torch.mean(torch.abs(b)).item())
                history["sample_u"].append(a.detach().numpy()[:10].copy())
                history["sample_v"].append(b.detach().numpy()[:10].copy())
            else:
                history["mean_abs_u"].append(np.nan)
                history["mean_abs_v"].append(np.nan)
                history["sample_u"].append(np.full(10, np.nan))
                history["sample_v"].append(np.full(10, np.nan))

    if record:
        log(0)
    for step in range(1, steps + 1):
        idx = (np.arange(problem.N) if batch_size == 0 else
               batches.integers(problem.N, size=batch_size))
        idx_t = torch.as_tensor(idx)
        xb = problem.X_t[idx_t]
        target = problem.y_t[idx_t] + fresh_noise * torch.as_tensor(noises.standard_normal(len(idx)))

        optimizer.zero_grad(set_to_none=True)
        w_now = current_w()
        loss = 0.5 * torch.mean((xb @ w_now - target) ** 2)  # half MSE
        if not parameterized:
            loss = loss + (lam / problem.d) * torch.sum((w_now ** 2 - 1) ** 2)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            if parameterized:
                if (not torch.isfinite(a).all() or not torch.isfinite(b).all()
                        or max(a.abs().max().item(), b.abs().max().item()) > cfg["latent_limit"]):
                    raise FloatingPointError(f"Latent instability at step {step}")
                w_now = h(a, b)
            else:
                w_now = w_param
            if not torch.isfinite(w_now).all() or w_now.abs().max().item() > cfg["weight_limit"]:
                raise FloatingPointError(f"Weight instability at step {step}")

            if measure_ratio:
                current_charge = charge(a, b)
                charge_sum += torch.linalg.vector_norm(current_charge - previous_charge).item()
                predictor_sum += torch.linalg.vector_norm(w_now - previous_w).item()
                window_count += 1
                previous_charge, previous_w = current_charge.clone(), w_now.clone()
                if window_count == cfg["ratio_window"] or step == steps:
                    if cfg["ratio_mode"] == "mean_step":
                        dc = charge_sum / window_count
                        dw = predictor_sum / window_count
                    else:
                        dc = torch.linalg.vector_norm(current_charge - start_charge).item()
                        dw = torch.linalg.vector_norm(w_now - start_w).item()
                    ratio = dc / (dw + cfg["ratio_epsilon"])
                    for key, value in dict(ratio_step=step, ratio_count=window_count,
                                            delta_charge=dc, delta_predictor=dw, change_ratio=ratio,
                                            ratio_reliable=dw > cfg["ratio_epsilon"]).items():
                        history[key].append(value)
                    start_charge, start_w = current_charge.clone(), w_now.clone()
                    charge_sum, predictor_sum, window_count = 0.0, 0.0, 0

        if record and (step % cfg["log_every"] == 0 or step == steps):
            log(step)

    with torch.no_grad():
        w_final = current_w().detach().numpy().copy()
    return w_final, {key: np.asarray(values) for key, values in history.items()}
