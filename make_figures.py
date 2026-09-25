import json

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, ScalarFormatter, NullLocator, LogLocator
from safetensors.numpy import load_file

from config import (CONFIG, PLOT, MODEL_COLORS, MODEL_LABELS, MODEL_ORDER, CHARGE_COLOR,
                     HISTORY_KEYS, FIGURES_DIR, RESULTS_JSON_PATH, WEIGHTS_PATH, HISTORIES_PATH)
from common import build_problem, h, charge

MODEL_LABELS = dict(MODEL_LABELS, parameterized="Parametrized")

plt.rcParams.update({
    "font.size": PLOT["font_size"],
    "axes.labelsize": PLOT["label_size"],
    "axes.titlesize": PLOT["title_size"],
    "axes.titleweight": "bold",
    "axes.linewidth": 1.3,
    "lines.linewidth": PLOT["line_width"],
    "xtick.labelsize": PLOT["font_size"],
    "ytick.labelsize": PLOT["font_size"],
    "xtick.major.size": 6,
    "ytick.major.size": 6,
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

def load_results():
    results = json.loads(RESULTS_JSON_PATH.read_text())
    saved_cfg = results["config"]
    if saved_cfg != CONFIG:
        print("WARNING: config.py has changed since these results were produced; "
              "figures will still be drawn from the saved arrays, but rerun "
              "search_best_settings.py + run_final_experiment.py to refresh them.")

    weights = load_file(str(WEIGHTS_PATH))
    flat_histories = load_file(str(HISTORIES_PATH))
    n_rep = saved_cfg["n_repeats"]

    def unflatten(name):
        return [{key: flat_histories[f"{name}_{key}_r{r}"] for key in HISTORY_KEYS}
                for r in range(n_rep)]

    model_repeats = {
        name: dict(W=list(weights[name]), H=unflatten(name), color=MODEL_COLORS[name])
        for name in MODEL_ORDER
    }
    problem = build_problem(saved_cfg)
    assert np.allclose(problem.w_star, weights["teacher"]), (
        "Saved teacher weights don't match config.py's seeds -- results.json "
        "and config.py are out of sync."
    )
    return saved_cfg, problem, model_repeats


def style_axes(ax, grid=True):
    ax.xaxis.set_major_locator(MaxNLocator(nbins=PLOT["tick_count"]))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=PLOT["tick_count"]))
    ax.minorticks_off()
    ax.set_axisbelow(True)
    if grid:
        ax.grid(alpha=0.15, linewidth=0.8)
    ax.tick_params(pad=6)


def inside_legend(ax, loc="best", **kwargs):
    return ax.legend(loc=loc, fontsize=PLOT["legend_size"],
                      frameon=True, facecolor="white", framealpha=0.9,
                      edgecolor="none", **kwargs)


def export_figure(fig, name):
    for extension in ("png", "pdf"):
        fig.savefig(FIGURES_DIR / f"{name}.{extension}",
                    dpi=PLOT["dpi"], bbox_inches="tight", facecolor="white")
    plt.close(fig)


def draw_fields(ax, title="Binary parametrization"):
    limit = PLOT["field_limit"]
    grid = np.linspace(-limit, limit, PLOT["field_grid"])
    U, V = np.meshgrid(grid, grid)
    H = (3 * U ** 2 * V - V ** 3 + 3 * V) / 2
    CHI = U ** 3 - 3 * U * V ** 2 + 3 * U
    blue, orange = MODEL_COLORS["parameterized"], MODEL_COLORS["regularized"]
    ax.contour(U, V, H, levels=[-2, -1.5, -1, -0.5, -0.2, 0, 0.2, 0.5, 1, 1.5, 2],
               colors=blue, linewidths=2.2, linestyles="solid")
    ax.contour(U, V, CHI, levels=[-6, -3, -1.5, -0.5, 0.5, 1.5, 3, 6],
               colors=orange, linewidths=2.2, linestyles="dashed")
    ax.axvline(0, color=orange, lw=3.5)
    branch = np.sqrt(1 + grid ** 2 / 3)
    ax.plot(grid, branch, color=orange, lw=3.5)
    ax.plot(grid, -branch, color=orange, lw=3.5)
    ax.scatter([0, 0], [1, -1], s=85, color="#15202B", edgecolor="white", linewidth=1.5, zorder=5)
    for v, text, offset in [(1, r"$h=+1$", (12, 12)), (-1, r"$h=-1$", (12, -25))]:
        ax.annotate(text, (0, v), xytext=offset, textcoords="offset points", fontsize=16,
                    bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=1.5))
    ax.set(xlim=(-limit, limit), ylim=(-limit, limit), xlabel=r"$u$", ylabel=r"$v$", title=title)
    style_axes(ax)
    ax.set_box_aspect(1)
    handles = [Line2D([], [], color=blue, lw=3, label=r"Predictor $h$"),
               Line2D([], [], color=orange, lw=3, ls="--", label=r"Charge $\chi$"),
               Line2D([], [], color=orange, lw=4, label=r"$\chi=0$")]
    inside_legend(ax, loc="upper left", handles=handles)


def draw_training(ax, cfg, model_repeats, title="Training and test"):
    train_key = "train_mse" if cfg["plot_clean_train"] else "train_observed_mse"
    train_label = "train" if cfg["plot_clean_train"] else "observed train"
    for name in MODEL_ORDER:
        info = model_repeats[name]
        color, label = info["color"], MODEL_LABELS[name]
        steps = info["H"][0]["step"]
        train_stack = np.stack([hist[train_key] for hist in info["H"]])
        test_stack = np.stack([hist["test_mse"] for hist in info["H"]])
        train_mean = np.mean(train_stack, axis=0)
        train_std = np.std(train_stack, axis=0)
        test_mean = np.mean(test_stack, axis=0)
        test_std = np.std(test_stack, axis=0)
        for mean_val, std_val, sub_label, line_style in [
            (train_mean, train_std, train_label, "-"),
            (test_mean, test_std, "test", "--"),
        ]:
            display_mean = np.maximum(mean_val, PLOT["mse_floor"])
            ax.plot(steps, display_mean, color=color, ls=line_style,
                    label=f"{label}: {sub_label}")
            ax.fill_between(steps, np.maximum(mean_val - std_val, PLOT["mse_floor"]),
                            mean_val + std_val, color=color, alpha=0.15, linewidth=0)
    ax.set(xlabel="SGD update", ylabel="Prediction MSE (mean over repeats)", title=title)
    style_axes(ax)
    ax.set_yscale("log")
    ax.set_yticks([1e0, 1e-4, 1e-8, 1e-12, 1e-16])
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_box_aspect(1)
    inside_legend(ax, loc="upper right")


def draw_matrices(axes, cax, problem, model_repeats, titles=None, repeat_idx=0):
    if titles is None:
        titles = ["Ground truth", "Parametrized", "Regularized", "Vanilla"]
    repeat0 = {name: model_repeats[name]["W"][repeat_idx] for name in MODEL_ORDER}
    all_w = [problem.w_star] + [repeat0[name] for name in MODEL_ORDER]
    limit = max(1.0, *(float(np.max(np.abs(w))) for w in all_w))
    for ax, weights, title in zip(axes, all_w, titles):
        image = ax.imshow(problem.matrix(weights), cmap="gray", vmin=-limit, vmax=limit,
                           interpolation="nearest")
        ax.set(xlabel="Spin j", ylabel="Spin i", title=title)
        style_axes(ax, grid=False)
        ticks = np.unique(np.linspace(0, problem.n - 1, 4).astype(int))
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
    colorbar = axes[0].figure.colorbar(image, cax=cax)
    colorbar.set_label("Raw interaction")
    colorbar.set_ticks([-limit, 0, limit])


def draw_ratio(ax, cfg, model_repeats, title="Charge and predictor changes"):
    ratio_stack = []
    for hist in model_repeats["parameterized"]["H"]:
        ratios = np.asarray(hist["change_ratio"], dtype=float).copy()
        reliable = np.asarray(hist["ratio_reliable"], dtype=bool)
        if cfg["ratio_mask_small"]:
            ratios[~reliable] = np.nan
        ratio_stack.append(ratios)
    ratio_step = model_repeats["parameterized"]["H"][0]["ratio_step"]
    stacked = np.stack(ratio_stack)
    with np.errstate(invalid="ignore"):
        mean_ratio = np.nanmean(stacked, axis=0)
        std_ratio = np.nanstd(stacked, axis=0)
    ax.plot(ratio_step, mean_ratio, color=CHARGE_COLOR, label=r"$\Delta\chi/(\Delta w+\varepsilon)$ (mean)")
    ax.fill_between(ratio_step, mean_ratio - std_ratio, mean_ratio + std_ratio,
                    color=CHARGE_COLOR, alpha=0.15, linewidth=0,
                    label=r"$\pm 1$ std over repeats")
    ax.set(xlabel="SGD update", ylabel="Charge / predictor change", title=title)
    style_axes(ax)
    ax.margins(y=0.08)
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-2, 2))
    ax.yaxis.set_major_formatter(formatter)
    inside_legend(ax, loc="upper right")


def draw_histogram(ax, w_star, model_repeats, errors=False, title=None, repeat_idx=0):
    pooled = []
    for name in MODEL_ORDER:
        info = model_repeats[name]
        single_w = info["W"][repeat_idx]
        if errors:
            values = w_star - single_w
        else:
            values = single_w
        pooled.append((MODEL_LABELS[name], values, info["color"]))
    number_of_bins = int(PLOT["histogram_bins"])
    if number_of_bins % 2 == 0:  # an odd bin count puts zero in the middle of a bin
        number_of_bins += 1
    limit = max(2.2 if errors else 1.5, 1.05 * max(float(np.max(np.abs(values))) for _, values, _ in pooled))
    edges = np.linspace(-limit, limit, number_of_bins + 1)
    for (label, values, color), ls in zip(pooled, ["-", "--", "-."]):
        fractions, _ = np.histogram(values, bins=edges)
        fractions = fractions / len(values)
        ax.stairs(fractions, edges, fill=True, color=color, alpha=0.12)
        ax.stairs(fractions, edges, color=color, lw=PLOT["line_width"], linestyle=ls, label=label)
    for x in ([-2, 0, 2] if errors else [-1, 1]):
        ax.axvline(x, color="0.5", ls=":", lw=1.5, zorder=0)
    ax.set(xlabel="Teacher - learned interaction" if errors else "Learned interaction",
           ylabel="Fraction of couplings per bin",
           title=title or (("Recovery errors" if errors else "Learned weights")
                            + f" (repeat {repeat_idx})"))
    style_axes(ax)
    ax.set_ylim(bottom=0)
    ax.margins(y=0.18)
    inside_legend(ax, loc="upper right")


def draw_charge(ax, model_repeats, title="Charge convergence"):
    charge_stack = np.stack([hist["charge_rms"] for hist in model_repeats["parameterized"]["H"]])
    steps = model_repeats["parameterized"]["H"][0]["step"]
    mean_values = np.maximum(np.mean(charge_stack, axis=0), PLOT["charge_floor"])
    std_values = np.std(charge_stack, axis=0)
    ax.plot(steps, mean_values, color=CHARGE_COLOR, label=r"$\|\chi\|_2/\sqrt{d}$ (mean)")
    ax.fill_between(steps, np.maximum(mean_values - std_values, PLOT["charge_floor"]),
                     mean_values + std_values, color=CHARGE_COLOR, alpha=0.15, linewidth=0,
                     label=r"$\pm 1$ std over repeats")
    ax.set(xlabel="SGD update", ylabel="Charge RMS", title=title)
    style_axes(ax)
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=5))
    ax.yaxis.set_minor_locator(NullLocator())
    inside_legend(ax, loc="upper right")


def draw_M2(ax, model_repeats, title="Charge second moment"):
    charge_stack = np.stack([hist["charge_rms"] for hist in model_repeats["parameterized"]["H"]])
    steps = model_repeats["parameterized"]["H"][0]["step"]
    m2_stack = charge_stack ** 2
    mean_values = np.maximum(np.mean(m2_stack, axis=0), PLOT["charge_floor"])
    std_values = np.std(m2_stack, axis=0)
    ax.plot(steps, mean_values, color=CHARGE_COLOR, label=r"$M_{2,w}(k)=\langle\chi^2\rangle$ (mean)")
    ax.fill_between(steps, np.maximum(mean_values - std_values, PLOT["charge_floor"]),
                     mean_values + std_values, color=CHARGE_COLOR, alpha=0.15, linewidth=0,
                     label=r"$\pm 1$ std over repeats")
    ax.set(xlabel="SGD update", ylabel="$M_{2,w}$", title=title)
    style_axes(ax)
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=5))
    ax.yaxis.set_minor_locator(NullLocator())
    inside_legend(ax, loc="upper right")


def draw_uv_mean(ax, model_repeats, title="Evolution of Parameter Magnitudes"):
    info = model_repeats["parameterized"]
    steps = info["H"][0]["step"]
    
    mean_abs_u_curve = np.mean([hist["mean_abs_u"] for hist in info["H"]], axis=0)
    mean_abs_v_curve = np.mean([hist["mean_abs_v"] for hist in info["H"]], axis=0)
    
    ax.plot(steps, mean_abs_u_curve, label=r'Mean $|u|$', color='blue', linewidth=PLOT["line_width"])
    ax.plot(steps, mean_abs_v_curve, label=r'Mean $|v|$', color='orange', linewidth=PLOT["line_width"])
    
    ax.axhline(0, color='blue', linestyle='--', alpha=0.5, label=r'Target $|u| = 0$')
    ax.axhline(1, color='orange', linestyle='--', alpha=0.5, label=r'Target $|v| = 1$')
    
    ax.set(xlabel="SGD update", ylabel="Parameter Magnitude", title=title)
    style_axes(ax)
    inside_legend(ax, loc="upper left")


def draw_uv_individual(ax, model_repeats, title="Individual Trajectories"):
    info = model_repeats["parameterized"]
    steps = info["H"][0]["step"]
    
    sample_u = info["H"][0]["sample_u"] 
    sample_v = info["H"][0]["sample_v"]
    
    colors = ['red', 'green', 'purple', 'brown', 'cyan']
    for i in range(5):
        ax.plot(steps, sample_u[:, i], color=colors[i], linestyle='-', linewidth=2, label=f'Coord {i+1}: $u$')
        ax.plot(steps, sample_v[:, i], color=colors[i], linestyle=':', linewidth=2, label=f'Coord {i+1}: $v$')
        
    ax.axhline(0, color='black', linewidth=1)
    ax.axhline(1, color='gray', linestyle='--')
    ax.axhline(-1, color='gray', linestyle='--')
    
    ax.set(xlabel="SGD update", ylabel="Parameter Value", title=title)
    style_axes(ax)
    inside_legend(ax, loc="upper left", bbox_to_anchor=(1.05, 1))


def main():
    cfg, problem, model_repeats = load_results()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 7), layout="constrained")
    draw_fields(ax)
    export_figure(fig, "01_fields_binary")

    fig, ax = plt.subplots(figsize=(7, 7), layout="constrained")
    draw_training(ax, cfg, model_repeats)
    export_figure(fig, "02_training_test")

    fig = plt.figure(figsize=(20, 5.5), layout="constrained")
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 1, 0.045])
    axes = [fig.add_subplot(gs[0, k]) for k in range(4)]
    cax = fig.add_subplot(gs[0, 4])
    draw_matrices(axes, cax, problem, model_repeats)
    export_figure(fig, "03_interaction_matrices")

    fig, ax = plt.subplots(figsize=PLOT["separate_size"], layout="constrained")
    draw_ratio(ax, cfg, model_repeats)
    export_figure(fig, "04_charge_predictor_change_ratio")

    fig, ax = plt.subplots(figsize=PLOT["separate_size"], layout="constrained")
    draw_histogram(ax, problem.w_star, model_repeats)
    export_figure(fig, "05_learned_weights")

    fig, ax = plt.subplots(figsize=PLOT["separate_size"], layout="constrained")
    draw_histogram(ax, problem.w_star, model_repeats, errors=True)
    export_figure(fig, "06_recovery_errors")

    fig, ax = plt.subplots(figsize=PLOT["separate_size"], layout="constrained")
    draw_charge(ax, model_repeats)
    export_figure(fig, "07_charge_convergence")

    fig, ax = plt.subplots(figsize=PLOT["separate_size"], layout="constrained")
    draw_M2(ax, model_repeats)
    export_figure(fig, "08_charge_second_moment")

    fig, ax = plt.subplots(figsize=PLOT["separate_size"], layout="constrained")
    draw_uv_mean(ax, model_repeats)
    export_figure(fig, "09_parameter_evolution_mean")

    fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")
    draw_uv_individual(ax, model_repeats)
    export_figure(fig, "10_parameter_evolution_individual")

    fig = plt.figure(figsize=PLOT["combined_size"], layout="constrained")
    rows = fig.add_gridspec(4, 1, height_ratios=[1.6, 0.85, 1.0, 1.0], hspace=0.08)

    top = rows[0].subgridspec(1, 2, wspace=0.08)
    draw_fields(fig.add_subplot(top[0, 0]), "(a) Binary parametrization")
    draw_training(fig.add_subplot(top[0, 1]), cfg, model_repeats, "(b) Training and test")

    middle = rows[1].subgridspec(1, 5, width_ratios=[1, 1, 1, 1, 0.035], wspace=0.08)
    axes = [fig.add_subplot(middle[0, k]) for k in range(4)]
    cax = fig.add_subplot(middle[0, 4])
    draw_matrices(axes, cax, problem, model_repeats,
                  ["(c) Ground truth", "Parametrized", "Regularized", "Vanilla"])

    diagnostics = rows[2].subgridspec(1, 3, wspace=0.08)
    draw_charge(fig.add_subplot(diagnostics[0, 0]), model_repeats, "(d) Charge convergence")
    draw_M2(fig.add_subplot(diagnostics[0, 1]), model_repeats, "(e) Charge second moment")
    draw_ratio(fig.add_subplot(diagnostics[0, 2]), cfg, model_repeats, "(f) Charge / predictor change")

    histograms = rows[3].subgridspec(1, 2, wspace=0.08)
    draw_histogram(fig.add_subplot(histograms[0, 0]), problem.w_star, model_repeats, title="(g) Learned weights")
    draw_histogram(fig.add_subplot(histograms[0, 1]), problem.w_star, model_repeats, errors=True,
                    title="(h) Recovery errors")

    export_figure(fig, "ising_overview_three_models")
    print(f"All figures saved as PNG and PDF in: {FIGURES_DIR.resolve()}")


if __name__ == "__main__":
    main()
