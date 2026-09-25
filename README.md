# Ising Interaction Recovery

This repository contains the source code for the empirical experiments detailed in our work. The code models the recovery of Ising interactions using three distinct approaches:
- **Parameterized Model**
- **Regularized Model** (Double-well penalty: $(w^2 - 1)^2$)
- **Vanilla Regression** (Unregularized)

The codebase is designed for complete reproducibility, relying on fixed pseudo-random number generator (PRNG) seeds for data generation, teacher network initialization, model initialization, and training noise. All configurable parameters (including seeds) are located in `config.py`.

## Requirements

You can easily set up the required environment using [Conda](https://docs.conda.io/en/latest/). We provide an `environment.yml` file with all dependencies.

```bash
conda env create -f environment.yml
conda activate ising_ib
```

If you prefer using `pip` directly, ensure you have Python 3.11+ installed along with:
```bash
pip install numpy>=1.26 pytorch>=2.2 matplotlib>=3.8 safetensors>=0.4
```

## Running the Experiments

The experimental pipeline is divided into three consecutive steps. You must run them in the following order:

### 1. Hyperparameter Search
Run the grid search to find the optimal hyperparameters for each of the three models based on the configurations specified in `config.py`. 

```bash
python search_best_settings.py
```
This script evaluates different combinations of learning rates, noise levels, and regularization strengths. It saves the search history and the selected configurations in `results/best_settings.json`.

### 2. Final Training & Evaluation
Once the optimal hyperparameters are found, run the final experiment. This will train each model multiple times (defined by `n_repeats` in `config.py`) to estimate the mean and standard deviation of the recovery metrics.

```bash
python run_final_experiment.py
```
This script will output performance metrics to the console and save the trained weights, learning trajectories (histories), and final summary metrics to `results/`.

### 3. Generate Figures
Finally, you can recreate the figures presented in the paper using the saved outputs from the final evaluation.

```bash
python make_figures.py
```
The figures will be saved in `results/figures/` in both PDF and PNG formats, visually comparing the parameterized model against the regularized and vanilla baselines.

## Note on Reproducibility

To ensure identical results across different machines, all seeds are explicitly set in `config.py`:
- `teacher_seed`: Used for generating the target parameters (w*).
- `data_seed`: Used to sample the dataset.
- `init_seed`: Controls the initial weights of the networks.
- `training_seed`: Sets the base seed for stochastic training dynamics (e.g., stochastic noise).