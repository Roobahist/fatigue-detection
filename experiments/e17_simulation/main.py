from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from tqdm.auto import tqdm

Array = np.ndarray


@dataclass(frozen=True)
class SimulationConfig:
    n: int
    p_signal: int
    p_noise: int
    signal_strength: float
    noise_sd: float
    n_components: int = 5
    recovery_threshold: float = 0.7


@dataclass(frozen=True)
class SimulationGrid:
    n: Sequence[int] = (50, 100, 300)
    p_signal: Sequence[int] = (5, 20, 50)
    p_noise: Sequence[int] = (0, 20, 100)
    signal_strength: Sequence[float] = (0.25, 0.5, 1.0, 2.0)
    noise_sd: Sequence[float] = (0.5, 1.0, 2.0)
    n_components: int = 5
    recovery_threshold: float = 0.7


@dataclass
class GeneratedData:
    X: Array
    z: Array
    signal_indices: List[int]
    noise_indices: List[int]
    true_loadings: Array


@dataclass
class PCAResult:
    scores: Array
    loadings: Array
    explained_variance_ratio: Array
    model: PCA


@dataclass(frozen=True)
class ProgressConfig:
    enabled: bool = True
    use_tqdm: bool = True
    print_every: int = 500


def make_rng(random_state: Optional[int | np.random.Generator]) -> np.random.Generator:
    if isinstance(random_state, np.random.Generator):
        return random_state
    return np.random.default_rng(random_state)


def generate_latent_signal(
    n: int,
    random_state: Optional[int | np.random.Generator] = None,
) -> Array:
    rng = make_rng(random_state)
    return rng.normal(0.0, 1.0, size=n)


def generate_observed_data(
    z: Array,
    p_signal: int,
    p_noise: int,
    signal_strength: float,
    noise_sd: float,
    random_state: Optional[int | np.random.Generator] = None,
) -> GeneratedData:
    rng = make_rng(random_state)
    n = z.shape[0]

    true_loadings = rng.uniform(0.5, 1.5, size=p_signal)

    signal_error = rng.normal(0.0, noise_sd, size=(n, p_signal))
    X_signal = signal_strength * np.outer(z, true_loadings) + signal_error

    X_noise = rng.normal(0.0, 1.0, size=(n, p_noise))
    X = np.concatenate([X_signal, X_noise], axis=1)

    signal_indices = list(range(p_signal))
    noise_indices = list(range(p_signal, p_signal + p_noise))

    return GeneratedData(
        X=X,
        z=z,
        signal_indices=signal_indices,
        noise_indices=noise_indices,
        true_loadings=true_loadings,
    )


def standardize_data(X: Array) -> Array:
    return StandardScaler().fit_transform(X)


def run_pca(X: Array, n_components: int) -> PCAResult:
    n_components = min(n_components, X.shape[0], X.shape[1])
    model = PCA(n_components=n_components)
    scores = model.fit_transform(X)

    return PCAResult(
        scores=scores,
        loadings=model.components_.T,
        explained_variance_ratio=model.explained_variance_ratio_,
        model=model,
    )


def absolute_correlation(x: Array, y: Array) -> float:
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(abs(np.corrcoef(x, y)[0, 1]))


def evaluate_recovery(
    z: Array,
    pca_result: PCAResult,
    signal_indices: Sequence[int],
    noise_indices: Sequence[int],
    recovery_threshold: float,
) -> Dict[str, float]:
    pc_correlations = [
        absolute_correlation(z, pca_result.scores[:, j])
        for j in range(pca_result.scores.shape[1])
    ]

    pc1_abs_corr = pc_correlations[0]
    best_abs_corr = float(np.nanmax(pc_correlations))
    best_pc_index = int(np.nanargmax(pc_correlations) + 1)

    pc1_abs_loadings = np.abs(pca_result.loadings[:, 0])
    signal_loading_mean = float(np.mean(pc1_abs_loadings[list(signal_indices)]))

    if len(noise_indices) > 0:
        noise_loading_mean = float(np.mean(pc1_abs_loadings[list(noise_indices)]))
        signal_loading_advantage = signal_loading_mean - noise_loading_mean
    else:
        noise_loading_mean = np.nan
        signal_loading_advantage = np.nan

    metrics: Dict[str, float] = {
        "pc1_abs_corr_with_z": pc1_abs_corr,
        "best_abs_corr_with_z": best_abs_corr,
        "best_pc_index": best_pc_index,
        "pc1_explained_variance": float(pca_result.explained_variance_ratio[0]),
        "signal_loading_mean_pc1": signal_loading_mean,
        "noise_loading_mean_pc1": noise_loading_mean,
        "signal_loading_advantage_pc1": signal_loading_advantage,
        "recovered_by_pc1": float(pc1_abs_corr >= recovery_threshold),
        "recovered_by_first_k": float(best_abs_corr >= recovery_threshold),
    }

    for j, corr in enumerate(pc_correlations, start=1):
        metrics[f"pc{j}_abs_corr_with_z"] = corr

    return metrics


def run_single_simulation(
    config: SimulationConfig,
    latent_signal_generator: Callable[..., Array] = generate_latent_signal,
    observed_data_generator: Callable[..., GeneratedData] = generate_observed_data,
    scaler: Callable[[Array], Array] = standardize_data,
    pca_runner: Callable[[Array, int], PCAResult] = run_pca,
    evaluator: Callable[..., Dict[str, float]] = evaluate_recovery,
    random_state: Optional[int | np.random.Generator] = None,
) -> Dict[str, float]:
    rng = make_rng(random_state)

    z = latent_signal_generator(config.n, random_state=rng)
    generated = observed_data_generator(
        z=z,
        p_signal=config.p_signal,
        p_noise=config.p_noise,
        signal_strength=config.signal_strength,
        noise_sd=config.noise_sd,
        random_state=rng,
    )

    X_scaled = scaler(generated.X)
    pca_result = pca_runner(X_scaled, config.n_components)

    metrics = evaluator(
        z=generated.z,
        pca_result=pca_result,
        signal_indices=generated.signal_indices,
        noise_indices=generated.noise_indices,
        recovery_threshold=config.recovery_threshold,
    )

    return {**asdict(config), **metrics}


def expand_grid(grid: SimulationGrid) -> List[SimulationConfig]:
    configs: List[SimulationConfig] = []

    for n in grid.n:
        for p_signal in grid.p_signal:
            for p_noise in grid.p_noise:
                for signal_strength in grid.signal_strength:
                    for noise_sd in grid.noise_sd:
                        configs.append(
                            SimulationConfig(
                                n=n,
                                p_signal=p_signal,
                                p_noise=p_noise,
                                signal_strength=signal_strength,
                                noise_sd=noise_sd,
                                n_components=grid.n_components,
                                recovery_threshold=grid.recovery_threshold,
                            )
                        )

    return configs


def format_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def print_progress_update(
    completed: int,
    total: int,
    start_time: float,
    config: SimulationConfig,
) -> None:
    elapsed = time.perf_counter() - start_time
    rate = completed / elapsed if elapsed > 0 else 0.0
    remaining = (total - completed) / rate if rate > 0 else 0.0
    percent = 100 * completed / total if total else 100.0

    print(
        f"Progress: {completed:,}/{total:,} ({percent:5.1f}%) | "
        f"elapsed {format_seconds(elapsed)} | "
        f"remaining {format_seconds(remaining)} | "
        f"current n={config.n}, p_signal={config.p_signal}, "
        f"p_noise={config.p_noise}, signal={config.signal_strength}, noise_sd={config.noise_sd}",
        flush=True,
    )


def run_simulation_grid(
    grid: SimulationGrid,
    n_repeats: int,
    seed: int,
    single_simulation_runner: Callable[..., Dict[str, float]] = run_single_simulation,
    progress: ProgressConfig = ProgressConfig(),
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: List[Dict[str, float]] = []
    configs = expand_grid(grid)
    total_runs = len(configs) * n_repeats
    completed = 0
    start_time = time.perf_counter()

    if progress.enabled:
        print(
            f"Starting simulation: {len(configs):,} conditions x {n_repeats:,} repeats = {total_runs:,} runs",
            flush=True,
        )

    iterator = configs
    progress_bar = None

    if progress.enabled and progress.use_tqdm and tqdm is not None:
        progress_bar = tqdm(total=total_runs, desc="Running PCA simulations", unit="run")

    for config_id, config in enumerate(iterator):
        for repeat in range(n_repeats):
            simulation_seed = int(rng.integers(0, 2**32 - 1))
            row = single_simulation_runner(config=config, random_state=simulation_seed)
            row["config_id"] = config_id
            row["repeat"] = repeat
            rows.append(row)

            completed += 1

            if progress_bar is not None:
                progress_bar.update(1)
                progress_bar.set_postfix(
                    {
                        "n": config.n,
                        "p_signal": config.p_signal,
                        "p_noise": config.p_noise,
                        "signal": config.signal_strength,
                        "noise": config.noise_sd,
                    }
                )
            elif progress.enabled and (
                completed == 1 or completed % progress.print_every == 0 or completed == total_runs
            ):
                print_progress_update(
                    completed=completed,
                    total=total_runs,
                    start_time=start_time,
                    config=config,
                )

    if progress_bar is not None:
        progress_bar.close()

    if progress.enabled:
        elapsed = time.perf_counter() - start_time
        print(f"Simulation runs finished in {format_seconds(elapsed)}.", flush=True)

    return pd.DataFrame(rows)


def summarize_by_condition(results_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "n",
        "p_signal",
        "p_noise",
        "signal_strength",
        "noise_sd",
        "n_components",
        "recovery_threshold",
    ]

    metric_cols = [
        "pc1_abs_corr_with_z",
        "best_abs_corr_with_z",
        "pc1_explained_variance",
        "signal_loading_advantage_pc1",
        "recovered_by_pc1",
        "recovered_by_first_k",
        "best_pc_index",
    ]

    summary = (
        results_df.groupby(group_cols, dropna=False)[metric_cols]
        .agg(["mean", "std", "median"])
        .reset_index()
    )

    summary.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col
        for col in summary.columns
    ]

    return summary


def make_focused_tables(results_df: pd.DataFrame, summary_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    tables: Dict[str, pd.DataFrame] = {}

    tables["overall_metrics"] = pd.DataFrame(
        {
            "metric": [
                "Mean PC1 recovery correlation",
                "Mean best recovery correlation among first k PCs",
                "PC1 recovery rate",
                "First-k-PC recovery rate",
                "Mean PC1 explained variance",
            ],
            "value": [
                results_df["pc1_abs_corr_with_z"].mean(),
                results_df["best_abs_corr_with_z"].mean(),
                results_df["recovered_by_pc1"].mean(),
                results_df["recovered_by_first_k"].mean(),
                results_df["pc1_explained_variance"].mean(),
            ],
        }
    )

    tables["by_signal_strength"] = (
        results_df.groupby("signal_strength", as_index=False)
        .agg(
            pc1_abs_corr_mean=("pc1_abs_corr_with_z", "mean"),
            best_abs_corr_mean=("best_abs_corr_with_z", "mean"),
            recovery_rate_first_k=("recovered_by_first_k", "mean"),
            pc1_explained_variance_mean=("pc1_explained_variance", "mean"),
        )
        .sort_values("signal_strength")
    )

    tables["by_noise_level"] = (
        results_df.groupby("noise_sd", as_index=False)
        .agg(
            pc1_abs_corr_mean=("pc1_abs_corr_with_z", "mean"),
            best_abs_corr_mean=("best_abs_corr_with_z", "mean"),
            recovery_rate_first_k=("recovered_by_first_k", "mean"),
            pc1_explained_variance_mean=("pc1_explained_variance", "mean"),
        )
        .sort_values("noise_sd")
    )

    tables["by_sample_size"] = (
        results_df.groupby("n", as_index=False)
        .agg(
            pc1_abs_corr_mean=("pc1_abs_corr_with_z", "mean"),
            best_abs_corr_mean=("best_abs_corr_with_z", "mean"),
            recovery_rate_first_k=("recovered_by_first_k", "mean"),
        )
        .sort_values("n")
    )

    tables["by_number_of_signal_variables"] = (
        results_df.groupby("p_signal", as_index=False)
        .agg(
            pc1_abs_corr_mean=("pc1_abs_corr_with_z", "mean"),
            best_abs_corr_mean=("best_abs_corr_with_z", "mean"),
            recovery_rate_first_k=("recovered_by_first_k", "mean"),
        )
        .sort_values("p_signal")
    )

    tables["by_number_of_irrelevant_variables"] = (
        results_df.groupby("p_noise", as_index=False)
        .agg(
            pc1_abs_corr_mean=("pc1_abs_corr_with_z", "mean"),
            best_abs_corr_mean=("best_abs_corr_with_z", "mean"),
            recovery_rate_first_k=("recovered_by_first_k", "mean"),
        )
        .sort_values("p_noise")
    )

    tables["signal_strength_by_noise_level"] = results_df.pivot_table(
        index="noise_sd",
        columns="signal_strength",
        values="recovered_by_first_k",
        aggfunc="mean",
    ).reset_index()

    tables["top_recovery_conditions"] = (
        summary_df.sort_values("recovered_by_first_k_mean", ascending=False)
        .head(20)
        .reset_index(drop=True)
    )

    tables["worst_recovery_conditions"] = (
        summary_df.sort_values("recovered_by_first_k_mean", ascending=True)
        .head(20)
        .reset_index(drop=True)
    )

    return tables


def save_tables(tables: Dict[str, pd.DataFrame], tables_dir: Path) -> None:
    tables_dir.mkdir(parents=True, exist_ok=True)

    for name, table in tables.items():
        table.to_csv(tables_dir / f"{name}.csv", index=False)

    with pd.ExcelWriter(tables_dir / "simulation_tables.xlsx") as writer:
        for name, table in tables.items():
            table.to_excel(writer, sheet_name=name[:31], index=False)


def plot_metric_by_factor(
    table: pd.DataFrame,
    x_col: str,
    y_col: str,
    output_path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(table[x_col], table[y_col], marker="o")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_signal_noise_heatmap(table: pd.DataFrame, output_path: Path) -> None:
    heatmap_data = table.set_index("noise_sd")

    plt.figure(figsize=(8, 5))
    plt.imshow(heatmap_data.values, aspect="auto")
    plt.colorbar(label="Recovery rate among first k PCs")

    plt.xticks(
        np.arange(len(heatmap_data.columns)),
        [str(col) for col in heatmap_data.columns],
    )
    plt.yticks(
        np.arange(len(heatmap_data.index)),
        [str(idx) for idx in heatmap_data.index],
    )

    plt.xlabel("Signal strength")
    plt.ylabel("Noise SD")
    plt.title("Recovery rate by signal strength and noise level")

    for i in range(heatmap_data.shape[0]):
        for j in range(heatmap_data.shape[1]):
            plt.text(j, i, f"{heatmap_data.values[i, j]:.2f}", ha="center", va="center")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_best_pc_distribution(results_df: pd.DataFrame, output_path: Path) -> None:
    counts = results_df["best_pc_index"].value_counts(normalize=True).sort_index()

    plt.figure(figsize=(8, 5))
    plt.bar(counts.index.astype(str), counts.values)
    plt.title("Which PC most often recovers the latent signal")
    plt.xlabel("Best recovering PC")
    plt.ylabel("Proportion of simulations")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_plots(tables: Dict[str, pd.DataFrame], results_df: pd.DataFrame, plots_dir: Path) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_metric_by_factor(
        table=tables["by_signal_strength"],
        x_col="signal_strength",
        y_col="best_abs_corr_mean",
        output_path=plots_dir / "01_recovery_by_signal_strength.png",
        title="PCA recovery improves as shared signal becomes stronger",
        xlabel="Signal strength",
        ylabel="Mean best absolute correlation with latent signal",
    )

    plot_metric_by_factor(
        table=tables["by_noise_level"],
        x_col="noise_sd",
        y_col="best_abs_corr_mean",
        output_path=plots_dir / "02_recovery_by_noise_level.png",
        title="PCA recovery decreases as noise increases",
        xlabel="Noise SD",
        ylabel="Mean best absolute correlation with latent signal",
    )

    plot_metric_by_factor(
        table=tables["by_sample_size"],
        x_col="n",
        y_col="best_abs_corr_mean",
        output_path=plots_dir / "03_recovery_by_sample_size.png",
        title="PCA recovery by sample size",
        xlabel="Sample size",
        ylabel="Mean best absolute correlation with latent signal",
    )

    plot_metric_by_factor(
        table=tables["by_number_of_signal_variables"],
        x_col="p_signal",
        y_col="best_abs_corr_mean",
        output_path=plots_dir / "04_recovery_by_signal_variables.png",
        title="PCA recovery by number of signal-carrying variables",
        xlabel="Number of signal variables",
        ylabel="Mean best absolute correlation with latent signal",
    )

    plot_metric_by_factor(
        table=tables["by_number_of_irrelevant_variables"],
        x_col="p_noise",
        y_col="best_abs_corr_mean",
        output_path=plots_dir / "05_recovery_by_irrelevant_variables.png",
        title="PCA recovery by number of irrelevant variables",
        xlabel="Number of irrelevant variables",
        ylabel="Mean best absolute correlation with latent signal",
    )

    plot_signal_noise_heatmap(
        table=tables["signal_strength_by_noise_level"],
        output_path=plots_dir / "06_recovery_heatmap_signal_strength_by_noise.png",
    )

    plot_best_pc_distribution(
        results_df=results_df,
        output_path=plots_dir / "07_best_recovering_pc_distribution.png",
    )


def write_readme(output_dir: Path, n_repeats: int, grid: SimulationGrid) -> None:
    text = f"""PCA Shared Signal Simulation Results

Purpose
This simulation evaluates whether PCA can recover a known latent shared signal from noisy observed variables.

Simulation settings
- Repeats per condition: {n_repeats}
- Sample sizes: {list(grid.n)}
- Signal variables: {list(grid.p_signal)}
- Irrelevant variables: {list(grid.p_noise)}
- Signal strengths: {list(grid.signal_strength)}
- Noise SD values: {list(grid.noise_sd)}
- Number of PCs evaluated: {grid.n_components}
- Recovery threshold: {grid.recovery_threshold}

Main interpretation
Use best_abs_corr_with_z and recovered_by_first_k as the main evaluation metrics. PC1 is also reported, but the signal can appear in PC2 or later components if another source of variance is stronger.

Key files
- raw/raw_simulation_results.csv: every simulation run
- summaries/summary_by_condition.csv: grouped results for every condition
- tables/simulation_tables.xlsx: all interpretation tables in one workbook
- tables/by_signal_strength.csv: recovery by signal strength
- tables/by_noise_level.csv: recovery by noise level
- tables/signal_strength_by_noise_level.csv: recovery-rate matrix
- plots/: publication-ready PNG plots
"""
    (output_dir / "README.txt").write_text(text, encoding="utf-8")


def main_simulation_study(
    grid: Optional[SimulationGrid] = None,
    n_repeats: int = 100,
    seed: int = 42,
    output_dir: str | Path = "results",
    progress: ProgressConfig = ProgressConfig(),
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame]]:
    if grid is None:
        grid = SimulationGrid()

    output_dir = Path(output_dir)
    raw_dir = output_dir / "raw"
    summary_dir = output_dir / "summaries"
    tables_dir = output_dir / "tables"
    plots_dir = output_dir / "plots"

    raw_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    results_df = run_simulation_grid(
        grid=grid,
        n_repeats=n_repeats,
        seed=seed,
        progress=progress,
    )

    if progress.enabled:
        print("Summarizing results and saving files...", flush=True)

    summary_df = summarize_by_condition(results_df)
    tables = make_focused_tables(results_df, summary_df)

    results_df.to_csv(raw_dir / "raw_simulation_results.csv", index=False)
    summary_df.to_csv(summary_dir / "summary_by_condition.csv", index=False)

    save_tables(tables=tables, tables_dir=tables_dir)
    save_plots(tables=tables, results_df=results_df, plots_dir=plots_dir)
    write_readme(output_dir=output_dir, n_repeats=n_repeats, grid=grid)

    if progress.enabled:
        print(f"All files saved in: {output_dir}", flush=True)

    return results_df, summary_df, tables


if __name__ == "__main__":
    grid = SimulationGrid(
        n=(50, 100, 300),
        p_signal=(5, 20, 50),
        p_noise=(0, 20, 100),
        signal_strength=(0.25, 0.5, 1.0, 2.0),
        noise_sd=(0.5, 1.0, 2.0),
        n_components=5,
        recovery_threshold=0.7,
    )

    raw_results, summary_results, interpretation_tables = main_simulation_study(
        grid=grid,
        n_repeats=100,
        seed=42,
        output_dir="pca_simulation_results",
        progress=ProgressConfig(
            enabled=True,
            use_tqdm=True,
            print_every=500,
        ),
    )

    print("Simulation complete.")
    print("Main results saved in: pca_simulation_results/")
    print("Overall metrics:")
    print(interpretation_tables["overall_metrics"])
