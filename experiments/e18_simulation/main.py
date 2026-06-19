
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


Array = np.ndarray


@dataclass(frozen=True)
class MultiSourceSimulationConfig:
    n: int
    n_sources: int
    features_per_source: int
    shared_dim: int
    source_specific_dim: int
    shared_strength: float
    source_specific_strength: float
    noise_sd: float
    n_components: int = 5
    recovery_threshold: float = 0.7


@dataclass(frozen=True)
class MultiSourceSimulationGrid:
    n: Sequence[int] = (50, 100, 300)
    n_sources: Sequence[int] = (2, 3, 5)
    features_per_source: Sequence[int] = (10, 30, 60)
    shared_dim: Sequence[int] = (1, 2)
    source_specific_dim: Sequence[int] = (1, 2)
    shared_strength: Sequence[float] = (0.25, 0.5, 1.0, 2.0)
    source_specific_strength: Sequence[float] = (0.25, 0.5, 1.0, 2.0)
    noise_sd: Sequence[float] = (0.5, 1.0, 2.0)
    n_components: int = 5
    recovery_threshold: float = 0.7


@dataclass
class MultiSourceGeneratedData:
    X: Array
    Z_shared: Array
    source_labels: List[int]
    source_feature_indices: Dict[int, List[int]]
    shared_loadings_by_source: Dict[int, Array]
    source_specific_scores_by_source: Dict[int, Array]
    source_specific_loadings_by_source: Dict[int, Array]


@dataclass
class PCAResult:
    scores: Array
    loadings: Array
    explained_variance_ratio: Array
    model: PCA


def make_rng(random_state: Optional[int | np.random.Generator]) -> np.random.Generator:
    if isinstance(random_state, np.random.Generator):
        return random_state
    return np.random.default_rng(random_state)


def generate_shared_structure(
    n: int,
    shared_dim: int,
    random_state: Optional[int | np.random.Generator] = None,
) -> Array:
    rng = make_rng(random_state)
    return rng.normal(0.0, 1.0, size=(n, shared_dim))


def generate_multi_source_data(
    Z_shared: Array,
    n_sources: int,
    features_per_source: int,
    source_specific_dim: int,
    shared_strength: float,
    source_specific_strength: float,
    noise_sd: float,
    random_state: Optional[int | np.random.Generator] = None,
) -> MultiSourceGeneratedData:
    rng = make_rng(random_state)

    n, shared_dim = Z_shared.shape
    X_blocks: List[Array] = []
    source_labels: List[int] = []
    source_feature_indices: Dict[int, List[int]] = {}
    shared_loadings_by_source: Dict[int, Array] = {}
    source_specific_scores_by_source: Dict[int, Array] = {}
    source_specific_loadings_by_source: Dict[int, Array] = {}

    start_col = 0

    for source_id in range(n_sources):
        shared_loadings = rng.normal(
            loc=0.0,
            scale=1.0,
            size=(shared_dim, features_per_source),
        )

        source_scores = rng.normal(
            loc=0.0,
            scale=1.0,
            size=(n, source_specific_dim),
        )

        source_loadings = rng.normal(
            loc=0.0,
            scale=1.0,
            size=(source_specific_dim, features_per_source),
        )

        noise = rng.normal(
            loc=0.0,
            scale=noise_sd,
            size=(n, features_per_source),
        )

        X_source = (
            shared_strength * (Z_shared @ shared_loadings)
            + source_specific_strength * (source_scores @ source_loadings)
            + noise
        )

        end_col = start_col + features_per_source
        source_feature_indices[source_id] = list(range(start_col, end_col))
        source_labels.extend([source_id] * features_per_source)

        shared_loadings_by_source[source_id] = shared_loadings
        source_specific_scores_by_source[source_id] = source_scores
        source_specific_loadings_by_source[source_id] = source_loadings

        X_blocks.append(X_source)
        start_col = end_col

    X = np.concatenate(X_blocks, axis=1)

    return MultiSourceGeneratedData(
        X=X,
        Z_shared=Z_shared,
        source_labels=source_labels,
        source_feature_indices=source_feature_indices,
        shared_loadings_by_source=shared_loadings_by_source,
        source_specific_scores_by_source=source_specific_scores_by_source,
        source_specific_loadings_by_source=source_specific_loadings_by_source,
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


def regression_subspace_r2(target: Array, predictors: Array) -> float:
    model = LinearRegression()
    model.fit(predictors, target)
    predicted = model.predict(predictors)

    ss_res = np.sum((target - predicted) ** 2)
    ss_tot = np.sum((target - target.mean(axis=0, keepdims=True)) ** 2)

    if ss_tot == 0:
        return np.nan

    return float(1.0 - ss_res / ss_tot)


def evaluate_shared_structure_recovery(
    Z_shared: Array,
    pca_result: PCAResult,
    source_feature_indices: Dict[int, List[int]],
    recovery_threshold: float,
) -> Dict[str, float]:
    pc_scores = pca_result.scores
    n_available_pcs = pc_scores.shape[1]

    shared_dim = Z_shared.shape[1]
    per_factor_best_corrs: List[float] = []
    per_factor_best_pcs: List[int] = []

    for factor_idx in range(shared_dim):
        correlations = [
            absolute_correlation(Z_shared[:, factor_idx], pc_scores[:, pc_idx])
            for pc_idx in range(n_available_pcs)
        ]
        per_factor_best_corrs.append(float(np.nanmax(correlations)))
        per_factor_best_pcs.append(int(np.nanargmax(correlations) + 1))

    pc1_factor_corrs = [
        absolute_correlation(Z_shared[:, factor_idx], pc_scores[:, 0])
        for factor_idx in range(shared_dim)
    ]

    first_k_subspace_r2 = regression_subspace_r2(Z_shared, pc_scores)
    pc1_subspace_r2 = regression_subspace_r2(Z_shared, pc_scores[:, [0]])

    source_loading_means = {}
    pc1_abs_loadings = np.abs(pca_result.loadings[:, 0])

    for source_id, indices in source_feature_indices.items():
        source_loading_means[source_id] = float(np.mean(pc1_abs_loadings[indices]))

    loading_values = np.array(list(source_loading_means.values()))
    loading_balance_ratio = float(
        np.min(loading_values) / np.max(loading_values)
        if np.max(loading_values) > 0
        else np.nan
    )

    metrics: Dict[str, float] = {
        "pc1_mean_abs_corr_with_shared_factors": float(np.nanmean(pc1_factor_corrs)),
        "best_mean_abs_corr_with_shared_factors": float(np.nanmean(per_factor_best_corrs)),
        "min_best_abs_corr_across_shared_factors": float(np.nanmin(per_factor_best_corrs)),
        "pc1_subspace_r2": pc1_subspace_r2,
        "first_k_subspace_r2": first_k_subspace_r2,
        "pc1_explained_variance": float(pca_result.explained_variance_ratio[0]),
        "first_k_explained_variance": float(np.sum(pca_result.explained_variance_ratio)),
        "source_loading_balance_ratio_pc1": loading_balance_ratio,
        "recovered_by_pc1_corr": float(np.nanmean(pc1_factor_corrs) >= recovery_threshold),
        "recovered_by_first_k_corr": float(np.nanmean(per_factor_best_corrs) >= recovery_threshold),
        "recovered_by_first_k_subspace": float(first_k_subspace_r2 >= recovery_threshold),
        "mean_best_pc_index": float(np.mean(per_factor_best_pcs)),
    }

    for factor_idx, corr in enumerate(pc1_factor_corrs, start=1):
        metrics[f"shared_factor_{factor_idx}_pc1_abs_corr"] = corr

    for factor_idx, corr in enumerate(per_factor_best_corrs, start=1):
        metrics[f"shared_factor_{factor_idx}_best_abs_corr"] = corr

    for factor_idx, pc_idx in enumerate(per_factor_best_pcs, start=1):
        metrics[f"shared_factor_{factor_idx}_best_pc_index"] = pc_idx

    for source_id, value in source_loading_means.items():
        metrics[f"source_{source_id}_pc1_abs_loading_mean"] = value

    return metrics


def run_single_multi_source_simulation(
    config: MultiSourceSimulationConfig,
    shared_structure_generator: Callable[..., Array] = generate_shared_structure,
    data_generator: Callable[..., MultiSourceGeneratedData] = generate_multi_source_data,
    scaler: Callable[[Array], Array] = standardize_data,
    pca_runner: Callable[[Array, int], PCAResult] = run_pca,
    evaluator: Callable[..., Dict[str, float]] = evaluate_shared_structure_recovery,
    random_state: Optional[int | np.random.Generator] = None,
) -> Dict[str, float]:
    rng = make_rng(random_state)

    Z_shared = shared_structure_generator(
        n=config.n,
        shared_dim=config.shared_dim,
        random_state=rng,
    )

    generated = data_generator(
        Z_shared=Z_shared,
        n_sources=config.n_sources,
        features_per_source=config.features_per_source,
        source_specific_dim=config.source_specific_dim,
        shared_strength=config.shared_strength,
        source_specific_strength=config.source_specific_strength,
        noise_sd=config.noise_sd,
        random_state=rng,
    )

    X_scaled = scaler(generated.X)
    pca_result = pca_runner(X_scaled, config.n_components)

    metrics = evaluator(
        Z_shared=generated.Z_shared,
        pca_result=pca_result,
        source_feature_indices=generated.source_feature_indices,
        recovery_threshold=config.recovery_threshold,
    )

    return {**asdict(config), **metrics}


def expand_grid(grid: MultiSourceSimulationGrid) -> List[MultiSourceSimulationConfig]:
    configs: List[MultiSourceSimulationConfig] = []

    for n in grid.n:
        for n_sources in grid.n_sources:
            for features_per_source in grid.features_per_source:
                for shared_dim in grid.shared_dim:
                    for source_specific_dim in grid.source_specific_dim:
                        for shared_strength in grid.shared_strength:
                            for source_specific_strength in grid.source_specific_strength:
                                for noise_sd in grid.noise_sd:
                                    configs.append(
                                        MultiSourceSimulationConfig(
                                            n=n,
                                            n_sources=n_sources,
                                            features_per_source=features_per_source,
                                            shared_dim=shared_dim,
                                            source_specific_dim=source_specific_dim,
                                            shared_strength=shared_strength,
                                            source_specific_strength=source_specific_strength,
                                            noise_sd=noise_sd,
                                            n_components=grid.n_components,
                                            recovery_threshold=grid.recovery_threshold,
                                        )
                                    )

    return configs


def format_seconds(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes > 0:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def progress_iterable(
    items: Sequence[Tuple[int, MultiSourceSimulationConfig, int]],
    show_progress: bool,
    progress_every: int,
):
    if not show_progress:
        yield from items
        return

    total = len(items)

    if tqdm is not None:
        yield from tqdm(items, total=total, desc="Running multi-source simulations")
        return

    start = perf_counter()

    for idx, item in enumerate(items, start=1):
        if idx == 1 or idx % progress_every == 0 or idx == total:
            elapsed = perf_counter() - start
            rate = idx / elapsed if elapsed > 0 else 0
            remaining = (total - idx) / rate if rate > 0 else 0
            _, config, repeat = item

            print(
                f"Progress: {idx:,}/{total:,} ({100 * idx / total:5.1f}%) | "
                f"elapsed {format_seconds(elapsed)} | "
                f"remaining {format_seconds(remaining)} | "
                f"repeat={repeat}, n={config.n}, sources={config.n_sources}, "
                f"features/source={config.features_per_source}, "
                f"shared_strength={config.shared_strength}, "
                f"source_specific_strength={config.source_specific_strength}, "
                f"noise_sd={config.noise_sd}"
            )

        yield item


def run_simulation_grid(
    grid: MultiSourceSimulationGrid,
    n_repeats: int,
    seed: int,
    show_progress: bool = True,
    progress_every: int = 500,
    single_simulation_runner: Callable[..., Dict[str, float]] = run_single_multi_source_simulation,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    configs = expand_grid(grid)

    jobs: List[Tuple[int, MultiSourceSimulationConfig, int]] = [
        (config_id, config, repeat)
        for config_id, config in enumerate(configs)
        for repeat in range(n_repeats)
    ]

    print(
        f"Starting multi-source simulation: {len(configs):,} conditions x "
        f"{n_repeats:,} repeats = {len(jobs):,} runs"
    )

    rows: List[Dict[str, float]] = []

    for config_id, config, repeat in progress_iterable(jobs, show_progress, progress_every):
        simulation_seed = int(rng.integers(0, 2**32 - 1))
        row = single_simulation_runner(config=config, random_state=simulation_seed)
        row["config_id"] = config_id
        row["repeat"] = repeat
        rows.append(row)

    return pd.DataFrame(rows)


def summarize_by_condition(results_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "n",
        "n_sources",
        "features_per_source",
        "shared_dim",
        "source_specific_dim",
        "shared_strength",
        "source_specific_strength",
        "noise_sd",
        "n_components",
        "recovery_threshold",
    ]

    metric_cols = [
        "pc1_mean_abs_corr_with_shared_factors",
        "best_mean_abs_corr_with_shared_factors",
        "min_best_abs_corr_across_shared_factors",
        "pc1_subspace_r2",
        "first_k_subspace_r2",
        "pc1_explained_variance",
        "first_k_explained_variance",
        "source_loading_balance_ratio_pc1",
        "recovered_by_pc1_corr",
        "recovered_by_first_k_corr",
        "recovered_by_first_k_subspace",
        "mean_best_pc_index",
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


def aggregate_by_factor(
    results_df: pd.DataFrame,
    factor: str,
) -> pd.DataFrame:
    return (
        results_df.groupby(factor, as_index=False)
        .agg(
            pc1_corr_mean=("pc1_mean_abs_corr_with_shared_factors", "mean"),
            best_corr_mean=("best_mean_abs_corr_with_shared_factors", "mean"),
            first_k_subspace_r2_mean=("first_k_subspace_r2", "mean"),
            recovery_rate_corr=("recovered_by_first_k_corr", "mean"),
            recovery_rate_subspace=("recovered_by_first_k_subspace", "mean"),
            pc1_explained_variance_mean=("pc1_explained_variance", "mean"),
            source_loading_balance_mean=("source_loading_balance_ratio_pc1", "mean"),
        )
        .sort_values(factor)
    )


def make_focused_tables(
    results_df: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    tables: Dict[str, pd.DataFrame] = {}

    tables["overall_metrics"] = pd.DataFrame(
        {
            "metric": [
                "Mean PC1 correlation with shared structure",
                "Mean best correlation among first k PCs",
                "Mean first-k subspace R2",
                "PC1 correlation recovery rate",
                "First-k correlation recovery rate",
                "First-k subspace recovery rate",
                "Mean PC1 explained variance",
                "Mean first-k explained variance",
                "Mean PC1 source-loading balance ratio",
            ],
            "value": [
                results_df["pc1_mean_abs_corr_with_shared_factors"].mean(),
                results_df["best_mean_abs_corr_with_shared_factors"].mean(),
                results_df["first_k_subspace_r2"].mean(),
                results_df["recovered_by_pc1_corr"].mean(),
                results_df["recovered_by_first_k_corr"].mean(),
                results_df["recovered_by_first_k_subspace"].mean(),
                results_df["pc1_explained_variance"].mean(),
                results_df["first_k_explained_variance"].mean(),
                results_df["source_loading_balance_ratio_pc1"].mean(),
            ],
        }
    )

    for factor in [
        "shared_strength",
        "source_specific_strength",
        "noise_sd",
        "n",
        "n_sources",
        "features_per_source",
        "shared_dim",
        "source_specific_dim",
    ]:
        tables[f"by_{factor}"] = aggregate_by_factor(results_df, factor)

    tables["shared_strength_by_noise_level"] = results_df.pivot_table(
        index="noise_sd",
        columns="shared_strength",
        values="recovered_by_first_k_subspace",
        aggfunc="mean",
    ).reset_index()

    tables["shared_strength_by_source_specific_strength"] = results_df.pivot_table(
        index="source_specific_strength",
        columns="shared_strength",
        values="recovered_by_first_k_subspace",
        aggfunc="mean",
    ).reset_index()

    tables["noise_by_source_specific_strength"] = results_df.pivot_table(
        index="noise_sd",
        columns="source_specific_strength",
        values="recovered_by_first_k_subspace",
        aggfunc="mean",
    ).reset_index()

    tables["top_recovery_conditions"] = (
        summary_df.sort_values("first_k_subspace_r2_mean", ascending=False)
        .head(25)
        .reset_index(drop=True)
    )

    tables["worst_recovery_conditions"] = (
        summary_df.sort_values("first_k_subspace_r2_mean", ascending=True)
        .head(25)
        .reset_index(drop=True)
    )

    return tables


def save_tables(tables: Dict[str, pd.DataFrame], tables_dir: Path) -> None:
    tables_dir.mkdir(parents=True, exist_ok=True)

    for name, table in tables.items():
        table.to_csv(tables_dir / f"{name}.csv", index=False)

    try:
        with pd.ExcelWriter(tables_dir / "multi_source_simulation_tables.xlsx") as writer:
            for name, table in tables.items():
                table.to_excel(writer, sheet_name=name[:31], index=False)
    except ImportError:
        print("openpyxl is not installed. Excel export skipped. CSV files were saved.")


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


def plot_heatmap(
    table: pd.DataFrame,
    index_col: str,
    output_path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    colorbar_label: str = "Recovery rate",
) -> None:
    heatmap_data = table.set_index(index_col)

    plt.figure(figsize=(8, 5))
    plt.imshow(heatmap_data.values, aspect="auto")
    plt.colorbar(label=colorbar_label)

    plt.xticks(
        np.arange(len(heatmap_data.columns)),
        [str(col) for col in heatmap_data.columns],
    )
    plt.yticks(
        np.arange(len(heatmap_data.index)),
        [str(idx) for idx in heatmap_data.index],
    )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    for i in range(heatmap_data.shape[0]):
        for j in range(heatmap_data.shape[1]):
            plt.text(j, i, f"{heatmap_data.values[i, j]:.2f}", ha="center", va="center")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_best_pc_distribution(results_df: pd.DataFrame, output_path: Path) -> None:
    pc_columns = [
        col
        for col in results_df.columns
        if col.startswith("shared_factor_") and col.endswith("_best_pc_index")
    ]

    values = []
    for col in pc_columns:
        values.extend(results_df[col].dropna().astype(int).tolist())

    counts = pd.Series(values).value_counts(normalize=True).sort_index()

    plt.figure(figsize=(8, 5))
    plt.bar(counts.index.astype(str), counts.values)
    plt.title("Which PC most often recovers shared factors")
    plt.xlabel("Best recovering PC")
    plt.ylabel("Proportion of shared factors")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_plots(
    tables: Dict[str, pd.DataFrame],
    results_df: pd.DataFrame,
    plots_dir: Path,
) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_metric_by_factor(
        table=tables["by_shared_strength"],
        x_col="shared_strength",
        y_col="first_k_subspace_r2_mean",
        output_path=plots_dir / "01_subspace_recovery_by_shared_strength.png",
        title="Shared-structure recovery improves as shared signal strengthens",
        xlabel="Shared signal strength",
        ylabel="Mean first-k subspace R2",
    )

    plot_metric_by_factor(
        table=tables["by_source_specific_strength"],
        x_col="source_specific_strength",
        y_col="first_k_subspace_r2_mean",
        output_path=plots_dir / "02_subspace_recovery_by_source_specific_strength.png",
        title="Shared-structure recovery as source-specific variation increases",
        xlabel="Source-specific strength",
        ylabel="Mean first-k subspace R2",
    )

    plot_metric_by_factor(
        table=tables["by_noise_sd"],
        x_col="noise_sd",
        y_col="first_k_subspace_r2_mean",
        output_path=plots_dir / "03_subspace_recovery_by_noise_level.png",
        title="Shared-structure recovery decreases as independent noise increases",
        xlabel="Noise SD",
        ylabel="Mean first-k subspace R2",
    )

    plot_metric_by_factor(
        table=tables["by_n_sources"],
        x_col="n_sources",
        y_col="first_k_subspace_r2_mean",
        output_path=plots_dir / "04_subspace_recovery_by_number_of_sources.png",
        title="Shared-structure recovery by number of information sources",
        xlabel="Number of sources",
        ylabel="Mean first-k subspace R2",
    )

    plot_metric_by_factor(
        table=tables["by_features_per_source"],
        x_col="features_per_source",
        y_col="first_k_subspace_r2_mean",
        output_path=plots_dir / "05_subspace_recovery_by_features_per_source.png",
        title="Shared-structure recovery by features per source",
        xlabel="Features per source",
        ylabel="Mean first-k subspace R2",
    )

    plot_metric_by_factor(
        table=tables["by_n"],
        x_col="n",
        y_col="first_k_subspace_r2_mean",
        output_path=plots_dir / "06_subspace_recovery_by_sample_size.png",
        title="Shared-structure recovery by sample size",
        xlabel="Sample size",
        ylabel="Mean first-k subspace R2",
    )

    plot_metric_by_factor(
        table=tables["by_shared_strength"],
        x_col="shared_strength",
        y_col="source_loading_balance_mean",
        output_path=plots_dir / "07_source_loading_balance_by_shared_strength.png",
        title="PC1 becomes more balanced across sources as shared signal strengthens",
        xlabel="Shared signal strength",
        ylabel="Mean source-loading balance ratio",
    )

    plot_heatmap(
        table=tables["shared_strength_by_noise_level"],
        index_col="noise_sd",
        output_path=plots_dir / "08_recovery_heatmap_shared_strength_by_noise.png",
        title="Subspace recovery rate by shared strength and noise",
        xlabel="Shared signal strength",
        ylabel="Noise SD",
    )

    plot_heatmap(
        table=tables["shared_strength_by_source_specific_strength"],
        index_col="source_specific_strength",
        output_path=plots_dir / "09_recovery_heatmap_shared_by_source_specific.png",
        title="Subspace recovery rate by shared and source-specific strength",
        xlabel="Shared signal strength",
        ylabel="Source-specific strength",
    )

    plot_heatmap(
        table=tables["noise_by_source_specific_strength"],
        index_col="noise_sd",
        output_path=plots_dir / "10_recovery_heatmap_noise_by_source_specific.png",
        title="Subspace recovery rate by noise and source-specific strength",
        xlabel="Source-specific strength",
        ylabel="Noise SD",
    )

    plot_best_pc_distribution(
        results_df=results_df,
        output_path=plots_dir / "11_best_recovering_pc_distribution.png",
    )


def write_readme(output_dir: Path, n_repeats: int, grid: MultiSourceSimulationGrid) -> None:
    n_conditions = len(expand_grid(grid))
    n_runs = n_conditions * n_repeats

    text = f"""Multi-Source PCA Shared Structure Simulation

Purpose
This experiment evaluates whether PCA can recover a known shared latent structure when several data sources contain the same underlying structure plus their own source-specific variation and independent noise.

Data-generating model
For each source s:

X_s = shared_strength * Z A_s + source_specific_strength * U_s B_s + E_s

where:
Z is the true shared latent structure.
A_s contains source-specific loadings for the shared structure.
U_s contains source-specific latent variation.
B_s contains source-specific loadings.
E_s is independent noise.

The combined data matrix is:

X = [X_1 | X_2 | ... | X_S]

PCA is applied to the standardized combined matrix.

Simulation settings
- Conditions: {n_conditions}
- Total runs: {n_runs}
- Repeats per condition: {n_repeats}
- Sample sizes: {list(grid.n)}
- Number of sources: {list(grid.n_sources)}
- Features per source: {list(grid.features_per_source)}
- Shared dimensions: {list(grid.shared_dim)}
- Source-specific dimensions: {list(grid.source_specific_dim)}
- Shared strengths: {list(grid.shared_strength)}
- Source-specific strengths: {list(grid.source_specific_strength)}
- Noise SD values: {list(grid.noise_sd)}
- Number of PCs evaluated: {grid.n_components}
- Recovery threshold: {grid.recovery_threshold}

Main metrics
- first_k_subspace_r2: how well the first k PCs recover the full shared latent structure.
- best_mean_abs_corr_with_shared_factors: average best correlation between each true shared factor and the first k PCs.
- pc1_mean_abs_corr_with_shared_factors: how well PC1 alone captures the shared structure.
- recovered_by_first_k_subspace: whether first_k_subspace_r2 exceeds the recovery threshold.
- source_loading_balance_ratio_pc1: whether PC1 has balanced loadings across sources. Values closer to 1 mean more balanced use of sources.

Recommended interpretation
Use first_k_subspace_r2 as the main metric when shared_dim > 1. Use best_mean_abs_corr_with_shared_factors as a simpler correlation-based metric. PC1 is useful, but it may not capture the whole shared structure when there are multiple shared factors or strong source-specific variation.

Key files
- raw/raw_multi_source_simulation_results.csv
- summaries/summary_by_condition.csv
- tables/multi_source_simulation_tables.xlsx
- tables/by_shared_strength.csv
- tables/by_source_specific_strength.csv
- tables/by_noise_sd.csv
- tables/shared_strength_by_noise_level.csv
- tables/shared_strength_by_source_specific_strength.csv
- tables/top_recovery_conditions.csv
- tables/worst_recovery_conditions.csv
- plots/
"""

    (output_dir / "README.txt").write_text(text, encoding="utf-8")


def main_multi_source_simulation_study(
    grid: Optional[MultiSourceSimulationGrid] = None,
    n_repeats: int = 100,
    seed: int = 42,
    output_dir: str | Path = "multi_source_pca_simulation_results",
    show_progress: bool = True,
    progress_every: int = 500,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame]]:
    if grid is None:
        grid = MultiSourceSimulationGrid()

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
        show_progress=show_progress,
        progress_every=progress_every,
    )

    summary_df = summarize_by_condition(results_df)
    tables = make_focused_tables(results_df, summary_df)

    results_df.to_csv(raw_dir / "raw_multi_source_simulation_results.csv", index=False)
    summary_df.to_csv(summary_dir / "summary_by_condition.csv", index=False)

    save_tables(tables=tables, tables_dir=tables_dir)
    save_plots(tables=tables, results_df=results_df, plots_dir=plots_dir)
    write_readme(output_dir=output_dir, n_repeats=n_repeats, grid=grid)

    return results_df, summary_df, tables


if __name__ == "__main__":
    grid = MultiSourceSimulationGrid(
        n=(50, 100, 300),
        n_sources=(2, 3, 5),
        features_per_source=(10, 30, 60),
        shared_dim=(1, 2),
        source_specific_dim=(1, 2),
        shared_strength=(0.25, 0.5, 1.0, 2.0),
        source_specific_strength=(0.25, 0.5, 1.0, 2.0),
        noise_sd=(0.5, 1.0, 2.0),
        n_components=5,
        recovery_threshold=0.7,
    )

    raw_results, summary_results, interpretation_tables = main_multi_source_simulation_study(
        grid=grid,
        n_repeats=100,
        seed=42,
        output_dir="multi_source_pca_simulation_results",
        show_progress=True,
        progress_every=500,
    )

    print("Multi-source simulation complete.")
    print("Main results saved in: multi_source_pca_simulation_results/")
    print("Overall metrics:")
    print(interpretation_tables["overall_metrics"])
