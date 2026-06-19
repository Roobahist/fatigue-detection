Multi-Source PCA Shared Structure Simulation

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
- Conditions: 5184
- Total runs: 518400
- Repeats per condition: 100
- Sample sizes: [50, 100, 300]
- Number of sources: [2, 3, 5]
- Features per source: [10, 30, 60]
- Shared dimensions: [1, 2]
- Source-specific dimensions: [1, 2]
- Shared strengths: [0.25, 0.5, 1.0, 2.0]
- Source-specific strengths: [0.25, 0.5, 1.0, 2.0]
- Noise SD values: [0.5, 1.0, 2.0]
- Number of PCs evaluated: 5
- Recovery threshold: 0.7

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
