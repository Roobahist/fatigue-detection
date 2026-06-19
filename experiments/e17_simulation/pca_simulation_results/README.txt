PCA Shared Signal Simulation Results

Purpose
This simulation evaluates whether PCA can recover a known latent shared signal from noisy observed variables.

Simulation settings
- Repeats per condition: 100
- Sample sizes: [50, 100, 300]
- Signal variables: [5, 20, 50]
- Irrelevant variables: [0, 20, 100]
- Signal strengths: [0.25, 0.5, 1.0, 2.0]
- Noise SD values: [0.5, 1.0, 2.0]
- Number of PCs evaluated: 5
- Recovery threshold: 0.7

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
