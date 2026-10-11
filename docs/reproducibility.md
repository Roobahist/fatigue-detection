# Reproducibility

1. Obtain UTA/DROZY datasets through authorized research channels; source participant datasets are not included here.
2. Record the code commit, dataset version/manifest, configuration, environment, operating system, driver/CUDA version and hardware for every new run.
3. Verify facial/EEG temporal alignment, memmap shapes, participant-level splitting, augmentation isolation and normalization fit within training folds before comparing results.
4. For E15, set `FATIGUE_DATA_ROOT` or pass `--data-root`. The original DROZY KSS matrix is under `DROZY/data/original/KSS.txt` relative to that root. `--kss-score-path` supports different layouts.
5. Inspect `src/jobs/submit_job.sh` and the historical `experiments/e15/configs/` before submitting HPC jobs. Never run SLURM scripts as part of ordinary CI.
6. Run unit, compile and shell checks from the README. These only verify portable components and syntax.

`env.yaml` is a frozen Linux/CUDA conda export and may not solve on other platforms. Reproducing research numbers requires original data, GPU environment, seed/fold checks, and comparison with saved predictions. Those checks have not been performed by this modernization.
