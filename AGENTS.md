# Engineering conventions

This repository is a scientific research archive. Preserve historical experiment directories, notebooks, model files, logs, plots, run settings, and result artifacts unless explicitly authorized to migrate them.

- Keep the historical `Zh` import prefix functional. Run modules with the checkout's parent on `PYTHONPATH`; avoid moving directories without testing external SLURM snapshot paths and experiment scripts.
- The E15 nested CV setup has scientific requirements: preserve subject split boundaries, augmentations, random seeds, label mapping, normalization scopes, evaluation methods and checkpoint formats.
- Keep outer test sets sealed during inner hyperparameter search. Use inner validation metrics for selection, and evaluate the outer test fold only on the final refitted model. New inner-fold test metrics are null for backward-compatible JSON serialization.
- Guard grouped splits against duplicate samples, empty folds and overlapping subjects; do not change historical fold assignments without a dataset-backed baseline.
- Inject operational paths and settings via the E15 CLI; path conventions are centralized in `src/utils/experiment_data.py`.
- Keep focused reusable functions and explicit dependencies; avoid speculative frameworks and excessive helpers.
- Do not add hardcoded user-specific paths, credentials, participant data, or confidential metadata to new code.
- From the checkout's parent run `PYTHONPATH=. python -m unittest discover -s Zh/tests -v`, compile changed Python, and `bash -n` on modified SLURM scripts.
- Lightweight CI is not a substitute for dataset-backed scientific reproduction; report unavailable training checks clearly.
- Never submit SLURM jobs, alter trained weights, or clean research outputs as part of automated tests.
- Record architectural changes and research limitations in `docs/`. Keep the diagram synchronized with the code.
