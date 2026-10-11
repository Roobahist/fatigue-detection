# Fatigue Detection Research

Research code and archived results for multimodal fatigue and drowsiness detection using facial video, EEG, and behavioral/self-report information. Includes preprocessing, feature extraction, video MIL training with nested cross-validation, exploratory notebooks, and PCA simulation studies. This is a research archive, not a production inference service.

## Structure

| Directory | Purpose |
| --- | --- |
| `src/preprocess/` | Video/EEG preparation and alignment |
| `src/process/` | Face embeddings, eventfulness, EEG scores, visualization |
| `src/utils/` | Shared functions and portable experiment configuration |
| `src/jobs/` | SLURM experiment submission and monitoring |
| `experiments/` | Distinct historical experiments, including E15 nested-CV video MIL |
| `models/`, `plots/`, `logs/`, `extra/` | Preserved models, figures, logs, notebooks and data products |
| `tests/`, `docs/` | Lightweight tests and engineering/reproducibility guides |

Do not delete or regenerate tracked model checkpoints, plots, or previous experimental results during routine refactoring.

## Setup

The repository historically imports `Zh.src` and `Zh.experiments`. Clone to a folder named `Zh` and run Python modules from its parent directory until those imports can be migrated with regression checks.

```bash
git clone https://github.com/Roobahist/fatigueDetection.git Zh
cd Zh
conda env create -f env.yaml
conda activate .cvenv
cd ..
python -m Zh.experiments.e15.main --help
```

`env.yaml` is a historical Linux/CUDA environment export, not a verified portable environment. Full model runs require authorized access to external datasets and compatible hardware.

## E15 configuration

E15 supports `FATIGUE_DATA_ROOT` and `FATIGUE_OUTPUT_ROOT`. Override these using `--data-root` and `--output-root`; individual resources can be overridden with `--embeddings-dir`, `--features-dir`, `--eventfulness-scores-dir`, `--normalization-info-path` and `--kss-score-path`. `--output_dir` continues to take priority when provided. The original data root remains a compatibility default only.

```bash
export FATIGUE_DATA_ROOT=/path/to/datasets
export FATIGUE_OUTPUT_ROOT=/path/to/outputs
python -m Zh.experiments.e15.main --help
```

For the complete research workflow, read [architecture](docs/architecture.md), [reproducibility](docs/reproducibility.md), and [migration notes](docs/migration-notes.md). Existing `experiments/e15/configs/*.conf` files are HPC-specific historical examples.

## Lightweight checks

From the parent of `Zh`:

```bash
PYTHONPATH=. python -m unittest discover -s Zh/tests -v
find Zh/src Zh/experiments/e15 -type f -name '*.py' ! -path '*/archive/*' -print0 | xargs -0 -r python -m py_compile
bash -n Zh/src/jobs/submit_job.sh Zh/src/jobs/manage_experiments.sh
```

These tests do not retrain models, verify original participant data, or establish numerical equivalence to published historical results.
