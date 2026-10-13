# Architecture

The diagram describes verified module roles for the E15 video workflow, not every experimental revision or an actual deployment topology.

```mermaid
flowchart TD
    R["UTA and DROZY datasets (external)"] --> P["src/preprocess: EEG/video preparation"]
    P --> E["src/process: embeddings, features, eventfulness"]
    E --> D["E15 dataset loading and memmap windows"]
    K["DROZY KSS labels"] --> D
    C["E15 CLI and dataset path resolution"] --> D
    D --> G["E15 subject-group split validation"]
    G --> CV["E15 nested cross-validation engine"]
    CV --> M["E15 MIL model and fold-wise fitting"]
    M --> O["Outer-test evaluation after final refit"]
    O --> R2["Experiment results and checkpoints"]
    H["SLURM scripts (optional HPC)"] --> C
```

`experiments/e15/main.py` builds the data, model, selection, training and CV configs; `src/utils/experiment_data.py` resolves paths and parses original labels independently of ML libraries. `experiments/e15/datasets.py` loads eventfulness-selected memmap windows; `video_mil_adapter.py` links the loaders and model to `nested_cv_engine.py`. `src/jobs/` contains optional SLURM job orchestration.

The diagram is manually-authored Mermaid grounded in inspected source code. Archify was unavailable, so no Archify output is claimed. Historical E1-E14 scripts and separate simulation studies are preserved without structural changes.
