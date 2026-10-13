# E15 nested cross-validation: evaluation integrity

This document describes the current E15 code path. It does not certify the
published scientific findings, reconstruct participant data, or claim a GPU
experiment has been rerun.

## Verified execution path

1. `experiments/e15/main.py` discovers video embedding IDs, resolves data
   paths, builds configs, and calls `nested_cv_video_mil`.
2. `VideoMILAdapter.split_fn` calls the historical
   `kfold_split_indices` with `dataset_name` and `video_ids`.
   UTA/DROZY partitioning groups videos by subject ID.
3. `src/utils/cv_splits.py` checks fold count, sample uniqueness,
   nonempty partitions, exact sample coverage, and disjoint subject groups.
   Its subject-key callback is injected by the adapter.
4. For each outer fold, `NestedCVEngine` searches hyperparameters using
   inner validation metrics. The adapter computes normalization statistics
   from the current training subject IDs, including when transforming
   validation or test samples.
5. After configuration selection, the engine trains a final model on the
   full outer training set and evaluates the corresponding outer test set.

The validation guards intentionally do not reorder groups, change the
existing fold assignments, or change training hyperparameters. They fail
early instead of accepting an invalid partition.

## Change to outer-test evaluation

**Previous behavior:** Both grid/random and Bayesian inner-CV loops ran
`evaluate_fn` on the outer test split for *every* trained inner-fold model.
The selected hyperparameter configuration was computed from inner
validation metrics rather than these outer-test scores, but the test scores
were repeatedly calculated, logged and saved. This creates an avoidable
opportunity for human-led model-selection bias and breaks the clean
separation expected of a sealed nested-CV holdout.

**New behavior:** The inner loops only compute training and validation
metrics. The existing `InnerFoldResult.outer_test_metric` key remains for
serialization compatibility but is `null` for newly computed inner folds.
Only the final refitted model produces the outer-fold test score. Historical
checkpoint records may still contain the old numeric inner-test metrics,
and historical visualizations can still show them with a legacy label.

**Important:** A historical run resumed from its old checkpoint can
contain a mixture of old numeric and new null inner-test fields. For
strictly sealed results, run into a fresh output directory rather than
resuming an earlier experiment.

This change does not retroactively alter any prior model parameters,
scores, model artifacts or figures, and cannot demonstrate that the
manuscript's reported estimates would change.

## Reproducibility and leakage checks requiring original data

Before claiming a freshly reproduced result:

- Record the exact code commit and configuration and build a private
  manifest of raw and processed data versions.
- Confirm that filenames encode stable participant identities, and
  that augmented views of a participant never cross train/validation/test.
- Record the membership of every outer and inner split and verify
  group isolation with independent participant metadata.
- Confirm all data-driven preprocessing, including PCA target construction
  and normalization, is fitted on the appropriate training data if required
  by the intended evaluation protocol. Precomputed inputs in this archive
  cannot be independently audited without the original data and pipeline.
- Confirm test folds are not used to choose architectures, stopping rules,
  hyperparameters or reportable variants.
- Check label construction, KSS indexing, and facial/EEG temporal
  alignment against the authorized source data.
- Reproduce fold-wise predictions and numerical metrics on compatible
  hardware; compare to archived outputs using appropriate tolerances.

Some target definitions are proxies, not direct physiological ground truth.
UTA-RLDD self-report and DROZY composite outcomes are defined on different
scales, so their MAE values should not be compared as though identical.

## Compatibility and local verification

The legacy `Zh` package prefix, output JSON field names, checkpoint
locations, fold assignment algorithm and preserved research artifacts remain.
Direct calls to `nested_cv_video_mil` without an explicit output path now
use `FATIGUE_OUTPUT_ROOT` (default: `outputs`) and an E15 timestamp,
instead of an author-specific absolute filesystem path.

Run the dependency-free synthetic unit suite and active-code compilation as
described in the README. Those checks exercise the validation utilities and
Python syntax, not the GPU training/evaluation pipeline. Archived
experiments and saved checkpoints should be treated as historical research
records, not as scientifically revalidated artifacts.
