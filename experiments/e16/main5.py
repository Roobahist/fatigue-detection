import os
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut

from Zh.src.utils.memmap_handler import MemmapHandler
from Zh.experiments.e15.datasets import get_target_from_video_id
from Zh.experiments.e15.helpers import get_subject_id_from_video_id
from Zh.src.utils.file_folder import get_file_paths

# DATASET_NAME = "DROZY"

# # Base videos live inside the base/ subfolder under each ROI
# ROI_DIRS = {
#     "ROI0": "/work/long_lab/Mojtaba/data/DROZY/data/videos/3.1-embedding/dinov2-base/cls/ROI0/base",
#     "ROI1": "/work/long_lab/Mojtaba/data/DROZY/data/videos/3.1-embedding/dinov2-base/cls/ROI1/base",
#     "ROI2": "/work/long_lab/Mojtaba/data/DROZY/data/videos/3.1-embedding/dinov2-base/cls/ROI2/base",
#     "ROI3": "/work/long_lab/Mojtaba/data/DROZY/data/videos/3.1-embedding/dinov2-base/cls/ROI3/base",
# }
# FEATURES_DIR = "/work/long_lab/Mojtaba/data/DROZY/data/videos/3-augmented_roi/masked_aligned_cropped0.05/features"


DATASET_NAME = "UTA"

# Base videos live inside the base/ subfolder under each ROI
ROI_DIRS = {
    "ROI0": "/work/long_lab/Mojtaba/data/UTA/3.1-embedding/dinov2-base/cls/ROI0/base",
    "ROI1": "/work/long_lab/Mojtaba/data/UTA/3.1-embedding/dinov2-base/cls/ROI1/base",
    "ROI2": "/work/long_lab/Mojtaba/data/UTA/3.1-embedding/dinov2-base/cls/ROI2/base",
    "ROI3": "/work/long_lab/Mojtaba/data/UTA/3.1-embedding/dinov2-base/cls/ROI3/base",
}
FEATURES_DIR = "/work/long_lab/Mojtaba/data/UTA/3-augmented_roi/masked_aligned_cropped0.05/features"


# ─────────────────────────────────────────────
# Pooling
# ─────────────────────────────────────────────

def pool_normalized(arr):
    """Within-video z-score then pool static + temporal dynamics."""
    if arr is None or arr.shape[0] < 2:
        return None
    mean = arr.mean(axis=0, keepdims=True)
    std  = arr.std(axis=0, keepdims=True) + 1e-8
    arr  = (arr - mean) / std

    diff1 = np.diff(arr, axis=0)

    return np.concatenate([
        arr.mean(axis=0),
        arr.std(axis=0),
        np.percentile(arr, 10, axis=0),
        np.percentile(arr, 50, axis=0),
        np.percentile(arr, 90, axis=0),
        np.abs(diff1).mean(axis=0),
        diff1.std(axis=0),
    ])


# ─────────────────────────────────────────────
# Load data — list files directly from ROI0/base
# ─────────────────────────────────────────────

ref_basenames = sorted([
    f for f in os.listdir(ROI_DIRS["ROI0"])
    if f.endswith(".dat")
])
print(f"Found {len(ref_basenames)} base videos in ROI0/base\n")

X_rois  = {roi: [] for roi in ROI_DIRS}
X_blend = []
X_other = []
y       = []
groups  = []

for basename in ref_basenames:
    try:
        # ── Load each ROI embedding ──
        roi_feats = {}
        skip = False
        for roi, roi_dir in ROI_DIRS.items():
            roi_basename = basename.replace("ROI0", roi)
            roi_path     = os.path.join(roi_dir, roi_basename)
            if not os.path.exists(roi_path):
                print(f"  MISSING {roi}: {roi_path}")
                skip = True
                break
            mm  = MemmapHandler(path=roi_path, mode="r")
            emb = mm.read(slice=(slice(None), slice(0, 768)))
            if emb is None or emb.shape[0] == 0:
                print(f"  EMPTY {roi}: {roi_path}")
                skip = True
                break
            feat = pool_normalized(emb)
            if feat is None:
                skip = True
                break
            roi_feats[roi] = feat

        if skip:
            continue

        # ── Load blendshapes + other features ──
        # basename e.g.: "1-1_15fps__masked_aligned_cropped0.05__ROI0__base__embedding_dinov2-base_strategy_cls.dat"
        # features file:  "1-1_15fps__features.dat"  (or inside a base/ subfolder)
        feat_prefix   = basename.split("__")[0]          # "1-1_15fps"
        feat_basename = feat_prefix + "__features.dat"

        # Try flat layout first, then base/ subfolder
        feat_path = os.path.join(FEATURES_DIR, feat_basename)
        if not os.path.exists(feat_path):
            feat_path = os.path.join(FEATURES_DIR, "base", feat_basename)
        if not os.path.exists(feat_path):
            print(f"  MISSING features: {feat_basename}")
            continue

        feat_mm = MemmapHandler(path=feat_path, mode="r")
        blend   = feat_mm.read(slice=(slice(None), slice(7, 59)))
        other   = feat_mm.read(slice=(slice(None), slice(1, 5)))

        if blend is None or blend.shape[0] == 0 or other is None or other.shape[0] == 0:
            print(f"  SKIP (empty features): {basename}")
            continue

        blend_feat = pool_normalized(blend)
        other_feat = pool_normalized(other)
        if blend_feat is None or other_feat is None:
            continue

        for roi in ROI_DIRS:
            X_rois[roi].append(roi_feats[roi])
        X_blend.append(blend_feat)
        X_other.append(other_feat)

        # Pass the full path so helpers can parse subject-session from filename
        full_roi0_path = os.path.join(ROI_DIRS["ROI0"], basename)
        y.append(get_target_from_video_id(full_roi0_path, DATASET_NAME))
        groups.append(get_subject_id_from_video_id(full_roi0_path, DATASET_NAME))

    except Exception as e:
        import traceback
        print(f"  ERROR on {basename}: {e}")
        traceback.print_exc()

if not X_rois["ROI0"]:
    raise RuntimeError(
        "No videos loaded at all.\n"
        "Likely causes:\n"
        "  1. ROI_DIRS paths are wrong — check that the base/ subfolder exists\n"
        "  2. get_target_from_video_id cannot parse filenames in base/ subfolder\n"
        "     (it may expect the old path format without base/ in the path)\n"
        "  3. Features dir layout differs from expected\n"
        "Run:  ls " + ROI_DIRS["ROI0"] + "  to verify filenames."
    )

# Stack arrays
for roi in ROI_DIRS:
    X_rois[roi] = np.stack(X_rois[roi])
X_blend = np.stack(X_blend)
X_other = np.stack(X_other)
y       = np.array(y)
groups  = np.array(groups)

print(f"Dataset: {len(y)} videos, {len(np.unique(groups))} subjects")
for roi in ROI_DIRS:
    print(f"  {roi} shape: {X_rois[roi].shape}")
print(f"  blend shape: {X_blend.shape}")
print(f"  other shape: {X_other.shape}\n")


# ─────────────────────────────────────────────
# Evaluation helper
# ─────────────────────────────────────────────

logo = LeaveOneGroupOut()

def evaluate(name, X):
    n_components = min(15, X.shape[0] - 1, X.shape[1])
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("pca",    PCA(n_components=n_components)),
        ("model",  RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0])),
    ])
    all_preds, all_true = [], []
    for train_idx, test_idx in logo.split(X, y, groups):
        pipe.fit(X[train_idx], y[train_idx])
        preds = pipe.predict(X[test_idx])
        all_preds.extend(preds.tolist())
        all_true.extend(y[test_idx].tolist())
    all_true  = np.array(all_true)
    all_preds = np.array(all_preds)
    mae = mean_absolute_error(all_true, all_preds)
    r2  = r2_score(all_true, all_preds)
    print(f"  {name:35s}  MAE={mae:.4f}  R²={r2:.4f}")
    return mae, r2


# ─────────────────────────────────────────────
# Experiments
# ─────────────────────────────────────────────

print("── Individual ROIs ──")
for roi in ROI_DIRS:
    evaluate(roi, X_rois[roi])

print("\n── ROI combinations ──")
evaluate("ROI0+ROI1",           np.hstack([X_rois["ROI0"], X_rois["ROI1"]]))
evaluate("ROI0+ROI2",           np.hstack([X_rois["ROI0"], X_rois["ROI2"]]))
evaluate("ROI0+ROI3",           np.hstack([X_rois["ROI0"], X_rois["ROI3"]]))
evaluate("ROI1+ROI2",           np.hstack([X_rois["ROI1"], X_rois["ROI2"]]))
evaluate("ROI1+ROI3",           np.hstack([X_rois["ROI1"], X_rois["ROI3"]]))
evaluate("ROI2+ROI3",           np.hstack([X_rois["ROI2"], X_rois["ROI3"]]))
evaluate("ROI0+ROI1+ROI2",      np.hstack([X_rois["ROI0"], X_rois["ROI1"], X_rois["ROI2"]]))
evaluate("ROI0+ROI1+ROI3",      np.hstack([X_rois["ROI0"], X_rois["ROI1"], X_rois["ROI3"]]))
evaluate("ROI0+ROI2+ROI3",      np.hstack([X_rois["ROI0"], X_rois["ROI2"], X_rois["ROI3"]]))
evaluate("ROI1+ROI2+ROI3",      np.hstack([X_rois["ROI1"], X_rois["ROI2"], X_rois["ROI3"]]))
evaluate("all ROIs",            np.hstack([X_rois[r] for r in ROI_DIRS]))

print("\n── All ROIs + other modalities ──")
all_rois = np.hstack([X_rois[r] for r in ROI_DIRS])
evaluate("all ROIs + blend",         np.hstack([all_rois, X_blend]))
evaluate("all ROIs + other",         np.hstack([all_rois, X_other]))
evaluate("all ROIs + blend + other", np.hstack([all_rois, X_blend, X_other]))

print("\n── Late fusion: per-ROI predictions averaged ──")

def get_loo_preds(X):
    n_components = min(15, X.shape[0] - 1, X.shape[1])
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("pca",    PCA(n_components=n_components)),
        ("model",  RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0])),
    ])
    preds_all, true_all = [], []
    for train_idx, test_idx in logo.split(X, y, groups):
        pipe.fit(X[train_idx], y[train_idx])
        preds_all.extend(pipe.predict(X[test_idx]).tolist())
        true_all.extend(y[test_idx].tolist())
    return np.array(preds_all), np.array(true_all)

roi_preds = {}
true_ = None
for roi in ROI_DIRS:
    roi_preds[roi], true_ = get_loo_preds(X_rois[roi])

# Equal weight across all ROIs
fused_equal = np.mean([roi_preds[roi] for roi in ROI_DIRS], axis=0)
print(f"  {'late fusion equal weight':35s}  "
      f"MAE={mean_absolute_error(true_, fused_equal):.4f}  "
      f"R²={r2_score(true_, fused_equal):.4f}")

# R²-weighted (better ROIs get higher weight, negative R² → weight 0)
individual_r2s = [r2_score(true_, roi_preds[roi]) for roi in ROI_DIRS]
print(f"  Individual R²s: { {roi: round(r2, 3) for roi, r2 in zip(ROI_DIRS, individual_r2s)} }")
weights = np.array([max(r2, 0.0) for r2 in individual_r2s])
if weights.sum() > 0:
    weights = weights / weights.sum()
    fused_weighted = sum(w * roi_preds[roi] for w, roi in zip(weights, ROI_DIRS))
    print(f"  {'late fusion R²-weighted':35s}  "
          f"MAE={mean_absolute_error(true_, fused_weighted):.4f}  "
          f"R²={r2_score(true_, fused_weighted):.4f}")
    print(f"  weights: { {roi: round(float(w), 3) for roi, w in zip(ROI_DIRS, weights)} }")

print(f"\n  {'baseline (predict mean)':35s}  "
      f"MAE={mean_absolute_error(y, np.full_like(y, y.mean())):.4f}")