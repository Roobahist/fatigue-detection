import os

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

from Zh.experiments.e15.datasets import get_target_from_video_id
from Zh.experiments.e15.helpers import get_subject_id_from_video_id
from Zh.src.utils.file_folder import get_file_paths
from Zh.src.utils.memmap_handler import MemmapHandler

# DATASET_NAME = "DROZY"  # or "UTA"
# EMBEDDINGS_DIR = (
#     "/work/long_lab/Mojtaba/data/DROZY/data/videos/3.1-embedding/dinov2-base/cls/ROI0"
# )

DATASET_NAME = "UTA"  # or "UTA"
EMBEDDINGS_DIR = (
    "/work/long_lab/Mojtaba/data/UTA/3.1-embedding/dinov2-base/cls/ROI0"
)

video_ids = get_file_paths(input_path=EMBEDDINGS_DIR, depth=3, EXT_LIST=".dat")

# Only use base (non-augmented) videos
# video_ids = [v for v in video_ids if "__base__" in v]

print(f"Found {len(video_ids)} base videos")

X = []
y = []
groups = []  # subject IDs for leave-one-subject-out CV

for vid in video_ids:
    try:
        memmap = MemmapHandler(path=vid, mode="r")
        # Read all frames, all 768 embedding dims
        emb = memmap.read(slice=(slice(None), slice(0, 768)))
        if emb is None or emb.shape[0] == 0:
            print(f"  SKIP (empty): {vid}")
            continue
        # Simple average pooling over all frames -> one vector per video
        avg_emb = emb.mean(axis=0)  # shape: (768,)
        X.append(avg_emb)

        target = get_target_from_video_id(vid, DATASET_NAME)
        y.append(target)

        subject_id = get_subject_id_from_video_id(vid, DATASET_NAME)
        groups.append(subject_id)

    except Exception as e:
        print(f"  ERROR on {vid}: {e}")

X = np.stack(X)  # (N_videos, 768)
y = np.array(y)  # (N_videos,)
groups = np.array(groups)

print(f"\nDataset: {X.shape[0]} videos, {len(np.unique(groups))} subjects")
print(
    f"Label distribution: min={y.min()}, max={y.max()}, mean={y.mean():.2f}, std={y.std():.2f}"
)

# --- Leave-One-Subject-Out Cross-Validation ---
logo = LeaveOneGroupOut()
scaler = StandardScaler()
model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0, 1000.0])

all_preds = []
all_true = []

for fold_i, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)

    all_preds.extend(preds.tolist())
    all_true.extend(y_test.tolist())

    subject = np.unique(groups[test_idx])[0]
    mae = mean_absolute_error(y_test, preds)
    print(
        f"  Subject {subject:2d}: MAE={mae:.3f}, true={y_test.tolist()}, pred={[round(p,2) for p in preds.tolist()]}"
    )

all_true = np.array(all_true)
all_preds = np.array(all_preds)

print(f"\n{'='*50}")
print(f"LOSO Results (Leave-One-Subject-Out)")
print(f"  MAE:  {mean_absolute_error(all_true, all_preds):.4f}")
print(f"  R²:   {r2_score(all_true, all_preds):.4f}")
print(
    f"  Baseline MAE (predict mean): {mean_absolute_error(all_true, np.full_like(all_true, all_true.mean())):.4f}"
)
print(f"{'='*50}")
