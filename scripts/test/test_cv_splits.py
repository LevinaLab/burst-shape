"""Confirm that cv splits are the same when repeating with the same seed."""
import numpy as np
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedShuffleSplit,
)
from tqdm import tqdm

cv_type = (
    "RepeatedStratifiedKFold"
    # "StratifiedShuffleSplit"
)


# %% prediction with xgboost
random_state = 1234567890
n_splits = 100


def _get_outer_cv(cv_type):
    match cv_type:
        case "StratifiedShuffleSplit":
            outer_cv = StratifiedShuffleSplit(
                n_splits=n_splits, test_size=0.2, random_state=random_state
            )
        case "RepeatedStratifiedKFold":
            outer_cv = RepeatedStratifiedKFold(
                # outer_cv = RepeatedKFold(
                n_splits=5,
                n_repeats=n_splits // 5,
                random_state=random_state,
            )
        case _:
            raise NotImplementedError(f"Unknown cv type: {cv_type}")
    return outer_cv


X1 = np.random.randn(100, 5)
X2 = np.random.randn(100, 3)
y = np.random.randint(0, 2, size=100)

outer_cv = _get_outer_cv(cv_type)
for train_idx1, test_idx1 in tqdm(
    outer_cv.split(X1, y), total=n_splits, desc="Outer loop of cv"
):
    pass

outer_cv = _get_outer_cv(cv_type)
for train_idx2, test_idx2 in tqdm(
    outer_cv.split(X2, y), total=n_splits, desc="Outer loop of cv"
):
    pass

outer_cv = _get_outer_cv(cv_type)
for train_idx3, test_idx3 in tqdm(
    outer_cv.split(np.zeros_like(y), y), total=n_splits, desc="Outer loop of cv"
):
    pass

print("Split index is identical", np.all(train_idx1 == train_idx2))
print("Split index is identical", np.all(train_idx1 == train_idx3))
