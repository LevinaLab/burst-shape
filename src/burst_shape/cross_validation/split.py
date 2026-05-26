import pandas as pd
from sklearn.model_selection import KFold


def split_training_and_validation_data(
    df_bursts, type="kfold", n_splits=5, shuffle=True, random_state=0
):
    """Split data into training and validation sets.

    Parameters
    ----------
    df_bursts : pd.DataFrame
        Dataframe with burst information.
    type : str, optional
        Type of cross-validation, by default "kfold"
    n_splits : int, optional
        Number of splits, by default 5
    shuffle : bool, optional
        Shuffle data, by default True
    random_state : int, optional
        Random state, by default 0

    Returns
    -------
    pd.DataFrame
        Dataframe with new columns for training and validation sets.
    """
    match type:
        case "kfold":
            kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        case _:
            raise NotImplementedError(f"Unknown cross-validation type: {type}")
    split = list(kf.split(df_bursts))

    # write "cv_i_train" columns, remember that index is multiindex
    for i_split in range(n_splits):
        df_bursts[f"cv_{i_split}_train"] = pd.Series(dtype=bool)
        df_bursts[f"cv_{i_split}_train"] = False
        df_bursts.loc[df_bursts.index[split[i_split][0]], f"cv_{i_split}_train"] = True
    return df_bursts
