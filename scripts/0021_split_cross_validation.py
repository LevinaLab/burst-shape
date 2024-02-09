import os
import json

import pandas as pd

from src.cross_validation.split import split_training_and_validation_data
from src.persistence import get_burst_folder, cv_params_to_string

burst_extraction_params = "burst_n_bins_50_extend_left_50_extend_right_50"
cv_params = {
    "type": "kfold",
    "n_splits": 5,
    "shuffle": True,
    "random_state": 0,
}

cv_string = cv_params_to_string(cv_params)

df_bursts = pd.read_pickle(
    os.path.join(
        get_burst_folder(burst_extraction_params),
        "002_wagenaar_bursts_df.pkl",
    )
)

df_bursts = split_training_and_validation_data(df_bursts, **cv_params)

# save cv_params
with open(
    os.path.join(get_burst_folder(burst_extraction_params), f"{cv_string}_params.json"),
    "w",
) as f:
    json.dump(cv_params, f, indent=4)

# save
df_bursts.to_pickle(
    os.path.join(
        get_burst_folder(burst_extraction_params),
        f"002_wagenaar_bursts_df_{cv_string}.pkl",
    )
)
