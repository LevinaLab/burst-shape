from src.cross_validation.split import split_training_and_validation_data
from src.persistence import load_df_bursts, save_cv_params, save_df_bursts

# parameters
burst_extraction_params = "burst_n_bins_50_extend_left_50_extend_right_50"
cv_params = {
    "type": "kfold",
    "n_splits": 5,
    "shuffle": True,
    "random_state": 0,
}

# load data
df_bursts = load_df_bursts(burst_extraction_params)

# split data
df_bursts = split_training_and_validation_data(df_bursts, **cv_params)

# save cv_params
save_cv_params(cv_params, burst_extraction_params)

# save
save_df_bursts(df_bursts, burst_extraction_params, cv_params=cv_params)
