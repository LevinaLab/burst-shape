from src.cross_validation.split import split_training_and_validation_data
from src.persistence import load_df_bursts, save_cv_params, save_df_bursts

# parameters
burst_extraction_params = (
    # "burst_n_bins_50_normalization_integral_min_length_30_min_firing_rate_3162_smoothing_kernel_4"
    # "burst_dataset_kapucu_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_500_minSburst_100_n_bins_50_normalization_integral_min_length_30_min_firing_rate_316_smoothing_kernel_4"
    # "burst_dataset_hommersom_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_100_minSburst_100_n_bins_50_normalization_integral_min_length_30"
    # "burst_dataset_inhibblock_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_100_minSburst_100_n_bins_50_normalization_integral_min_length_30"
    # "burst_dataset_mossink_maxISIstart_50_maxISIb_50_minBdur_100_minIBI_500_minSburst_100_n_bins_50_normalization_integral_min_length_30"
    "burst_dataset_mossink_maxISIstart_100_maxISIb_50_minBdur_100_minIBI_500_n_bins_50_normalization_integral_min_length_30"
)
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
