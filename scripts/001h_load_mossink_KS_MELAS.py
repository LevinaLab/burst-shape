"""
Load mossink results and subsample for binary comparison.

Create two new datasets.
One with "KS" and "Control", and one with "MELAS" and "Control".
"""
from src.persistence import (
    load_burst_matrix,
    load_df_bursts,
    load_df_cultures,
    save_burst_matrix,
    save_df_bursts,
    save_df_cultures,
)

burst_extraction_params = "burst_dataset_mossink_maxISIstart_100_maxISIb_50_minBdur_100_minIBI_500_n_bins_50_normalization_integral_min_length_30"

df_cultures = load_df_cultures(burst_extraction_params)
df_bursts = load_df_bursts(burst_extraction_params)
burst_matrix = load_burst_matrix(burst_extraction_params)
print("Original size")
print("df_cultures", len(df_cultures))
print("df_bursts", len(df_bursts))
print("burst_matrix", burst_matrix.shape)

for selected_group in ["KS", "MELAS"]:
    dataset = f"mossink_{selected_group}"
    burst_extraction_params_subset = f"burst_dataset_{dataset}"
    selection_df_cultures = df_cultures.index.get_level_values("group").isin(
        ["Control", selected_group]
    )
    df_cultures_subset = df_cultures[selection_df_cultures]
    selection_df_bursts = df_bursts.index.get_level_values("group").isin(
        ["Control", selected_group]
    )
    df_bursts_subset = df_bursts[selection_df_bursts]
    burst_matrix_subset = burst_matrix[selection_df_bursts]

    print(f"\n{selected_group}")
    print("Reduced size")
    print("df_cultures", len(df_cultures_subset))
    print("df_bursts", len(df_bursts_subset))
    print("burst_matrix", burst_matrix_subset.shape)

    save_df_cultures(df_cultures_subset, burst_extraction_params_subset)
    save_df_bursts(df_bursts_subset, burst_extraction_params_subset)
    save_burst_matrix(burst_matrix_subset, burst_extraction_params_subset)
