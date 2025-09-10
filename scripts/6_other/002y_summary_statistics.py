"""
Print summary statistics for publication.

Print summary statistics like number of recordings, bursts, ...
Print it in a table that can be copy-pasted to latex.
"""
import pandas as pd

from src.persistence import load_df_cultures
from src.settings import get_dataset_from_burst_extraction_params

burst_extraction_params_list = [
    "burst_dataset_inhibblock_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_100_minSburst_100_n_bins_50_normalization_integral_min_length_30",
    "burst_dataset_hommersom_binary_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_100_minSburst_100_n_bins_50_normalization_integral_min_length_30",
    "burst_dataset_mossink_maxISIstart_100_maxISIb_50_minBdur_100_minIBI_500_n_bins_50_normalization_integral_min_length_30",
    "burst_n_bins_50_normalization_integral_min_length_30_min_firing_rate_3162_smoothing_kernel_4",
    "burst_dataset_kapucu_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_500_minSburst_100_n_bins_50_normalization_integral_min_length_30_min_firing_rate_316_smoothing_kernel_4",
    # "burst_dataset_mossink_KS",
    # "burst_dataset_mossink_MELAS",
]

statistics_list = []
for burst_extraction_params in burst_extraction_params_list:
    dataset = get_dataset_from_burst_extraction_params(burst_extraction_params)
    print(f"Processing dataset: {dataset}")
    df_cultures = load_df_cultures(burst_extraction_params)
    statistics_list.append(
        {
            "dataset": dataset,
            "n_cultures": df_cultures.shape[0],
            "n_bursts": df_cultures["n_bursts"].sum(),
            "max_time": df_cultures["burst_start_end"]
            .apply(lambda x: x[-1][1] if len(x) > 0 else 0)
            .max()
            / 1000
            / 60,  # convert to minutes
        }
    )

df_statistics = pd.DataFrame(statistics_list)
df_statistics["max_time_rounded"] = df_statistics["max_time"].round(0).astype(int)
df_statistics["total_time"] = (
    df_statistics["n_cultures"] * df_statistics["max_time_rounded"]
)

# %% print
print(df_statistics.to_string(line_width=1000))
total_n_cultures = df_statistics["n_cultures"].sum()
total_n_bursts = df_statistics["n_bursts"].sum()
total_time = df_statistics["total_time"].sum()
print(f"Total number of cultures: {total_n_cultures}")
print(f"Total number of bursts: {total_n_bursts}")
print(f"Total time in minutes: {total_time}")
print(f"Total time in hours: {total_time / 60:.2f}")

# %% print latex table
dataset_mapping = {
    "inhibblock": "Blocked inhibition",
    "hommersom_binary": "CACNA1A",
    "wagenaar": "Developing rat",
    "kapucu": "Rat vs hPSC",
    "mossink": "hPSC disease",
}
print(
    df_statistics.to_latex(
        index=False,
        float_format="%.2f",
        column_format="l|c|c|c|c",
        columns=["dataset", "n_cultures", "n_bursts", "max_time_rounded", "total_time"],
        formatters={"dataset": lambda x: dataset_mapping.get(x, x)},
        header=[
            "Dataset",
            "\#cultures",
            "\#bursts",
            "Time (min)",
            "Total time (min)",
        ],
        caption="Summary statistics of datasets used in the analysis.",
        label="tab-supp:datasets",
        position="ht",
        escape=False,  # allow LaTeX formatting
        na_rep="--",  # replace NaN with --
    )
)
