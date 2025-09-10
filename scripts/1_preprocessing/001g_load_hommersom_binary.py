import os

import pandas as pd

from src.folders import get_data_hommersom_binary_folder, get_data_hommersom_folder

df_cultures = pd.read_pickle(
    os.path.join(get_data_hommersom_folder(), "df_hommersom.pkl")
)

# %%
df_cultures = df_cultures[df_cultures["group"] != "Other"]
df_cultures["group"] = pd.Categorical(
    df_cultures["group"], categories=["Control", "CACNA1A"]
)

print(df_cultures["group"].value_counts())

# %%
os.makedirs(get_data_hommersom_binary_folder(), exist_ok=True)
df_cultures.to_pickle(
    os.path.join(
        get_data_hommersom_binary_folder(),
        "df_hommersom_binary.pkl",
    )
)
