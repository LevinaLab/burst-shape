"""

Download data from here: https://data.mendeley.com/datasets/bvt5swtc5h/1
"""

import os
import re

import numpy as np
import pandas as pd
from scipy.io import loadmat
from tqdm import tqdm

from src.folders import get_data_mossink_folder

na = np.array

load_spikes = True

print("Top folder exists", os.path.exists(get_data_mossink_folder()))
path = os.path.join(get_data_mossink_folder(), "Peak trains")
print("Data folder exists: ", os.path.exists(path))

save_path = os.path.join(
    get_data_mossink_folder(),
    "df_mossink.pkl",
)

if load_spikes is True and os.path.exists(save_path):
    print("Loading spikes from", save_path)
    df = pd.read_pickle(save_path)
else:
    print("Building dataframe ...")
    res = list(os.walk(path, topdown=True))
    folders = res[0][1]

    def _get_mossink_subject_info(group, subject_id, folder):
        isogenic_control = None
        match group:
            case "Control":
                coating = "Human" if subject_id in [1, 2, 3, 4, 5, 7] else "Mouse"
                if subject_id == 6:
                    if "mLAM" in folder:
                        coating = "Mouse"
                    else:
                        coating = "Human"
                        assert "hLAM" in folder
                match subject_id:
                    case 1:
                        gender, age, genotype = "F", 36, "Control"
                    case 2:
                        gender, age, genotype = "F", 17, "0% HP"
                    case 3:
                        gender, age, genotype = "M", 30, "0% HP"
                    case 4:
                        gender, age, genotype = "F", 45, "0% HP"
                    case 5:
                        gender, age, genotype = "M", 42, "0% HP"
                    case 6:
                        gender, age, genotype = "M", 30, "Control"
                    case 7:
                        gender, age, genotype = "F", 41, "Control"
                    case 8:
                        gender, age, genotype = "M", 9, "Control"
                    case 9:
                        gender, age, genotype = "F", 34, "MOS_WT"
                    case 10:
                        gender, age, genotype = "M", 51, "Control"
            case "MELAS":
                coating = "Human"
                match subject_id:
                    case 1:
                        gender, age, genotype = "F", 17, "m.3243A > G 66% HP"
                        isogenic_control = 2
                    case 2:
                        gender, age, genotype = "F", 45, "m.3243A > G 65% HP"
                        isogenic_control = 4
                    case 3:
                        gender, age, genotype = "M", 42, "m.3243A > G 61% HP"
                        isogenic_control = 5
            case "KS":
                coating = "Mouse"
                match subject_id:
                    case 1:
                        gender, age, genotype = "F", 13, "c.3181-80_3233del"
                    case 2:
                        gender, age, genotype = "F", 12, "c.3125G > A"
                    case 3:
                        gender, age, genotype = (
                            "F",
                            34,
                            "Heterozygous 233 kbp deletion of chromosome 9",
                        )
                        isogenic_control = 9
                    case 4:
                        gender, age, genotype = (
                            "M",
                            51,
                            "Heterozygous EHMT1 mutation in Exon 2",
                        )
                        isogenic_control = 10
        return gender, age, genotype, isogenic_control, coating

    groups = []
    subject_ids = []
    genders = []
    ages = []
    genotypes = []
    isogenic_controls = []
    coatings = []
    batches = []
    wells = []
    noBDNF_list = []
    sts = []
    gids = []
    # TODO add coating info
    # TODO check for more info from paper
    # Pretty code, but good for debugging
    # Get each control
    for folder in tqdm(folders):
        # Get the name
        if "Control" in folder:
            group = "Control"
        elif "MELAS" in folder:
            group = "MELAS"
        elif "KS" in folder:
            group = "KS"
        else:
            raise ValueError(f"{folder} is not a valid folder")
        subject_id = int(re.search(r"\d+", folder).group())
        gender, age, genotype, isogenic_control, coating = _get_mossink_subject_info(
            group, subject_id, folder
        )

        local_path = path + "/" + folder
        res_ = list(os.walk(local_path, topdown=True))
        folders_local = res_[0][1]
        # Take each well and each day
        for folder_local in folders_local:
            folder_local_split = folder_local.split("_")
            batch = folder_local_split[0]
            well = folder_local_split[1]
            well_id = well[-2:]
            if len(folder_local_split) == 3:
                noBDNF = True
            else:
                noBDNF = False
            assert folder_local == f"{batch}_Well{well_id}{'_noBDNF' if noBDNF else ''}"
            noBDNF_list.append(noBDNF)

            recording_path = path + "/" + folder + "/" + folder_local
            # Filter irrelevant analysis folders that sometimes appeaer
            res_ = list(os.walk(recording_path, topdown=True))
            head_folders = [r for r in res_ if "ptrain" in r[0]]
            internal_path = head_folders[0][0]
            channel_files = head_folders[0][2]
            batches.append(int(batch))
            wells.append(well_id)
            groups.append(group)
            subject_ids.append(subject_id)
            genders.append(gender)
            ages.append(age)
            genotypes.append(genotype)
            isogenic_controls.append(isogenic_control)
            coatings.append(coating)

            # Stack all channgels
            if load_spikes:
                st = []
                gid = []
                for channel in channel_files:
                    uid = channel.split("_")[1]
                    uid = int(uid.split(".")[0])
                    channel_path = internal_path + "/" + channel
                    # channel_path = '/home/ovinogradov/Projects/ReducedBursting/data/Moss/Peak_Trains/Control 2/3_WellA6/WellA6_PeakDetectionMAT_PLP2ms_RP1ms/WellA6_ptrain_/ptrain_43.mat'
                    mat = loadmat(channel_path)
                    peaks = na(mat["peak_train"].todense())
                    spikes = peaks[:, 0]
                    # fs =
                    st_ = np.where(spikes)[0]
                    st.extend(st_)
                    gid.extend([uid] * len(st_))
                sts.append(st.copy())
                gids.append(gid.copy())
            else:
                sts.append([])
                gids.append([])

    df = pd.DataFrame(
        {
            "group": groups,
            "subject_id": subject_ids,
            "gender": genders,
            "age": ages,
            "genotype": genotypes,
            "isogenic_control": isogenic_controls,
            "coating": coatings,
            "batch": batches,
            "well": wells,
            "noBDNF": noBDNF_list,
            "times": sts,
            "gid": gids,
        }
    )

    for index in tqdm(df.index, desc="Sorting"):
        order = np.argsort(df.at[index, "times"])
        df.at[index, "times"] = np.array(df.at[index, "times"])[order]
        df.at[index, "gid"] = np.array(df.at[index, "gid"])[order]

        df.at[index, "times"] = (
            df.at[index, "times"] / 10_000
        )  # convert to seconds from 10kHz sampling frequency

    df["well_idx"] = pd.Series(0, dtype=int)
    for row in df[["group", "subject_id"]].drop_duplicates().itertuples():
        group, subject_id = row.group, row.subject_id
        n_wells = len(df.loc[(df["group"] == group) & (df["subject_id"] == subject_id)])
        df.loc[
            (df["group"] == group) & (df["subject_id"] == subject_id),
            "well_idx",
        ] = np.arange(n_wells)
    df["well_idx"] = df["well_idx"].astype(int)

    df.set_index(["group", "subject_id", "well_idx"], inplace=True, drop=True)

    if load_spikes is True:
        df.to_pickle(save_path)
        print(f"Saved to {save_path}")
# %%
df.reset_index(inplace=True)
print(df["group"].value_counts())
print(df["subject_id"].value_counts())
print(df["batch"].value_counts())
print(df["well"].value_counts())
print(df["noBDNF"].value_counts())
print(df[["group", "subject_id"]].value_counts())
