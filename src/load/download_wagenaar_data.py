import os
import subprocess

import numpy as np
from tqdm import tqdm

from src.folders import get_data_folder


def download(
    target_folder=None,
    days=None,
    extract=True,
    density="dense",
    wget_append="--no-check-certificate",
):
    if target_folder is None:
        target_folder = get_data_folder()
    if days is None:
        days = list(np.arange(7, 35, 1))
    print("Downloading Wagenaar data...")
    print(f"Data will be saved in {get_data_folder()}")
    print(f"Folder exists: {os.path.isdir(get_data_folder())}")

    load = True
    path = target_folder  # "../../data/WagenaarData/" #path to store the data
    path_lists = os.path.join(path, "lists")
    path_raw = os.path.join(path, "raw")
    path_extracted = os.path.join(path, "extracted")
    for _path in [path_lists, path_raw, path_extracted]:
        os.makedirs(_path, exist_ok=True)

    # list of all file names for daily spong activity
    list_path = os.path.join(path_lists, f"daily.spont.{density}.text.0.0.0.list")
    list_exists = os.path.isfile(list_path)
    big_list = f"http://neurodatasharing.bme.gatech.edu/development-data/html/wget/daily.spont.{density}.text.0.0.0.list"
    url = big_list
    # download the list of all files
    if load and not list_exists:
        bashCommand = "wget %s -P %s %s" % (url, path_lists, wget_append)
        subprocess.call(bashCommand, shell=True)  # stdout=subprocess.PIPE)
    file_names = []
    with open(list_path, "r") as file:
        for line in file:
            file_names.append(line.split()[0])
    res = list(os.walk(path_raw, topdown=True))
    already_donwnloaded_files = res[0][2]  # all file names

    meta_data = []  # file to store days
    # loop over days
    for day in tqdm(days):
        day_urls = [file for file in file_names if "-%s.spk." % day in file]
        if load:
            day_urls = [
                file
                for file in day_urls
                if file.rsplit("/", 1)[-1] not in already_donwnloaded_files
            ]
        # clear the files that are already downloaded

        for url in day_urls:
            batch_culture = (
                url.split("/")[-1].split("-")[0]
                + "-"
                + url.split("/")[-1].split("-")[1]
            )
            meta_data.append(
                [day, batch_culture, url.split("/")[-1]]
            )  # store day and a file name
            if load:
                bashCommand = "wget %s -P %s %s" % (url, path_raw, wget_append)
                subprocess.call(bashCommand, shell=True)  # stdout=subprocess.PIPE

    # extract .bz2 files from path_raw to path_extracted
    if extract is True:
        print("Extracting files...")
        existing_files = os.listdir(path_extracted)
        files_to_extract = [
            file
            for file in os.listdir(path_raw)
            if file.removesuffix(".bz2") not in existing_files
        ]
        for file in tqdm(files_to_extract):
            bashCommand = f"bzip2 -d {os.path.join(path_raw, file)} -c > {os.path.join(path_extracted, file.removesuffix('.bz2'))}"
            subprocess.call(bashCommand, shell=True)
    print("Manually fix file content...")
    _manually_fix_file_content(path_extracted)
    print("Done.")
    return


def _manually_fix_file_content(path_extracted):
    # batch 8, culture 3, day 17: delete last line if spike time is larger than 1e6
    file = os.path.join(path_extracted, "8-3-17.spk.txt")
    spike_times = np.loadtxt(file)
    if spike_times[-1, 0] > 1e6:
        print(f"Deleting last line in {file}...")
        # delete last line
        with open(file, "r") as f:
            lines = f.readlines()
        with open(file, "w") as f:
            f.writelines(lines[:-1])
            print(f"Delete in {file}: {lines[-1]}")
