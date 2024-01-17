import os

from src.folders import get_data_folder


def download(target_folder=None):
    print("Downloading Wagenaar data...")
    if target_folder is None:
        data_folder = get_data_folder()
    print(f"Data will be saved in {get_data_folder()}")
    print(f"Folder exists: {os.path.isdir(get_data_folder())}")
    raise NotImplementedError("Download not implemented yet.")
    print("Done.")
    return
