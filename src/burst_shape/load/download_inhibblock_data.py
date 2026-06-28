"""Download the inhibition-block (Vinogradov et al., 2024) spike data.

The spike CSV files used in the tutorial are hosted on figshare
(https://doi.org/10.6084/m9.figshare.27110542, record 27110542). They are
downloaded into the folder structure under ``data/data_inhibblock/`` expected by
``scripts/1_preprocessing/001d_preload_inhibblock.py`` and the tutorial, one
spike CSV per recording day (17 and 18).

Files that already exist are skipped, which makes :func:`download` idempotent.
"""

import os
import urllib.request

from burst_shape.folders import get_data_inhibblock_folder

# day -> file metadata. Direct download urls obtained from the figshare API for
# record 27110542.
_FIGSHARE_FILES = {
    17: {
        "filename": "day17_potassium4.2_spikes.csv",
        "day_folder": "ctx_14.03.22_Hertie",
        "url": "https://ndownloader.figshare.com/files/49426732",
        "size_mb": 1739,
    },
    18: {
        "filename": "day18_potassium4.2_spikes2.csv",
        "day_folder": "ctx_03.04.22_Hertie",
        "url": "https://ndownloader.figshare.com/files/49426729",
        "size_mb": 343,
    },
}


def _progress_hook(filename):
    def hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, downloaded * 100 / total_size)
            print(
                f"\r  {filename}: {percent:5.1f}% "
                f"({downloaded / 1e6:.0f} / {total_size / 1e6:.0f} MB)",
                end="",
                flush=True,
            )

    return hook


def download(days=(17, 18), target_folder=None, overwrite=False):
    """Download the inhibblock spike CSVs from figshare.

    Args:
        days: iterable of recording days to download (subset of ``{17, 18}``).
            Use ``[18]`` for the smaller ~0.3 GB file only.
        target_folder: destination ``data_inhibblock`` folder. Defaults to
            :func:`burst_shape.folders.get_data_inhibblock_folder`.
        overwrite: if ``False`` (default), existing files are skipped.

    Returns:
        dict mapping day to the absolute path the file was saved to.
    """
    if target_folder is None:
        target_folder = get_data_inhibblock_folder()

    print("Downloading inhibblock data (Vinogradov et al., 2024) from figshare...")
    paths = {}
    for day in days:
        if day not in _FIGSHARE_FILES:
            raise ValueError(
                f"No figshare file for day {day}; choose from {list(_FIGSHARE_FILES)}."
            )
        info = _FIGSHARE_FILES[day]
        dest_dir = os.path.join(target_folder, info["day_folder"], "extracted_data")
        os.makedirs(dest_dir, exist_ok=True)
        dest = os.path.join(dest_dir, info["filename"])
        paths[day] = dest

        if os.path.exists(dest) and not overwrite:
            print(f"  {info['filename']} already exists, skipping.")
            continue

        print(f"  {info['filename']} (~{info['size_mb']} MB) <- {info['url']}")
        urllib.request.urlretrieve(
            info["url"], dest, reporthook=_progress_hook(info["filename"])
        )
        print()  # newline after the progress indicator
    print("Done.")
    return paths
