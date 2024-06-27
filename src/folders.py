import os

# Set the project folder dynamically based on the location of this file
_project_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_data_folder = "data"
_fig_folder = "figures"
_results_folder = "results"


def get_data_folder():
    """Returns the path to the data folder.

    Returns:
        str: path to the data folder
    """
    return os.path.join(_project_folder, _data_folder)


def get_fig_folder():
    """Returns the path to the figure folder.

    Returns:
        str: path to the figure folder
    """
    return os.path.join(_project_folder, _fig_folder)


def get_results_folder():
    """Returns the path to the results folder.

    Returns:
        str: path to the results folder
    """
    return os.path.join(_project_folder, _results_folder)
