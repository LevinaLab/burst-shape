import numpy as np
import pandas as pd


def get_burst_level_predictions(
    df_cultures, df_bursts, distance_matrix_square, kth, target_label="target_label"
):
    """Compute the burst-level predictions of KNN-Clustering.

    :param df_cultures: dataframe of cultures, must have a target column named "target_label"
    :param df_bursts: dataframe of bursts, must have a target column named "target_label"
    :param distance_matrix_square: distance matrix of bursts
    :param kth: number of neighbors to consider
    :param target_label: use this to specify if the column "target_label" is named differently
    :return:
        class_labels: list of unique labels present in target column
        relative_votes: matrix of size (n_bursts, len(class_labels)) with relative votes
        true_labels: true labels of individual bursts from target column
        predicted_labels: predicted labels as the maximum votes of relative votes matrix
    """
    n_bursts = len(df_bursts)
    class_frequencies = df_cultures[target_label].value_counts().sort_index()
    class_labels = class_frequencies.index.values
    weight_matrix = np.ones(n_bursts)
    for index in df_cultures.index:
        mask_recording = get_recording_mask(df_bursts, index)
        # normalize by n_bursts per recording
        weight_matrix[mask_recording] /= df_cultures.at[index, "n_bursts"]
        # normalize by number of recordings by class
        weight_matrix[mask_recording] /= class_frequencies[
            df_cultures.at[index, target_label]
        ]
    # weight_matrix = np.ones(n_bursts)
    true_labels = df_bursts[target_label].values

    votes = np.zeros((n_bursts, len(class_labels)))
    for index in df_cultures.index:
        mask_recording = get_recording_mask(df_bursts, index)
        distances_recording = distance_matrix_square[mask_recording]
        distances_recording[:, mask_recording] = np.inf
        nearest_neighbours = np.argpartition(distances_recording, kth=kth, axis=1)[
            :, :kth
        ]
        weights_neighbours = weight_matrix[nearest_neighbours]
        labels_neighbours = true_labels[nearest_neighbours]
        for i_class, class_label in enumerate(class_labels):
            votes[mask_recording, i_class] = np.sum(
                weights_neighbours * (labels_neighbours == class_label),
                axis=1,
            )

    relative_votes = votes / np.sum(votes, axis=1, keepdims=True)
    predicted_labels = class_labels[np.argmax(relative_votes, axis=1)]
    return class_labels, relative_votes, true_labels, predicted_labels


def get_culture_level_predictions(df_cultures, df_bursts, relative_votes, class_labels):
    """Aggregates the results from burst-level KNN clustering to the culture level.

    :param df_cultures: dataframe of cultures where the column "relative_votes" and "predicted_label" will be added
    :param df_bursts: dataframe of bursts
    :param relative_votes: relative votes on individual burst level, result from get_burst_level_predictions()
    :param class_labels: unique class labels present in target column, must be the result from get_burst_level_predictions() to ensure consistency with relative_votes!
    :return: df_cultures with additional columns "relative_votes" and "predicted_label"
    """
    df_cultures["relative_votes"] = pd.Series(dtype="object")
    for index in df_cultures.index:
        mask_recording = get_recording_mask(df_bursts, index)
        df_cultures.at[index, "relative_votes"] = relative_votes[mask_recording].mean(
            axis=0
        )
    df_cultures["predicted_label"] = df_cultures["relative_votes"].apply(
        lambda x: class_labels[np.argmax(x)]
    )

    return df_cultures


def get_recording_mask(df_bursts, culture_index):
    """Find the corresponding indices in df_bursts from a specific recording (culture_index).

    :param df_bursts: datafame of bursts
    :param culture_index: index from df_cultures for which the corresponding indices in df_burst should be found
    :return: boolean mask for df_bursts of size (n_bursts)
    """
    # Number of levels to match
    N = len(culture_index)
    # Create a boolean mask by comparing index levels
    return (
        df_bursts.index.to_frame(index=False).iloc[:, :N].eq(culture_index).all(axis=1)
    )
