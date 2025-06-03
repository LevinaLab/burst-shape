def get_dataset_from_burst_extraction_params(burst_extraction_params):
    if isinstance(burst_extraction_params, dict):
        return burst_extraction_params["dataset"]
    else:
        if "kapucu" in burst_extraction_params:
            dataset = "kapucu"
        elif "hommersom" in burst_extraction_params:
            dataset = "hommersom"
        elif "inhibblock" in burst_extraction_params:
            dataset = "inhibblock"
        elif "mossink" in burst_extraction_params:
            dataset = "mossink"
        else:
            assert (
                "dataset" not in burst_extraction_params
            ), "Unknown dataset, cannot use this function."
            dataset = "wagenaar"
        return dataset


def get_chosen_spectral_embedding_params(dataset):
    match dataset:
        case "kapucu":
            clustering_params = (
                "spectral_affinity_precomputed_metric_wasserstein_n_neighbors_150"
            )
        case "hommersom":
            clustering_params = (
                "spectral_affinity_precomputed_metric_wasserstein_n_neighbors_6"
            )
        case "inhibblock":
            clustering_params = (
                "spectral_affinity_precomputed_metric_wasserstein_n_neighbors_85"
            )
        case "mossink":
            clustering_params = (
                "spectral_affinity_precomputed_metric_wasserstein_n_neighbors_85"
            )
        case "wagenaar":
            clustering_params = (
                "spectral_affinity_precomputed_metric_wasserstein_n_neighbors_150"
            )
        case _:
            raise NotImplementedError(f"Dataset {dataset} not implemented.")
    return clustering_params


def get_chosen_spectral_clustering_params(dataset):
    match dataset:
        case "kapucu":
            n_clusters = 4
        case "hommersom":
            n_clusters = 4
        case "inhibblock":
            n_clusters = 4
        case "mossink":
            n_clusters = 4
        case "wagenaar":
            n_clusters = 6
        case _:
            raise NotImplementedError(f"Dataset {dataset} not implemented.")
    clustering_params = get_chosen_spectral_embedding_params(dataset)
    return clustering_params, n_clusters
