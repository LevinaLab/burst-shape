import warnings


def get_dataset_from_burst_extraction_params(burst_extraction_params):
    if isinstance(burst_extraction_params, dict):
        return burst_extraction_params["dataset"]
    else:
        if "kapucu" in burst_extraction_params:
            dataset = "kapucu"
        elif "hommersom_test" in burst_extraction_params:
            dataset = "hommersom_test"
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
        case "hommersom_test":
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
        case "hommersom_test":
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


def get_citation_doi_link(dataset):
    match dataset:
        case "kapucu":
            citation = "Kapucu et al. (2022)"
            doi_link = "https://doi.org/10.1038/s41597-022-01242-4"
        case "hommersom_test":
            citation = "Hommersom et al. (2024)"
            doi_link = "https://doi.org/10.1101/2024.03.18.585506"
        case "inhibblock":
            citation = "Vinogradov et al. (2024)"
            doi_link = "https://doi.org/10.1101/2024.08.21.608974"
        case "mossink":
            citation = "Mossink et al. (2021)"
            doi_link = "https://doi.org/10.17632/bvt5swtc5h.1"
        case "wagenaar":
            citation = "Wagenaar et al. (2006)"
            doi_link = "https://doi.org/10.1186/1471-2202-7-11"
        case _:
            citation = "the relevant literature"
            doi_link = None
            warnings.warn(f"Citation and doi link not defined for dataset {dataset}.")
    return citation, doi_link
