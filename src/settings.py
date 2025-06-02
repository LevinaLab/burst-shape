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
