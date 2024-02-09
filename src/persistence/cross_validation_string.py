_cv_default_params = {
    "type": "kfold",
    "n_splits": 5,
    "shuffle": True,
    "random_state": 0,
}


def _cv_params_to_string(params: str or dict, i_split=None):
    if isinstance(params, dict):
        name = "cv"
        for key, value in params.items():
            if key in _cv_default_params and value != _cv_default_params[key]:
                name += f"_{key}_{value}"
    else:
        name = params
    if i_split is not None:
        name += f"_split_{i_split}"
    return name
