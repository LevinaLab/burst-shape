_cv_default_params = {
    "type": "kfold",
    "n_splits": 5,
    "shuffle": True,
    "random_state": 0,
}


def cv_params_to_string(params):
    name = "cv"
    for key, value in params.items():
        if key in _cv_default_params and value != _cv_default_params[key]:
            name += f"_{key}_{value}"
    return name
