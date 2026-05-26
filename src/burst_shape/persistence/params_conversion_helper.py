import ast
from typing import Any


def _parse_param_value(value_str: str):
    try:
        return ast.literal_eval(value_str)
    except ValueError or SyntaxError:
        return value_str


def params_dict_to_string(params: dict, defaults: dict, startswith: str | None = None):
    if not isinstance(params, dict):
        raise TypeError("params must be a dict")

    name = "" if startswith is None else startswith
    for key, value in params.items():
        if key in defaults and value != defaults[key]:
            name += f"_{key}_{value}"
    return name


def replace_key_value_in_params(params: dict, key: str, value: Any):
    if key not in params:
        raise KeyError(f"Key {key} not found in parameters.")
    new_params = dict(params)
    new_params[key] = value
    return new_params


def params_string_to_dict(burst_params: str, defaults: dict, startswith=None):
    if not isinstance(burst_params, str):
        raise TypeError("burst_params must be a string")

    if startswith is not None:
        if burst_params == startswith:
            remainder = ""
        elif burst_params.startswith(f"{startswith}_"):
            remainder = burst_params[len(startswith) :]
        else:
            raise ValueError(f"burst_params must start with '{startswith}'")
    else:
        remainder = burst_params

    parsed_params = dict(defaults)

    if not remainder:
        return parsed_params

    keys_by_length = sorted(defaults.keys(), key=len, reverse=True)
    index = 0

    while index < len(remainder):
        matched_key = None
        for key in keys_by_length:
            marker = f"_{key}_"
            if remainder.startswith(marker, index):
                matched_key = key
                index += len(marker)
                break

        if matched_key is None:
            raise ValueError(
                f"Unknown parameter key in: {burst_params} (remainder: {remainder})"
            )

        next_key_index = None
        for key in keys_by_length:
            marker = f"_{key}_"
            candidate = remainder.find(marker, index)
            if candidate != -1 and (
                next_key_index is None or candidate < next_key_index
            ):
                next_key_index = candidate

        if next_key_index is None:
            value_str = remainder[index:]
            index = len(remainder)
        else:
            value_str = remainder[index:next_key_index]
            index = int(next_key_index)

        parsed_params[matched_key] = _parse_param_value(value_str)

    return parsed_params
