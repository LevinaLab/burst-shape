import ast
from typing import Any

# Short aliases used ONLY in the serialized parameter string (to keep folder
# names under the filesystem's 255-byte component limit). These keys are new
# (no pre-existing folder contains them), so abbreviating them does not affect
# parsing of any already-saved folder. The in-memory dict always uses the full
# key names; abbreviation happens only at the string boundary.
_KEY_ABBREVIATIONS = {
    "mcs_min_simultaneous": "msim",
    "mcs_min_participating": "mpart",
}


def _string_key(key: str) -> str:
    return _KEY_ABBREVIATIONS.get(key, key)


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
            name += f"_{_string_key(key)}_{value}"
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

    # (marker, full_key) pairs, using the abbreviated string key, longest first
    markers = [(f"_{_string_key(key)}_", key) for key in defaults.keys()]
    markers.sort(key=lambda mk: len(mk[0]), reverse=True)
    index = 0

    while index < len(remainder):
        matched_key = None
        for marker, full_key in markers:
            if remainder.startswith(marker, index):
                matched_key = full_key
                index += len(marker)
                break

        if matched_key is None:
            raise ValueError(
                f"Unknown parameter key in: {burst_params} (remainder: {remainder})"
            )

        next_key_index = None
        for marker, _full_key in markers:
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
