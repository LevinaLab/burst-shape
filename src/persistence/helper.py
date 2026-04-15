import ast


def _parse_param_value(value_str: str):
    try:
        return ast.literal_eval(value_str)
    except ValueError or SyntaxError:
        return value_str


def parse_burst_params_string(burst_params: str, defaults: dict):
    if not isinstance(burst_params, str):
        raise TypeError("burst_params must be a string")
    if not burst_params.startswith("burst"):
        raise ValueError("burst_params must start with 'burst'")

    parsed_params = dict(defaults)
    remainder = burst_params[len("burst") :]
    if not remainder:
        return parsed_params

    keys_by_length = sorted(defaults.keys(), key=len, reverse=True)
    index = 0

    while index < len(remainder):
        if remainder[index] != "_":
            raise ValueError(f"Invalid burst parameter string: {burst_params}")

        matched_key = None
        for key in keys_by_length:
            marker = f"_{key}_"
            if remainder.startswith(marker, index):
                matched_key = key
                index += len(marker)
                break

        if matched_key is None:
            raise ValueError(f"Unknown parameter key in: {burst_params}")

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
