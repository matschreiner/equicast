def cast_dict(x, target_type):
    if isinstance(x, dict):
        return {k: cast_dict(v, target_type) for k, v in x.items()}
    elif isinstance(x, list):
        return [cast_dict(v, target_type) for v in x]
    elif isinstance(x, tuple):
        return tuple(cast_dict(v, target_type) for v in x)
    else:
        return target_type(x)
