from typing import Any, Callable


def cast_dict(x: Any, target_type: Callable) -> Any:
    """
    Recursively cast values in nested dict/list/tuple structures.

    Args:
        x: Input structure (dict, list, tuple, or scalar)
        target_type: Callable to cast scalar values

    Returns:
        Structure with all scalar values cast to target_type
    """
    if isinstance(x, dict):
        return {k: cast_dict(v, target_type) for k, v in x.items()}
    elif isinstance(x, list):
        return [cast_dict(v, target_type) for v in x]
    elif isinstance(x, tuple):
        return tuple(cast_dict(v, target_type) for v in x)
    else:
        return target_type(x)

