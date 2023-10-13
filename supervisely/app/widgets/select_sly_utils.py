import os


def _get_int_env(env_key: str) -> int:
    res = os.environ.get(env_key)
    if res is not None:
        res = int(res)
    return res


def _get_int_or_env(value: int, env_key: str) -> int:
    if value is not None:
        return int(value)
    return _get_int_env(env_key)
