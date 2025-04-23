from functools import wraps


def empty_decorator(func):
    @wraps(func)
    def defunct(*args, **kwargs):
        return func(*args, **kwargs)
    return defunct
