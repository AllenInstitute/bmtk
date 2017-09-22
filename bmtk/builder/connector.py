import functor_cache


def create(connector, **params):
    return CONNECTOR_CACHE.create(connector, **params)


def register(name, func):
    CONNECTOR_CACHE.register(name, func)


CONNECTOR_CACHE = functor_cache.FunctorCache()
register('passthrough', lambda *_: {})
