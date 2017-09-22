import functools


class FunctorCache(object):
    def __init__(self):
        self.cache = {}

    def create(self, connector, **params):
        if params is None:
            params = {}

        if isinstance(connector, basestring):
            # TODO: don't do this, a user may want to return a string in connection_map params
            func = self.cache[connector]
            return functools.partial(func, **params)

        elif isinstance(connector, dict):
            return lambda *args: connector

        elif isinstance(connector, list):
            # for the iterator we want to pass backs lists as they are
            return connector

        elif callable(connector):
            return functools.partial(connector, **params)

        else:
            # should include all numericals, non-callable objects and tuples
            return lambda *args: connector

    def register(self, name, func):
        self.cache[name] = func
