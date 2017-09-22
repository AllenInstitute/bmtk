import threading


class IDGenerator(object):
    """ A simple class for fetching global ids. To get a unqiue global ID class next(), which should be thread-safe. It
    Also has a remove_id(gid) in which case next() will never return the gid. The remove_id function is used for cases
    when using imported networks and we want to elimnate previously created id.

    TODO:
     * Implement a bit array to keep track of already existing gids
     * It might be necessary to implement with MPI support?
    """
    def __init__(self, init_val=0):
        self.__counter = init_val
        self.__taken = set()
        self.__lock = threading.Lock()

    def remove_id(self, gid):
        assert isinstance(gid, (int, long))
        if gid >= self.__counter:
            self.__taken.add(gid)

    def next(self):
        self.__lock.acquire()
        while self.__counter in self.__taken:
            self.__taken.remove(self.__counter)
            self.__counter += 1

        nid = self.__counter
        self.__counter += 1
        self.__lock.release()

        return nid

    def __contains__(self, gid):
        return gid < self.__counter

    def __call__(self, *args, **kwargs):
        if len(args) == 1:
            N = args[0]
        elif 'N' in 'kwargs':
            N = args['N']

        assert(isinstance(N, (int, long)))
        return [self.next() for _ in xrange(N)]

