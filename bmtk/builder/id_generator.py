# Copyright 2017. Allen Institute. All rights reserved
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
# following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following
# disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following
# disclaimer in the documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote
# products derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
import threading
import numpy as np


class IDGenerator(object):
    """ A simple class for fetching global ids. To get a unqiue global ID class next(), which should be thread-safe. It
    Also has a remove_id(gid) in which case next() will never return the gid. The remove_id function is used for cases
    when using imported networks and we want to eliminate previously created id.

    TODO:
     * Implement a bit array to keep track of already existing gids
     * It might be necessary to implement with MPI support?
    """
    def __init__(self, init_val=0):
        self.__counter = init_val
        self.__taken = set()
        self.__lock = threading.Lock()

    def remove_id(self, gid):
        assert(np.issubdtype(type(gid), np.integer))
        self.__taken.add(gid)

    def next(self):
        self.__lock.acquire()
        while self.__counter in self.__taken:
            self.__counter += 1

        gid = self.__counter
        self.remove_id(gid)
        self.__counter += 1
        self.__lock.release()
        return gid

    def get_ids(self, size):
        return [self.next() for _ in range(size)]

    def __contains__(self, gid):
        return gid in self.__taken

    def __call__(self, *args, **kwargs):
        if len(args) == 0 and len(kwargs) == 0:
            return self.next()

        if len(args) == 1:
            return self.get_ids(size=args[0])

        elif 'N' in kwargs:
            return self.get_ids(size=kwargs['N'])

        else:
            raise ValueError('Uknown input to id_generator(): {}'.format(kwargs))
