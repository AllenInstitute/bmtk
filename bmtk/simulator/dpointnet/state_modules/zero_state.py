
class ZeroStateModule:
    def __init__(self, rnn, **kwargs):
        self._rnn = rnn
        # self._batch_size = self._rnn.batch_size
        self._dtype = self._rnn.dtype

    def get_state(self, batch_size=None, **kwargs):
        return self._rnn.cell.zero_state(batch_size or self._rnn.adjusted_batch_size, self._dtype)
