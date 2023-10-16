import numpy as np
from typing import List, Optional, Tuple

__all__ = ['AverageMeter', 'HistoryBuffer']


class AverageMeter(object):
    def __init__(self, n):
        if isinstance(n, (list, tuple)):
            self.len = len(n)
            for i in n:
                assert isinstance(i, str)
            self.name = n
        elif isinstance(n, int):
            self.len = n
        else:
            raise TypeError(type(n))
        self.reset()

    def update(self, value):
        self._value_vector += np.asarray(value, dtype=self._value_vector.dtype)
        self.count += 1

    def reset(self):
        self._value_vector = np.zeros(self.len, dtype=np.float32)
        self.count = 0

    def __getitem__(self, item):
        return np.round(self._value_vector[item] / self.count, 6)

    @property
    def mean(self):
        return np.round(self._value_vector / self.count, 6)

    def __len__(self):
        return self.len

    def __iter__(self):
        output = []
        for i in range(self.len):
            if hasattr(self, 'name'):
                output.append((self.name[i], self[i]))
            else:
                output.append(self[i])

        return iter(output)


class HistoryBuffer:
    """
    Track a series of scalar values and provide access to smoothed values over a
    window or the global average of the series.
    """

    def __init__(self, max_length: int = 1000000) -> None:
        """
        Args:
            max_length: maximal number of values that can be stored in the
                buffer. When the capacity of the buffer is exhausted, old
                values will be removed.
        """
        self._max_length: int = max_length
        self._data: List[Tuple[float, float]] = []  # (value, iteration) pairs
        self._count: int = 0
        self._global_avg: float = 0

    def update(self, value: float, iteration: Optional[float] = None) -> None:
        """
        Add a new scalar value produced at certain iteration. If the length
        of the buffer exceeds self._max_length, the oldest element will be
        removed from the buffer.
        """
        if iteration is None:
            iteration = self._count
        if len(self._data) == self._max_length:
            self._data.pop(0)
        self._data.append((value, iteration))

        self._count += 1
        self._global_avg += (value - self._global_avg) / self._count

    def latest(self) -> float:
        """
        Return the latest scalar value added to the buffer.
        """
        return self._data[-1][0]

    def median(self, window_size: int) -> float:
        """
        Return the median of the latest `window_size` values in the buffer.
        """
        return np.median([x[0] for x in self._data[-window_size:]])

    def avg(self, window_size: int) -> float:
        """
        Return the mean of the latest `window_size` values in the buffer.
        """
        return np.mean([x[0] for x in self._data[-window_size:]])

    def global_avg(self) -> float:
        """
        Return the mean of all the elements in the buffer. Note that this
        includes those getting removed due to limited buffer storage.
        """
        return self._global_avg

    def values(self) -> List[Tuple[float, float]]:
        """
        Returns:
            list[(number, iteration)]: content of the current buffer.
        """
        return self._data