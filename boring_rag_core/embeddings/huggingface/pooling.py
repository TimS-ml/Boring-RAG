from enum import Enum
import numpy as np

class Pooling(str, Enum):
    CLS = "cls"
    MEAN = "mean"
    LAST = "last"

    def __call__(self, array: np.ndarray) -> np.ndarray:
        if self == self.CLS:
            return self.cls_pooling(array)
        elif self == self.LAST:
            return self.last_pooling(array)
        return self.mean_pooling(array)

    @classmethod
    def cls_pooling(cls, array: np.ndarray) -> np.ndarray:
        pass

    @classmethod
    def mean_pooling(cls, array: np.ndarray) -> np.ndarray:
        pass

    @classmethod
    def last_pooling(cls, array: np.ndarray) -> np.ndarray:
        pass
