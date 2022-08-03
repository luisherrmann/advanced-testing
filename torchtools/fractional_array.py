import pickle
import numpy as np

def _is_int(x):
    return str(x.dtype).startswith('int')

class FractionalArray(object):

    def __init__(self, a: np.ndarray, b: np.ndarray):
        super().__init__()
        if not isinstance(a, np.ndarray): raise ValueError
        if not _is_int(a):  raise TypeError
        if not isinstance(b, np.ndarray): raise ValueError
        if not _is_int(b):  raise TypeError
        if not a.shape == b.shape:
            raise ValueError(f"Tensor shapes do not match: {a.shape} and {b.shape}!")
        self.a = a
        self.b = b

    @classmethod
    def from_pkl(cls, file_a: str, file_b: str):
        with open(file_a, 'rb') as f:
            a = pickle.load(f)
        with open(file_b, 'rb') as f:
            b = pickle.load(f)
        return cls(a, b)

    def resolve(self) -> np.ndarray:
        a_, b_ = self.a.reshape(-1), self.b.reshape(-1)
        zeros_mask = b_ == 0
        c = np.empty(a_.shape, dtype=np.int32)
        c[zeros_mask] = np.nan
        c[~zeros_mask] = a_[~zeros_mask] / b_[~zeros_mask]
        c = c.reshape(self.a.shape)
        return c