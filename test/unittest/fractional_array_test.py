from multiprocessing import context
import shutil
import unittest
import os
from contextlib import ExitStack
import numpy as np
import pickle
import tempfile

from torchtools.fractional_array import FractionalArray


class FractionalArrayTest(unittest.TestCase):

    def setUp(self) -> None:
        if not os.path.exists('tmp'):
            os.makedirs('tmp')

    def tearDown(self) -> None:
        shutil.rmtree('tmp')

    def test__init__(self):
        inputs = [
            # a, b
            (np.array([2], dtype=int), np.array([1], dtype=int), None),
            (np.array([], dtype=int), np.array([], dtype=int), None),
            (np.array([2], dtype=int), np.array([2, 1], dtype=int), ValueError),
            (
                np.array([2, 2], dtype=int).reshape([2, 1]), 
                np.array([1, 1], dtype=int).reshape([1, 2]),
                ValueError
            ),
            (np.array([2], dtype=int), np.array([1], dtype=float), TypeError),
            (np.array([2], dtype=float), np.array([1], dtype=int), TypeError)
        ]
        for input in inputs:
            a, b, err_type = input
            if err_type is None:
                FractionalArray(a, b)
            else:
                self.assertRaises(err_type, lambda: FractionalArray(a, b))


    def test_from_pkl(self):
        with ExitStack() as stack:
            fa = stack.enter_context(tempfile.NamedTemporaryFile('wb', dir='tmp', suffix='.pkl'))
            fb = stack.enter_context(tempfile.NamedTemporaryFile('wb', dir='tmp', suffix='.pkl'))
            a = np.array([2, 4, 6], dtype=int)
            pickle.dump(a, open(fa.name, 'wb'))
            name_a = fa.name
            b = np.array([1, 2, 3], dtype=int)
            pickle.dump(b, open(fb.name, 'wb'))
            name_b = fb.name
            ftensor = FractionalArray.from_pkl(name_a, name_b)
        self.assertIsInstance(ftensor, FractionalArray)
