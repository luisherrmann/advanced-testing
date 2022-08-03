from multiprocessing import context
import shutil
import pytest
import os
from contextlib import ExitStack
import numpy as np
import pickle
import tempfile
from torchtools.fractional_array import FractionalArray


class TestFractionalTensor(object):

    @pytest.fixture
    def handleTmp(self):
        if not os.path.exists('tmp'):
            os.makedirs('tmp')
        yield
        shutil.rmtree('tmp')

    @pytest.fixture
    def tmpFiles(self, handleTmp):
        with ExitStack() as stack:
            fa = stack.enter_context(tempfile.NamedTemporaryFile('wb', dir='tmp', suffix='.pkl'))
            fb = stack.enter_context(tempfile.NamedTemporaryFile('wb', dir='tmp', suffix='.pkl'))
            yield fa, fb

    @pytest.fixture(params=[
        (
            np.array([2, 4, 6], dtype=int), 
            np.array([1, 2, 3], dtype=int)
        )
    ])
    def dumpPkl(self, tmpFiles, request):
        fa, fb = tmpFiles
        a, b = request.param
        pickle.dump(a, open(fa.name, 'wb'))
        name_a = fa.name
        pickle.dump(b, open(fb.name, 'wb'))
        name_b = fb.name
        return name_a, name_b

    @pytest.mark.parametrize("a,b,err_type", [
        (
            np.array([2], dtype=int), 
            np.array([1], dtype=int), 
            None
        ),
        (
            np.array([], dtype=int), 
            np.array([], dtype=int), 
            None
        ),
        (
            np.array([2], dtype=int),
            np.array([2, 1], dtype=int),
            ValueError
        ),
        (
            np.array([2, 2], dtype=int).reshape([2, 1]),
            np.array([1, 1], dtype=int).reshape([1, 2]),
            ValueError
        ),
        (
            np.array([2], dtype=int), 
            np.array([1], dtype=float), 
            TypeError
        ),
        (
            np.array([2], dtype=float), 
            np.array([1], dtype=int), 
            TypeError
        )
    ])
    def test__init__(self, a, b, err_type):
        if err_type is None:
            c = FractionalArray(a, b)
        else:
            with pytest.raises(err_type):
                c = FractionalArray(a, b)

    @pytest.mark.skipif(os.environ.get('CI_CD') == '1', reason='CI/CD environment')
    def test_from_pkl(self, dumpPkl):
        name_a, name_b = dumpPkl
        ftensor = FractionalArray.from_pkl(name_a, name_b)
        assert isinstance(ftensor, FractionalArray)
