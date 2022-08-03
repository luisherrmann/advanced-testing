import math
import unittest
import pytest

from torchtools.utils import divide_int

class UtilsTest(unittest.TestCase):

    def setUp(self) -> None:
        return super().setUp()     

    def tearDown(self) -> None:
        return super().tearDown()

    # @pytest.mark.raises
    def test_divide_int(self):
        params = [
            # (a, b)
            # ("0:0", 0, 0),
            ("0:x", 0, 3),
            # ("x:0", 3, 0),
            ("pos:pos", 4, 2),
            ("pos:neg", 4, -2),
            ("neg:pos", -4, 2),
            ("neg:neg", -4, -2),
        ]
        results = [
            # math.nan,
            0,
            # math.nan,
            2,
            -2,
            -2,
            2
        ]
        for params, res in zip(params, results):
            name, a, b = params
            with self.subTest(name):
                self.assertEqual(divide_int(a, b), res)
        with self.subTest("0:0"):
            a, b = 0, 0
            self.assertTrue(math.isnan(divide_int(a, b)))
        with self.subTest("0:x"):
            a, b = 3, 0
            self.assertTrue(math.isnan(divide_int(a, b)))
        
        params_err = [
            ("int:other", 4, 2.0),
            ("other:int", 4.0, 2),
            ("other:other", 4.0, 2.0)
        ]
        for name, a, b in params_err:
            with self.subTest(name):
                self.assertRaises(TypeError, lambda: divide_int(a, b))
