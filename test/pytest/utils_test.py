import math
import pytest

from torchtools.utils import divide_int

class TestUtils(object):

    @pytest.mark.parametrize("dividend,divisor,quotient",
        [
            # (0, 0, math.nan),
            (0, 3, 0),
            # (3, 0, math.nan),
            (4, 2, 2),
            (4, -2, -2),
            ( -4, 2, -2),
            ( -4, -2, 2),
        ],
        ids=[
            # "0:0"
            "0:x",
            # "x:0",
            "pos:pos",
            "pos:neg",
            "neg:pos",
            "neg:neg",
        ])
    def test_divide_int(self, dividend, divisor, quotient):
        assert divide_int(dividend, divisor) == quotient

    @pytest.mark.nan
    @pytest.mark.parametrize("dividend,divisor", [
        (0, 0),
        (3, 0)
    ],
    ids=["0:0", "0:x"])
    def test_divide_int_nan(self, dividend, divisor):
        assert math.isnan(divide_int(dividend, divisor))

    @pytest.mark.raises
    @pytest.mark.parametrize("dividend,divisor,type", [
        (4, 2.0, TypeError),
        (4.0, 2, TypeError),
        (4.0, 2.0, TypeError)
    ],
    ids=[
        "int:other",
        "other:int",
        "other:other"
    ])
    def test_divide_int_err(self, dividend, divisor, type):
        with pytest.raises(type) as e:
            divide_int(dividend, divisor)
