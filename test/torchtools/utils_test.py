import math
import pytest
from inspect import issubclass

from torchtools.utils import divide_int

class TestUtils(object):

    ### Parametrizing tests

    # There are three diffent mechanisms to parametrize tests in pytest:
    # 1. Using the `@pytest.mark.parametrize` decorator to annotate a test function
    # 2. Using the `parametrize` argument of the `@pytest.fixture` decorator
    # 3. Using the `pytest_generate_tests` hook to generate tests dynamically

    ## Parametrizing tests with `@pytest.mark.parametrize`:

    # We can use the @pytest.mark.parametrize decorator to parametrize tests and run them with different inputs.
    # The test below covers all equivalence classes of the (input, output) space for the division of two integers:
    #
    # For dividend and divisor we can consider 4 classes of the input domain:
    # 1. 0 (neutral element of summation)
    # 2. 1 (neutral element of multiplication)
    # 3. pos (positive integers)
    # 4. neg (negative integers)
    #
    # Furthermore, for the relation of division, we can consider 2 classes regarding divisibility:
    # 1. x mod y == 0 (divisible)
    # 2. x mod y != 0 (not divisible)
    #
    # To exhaustively test the function, we would need to test all possible combinations of these classes.
    # To do so, we select a representative from each class and test all possible combinations of these representatives.
    #
    # We select the following representatives:

    
    @pytest.mark.parametrize(
        ("dividend", "divisor", "quotient"),
        [
            # (0, 0, math.nan),
            (0, 3, 0),
            (0, -3, 0),
            # (3, 0, math.nan),
            #

            (4, 2, 2),
            (4, -2, -2),
            ( -4, 2, -2),
            ( -4, -2, 2),
        ],
        ids=[
            "0:0",
            "0:x",
            "x:0",
            "pos:pos",
            "pos:neg",
            "neg:pos",
            "neg:neg",
        ])
    def test_divide_int(self, dividend, divisor, quotient):
        if math.isnan(quotient):
            assert math.isnan(divide_int(dividend, divisor))
        else:
            assert divide_int(dividend, divisor) == quotient


    ### Handling errors
    # You can use the `@pytest` decorator to test for exceptions.



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


    def _test_divide_int_vmerged(self, inputs, expected):
        dividend, divisor = inputs
        quotient = expected
        if math.isnan(quotient):
            assert math.isnan(divide_int(dividend, divisor))
        else:
            assert divide_int(dividend, divisor) == quotient


    # We can also aggregate all of these into a single test function.
    # A possible approach is to combine the inputs and the expected output, and then treat the errors a special case.
    def test_divide_int_vmerged(self, inputs, expected):
        self._test_divide_int_vmerged(inputs, expected)

    ## Parametrizing fixtures

    # Another possibility to parametrize the tests is to directly parametrize a dependent fixture like so:
    @pytest.fixture(params=[
        ((0, 0), math.nan),
        ((3, 0), math.nan),
        ((4, 2), 2),
        ((4, -2), -2),
        ((-4, 2), -2),
        ((-4, -2), 2)
    ])
    def args_divide_int(self, request):
        return request.param
    
    def test_divide_int_vfixture(self, args_divide_int):
        inputs, expected = args_divide_int
        self._test_divide_int_vmerged(inputs, expected)

    # A very powerful feature for parametrizing features
    

    def test_divide_int_vhook(self, inputs, expected):
        self._test_divide_int_vmerged(inputs, expected)

    
    def pytest_generate_tests(metafunc):
        

    ## Fixture Composition: pytest-lazy-fixture
    # But what if I want to compose a fixture from multiple inputs?