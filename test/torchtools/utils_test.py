import math
import random
import torch
import pytest
from pytest import lazy_fixture
import itertools as it

from torchtools.utils import divide_int, CustomModel


random.seed(42)
torch.manual_seed(42)


### Parametrizing tests

# There are three diffent mechanisms to parametrize tests in pytest:
# 1. Using the `@pytest.mark.parametrize` decorator to annotate a test function
# 2. Using the `params` argument of the `@pytest.fixture` decorator
# 3. Using the `pytest_generate_tests` hook to generate tests dynamically

## Parametrizing tests with `@pytest.mark.parametrize`:

# We can use the @pytest.mark.parametrize decorator to parametrize tests and run them with different inputs.
# The test below covers all equivalence classes of the (input, output) space for the division of two integers:
#
# We can consider two classes of inputs regarding validity:
# 1. int (valid input)
# 2. other (invalid input)
#
# Additionally, we can further subdivide the valid input domain into 4 meaningful classes over the ring (Z, +, *), namely:
# 1. 0 (neutral element of summation)
# 2. 1 (neutral element of multiplication)
# 3. pos (positive integers)
# 4. neg (negative integers)
#
# Finally, we can consider 3 classes regarding divisibility:
# 1. x mod y == 0 (divisible)
# 2. x mod y != 0 (not divisible)
# 3. x / y == nan (not defined)
#
# To exhaustively test the function, we would need to test all possible combinations of these classes.
# To do so, we select a representative from each class and test all possible combinations of these representatives.
#
# We select the following representatives:
#
# Validity:
# 1. 2 (int)
# 2. 2.0 (float)
#
# Integer range:
# 1. 0
# 2. 1
# 3. 2
# 4. -2
#
# Divisibility:
# 1. (4, 2)
# 2. (3, 2)
# 3. (4, 0)
#
# The brute-force way to cover all possible combinations would be to consider the cartesian product of the representatives:
# (2.0, 4, 3, 2, 1, 0, -1, -2, -3, -4) x (2.0, 4, 3, 2, 1, 0, -1, -2, -3, -4)
#
# (dividend, divisor, quotient)
# ---
# (2.0, 4, TypeError)
# ...
# (2.0, -4, TypeError)
# (4, 4, 1)
# (4, 3, 1)
# (4, 2, 2)
# (4, 1, 4)
# (4, 0, math.nan)
# (4, -1, -4)
# (4, -2, -2)
# (4, -3, -1)
# (4, -4, -1)
# (3, 4, 0)
# (3, 3, 1)
# (3, 2, 1)
# (3, 1, 3)
# (3, 0, math.nan) !
# (3, -1, -3)
# (3, -2, -1)
# (3, -3, -1)
# (3, -4, -1)
# (2, 4, 0) !
# (2, 3, 0)
# (2, 2, 1)
# (2, 1, 2)
# (2, 0, math.nan) !
# (2, -1, -2)
# (2, -2, -1)
# (2, -3, -1)
# (2, -4, -1) !
# (1, 4, 0) !
# (1, 3, 0) !
# (1, 2, 0)
# (1, 1, 1)
# (1, -1, -1)
# (1, -2, -1)
# (1, -3, -1) !
# (1, -4, -1) !
# (0, 4, 0) !
# (0, 3, 0) !
# (0, 2, 0)
# (0, 1, 0)
# (0, 0, math.nan)
# (0, -1, 0)
# (0, -2, 0)
# (0, -3, 0) !
# (0, -3, 0) !
# ...
# (-4, 4, 1)
# (-4, 3, -2)
# (-4, 2, -2)
# (-4, 1, -4)
# (-4, 0, math.nan) !
# (-4, -1, 4)
# (-4, -2, 2)
# (-4, -3, 1)
# (-4, -4, 1)
#
# However, many of these problems are equivalent and, given our knowledge about the behaviour and implementation,
# do not provide any additional information regarding the correctness. I have highlighted some of these with a !.
#
# Perhaps a more succinct selection of tests would be the following:
#
# (dividend, divisor, quotient)
# ---
# (2.0, 4, TypeError)
# (4, 2.0, TypeError)
# (4, 4, 1)
# (4, 3, 1)
# (4, 2, 2)
# (4, 1, 4)
# (4, 0, math.nan)
# (4, -1, -4)
# (4, -2, -2)
# (4, -3, -2)
# (4, -4, -1)
# (3, 4, 0)
# (3, 3, 1)
# (3, 2, 1)
# (3, 1, 3)
# (3, -1, -3)
# (3, -2, -2)
# (3, -3, -1)
# (3, -4, -1)
# (2, 3, 0)
# (2, 2, 1)
# (2, 1, 2)
# (2, -1, -2)
# (2, -2, -1)
# (2, -3, -1)
# (1, 2, 0)
# (1, 1, 1)
# (1, -1, -1)
# (1, -2, -1)
# (0, 2, 0)
# (0, 1, 0)
# (0, 0, math.nan)
# (0, -1, 0)
# (0, -2, 0)
# (-1, 2, -1)
# (-1, 1, -1)
# (-1, -1, 1)
# (-1, -2, 0)
# (-2, 3, -1)
# (-2, 2, -1)
# (-2, 1, -2)
# (-2, -1, 2)
# (-2, -2, 1)
# (-2, -3, 0)
# (-3, 4, -1)
# (-3, 3, -1)
# (-3, 2, -2)
# (-3, 1, -3)
# (-3, -1, 3)
# (-3, -2, 1)
# (-3, -3, 1)
# (-3, -4, 0)
# (-4, 4, -1)
# (-4, 3, -2)
# (-4, 2, -2)
# (-4, 1, -4)
# (-4, 0, math.nan)
# (-4, -1, 4)
# (-4, -2, 2)
# (-4, -3, 1)
# (-4, -4, 1)
#
# Of course it is impossible to test all possible combinations, and one could decide to drop some of the test cases above.
# This is a design decision that should be made based on the requirements of the project and the expected behaviour of the function.
# Now let's see how we can implement this in pytest:

TEST_DIVIDE_INT_CASES = [
    (4, 4, 1),
    (4, 3, 1),
    (4, 2, 2),
    (4, 1, 4),
    (4, -1, -4),
    (4, -2, -2),
    (4, -3, -2),
    (4, -4, -1),
    (3, 4, 0),
    (3, 3, 1),
    (3, 2, 1),
    (3, 1, 3),
    (3, -1, -3),
    (3, -2, -2),
    (3, -3, -1),
    (3, -4, -1),
    (2, 3, 0),
    (2, 2, 1),
    (2, 1, 2),
    (2, -1, -2),
    (2, -2, -1),
    (2, -3, -1),
    (1, 2, 0),
    (1, 1, 1),
    (1, -1, -1),
    (1, -2, -1),
    (0, 2, 0),
    (0, 1, 0),
    (0, -1, 0),
    (0, -2, 0),
    (-1, 2, -1),
    (-1, 1, -1),
    (-1, -1, 1),
    (-1, -2, 0),
    (-2, 3, -1),
    (-2, 2, -1),
    (-2, 1, -2),
    (-2, -1, 2),
    (-2, -2, 1),
    (-2, -3, 0),
    (-3, 4, -1),
    (-3, 3, -1),
    (-3, 2, -2),
    (-3, 1, -3),
    (-3, -1, 3),
    (-3, -2, 1),
    (-3, -3, 1),
    (-3, -4, 0),
    (-4, 4, -1),
    (-4, 3, -2),
    (-4, 2, -2),
    (-4, 1, -4),
    (-4, -1, 4),
    (-4, -2, 2),
    (-4, -3, 1),
    (-4, -4, 1),
]

TEST_DIVIDE_INT_NAN_CASES = [
    (4, 0, math.nan),
    (0, 0, math.nan),
    (-4, 0, math.nan),
]

TEST_DIVIDE_INT_ERR_CASES = [
    (2.0, 2.0, TypeError),
    (2.0, 4, TypeError),
    (4, 2.0, TypeError),
]


@pytest.mark.parametrize(("dividend", "divisor", "quotient"), TEST_DIVIDE_INT_CASES)
def test_divide__int(dividend, divisor, quotient):
    assert divide_int(dividend, divisor) == quotient


@pytest.mark.nan
@pytest.mark.parametrize(("dividend", "divisor", "quotient"), TEST_DIVIDE_INT_NAN_CASES)
def test_divide_int__nan(dividend, divisor, quotient):
    assert math.isnan(divide_int(dividend, divisor))


### Handling errors
# You can use the `@pytest` decorator to test for exceptions.


@pytest.mark.raises
@pytest.mark.parametrize(("dividend", "divisor", "type"), TEST_DIVIDE_INT_ERR_CASES)
def test_divide_int__err(dividend, divisor, type):
    with pytest.raises(type) as e:
        divide_int(dividend, divisor)


# We can also aggregate all of these into a single test function.
# A possible approach is to combine the inputs and the expected output, and then treat the errors a special case.


def _test_divide_int__vmerged(dividend, divisor, expected):
    if isinstance(expected, type) and issubclass(expected, Exception):
        with pytest.raises(expected):
            divide_int(dividend, divisor)
    elif math.isnan(expected):
        assert math.isnan(divide_int(dividend, divisor))
    else:
        assert divide_int(dividend, divisor) == expected


@pytest.mark.parametrize(
    ("dividend", "divisor", "expected"),
    [
        *TEST_DIVIDE_INT_CASES,
        *TEST_DIVIDE_INT_NAN_CASES,
        *TEST_DIVIDE_INT_ERR_CASES,
    ],
)
def test_divide_int__vmerged(dividend, divisor, expected):
    _test_divide_int__vmerged(dividend, divisor, expected)


## Parametrizing fixtures

# Another possibility to parametrize the tests is to directly parametrize a dependent fixture like so:
@pytest.fixture(
    params=[
        *TEST_DIVIDE_INT_CASES,
        *TEST_DIVIDE_INT_NAN_CASES,
        *TEST_DIVIDE_INT_ERR_CASES,
    ]
)
def args_divide_int(request):
    return request.param


def test_divide_int__vfixture(args_divide_int):
    dividend, divisor, quotient = args_divide_int
    _test_divide_int__vmerged(dividend, divisor, quotient)


## Parametrize using hooks
#
# A very powerful feature for parametrizing tests is to use the `pytest_generate_tests` hook (see top of script).
# This hook allows you to generate tests dynamically based on some input.
# It will run every time that you call a test function within the scope where the hook is defined.


def pytest_generate_tests(metafunc):
    if "test_divide_int__vhook" in metafunc.function.__name__:
        all_cases = [
            *TEST_DIVIDE_INT_CASES,
            *TEST_DIVIDE_INT_ERR_CASES,
        ]
        metafunc.parametrize("divide_int_cases", all_cases)


def test_divide_int__vhook(divide_int_cases):
    dividend, divisor, quotient = divide_int_cases
    _test_divide_int__vmerged(dividend, divisor, quotient)


## Dynamic test generation
#
# Sometimes, we want to generate tests dynamically based on some input.
# For instance, it is very cumbersome to write all possible combinations of inputs and outputs for divide_int.
#
# However, suppose we had a way to generate these tests automatically.
# That is, for a subset of (input, output) pairs, we can define a function f s.t. f(input) = output.
# Obviously, for all possible (input, output) pairs, this function would be the implementation of divide_int.
# However, in some cases we can define an generating function for a subset that is easier to write and understand,
# or which is already tested and verified.
#
# For instance, we know that the internal implementation of / and math.floor in Python is correct.
# So we can use this to generate tests for divide_int:


@pytest.fixture(
    params=[
        *[
            (a, b, math.floor(a / b))
            for a, b in it.product(range(-4, 5), range(-4, 5))
            if b != 0
        ],
        *TEST_DIVIDE_INT_NAN_CASES,
        *TEST_DIVIDE_INT_ERR_CASES,
    ]
)
def divide_int_cases_vgen(request):
    return request.param


def test_divide_int__vgen(divide_int_cases_vgen):
    dividend, divisor, quotient = divide_int_cases_vgen
    _test_divide_int__vmerged(dividend, divisor, quotient)


# Of course, this is not very useful for divide_int, but it can be very useful for more complex functions.
# For instance, we could use this for a function that computes the point-wise integer division on tensors
# and use this implementation of the division to check the correctness of every single element of the outputs tensor.

## Fixture Composition: pytest-lazy-fixture
# Let's say we want to compose a test from multiple fixtures.


@pytest.fixture(params=TEST_DIVIDE_INT_CASES)
def divide_int_base_cases(request):
    return request.param


@pytest.fixture(params=TEST_DIVIDE_INT_NAN_CASES)
def divide_int_nan_cases(request):
    return request.param


@pytest.fixture(params=TEST_DIVIDE_INT_ERR_CASES)
def divide_int_err_cases(request):
    return request.param


@pytest.fixture()
def divide_int_random():
    random.seed(42)
    dividend = random.randint(-100, 100)
    divisor = random.randint(-100, 100)
    divisor = divisor if divisor != 0 else 1
    quotient = math.floor(dividend / divisor)
    return (dividend, divisor, quotient)


# Unforunately, there is no way we can pass these fixtures directly to a parametrized test.
# We cannot pass fixtures to params of parametrized fixtures:

# @pytest.fixture(params=[
#     *divide_int_base_cases,
#     *divide_int_nan_cases,
#     *divide_int_err_cases,
#     *divide_int_random
# ])
# def divide_int_all_cases(request):
#     return request.params

# And we cannot pass the fixture directly to @pytest.mark.parametrize decorator of the test function:

# @pytest.mark.parametrize(
#     ("dividend", "divisor", "quotient"),
#     divide_int_base_cases,
#     divide_int_nan_cases,
#     divide_int_err_cases,
#     divide_int_random
# )
# def test_divide_int__comp_(dividend, divisor, quotient):
#     _test_divide_int__vmerged(dividend, divisor, quotient)

# Of course, we could pass the fixtures to the test function like so:
# def test_divide_int__comp_(divide_int_base_cases, divide_int_nan_cases):
#     dividend, divisor, quotient = divide_int_base_cases
#     _test_divide_int__vmerged(dividend, divisor, quotient)

# But this will not lead to the desired behaviour because we will get the cartesian product
# of the fixtures, which is not what we want.

# However, we can use the pytest-lazy-fixture plugin to achieve the desired behaviour:


@pytest.mark.parametrize(
    ("args"),
    [
        (5, 5, 1),
        lazy_fixture("divide_int_base_cases"),
        lazy_fixture("divide_int_nan_cases"),
        lazy_fixture("divide_int_err_cases"),
        lazy_fixture("divide_int_random"),
    ],
)
def test_divide_int__comp(args):
    dividend, divisor, quotient = args
    _test_divide_int__vmerged(dividend, divisor, quotient)


# We can also pass lazy_fixtures to params of parametrized fixtures:
@pytest.fixture(
    params=[
        lazy_fixture("divide_int_base_cases"),
        lazy_fixture("divide_int_nan_cases"),
        lazy_fixture("divide_int_nan_cases"),
        lazy_fixture("divide_int_random"),
    ]
)
def divide_int_ext_cases(request):
    return request.param


def test_divide_int__ext(divide_int_ext_cases):
    dividend, divisor, quotient = divide_int_ext_cases
    _test_divide_int__vmerged(dividend, divisor, quotient)


# We can also use lazy-fixture on a single argument:
@pytest.fixture(params=[random.randint(-100, 100) for _ in range(10)])
def random_nonzero(request):
    return request.param if request.param != 0 else 1


@pytest.fixture(params=[*range(10)])
def range10(request):
    return request.param


@pytest.fixture(
    params=[
        lazy_fixture("range10"),
    ]
)
def multiples2(request):
    return 2 * request.param


@pytest.mark.parametrize(
    ("dividend", "divisor", "quotient"),
    [
        (0, lazy_fixture("random_nonzero"), 0),
        (lazy_fixture("random_nonzero"), 1, lazy_fixture("random_nonzero")),
        (lazy_fixture("multiples2"), 2, lazy_fixture("range10")),
    ],
)
def test_divide_int__single(dividend, divisor, quotient):
    _test_divide_int__vmerged(dividend, divisor, quotient)


# Of course, you need to make sure that the parametrized lazy fixtures have the same length and that they are compatible.
# Basically, you can understand the behaviour of lazy_fixture in @pytest.mark.parametrize as follows:
#
# 1. (lazy_fixture("param_fixture1"), single_value1) becomes:
#
#       (param_fixture1[0], single_value1),
#       ...
#       (param_fixture1[n], single_value1)
#
#   inside the test function.
#
# 2. (lazy_fixture("param_fixture1"), lazy_fixture("val_fixture1")) becomes:
#
#       (param_fixture1[0], val_fixture1),
#       ...
#       (param_fixture1[n], val_fixture1)
#
# 3. (lazy_fixture("param_fixture1"), lazy_fixture("param_fixture2")) becomes:
#
#       (param_fixture1[0], param_fixture2[0]),
#       ...
#       (param_fixture1[n], param_fixture2[n])
#
#   assuming that param_fixture1 and param_fixture2 have the same length AND dependencies!
#   Otherwise, we obtain the cartesian product of the two fixtures, i.e.:
#
# 4. (lazy_fixture("param_fixture1"), lazy_fixture("param_fixture2")) becomes:
#
#       (param_fixture1[0], param_fixture2[0]),
#       ...
#       (param_fixture1[0], param_fixture2[n]),
#       (param_fixture1[1], param_fixture2[0]),
#       ...
#       (param_fixture1[m], param_fixture2[n])


class TestCustomModel(object):

    ## Complex tests: Test case composition
    # Sometimes, a function has multiple inputs and even outputs that need to be tested.
    # For example, consider a function that initializes a custom neural network model.
    #
    # The function takes the following inputs:
    # 1. input_dim (int)
    # 2. hidden_dim (int)
    # 3. output_dim (int)
    # 4. activation (str)
    # 5. blocks (int)
    #
    # While this is still quite well-behaved, it's easy to see how the number of parameters
    # can quickly grow for more complex networks and make it difficult to test.
    #
    # In many cases, the test cases will all be variations of a base configuration, where
    # only a few parameters are changed. To handle this with less verbosity, we can use the override pattern.

    @pytest.fixture()
    def base_args(self):
        return {
            "input_dim": 5,
            "hidden_dim": 10,
            "output_dim": 1,
            "activation": "relu",
            "blocks": 1,
        }

    @pytest.fixture()
    def base_expected(self):
        return {
            "length": 5,
        }

    @pytest.fixture(params=[0, 2])
    def blocks_args(self, request):
        return {"blocks": request.param}

    @pytest.fixture(params=[lazy_fixture("blocks_args")])
    def blocks_expected(self, request):
        return {"length": 3 + 2 * request.param["blocks"]}

    @pytest.mark.parametrize(
        ("args", "args_overrides", "expected", "expected_overrides"),
        [
            (lazy_fixture("base_args"), {}, lazy_fixture("base_expected"), {}),
            (
                lazy_fixture("base_args"),
                {"input_dim": 10},
                lazy_fixture("base_expected"),
                {},
            ),
            (
                lazy_fixture("base_args"),
                {"hidden_dim": 20},
                lazy_fixture("base_expected"),
                {},
            ),
            (
                lazy_fixture("base_args"),
                {"output_dim": 2},
                lazy_fixture("base_expected"),
                {},
            ),
            (
                lazy_fixture("base_args"),
                {"activation": "relu"},
                lazy_fixture("base_expected"),
                {},
            ),
            # These two test inputs do actually change the length of the model:
            (
                lazy_fixture("base_args"),
                {"blocks": 0},
                lazy_fixture("base_expected"),
                {"length": 3},
            ),
            (
                lazy_fixture("base_args"),
                {"blocks": 2},
                lazy_fixture("base_expected"),
                {"length": 7},
            ),
            # Easier way to accomplish the same as with the above two tests:
            (
                lazy_fixture("base_args"),
                lazy_fixture("blocks_args"),
                lazy_fixture("base_expected"),
                lazy_fixture("blocks_expected"),
            ),
        ],
    )
    def test__init__(self, args, args_overrides, expected, expected_overrides):
        args = {**args, **args_overrides}
        expected = {**expected, **expected_overrides}
        model = CustomModel(**args)
        assert len(model.layers) == expected["length"]

    # Now for a more complex example where we want to test the forward method of the model.
    # We start by defining a base test case configuration:

    @pytest.fixture()
    def base_args_forward(self, base_args):
        input = torch.rand(1, 5)
        model = CustomModel(**base_args)
        return {"model": model, "input": input}

    @pytest.fixture()
    def base_expected_forward(self):
        return {"output_shape": (1, 1)}

    # Next, we define test cases with different batch sizes.
    # These will use the base test case configuration and only change the batch size.

    @pytest.fixture(params=[1, 2, 3, 10])
    def batch_sizes(self, request):
        return request.param

    @pytest.fixture(params=[lazy_fixture("batch_sizes")])
    def args_batch_sizes(self, base_args, request):
        return {
            "input": torch.rand(request.param, base_args["input_dim"]),
            "model": CustomModel(**base_args),
        }

    # We need to adapt the expected output shapes accordingly:

    @pytest.fixture(params=[lazy_fixture("batch_sizes")])
    def expected_batch_sizes(self, request):
        return {"output_shape": (request.param, 1)}

    # Next, we define test cases with different input dimensions.

    @pytest.fixture(params=[1, 2, 5, 10])
    def input_dims(self, request):
        return request.param

    @pytest.fixture(params=[lazy_fixture("input_dims")])
    def args_input_dims(self, base_args, request):
        args = {**base_args, "input_dim": request.param}
        return {"input": torch.rand(1, request.param), "model": CustomModel(**args)}

    @pytest.fixture(params=[lazy_fixture("input_dims")])
    def expected_input_dims(self, request):
        return {"output_shape": (1, 1)}

    # Finally, we define test cases with different output dimensions.

    @pytest.fixture(params=[1, 2, 7, 10])
    def output_dims(self, request):
        return request.param

    @pytest.fixture(params=[lazy_fixture("output_dims")])
    def args_output_dims(self, base_args, request):
        args = {**base_args, "output_dim": request.param}
        return {"input": torch.rand(1, 5), "model": CustomModel(**args)}

    @pytest.fixture(params=[lazy_fixture("output_dims")])
    def expected_output_dims(self, request):
        return {"output_shape": (1, request.param)}

    # And to close it off, let us define some test cases that modify the batch size and the output dimension jointly.

    @pytest.fixture(params=it.product([1, 2, 3, 10], [1, 2, 5, 10]))
    def product_batch_output(self, request):
        return request.param

    @pytest.fixture(params=[lazy_fixture("product_batch_output")])
    def args_batch_output(self, base_args, request):
        batch_size, output_dim = request.param
        input_dim = base_args["input_dim"]
        input = torch.rand(batch_size, input_dim)
        args = {**base_args, "output_dim": output_dim}
        model = CustomModel(**args)
        return {"model": model, "input": input}

    @pytest.fixture(params=[lazy_fixture("product_batch_output")])
    def expected_batch_output(self, request):
        batch_size, output_dim = request.param
        return {"output_shape": (batch_size, output_dim)}

    # Let us pass all these fixture to the forward test function:

    @pytest.mark.parametrize(
        ("args", "args_overrides", "expected", "expected_overrides"),
        [
            (
                lazy_fixture("base_args_forward"),
                {},
                lazy_fixture("base_expected_forward"),
                {},
            ),
            # Modify batch size only
            (
                lazy_fixture("base_args_forward"),
                lazy_fixture("args_batch_sizes"),
                lazy_fixture("base_expected_forward"),
                lazy_fixture("expected_batch_sizes"),
            ),
            # Modify input dim only
            (
                lazy_fixture("base_args_forward"),
                lazy_fixture("args_input_dims"),
                lazy_fixture("base_expected_forward"),
                lazy_fixture("expected_input_dims"),
            ),
            # Modify output dim only
            (
                lazy_fixture("base_args_forward"),
                lazy_fixture("args_output_dims"),
                lazy_fixture("base_expected_forward"),
                lazy_fixture("expected_output_dims"),
            ),
            # Modify batch size and output dim
            (
                lazy_fixture("base_args_forward"),
                lazy_fixture("args_batch_output"),
                lazy_fixture("base_expected_forward"),
                lazy_fixture("expected_batch_output"),
            ),
        ],
    )
    def test_forward(self, args, args_overrides, expected, expected_overrides):
        args = {**args, **args_overrides}
        expected = {**expected, **expected_overrides}
        model, input = args["model"], args["input"]
        output = model.forward(input)
        assert output.shape == expected["output_shape"]

    # And that's it! Thanks for taking this advanced pytest workshop and happy testing! :)
