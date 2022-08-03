from multiprocessing.sharedctypes import Value


import math


def divide_int(a: int, b: int) -> int:
    if not isinstance(a, int):
        raise TypeError
    if not isinstance(b, int):
        raise TypeError
    if b == 0:
        return math.nan
    else:
        return a // b