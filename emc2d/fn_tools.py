from itertools import accumulate, repeat, islice
from functools import partial


def iterate(f, x):
    """
    Generates the sequence: x, f(x), f(f(x)) ...
    """
    return accumulate(repeat(x), lambda fx, _: f(fx))


def take(n, it):
    return [x for x in islice(it, n)]


def drop(n, it):
    return islice(it, n, None)


head = next


tail = partial(drop, 1)
