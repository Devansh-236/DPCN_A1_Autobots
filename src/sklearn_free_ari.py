"""Adjusted Rand Index without pulling in scikit-learn."""
from collections import Counter
from math import comb


def adjusted_rand(a, b):
    assert len(a) == len(b)
    n = len(a)
    if n < 2:
        return 0.0
    cont = Counter(zip(a, b))
    ra, rb = Counter(a), Counter(b)
    index = sum(comb(v, 2) for v in cont.values())
    ea = sum(comb(v, 2) for v in ra.values())
    eb = sum(comb(v, 2) for v in rb.values())
    expected = ea * eb / comb(n, 2)
    maximum = (ea + eb) / 2
    return 0.0 if maximum == expected else (index - expected) / (maximum - expected)
