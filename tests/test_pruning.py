r"""Tests for the pruned-read helpers: ``num_blocks`` and the two display functions."""

import sympy

from dfthcbe.utils import aligned_blocks, num_blocks, popcount


def _greedy(a, b):
    """From ``a``, repeatedly take the largest aligned block that starts there and fits."""
    n = 0
    while a < b:
        size = a & -a if a else 1 << (b - 1).bit_length()
        while size > b - a:
            size //= 2
        a, n = a + size, n + 1
    return n


def test_num_blocks_is_the_canonical_decomposition():
    assert all(num_blocks(a, b) == _greedy(a, b) for a in range(65) for b in range(a, 70))


def test_a_prefix_is_one_block_per_set_bit():
    assert all(num_blocks(0, d) == bin(d).count('1') for d in range(70))


def test_display_functions_print_and_evaluate():
    """Symbolic they read ``s(RC)`` and ``beta(RC, N + RC)``; substituted they are integers."""
    rc, n = sympy.symbols('RC N', positive=True, integer=True)
    assert str(popcount(rc)) == 's(RC)'
    assert str(aligned_blocks(rc, rc + n)) == 'beta(RC, N + RC)'
    assert popcount(rc).subs(rc, 270) == 4
    assert aligned_blocks(rc, rc + n).subs({rc: 270, n: 54}) == 4
