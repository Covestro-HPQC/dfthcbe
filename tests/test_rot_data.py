r"""Tests for the rotation-angle data: the two loads and the erase (steps 10, 11, 18).

The loads are plain Qualtran ``QROM``s, so ``call_classically`` reads them exactly -- no
contraction needed. Two cost regressions pin the pruning: nesting
``b`` outer so the all-zero ``b = B`` row prunes its whole inner ``r`` loop, and zero-padding
the SF range of the ``x_o``-addressed tail so it prunes down to ``~N``.
"""

import numpy as np
import pytest
import sympy

from qualtran.bloqs.data_loading.qrom import QROM
from qualtran.resource_counting import get_cost_value, QECGatesCost
from qualtran.symbolics import Shaped

from dfthcbe import (
    angles_from_vector,
    discretize_angles,
    RotDataErase,
    sf_rotation_qrom,
    tail_rotation_qrom,
)
from dfthcbe.utils import popcount

B_ROT = 5
SF = np.array([[[3, 9], [12, 6]], [[1, 25], [7, 2]]])  # (R=2, B=2, N-1=2)
TAIL = np.array([[5, 2], [8, 11], [4, 30]])            # (N=3, N-1=2): 2 x D1 then 1 x Q1
NUM_SF, NUM_POS = 4, 2                                 # RC = 4, N_+ = 2

# FeMoco-54, the README's worked example.
N54, R54, B54, C54, BROT54 = 54, 10, 27, 27, 15


def _ccz(bloq):
    return get_cost_value(bloq, QECGatesCost()).total_t_and_ccz_count()['n_ccz']


def _names(n_rungs, n_extra):
    return [f'target{k}_' for k in range(n_rungs + n_extra)]


def _run(bloq, **kwargs):
    return dict(zip([r.name for r in bloq.signature], bloq.call_classically(**kwargs)))


def _big_tables(seed=0):
    rng = np.random.default_rng(seed)
    return (rng.integers(1, 2**BROT54, size=(R54, B54, N54 - 1)),
            rng.integers(1, 2**BROT54, size=(N54, N54 - 1)))


# --- the SF load (step 11) -----------------------------------------------------------


def test_sf_load_reads_angles_and_writes_the_identity_flag():
    """Rows ``b < B`` load ``angle_data[r, b]`` and set ``b != B``; row ``b = B`` is identity."""
    qrom = sf_rotation_qrom(SF, B_ROT)
    *angles, flag = _names(SF.shape[2], 1)
    for r in range(SF.shape[0]):
        for b in range(SF.shape[1] + 1):
            out = _run(qrom, control=1, selection0=b, selection1=r,
                       **{n: 0 for n in angles + [flag]})
            loaded = [int(out[n]) for n in angles]
            if b == SF.shape[1]:
                assert (loaded, out[flag]) == ([0, 0], 0)
            else:
                assert (loaded, out[flag]) == (list(SF[r, b]), 1)


def test_sf_load_is_inert_when_not_selected():
    """It fires on its control only -- in the circuit that control is ``G_0 = 0`` (X-sandwiched)."""
    qrom = sf_rotation_qrom(SF, B_ROT)
    *angles, flag = _names(SF.shape[2], 1)
    out = _run(qrom, control=0, selection0=0, selection1=0, **{n: 0 for n in angles + [flag]})
    assert [int(out[n]) for n in angles + [flag]] == [0, 0, 0]


def test_sf_load_b_outer_nesting_prunes_the_identity_row():
    """``b`` outer / ``r`` inner costs ``~RB``; ``r`` outer / ``b`` inner costs ``R(B+1)-1``.

    The ``b = B`` row is constant, so ``QROM._break_early`` deletes its entire inner ``r``
    iteration -- only possible with ``b`` on the outside.
    """
    sf, _ = _big_tables()
    ours = sf_rotation_qrom(sf, BROT54)
    # The same table with the axes the other way round: (r, b+1) instead of (b+1, r).
    padded = np.zeros((R54, B54 + 1, N54 - 1), dtype=int)
    padded[:, :B54] = sf
    flag = np.ones((R54, B54 + 1), dtype=int)
    flag[:, B54] = 0
    r_outer = QROM.build_from_data(
        *[padded[:, :, k] for k in range(N54 - 1)], flag,
        target_bitsizes=(BROT54,) * (N54 - 1) + (1,), num_controls=1,
    )
    assert _ccz(r_outer) == R54 * (B54 + 1) - 1 == 279
    assert _ccz(ours) == 271  # RB + s(R) - 1: the zero row splits into [0, 8) and [8, 10)
    assert _ccz(ours) < _ccz(r_outer)


# --- the D1/Q1 tail load (step 10) --------------------------------------------------


def test_tail_load_reads_directions_and_writes_the_family():
    """SF rows (``x_o < RC``) load nothing; row ``RC + j`` loads the ``j``-th D1/Q1 direction.

    The family selector comes out ``00`` on SF, ``10`` on D1 and ``11`` on Q1 -- so ``G = 01``
    never occurs, which is what lets :class:`Maj` leave its ``G1`` gate uncontrolled.
    """
    qrom = tail_rotation_qrom(TAIL, NUM_SF, NUM_POS, B_ROT)
    *angles, G0, G1 = _names(TAIL.shape[1], 2)
    for x_o in range(NUM_SF + TAIL.shape[0]):
        out = _run(qrom, selection=x_o, **{n: 0 for n in angles + [G0, G1]})
        loaded = [int(out[n]) for n in angles]
        family = (int(out[G0]), int(out[G1]))
        if x_o < NUM_SF:
            assert (loaded, family) == ([0, 0], (0, 0))
        else:
            j = x_o - NUM_SF
            assert loaded == list(TAIL[j])
            assert family == (1, int(j >= NUM_POS))


def test_tail_load_prunes_the_sf_prefix():
    """The all-zero SF rows cost a handful of ANDs, not ``RC`` of them."""
    _, tail = _big_tables()
    ours = tail_rotation_qrom(tail, R54 * C54, N54 // 2, BROT54)
    # Same address space, but every row carries data: nothing to prune.
    dense = np.tile(tail[:1], (R54 * C54 + N54, 1))
    unpruned = QROM.build_from_data(
        *[dense[:, k] for k in range(N54 - 1)], np.arange(R54 * C54 + N54) % 2,
        np.arange(R54 * C54 + N54) % 3, target_bitsizes=(BROT54,) * (N54 - 1) + (1, 1),
    )
    assert _ccz(ours) == 56  # N + s(RC) - 2 = 54 + 4 - 2
    assert _ccz(unpruned) == R54 * C54 + N54 - 2 == 322
    assert _ccz(ours) < _ccz(unpruned)


def test_exact_erase_undoes_both_loads():
    """Load tail then SF, run both backwards: the angles and ``G`` return to ``|0>``."""
    sf_load = sf_rotation_qrom(SF, B_ROT)
    tail_load = tail_rotation_qrom(TAIL, NUM_SF, NUM_POS, B_ROT)
    *angles, G0, G1 = _names(TAIL.shape[1], 2)
    flag = _names(SF.shape[2], 1)[-1]
    for x_o in range(NUM_SF + TAIL.shape[0]):
        for b, r in [(0, 1), (2, 0)]:
            out = _run(tail_load, selection=x_o, **{n: 0 for n in angles + [G0, G1]})
            family = {n: int(out[n]) for n in (G0, G1)}
            ctrl = 1 - family[G0]  # the X-sandwiched G_0 = 0 control
            state = {n: int(out[n]) for n in angles}
            for _ in range(2):  # forward read, then the same read again to undo it
                out = _run(sf_load, control=ctrl, selection0=b, selection1=r, **state,
                           **{flag: 0 if _ == 0 else int(out[flag])})
                state = {n: int(out[n]) for n in angles}
            out = _run(tail_load, selection=x_o, **state, **family)
            assert [int(out[n]) for n in angles + [G0, G1]] == [0, 0, 0, 0]


# --- exact walk costs, numeric and symbolic -------------------------------------------

N_, RC_, R_, B_ = sympy.symbols('N RC R B', positive=True, integer=True)


def _distinct(shape):
    """Distinct entries, so that no run of live rows is constant and only the dead rows prune."""
    return np.arange(1, int(np.prod(shape)) + 1).reshape(shape)


@pytest.mark.parametrize("num_orb", [2, 5])
@pytest.mark.parametrize("num_sf", [0, 1, 2, 3, 4, 7, 8, 9, 15, 16])
def test_tail_load_costs_one_leaf_per_aligned_dead_block(num_orb, num_sf):
    """``N + s(RC) - 2`` ANDs, and the symbolic bloq gives the same once substituted.

    ``RC`` covers no dead rows, one aligned block, several, the worst prefix ``2**h - 1`` and
    the prefix one past it.
    """
    built = tail_rotation_qrom(_distinct((num_orb, num_orb - 1)), num_sf, num_orb // 2, 8)
    symbolic = tail_rotation_qrom(Shaped((N_, N_ - 1)), RC_, 0, 8)
    assert _ccz(built) == num_orb - 2 + popcount(num_sf)
    assert _ccz(symbolic).subs({N_: num_orb, RC_: num_sf}) == _ccz(built)


@pytest.mark.parametrize("num_r", [2, 3, 4, 7, 8, 10])
@pytest.mark.parametrize("num_b", [1, 2, 5])
def test_sf_load_costs_rb_plus_the_zero_rows_blocks(num_r, num_b):
    """``RB + s(R) - 1``: ``B`` live rows of ``R`` leaves, the zero row ``b = B`` split into its
    ``s(R)`` aligned blocks over ``r``, and one control.
    """
    built = sf_rotation_qrom(_distinct((num_r, num_b, 2)), 8)
    symbolic = sf_rotation_qrom(Shaped((R_, B_, 2)), 8)
    assert _ccz(built) == num_r * num_b + popcount(num_r) - 1
    assert _ccz(symbolic).subs({R_: num_r, B_: num_b}) == _ccz(built)


# --- the cheap erase (step 18) -------------------------------------------------------


def test_erase_is_cheap_and_b_rot_free():
    """The two phase fixups cost ``R + B - 3`` and the tail load's walk, not the ``N + RB`` load.

    That asymmetry -- and its independence of ``b_rot`` -- is the whole point of splitting the
    load out of the ladder.
    """
    sf, tail = _big_tables()
    tail_load = _ccz(tail_rotation_qrom(tail, R54 * C54, 27, BROT54))
    load = _ccz(sf_rotation_qrom(sf, BROT54)) + tail_load
    for b_rot in (BROT54, 2 * BROT54):
        erase = _ccz(
            RotDataErase(num_r=R54, num_b=B54, num_orb=N54, num_sf=R54 * C54, b_rot=b_rot)
        )
        assert erase == (R54 + B54 - 3) + tail_load == 34 + 56  # x_o fixup = the tail's walk
        assert erase < load


# --- angles <-> directions (pure numpy, no simulation) -------------------------------


@pytest.mark.parametrize("n", [3, 5, 8, 20])
def test_angles_from_vector_continuum(n):
    """The exact (undiscretised) angles rotate ``e_0`` exactly onto ``u_vec``."""
    u = np.random.default_rng(n).standard_normal(n)
    u /= np.linalg.norm(u)
    theta = angles_from_vector(u)
    assert theta.shape == (n - 1,)
    Q = np.eye(n)
    for k in range(n - 1):
        c, s = np.cos(theta[k]), np.sin(theta[k])
        rung = np.eye(n)
        rung[np.ix_([k, k + 1], [k, k + 1])] = [[c, s], [s, -c]]
        Q = rung @ Q
    assert np.allclose(Q[:, 0], u, atol=1e-12)


def test_angles_from_vector_rejects_unnormalized():
    with pytest.raises(ValueError):
        angles_from_vector(np.array([1.0, 2.0]))


def test_discretize_angles_factor():
    """Discretisation uses ``2**(b_rot-1)`` steps per turn."""
    b_rot = 10
    (x,) = discretize_angles(np.array([np.pi / 2]), b_rot)
    assert x == 2 ** (b_rot - 1) // 4
    assert np.isclose(4 * np.pi * x / 2**b_rot, np.pi / 2)
