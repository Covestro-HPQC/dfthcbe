r"""Tests for the corrected real Givens rotation bloq (:mod:`dfthcbe.givens`).

Two complementary checks:

* **Correctness** by dense tensor contraction at *tiny* ``b_rot`` (rotations map basis
  states to superpositions, so ``call_classically`` is unavailable -- a small contraction
  is the only way to see the real operator). We assert the bloq applies
  ``[[c, -s], [s, c]]`` on the ``{01, 10}`` subspace and fixes ``|00>``/``|11>``.
* **Cost** by the call graph (``QECGatesCost``): ``2*b_rot - 2`` Toffolis, numeric and
  symbolic -- matching Qualtran's phase-gradient atom (the drop-in it replaces).
"""

import numpy as np
import pytest
import sympy

from qualtran import BloqBuilder
from qualtran.bloqs.basic_gates import IntEffect, IntState
from qualtran.bloqs.rotations.phase_gradient import PhaseGradientState
from qualtran.resource_counting import QECGatesCost, get_cost_value

from dfthcbe.givens import GivensRotationByPhaseGradient


def _givens_op(b_rot: int, x: int) -> np.ndarray:
    """Dense 4x4 operator the bloq applies for integer angle ``x``.

    Prepares/disposes the shared phase-gradient state and the ``rom_data`` register so
    only the two mode qubits ``|target_i target_j>`` survive.
    """
    g = GivensRotationByPhaseGradient(phasegrad_bitsize=b_rot)
    bb = BloqBuilder()
    ti = bb.add_register('target_i', 1)
    tj = bb.add_register('target_j', 1)
    pg = bb.add(PhaseGradientState(bitsize=b_rot))
    rom = bb.add(IntState(val=x, bitsize=b_rot))
    ti, tj, rom, pg = bb.add(g, target_i=ti, target_j=tj, rom_data=rom, phase_gradient=pg)
    bb.add(PhaseGradientState(bitsize=b_rot).adjoint(), phase_grad=pg)
    bb.add(IntEffect(val=x, bitsize=b_rot), val=rom)
    return bb.finalize(target_i=ti, target_j=tj).tensor_contract()


# --- correctness (small dense contraction; b_rot kept tiny per no-dense-contraction rule) ---


@pytest.mark.parametrize("x", range(5))  # angles 0, pi/2, pi, 3pi/2, 2pi at b_rot=4
def test_is_real_givens_rotation(x):
    """The bloq applies [[c, -s], [s, c]] on {01, 10} and fixes |00>, |11| (Givens)."""
    b_rot = 4
    theta = 4 * np.pi * x / 2**b_rot
    c, s = np.cos(theta), np.sin(theta)
    want = np.eye(4, dtype=complex)
    want[np.ix_([1, 2], [1, 2])] = [[c, -s], [s, c]]  # basis |01>=1, |10>=2
    assert np.allclose(_givens_op(b_rot, x), want, atol=1e-9)


def test_decomposition_matches_bloq():
    """build_composite_bloq implements what the top-level bloq claims (dense equivalence)."""
    from qualtran.simulation.tensor import bloq_to_dense

    g = GivensRotationByPhaseGradient(phasegrad_bitsize=3)
    assert np.allclose(bloq_to_dense(g), bloq_to_dense(g.decompose_bloq()))


# --- cost --------------------------------------------------------------------------


def test_toffoli_cost_numeric():
    """Two Rz-by-addition => 2*b_rot - 2 Toffolis (matches Qualtran's atom)."""
    b_rot = 5
    cost = get_cost_value(
        GivensRotationByPhaseGradient(phasegrad_bitsize=b_rot), QECGatesCost()
    )
    assert cost.and_bloq == 2 * b_rot - 2


def test_toffoli_cost_symbolic():
    """Symbolically the Toffoli count is 2*b_rot - 2."""
    b = sympy.Symbol('b_rot', positive=True, integer=True)
    got = get_cost_value(
        GivensRotationByPhaseGradient(phasegrad_bitsize=b), QECGatesCost()
    ).and_bloq
    assert sympy.simplify(got - (2 * b - 2)) == 0
