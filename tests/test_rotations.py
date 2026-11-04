r"""Tests for :class:`GivensRotationLadder`, the Givens ladder (steps 14, 16).

The ladder alone is a pure unitary, so **correctness** is checked by tensor simulation at small
``N`` with the ``angles``/``phase_grad`` registers pinned via ``IntState``/``IntEffect``
(leaving them open blows the contraction up).
**Cost** comes from the call graph, numeric and symbolic.
"""

import numpy as np
import pytest
import sympy

from qualtran import BloqBuilder, DecomposeTypeError, QAny
from qualtran.bloqs.basic_gates import IntEffect, IntState
from qualtran.bloqs.rotations.phase_gradient import PhaseGradientState
from qualtran.resource_counting import get_cost_value, QECGatesCost

from dfthcbe import angles_from_vector, discretize_angles, GivensRotationLadder, vector_from_angles
from dfthcbe.givens import GivensRotationByPhaseGradient

N = 3
B_ROT = 5
ANGLE_TABLE = np.array([[3, 9], [12, 6], [1, 25]])  # (3 rotations, N-1=2 angles each)


def _ccz(bloq):
    return get_cost_value(bloq, QECGatesCost()).total_t_and_ccz_count()['n_ccz']


def _ladder_system_op(ladder: GivensRotationLadder, angle_row):
    """The operator the ladder applies to ``system`` for one fixed loaded ``angle_row``."""
    b, n = ladder.b_rot, ladder.num_orb
    bb = BloqBuilder()
    system = bb.add_register('system', QAny(n))
    pg = bb.add(PhaseGradientState(bitsize=b))
    angs = [bb.add(IntState(val=int(a), bitsize=b)) for a in angle_row]
    system, angs_out, pg = bb.add(ladder, system=system, angles=np.array(angs), phase_grad=pg)
    bb.add(PhaseGradientState(bitsize=b).adjoint(), phase_grad=pg)
    for k, a in enumerate(angle_row):
        bb.add(IntEffect(val=int(a), bitsize=b), val=angs_out[k])
    return bb.finalize(system=system).tensor_contract()


def _reference_ladder_op(angle_row, b_rot, n_orb):
    """Ground-truth dense operator the ladder applies: the rungs built atom-for-atom, in the
    same bottom-up order (``k = N-2 .. 0``) as ``GivensRotationLadder``.
    """
    givens = GivensRotationByPhaseGradient(phasegrad_bitsize=b_rot)
    bb = BloqBuilder()
    qs = [bb.add_register(f'q{i}', 1) for i in range(n_orb)]
    pg = bb.add(PhaseGradientState(bitsize=b_rot))
    roms = [bb.add(IntState(val=int(a), bitsize=b_rot)) for a in angle_row]
    for k in reversed(range(n_orb - 1)):
        qs[k], qs[k + 1], roms[k], pg = bb.add(
            givens, target_i=qs[k], target_j=qs[k + 1], rom_data=roms[k], phase_gradient=pg
        )
    bb.add(PhaseGradientState(bitsize=b_rot).adjoint(), phase_grad=pg)
    for k, a in enumerate(angle_row):
        bb.add(IntEffect(val=int(a), bitsize=b_rot), val=roms[k])
    return bb.finalize(**{f'q{i}': qs[i] for i in range(n_orb)}).tensor_contract()


def _majorana(j, n):
    """gamma_j = Z_0 .. Z_{j-1} X_j (Jordan-Wigner, qubit 0 = MSB)."""
    I2 = np.eye(2)
    X = np.array([[0, 1], [1, 0]], complex)
    Z = np.array([[1, 0], [0, -1]], complex)
    op = np.array([[1]], complex)
    for m in [Z] * j + [X] + [I2] * (n - 1 - j):
        op = np.kron(op, m)
    return op


# --- correctness (tensor sim, pinned angles) ----------------------------------------


def test_ladder_is_unitary():
    U = _ladder_system_op(GivensRotationLadder(num_orb=N, b_rot=B_ROT), ANGLE_TABLE[0])
    assert U.shape == (2**N, 2**N)
    assert np.allclose(U.conj().T @ U, np.eye(2**N))


def test_ladder_matches_reference_ladder():
    """``GivensRotationLadder`` applies exactly the raw Givens ladder built from the same angles."""
    ladder = GivensRotationLadder(num_orb=N, b_rot=B_ROT)
    for row in ANGLE_TABLE:
        assert np.allclose(_ladder_system_op(ladder, row), _reference_ladder_op(row, B_ROT, N))


def test_ladder_adjoint_inverts():
    ladder = GivensRotationLadder(num_orb=N, b_rot=B_ROT)
    bb = BloqBuilder()
    system = bb.add_register('system', QAny(N))
    pg = bb.add(PhaseGradientState(bitsize=B_ROT))
    angs = [bb.add(IntState(val=int(a), bitsize=B_ROT)) for a in ANGLE_TABLE[0]]
    system, angs, pg = bb.add(ladder, system=system, angles=np.array(angs), phase_grad=pg)
    system, angs, pg = bb.add(ladder.adjoint(), system=system, angles=angs, phase_grad=pg)
    bb.add(PhaseGradientState(bitsize=B_ROT).adjoint(), phase_grad=pg)
    for k, a in enumerate(ANGLE_TABLE[0]):
        bb.add(IntEffect(val=int(a), bitsize=B_ROT), val=angs[k])
    assert np.allclose(bb.finalize(system=system).tensor_contract(), np.eye(2**N))


def test_rotates_majorana_onto_direction():
    """The defining identity: ``U(u)^dag gamma_0 U(u) = sum_j u_j gamma_j``, to the angle
    rounding, and exactly for ``vector_from_angles`` of the loaded angles.

    Pins both the proper-rotation atom and the bottom-up rung order.
    """
    n, b_rot = 3, 10
    rng = np.random.default_rng(0)
    u_vecs = rng.standard_normal((3, n))
    u_vecs /= np.linalg.norm(u_vecs, axis=1)[:, None]
    gammas = [_majorana(j, n) for j in range(n)]
    tol = 8 * n * 2.0 ** -(b_rot - 1)
    ladder = GivensRotationLadder(num_orb=n, b_rot=b_rot)
    for u in u_vecs:
        angles = discretize_angles(angles_from_vector(u), b_rot)
        U = _ladder_system_op(ladder, angles)
        rotated = U.conj().T @ gammas[0] @ U
        assert np.allclose(rotated, sum(u[j] * gammas[j] for j in range(n)), atol=tol)
        v = vector_from_angles(angles, b_rot)
        assert np.allclose(rotated, sum(v[j] * gammas[j] for j in range(n)), atol=1e-10)


# --- cost ---------------------------------------------------------------------------


def test_ladder_toffoli_cost_numeric():
    """``N-1`` Givens rungs, each ``2*b_rot - 2`` Toffolis (two ``b_rot``-bit additions)."""
    ladder = GivensRotationLadder(num_orb=N, b_rot=B_ROT)
    assert _ccz(ladder.givens) == 2 * B_ROT - 2
    assert _ccz(ladder) == (N - 1) * (2 * B_ROT - 2)


def test_ladder_toffoli_cost_symbolic():
    n, b = sympy.symbols('N b_rot', positive=True, integer=True)
    ladder = GivensRotationLadder(num_orb=n, b_rot=b)
    assert sympy.simplify(_ccz(ladder) - (n - 1) * (2 * b - 2)) == 0


def test_ladder_symbolic_cannot_decompose():
    n, b = sympy.symbols('N b_rot', positive=True, integer=True)
    with pytest.raises(DecomposeTypeError):
        GivensRotationLadder(num_orb=n, b_rot=b).decompose_bloq()
