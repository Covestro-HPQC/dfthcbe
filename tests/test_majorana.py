r"""Tests for :class:`Maj`, the single-qubit Majorana gate (step 15).

Seven qubits, so exact contraction is cheap: we read off the operator applied to ``target`` for
every control setting. ``G_0 = 0`` is SF and ``G_0 = 1``
the D1/Q1 pair, as the SF load fires on ``G_0 = 0``.

``flavour`` arrives ``|+>``, so the operator applied is the ``<+|Maj|+>`` block on that wire.
"""

import numpy as np

from qualtran.resource_counting import get_cost_value, QECGatesCost

from dfthcbe import Maj

X = np.array([[0, 1], [1, 0]], complex)
Y = np.array([[0, -1j], [1j, 0]], complex)
Z = np.array([[1, 0], [0, -1]], complex)
I2 = np.eye(2, dtype=complex)

# (G0, G1, b_neq_B) -> the operator on `target`. The 1/2 on D1/Q1 is the |+>-ancilla LCU
# subnormalization (bookkept by PREPARE); the SF -Z branch stays weight 1. `G = 01` is absent
# because it is unreachable: the tail load sets G1 only where G0 = 1 (test_rot_data).
CASES = {
    (0, 0, 1): -Z,                # SF, b != B        -> -Z = i gamma_0 gamma_1
    (0, 0, 0): I2,                # SF, b  = B        -> the identity term
    (1, 0, 1): (X + 1j * Y) / 2,  # D1                -> raising
    (1, 1, 1): (X - 1j * Y) / 2,  # Q1                -> lowering
    (1, 0, 0): (X + 1j * Y) / 2,  # D1/Q1 ignores b_neq_B
    (1, 1, 0): (X - 1j * Y) / 2,
}


def _ccz(bloq):
    return get_cost_value(bloq, QECGatesCost()).total_t_and_ccz_count()['n_ccz']


def _target_op(U, G0, G1, b_neq_B, succ=(1, 1)):
    """The 2x2 operator ``U`` applies to ``target``, i.e. ``<+|U|+>`` on ``flavour``.

    Bit order is the signature order, MSB first: ``succ_x_o, succ_b, G0, G1, b_neq_B, flavour,
    target``.
    """
    head = 0
    for bit in (*succ, G0, G1, b_neq_B):
        head = head << 1 | bit
    ix = lambda flavour, target: ((head << 1 | flavour) << 1) | target
    plus = lambda out, inp: sum(U[ix(i, out), ix(j, inp)] for i in (0, 1) for j in (0, 1)) / 2
    return np.array([[plus(out, inp) for inp in (0, 1)] for out in (0, 1)])


def test_maj_applies_correct_operator():
    """``-Z`` for SF (gated by ``b != B``), ``(X +- iY)/2`` for D1/Q1 with the sign from ``G1``.
    Unitary, so both ``And``s come back clean."""
    U = Maj().tensor_contract()
    assert np.allclose(U.conj().T @ U, np.eye(2**7))
    for controls, expect in CASES.items():
        assert np.allclose(_target_op(U, *controls), expect), controls


def test_maj_fires_on_succ_x_o_and_g0_or_succ_b():
    """SF needs both success flags, D1/Q1 only ``succ_x_o``; otherwise the identity."""
    U = Maj().tensor_contract()
    for (G0, G1, b_neq_B), expect in CASES.items():
        for succ_x_o, succ_b in [(0, 0), (0, 1), (1, 0)]:
            want = expect if succ_x_o and (G0 or succ_b) else I2
            got = _target_op(U, G0, G1, b_neq_B, succ=(succ_x_o, succ_b))
            assert np.allclose(got, want), (G0, G1, b_neq_B, succ_x_o, succ_b)


def test_maj_toffoli_cost():
    """Two ``And``s and three ``CCZ``s."""
    assert _ccz(Maj()) == _ccz(Maj().adjoint()) == 5
