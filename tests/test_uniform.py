r"""``UniformViaPhaseGradient``, steps 1 and 5: Qualtran's ``Uniform`` with its two ``Rz`` on a
phase gradient of ``s`` bits, and the success flag ``succ = [x < L]``."""

import numpy as np
import pytest
import sympy

from qualtran import BloqBuilder
from qualtran.bloqs.basic_gates import IntState
from qualtran.bloqs.rotations.phase_gradient import PhaseGradientState
from qualtran.resource_counting import get_cost_value, QECGatesCost

from dfthcbe.prepare import UniformViaPhaseGradient
from test_prepare import _toffoli


@pytest.mark.parametrize("s", [3, 4, 5])
@pytest.mark.parametrize("L", [3, 5, 6, 7, 12])
def test_uniform_on_success_and_its_failure_probability(L, s):
    """With ``succ = 1`` the state is exactly uniform over ``[0, L)``, and ``P(succ = 0)`` is
    ``failure_probability()``, the numpy model the sizes below rely on, spread evenly over
    ``[L, 2**lg L)``."""
    uniform = UniformViaPhaseGradient(L, s=s)
    n = uniform.signature.get_left('target').total_bits()
    bb = BloqBuilder()
    out = bb.add_d(
        uniform, target=bb.add(IntState(0, n)), succ=bb.add(IntState(0, 1)),
        phase_grad=bb.add(PhaseGradientState(s)),
    )
    bb.add(PhaseGradientState(s).adjoint(), phase_grad=out.pop('phase_grad'))
    psi = bb.finalize(**out).tensor_contract().reshape(2**n, 2)  # target, succ
    assert np.allclose(psi[:L, 1], psi[0, 1]) and np.allclose(psi[L:, 1], 0)
    assert np.allclose(psi[:L, 0], 0)
    assert np.isclose(np.sum(np.abs(psi[:, 0]) ** 2), uniform.failure_probability())
    assert np.allclose(np.abs(psi[L:, 0]), np.abs(psi[L, 0]))


def test_failure_probability_over_s_and_L():
    """``P(succ = 0)`` never grows with ``s``, and at the default ``s = 7`` it is below 0.1% for
    FeMoco's ``B + 1 = 28`` and ``N + RC = 324``, the paper's choice. ``-s`` prints the table."""
    sizes, bits = (3, 5, 7, 28, 81, 324), range(3, 11)
    table = np.array(
        [[UniformViaPhaseGradient(L, s=s).failure_probability() for s in bits] for L in sizes]
    )
    print('\nP(succ = 0)' + ''.join(f'{f"s={s}":>9}' for s in bits))
    for L, row in zip(sizes, table):
        print(f'L = {L:<7d}' + ''.join(f'{p:9.1e}' for p in row))
    assert np.all(np.diff(table, axis=1) <= 1e-15)
    assert all(UniformViaPhaseGradient(L).failure_probability() < 1e-3 for L in (28, 324))


@pytest.mark.parametrize("s", [3, 7])
def test_symbolic_count_is_the_built_one(s):
    """Over odd, even and power-of-two ``L``: Qualtran's symbolic count takes every ``L`` odd."""
    L = sympy.Symbol('L', positive=True, integer=True)
    symbolic = sympy.sympify(_toffoli(UniformViaPhaseGradient(L, s=s)))
    for n in (2, 3, 4, 5, 6, 7, 8, 12, 27, 28, 81, 324):
        assert symbolic.subs(L, n) == int(_toffoli(UniformViaPhaseGradient(n, s=s))), n


def test_no_rotation_left():
    """Both rotations are phase-gradient additions: nothing is left in the ``rotation`` bucket."""
    L = sympy.Symbol('L', positive=True, integer=True)
    for uniform in (UniformViaPhaseGradient(324), UniformViaPhaseGradient(L)):
        assert get_cost_value(uniform, QECGatesCost()).rotation == 0


def test_s_needs_a_wide_enough_phase_gradient():
    with pytest.raises(ValueError):
        UniformViaPhaseGradient(5, s=7, phase_grad_bitsize=4)
