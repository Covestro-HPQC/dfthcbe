r"""The walk's two reflections, and which registers they span.

``REF_T2`` (step 23) sits *inside* the block encoding and does the oblivious amplitude
amplification that squares it; ``REF_aB`` (step 28) sits outside and promotes it to a
qubitization walk. Both are Qualtran's ``reflection_around_zero``, on ``n`` qubits costing
``n - 1`` Toffolis whether controlled or not.
"""

import sympy

from qualtran.resource_counting import get_cost_value, QECGatesCost
from qualtran.symbolics import bit_length

from dfthcbe import DFTHCWalk

N, R, B, C, b_rot, b_k1, b_k2, k_1, k_2, k_e1, k_e2 = sympy.symbols(
    'N R B C b_rot b_k1 b_k2 k_1 k_2 k_e1 k_e2', positive=True, integer=True
)
FEMOCO = {
    N: 54, R: 10, B: 27, C: 27, b_rot: 15, b_k1: 15, b_k2: 15, k_1: 2, k_2: 4, k_e1: 4, k_e2: 7
}
WALK = DFTHCWalk.symbolic(N, R, B, C, b_rot, b_k1, b_k2, k_1, k_2, k_e1, k_e2)


def _ccz(bloq):
    return get_cost_value(bloq, QECGatesCost()).total_t_and_ccz_count()['n_ccz']


def test_ref_t2_spans_the_inner_ancillas():
    """``b, m_b, sigma_0``: ``ceil lg(B+1) + b_k2 + 1`` qubits, so 20 at FeMoco-54.

    No rotated qubit: Qualtran's ``Uniform(b)`` leaves its ancillas clean.
    """
    reflected = bit_length(B) + b_k2 + 1
    assert sympy.simplify(_ccz(WALK.block_encoding.ref_t2) - (reflected - 1)) == 0
    assert _ccz(WALK.block_encoding.ref_t2).subs(FEMOCO) == 20


def test_ref_ab_spans_both_ancilla_sets():
    """``a u B``: ``ceil lg(N+RC) + ceil lg(B+1) + b_k1 + b_k2 + 2`` qubits, so 45.

    The ``+ 2`` are the two spins. Controlled, it still costs 45. The paper's one Toffoli more is
    the unary iteration on the phase-estimation register, which belongs to phase estimation.
    """
    reflected = bit_length(N + R * C - 1) + bit_length(B) + b_k1 + b_k2 + 2
    assert sympy.simplify(_ccz(WALK.ref_ab) - (reflected - 1)) == 0
    assert _ccz(WALK.ref_ab).subs(FEMOCO) == 45
