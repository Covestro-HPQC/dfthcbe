r""":class:`GivensRotationLadder` -- the Givens ladder that changes basis (steps 14, 16)."""

from __future__ import annotations

from functools import cached_property
from typing import Dict, TYPE_CHECKING

import numpy as np
from attrs import frozen

from qualtran import Bloq, BloqBuilder, DecomposeTypeError, QAny, QFxp, Register, Signature, SoquetT
from qualtran.symbolics import is_symbolic

from dfthcbe.givens import GivensRotationByPhaseGradient

if TYPE_CHECKING:
    from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator
    from qualtran.symbolics import SymbolicInt


@frozen
class GivensRotationLadder(Bloq):
    r"""The ``N-1`` Givens-rotation ladder ``U(u_vec)``, the paper's ``U_Rot`` (Eq. B20): it
    applies the orbital rotation whose ``N-1`` angles arrive on the ``angles`` register::

        U_Rot |system> |angles> = U(u(angles)) |system> |angles>

    Rung ``k`` is a :class:`GivensRotationByPhaseGradient` on modes ``(q_k, q_{k+1})``,
    driven by ``angles[k]`` via addition into the shared ``phase_grad`` catalyst.

    Realises ``U^dag(u) gamma_0 U(u) = sum_j u_j gamma_j``.

    Args:
        num_orb: ``N`` -- number of spin-orbitals / system qubits. May be symbolic.
        b_rot: bits of precision per angle; also the width of each ``angles[k]`` and of the
            shared phase-gradient register. May be symbolic.

    Registers:
        system: the ``N`` system qubits ``q_0 .. q_{N-1}`` that get rotated.
        angles: the ``N-1`` loaded angles (THRU), shape ``(N-1,)``, each ``b_rot`` bits.
        phase_grad: the shared ``b_rot``-bit phase-gradient state (a catalyst).
    """

    num_orb: 'SymbolicInt'
    b_rot: 'SymbolicInt'

    @property
    def is_symbolic_(self) -> bool:
        return is_symbolic(self.num_orb, self.b_rot)

    @cached_property
    def givens(self) -> GivensRotationByPhaseGradient:
        """The single two-mode Givens atom, shared by every rung of the ladder."""
        return GivensRotationByPhaseGradient(phasegrad_bitsize=self.b_rot)

    @cached_property
    def signature(self) -> Signature:
        # angles as QAny(b_rot) to line up bit-for-bit with the QROM angle targets.
        return Signature(
            [
                Register('system', QAny(self.num_orb)),
                Register('angles', QAny(self.b_rot), shape=(self.num_orb - 1,)),
                Register('phase_grad', QFxp(self.b_rot, self.b_rot, signed=False)),
            ]
        )

    def build_composite_bloq(
        self, bb: BloqBuilder, system: SoquetT, angles: SoquetT, phase_grad: SoquetT
    ) -> Dict[str, SoquetT]:
        if self.is_symbolic_:
            raise DecomposeTypeError(f"Cannot decompose symbolic {self}.")
        sys_qubits = list(bb.split(system))
        angles = list(angles)

        for k in reversed(range(self.num_orb - 1)):
            sys_qubits[k], sys_qubits[k + 1], angles[k], phase_grad = bb.add(
                self.givens,
                target_i=sys_qubits[k],
                target_j=sys_qubits[k + 1],
                rom_data=angles[k],
                phase_gradient=phase_grad,
            )

        system = bb.join(np.array(sys_qubits), dtype=QAny(self.num_orb))
        return {'system': system, 'angles': np.array(angles), 'phase_grad': phase_grad}

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        """The ladder is ``N-1`` Givens rungs (each ``2*b_rot - 2`` Toffolis)."""
        return {self.givens: self.num_orb - 1}
