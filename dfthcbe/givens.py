r""":class:`GivensRotationByPhaseGradient` -- the real Givens rotation of each ladder rung.
The bloq applies a genuine Givens rotation on the ``{01, 10}`` subspace of the two mode
qubits and leaves ``|00>`` / ``|11>`` fixed::

    |01> ->  c|01> + s|10>
    |10> -> -s|01> + c|10>            c = cos(theta),  s = sin(theta)

i.e. the block ``[[c, -s], [s, c]]`` -- a proper `SO(2)` rotation, not the imaginary
bloq of the original.
"""

from functools import cached_property
from typing import Dict

from attrs import frozen

from qualtran import Bloq, BloqBuilder, QBit, QFxp, Signature, SoquetT
from qualtran.bloqs.basic_gates import CNOT, Hadamard, SGate
from qualtran.bloqs.basic_gates.rotation import XPowGate
from qualtran.bloqs.rotations.rz_via_phase_gradient import RzViaPhaseGradient


@frozen
class GivensRotationByPhaseGradient(Bloq):
    r"""Real Givens rotation on two mode qubits, angle loaded via a phase gradient.

    Rotates the ``{01, 10}`` subspace by ``[[c, -s], [s, c]]`` (``c=cos theta``,
    ``s=sin theta``) and fixes ``|00>``/``|11>``. The angle is ``theta = 4 pi x / 2**b``,
    where ``x`` is the integer in ``rom_data`` and ``b = phasegrad_bitsize`` (the
    ``RzViaPhaseGradient`` convention ``Rz(4 pi x)``).

    Circuit (``i`` = ``target_i`` top, ``j`` = ``target_j`` bottom; boxes are matrix
    products, i.e. the rightmost label acts first)::

        i: ──[√X†]──●──[H]──[Rz(θ)]──[H]──●──[√X]─────
                    │                     │
        j: ─[√X†S]──X───────[Rz(θ)]───────X──[S†√X]───

    Both ``Rz(theta)`` are applied by addition into the shared phase-gradient register,
    each costing ``b - 1`` Toffolis -- ``2*b - 2`` in total, matching Qualtran's atom.

    Args:
        phasegrad_bitsize: width ``b`` of the phase-gradient register, and of the
            ``rom_data`` register holding the angle's ``b``-bit binary fraction.

    Registers:
        target_i: 1st mode qubit.
        target_j: 2nd mode qubit.
        rom_data: ``QFxp`` binary fraction of the rotation angle.
        phase_gradient: ``QFxp`` phase-gradient catalyst (passed in, returned unchanged).

    References:
        [Compilation of Fault-Tolerant Quantum Heuristics for Combinatorial Optimization](https://arxiv.org/abs/2007.07391).
        Appendix A: Addition for controlled rotations.
    """

    phasegrad_bitsize: int

    @cached_property
    def signature(self) -> Signature:
        return Signature.build_from_dtypes(
            target_i=QBit(),
            target_j=QBit(),
            rom_data=self.phasegrad_dtype,
            phase_gradient=self.phasegrad_dtype,
        )

    @cached_property
    def phasegrad_dtype(self) -> QFxp:
        return QFxp(self.phasegrad_bitsize, self.phasegrad_bitsize, signed=False)

    @property
    def add_into_phasegrad(self) -> RzViaPhaseGradient:
        """Rz-by-addition into the phase gradient: applies ``Rz(4 pi x)`` for angle ``x``."""
        return RzViaPhaseGradient(
            angle_dtype=self.phasegrad_dtype, phasegrad_dtype=self.phasegrad_dtype
        )

    def build_composite_bloq(
        self,
        bb: BloqBuilder,
        target_i: SoquetT,
        target_j: SoquetT,
        rom_data: SoquetT,
        phase_gradient: SoquetT,
    ) -> Dict[str, SoquetT]:
        sx, sx_dag = XPowGate(exponent=0.5), XPowGate(exponent=-0.5)  # √X, √X†

        # Pre-rotation basis change (Clifford). On j the box √X†S means: S first, then √X†.
        target_i = bb.add(sx_dag, q=target_i)
        target_j = bb.add(SGate(), q=target_j)
        target_j = bb.add(sx_dag, q=target_j)
        target_i, target_j = bb.add(CNOT(), ctrl=target_i, target=target_j)
        target_i = bb.add(Hadamard(), q=target_i)

        # Two parallel Rz(theta), both driven by the same angle in rom_data.
        target_i, rom_data, phase_gradient = bb.add(
            self.add_into_phasegrad, q=target_i, angle=rom_data, phase_grad=phase_gradient
        )
        target_j, rom_data, phase_gradient = bb.add(
            self.add_into_phasegrad, q=target_j, angle=rom_data, phase_grad=phase_gradient
        )

        # Post-rotation basis change (Clifford). On j the box S†√X means: √X first, then S†.
        target_i = bb.add(Hadamard(), q=target_i)
        target_i, target_j = bb.add(CNOT(), ctrl=target_i, target=target_j)
        target_i = bb.add(sx, q=target_i)
        target_j = bb.add(sx, q=target_j)
        target_j = bb.add(SGate(is_adjoint=True), q=target_j)

        return {
            'target_i': target_i,
            'target_j': target_j,
            'rom_data': rom_data,
            'phase_gradient': phase_gradient,
        }
