r""":class:`Maj` -- the single-qubit Majorana application (step 15)."""

from __future__ import annotations

from functools import cached_property
from typing import Dict

import numpy as np
from attrs import frozen

from qualtran import Bloq, BloqBuilder, QBit, Signature, SoquetT
from qualtran.bloqs.basic_gates import CNOT
from qualtran.bloqs.mcmt import And, MultiControlZ


@frozen
class Maj(Bloq):
    r"""Single-qubit Majorana application: ``-Z`` for SF, ``(X +- iY)/2`` for D1/Q1, gated on
    ``succ_x_o and (G0 or succ_b)``.

    After the Givens ladder the generator acts on ``target`` alone, where ``gamma_0 = X`` and
    ``gamma_1 = Y``. Two ``And``s give the D1/Q1 condition ``a = succ_x_o and G0`` and the SF
    condition ``s = succ_x_o and not G0 and succ_b``, then four moments::

        (1) -Z on target    if s, b_neq_B = 1          SF: i gamma_0 gamma_1
        (2)  X on target    if a                       D1/Q1: gamma_0
        (3)  Z on target    if a, flavour = 1          ZX = iY: i gamma_1
        (4)  Z on flavour   if a, G1 = 1               the sign of iY, so Q1

    ``flavour`` arrives in ``|+>``, so on D1/Q1 ``target`` sees ``<+|Maj|+> = (X +- iY)/2``, the
    ``a_u`` and ``a_u^dag`` of Eq. 28. In Fig. 4, (1) is ``+Z = i gamma_1 gamma_0``, the wrong sign
    for Eq. 29. Here ``-Z = diag(-1, 1)`` is a phase on ``target = 0``: a ``CCZ`` with ``target``
    as an open control and ``b_neq_B`` as the Z target.

    D1/Q1 ignores ``succ_b``, so a failed ``Uniform(b)`` there still applies ``a_u``. Both ``And``s
    are uncomputed by measurement: 5 Toffolis.

    Registers:
        succ_x_o, succ_b: the success flags of ``Uniform(x_o)`` and ``Uniform(b)``.
        G0: 0 selects SF, 1 the D1/Q1 pair.
        G1: for D1/Q1, 0 is ``D1`` and 1 is ``Q1``
        b_neq_B: for SF, 1 applies ``-Z``, 0 identity
        flavour: 1-qubit LCU ancilla picking ``X`` over ``Y``
    """

    @cached_property
    def signature(self) -> Signature:
        return Signature.build_from_dtypes(
            succ_x_o=QBit(), succ_b=QBit(), G0=QBit(), G1=QBit(), b_neq_B=QBit(), flavour=QBit(),
            target=QBit(),
        )

    def build_composite_bloq(
        self, bb: BloqBuilder, succ_x_o: SoquetT, succ_b: SoquetT, G0: SoquetT, G1: SoquetT,
        b_neq_B: SoquetT, flavour: SoquetT, target: SoquetT,
    ) -> Dict[str, SoquetT]:
        # a = succ_x_o and G0; the CNOT then leaves succ_x_o and not G0 for s.
        (succ_x_o, G0), a = bb.add(And(), ctrl=np.array([succ_x_o, G0]))
        a, succ_x_o = bb.add(CNOT(), ctrl=a, target=succ_x_o)
        (succ_x_o, succ_b), s = bb.add(And(), ctrl=np.array([succ_x_o, succ_b]))

        # (1) SF with b != B: -Z on target, i.e. a phase on target = 0.
        (s, target), b_neq_B = bb.add(
            MultiControlZ(cvs=(1, 0)), controls=np.array([s, target]), target=b_neq_B
        )
        # (2), (3) D1/Q1: X, then Z on flavour = 1.
        a, target = bb.add(CNOT(), ctrl=a, target=target)
        (a, flavour), target = bb.add(
            MultiControlZ(cvs=(1, 1)), controls=np.array([a, flavour]), target=target
        )
        # (4) Q1 vs D1 sign.
        (a, G1), flavour = bb.add(
            MultiControlZ(cvs=(1, 1)), controls=np.array([a, G1]), target=flavour
        )

        succ_x_o, succ_b = bb.add(And().adjoint(), ctrl=np.array([succ_x_o, succ_b]), target=s)
        a, succ_x_o = bb.add(CNOT(), ctrl=a, target=succ_x_o)
        succ_x_o, G0 = bb.add(And().adjoint(), ctrl=np.array([succ_x_o, G0]), target=a)

        return {
            'succ_x_o': succ_x_o, 'succ_b': succ_b, 'G0': G0, 'G1': G1, 'b_neq_B': b_neq_B,
            'flavour': flavour, 'target': target,
        }
