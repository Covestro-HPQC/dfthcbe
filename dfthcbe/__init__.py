"""dfthcbe -- Qualtran block encoding of the DFTHC Hamiltonian.

Step numbers refer to the circuit in ``README.md``:

* :class:`InnerSelect` -- steps 10-18: the two angle loads, the multiplexed spin, ``USwap``, the
  Givens ladder -> ``Maj`` -> the ladder undone, and the erase.
* :class:`GivensRotationLadder` -- the Givens ladder that changes basis given loaded angles.
* :class:`Maj` -- the single-qubit Majorana application.
* :class:`RotDataErase` and the two QROM factories -- the rotation-angle data.
* :class:`OuterPrepare` / :class:`InnerPrepare` -- steps 1-4 and 5-8: coherent alias
  sampling over ``x_o`` and, nested inside it, over ``b``; :class:`InnerSigns` -- step 9.
* :class:`DFTHCBlockEncoding` -- steps 1-27: the block encoding of ``H_SA / Lambda - I``.
* :class:`DFTHCWalk` -- all 28 steps: the qubitization walk operator.
* :mod:`dfthcbe.alias_data` -- the alias read and its erase (:class:`QROAMCleanHetero`), and
  exact symbolic costs of the pruned reads.
* :mod:`dfthcbe.utils` -- shared helpers: ``s(.)`` (popcount) and ``beta(a, b)`` (aligned blocks)
  as display functions; :func:`reconstruct`, the integrals of a decomposed Hamiltonian; and the
  walk's view of its factors: :func:`walk_inputs`, :func:`walk_e_sos` and :func:`walk_integrals`;
  and :func:`hamiltonian_matrix`, a Hamiltonian as a matrix on the system registers.
"""

from dfthcbe.alias_data import PrunedQROM, QROAMCleanHetero
from dfthcbe.majorana import Maj
from dfthcbe.prepare import (
    alias_probabilities,
    alias_table,
    InnerPrepare,
    InnerSigns,
    OuterPrepare,
    signed_alias_table,
)
from dfthcbe.rot_data import (
    angles_from_vector,
    discretize_angles,
    RotDataErase,
    sf_rotation_qrom,
    tail_rotation_qrom,
    vector_from_angles,
)
from dfthcbe.rotations import GivensRotationLadder
from dfthcbe.select import InnerSelect
from dfthcbe.block_encoding import DFTHCBlockEncoding, DFTHCWalk
from dfthcbe.utils import (
    aligned_blocks,
    hamiltonian_matrix,
    majoranas,
    num_blocks,
    popcount,
    reconstruct,
    walk_e_sos,
    walk_inputs,
    walk_integrals,
)
