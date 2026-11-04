import pytest
from dfthcbe.prepare import OuterPrepare
from qualtran.linalg.lcu_util import sub_bit_prec_from_epsilon, preprocess_probabilities_for_reversible_sampling
import numpy as np
from qualtran import BloqBuilder
from qualtran.bloqs.rotations.phase_gradient import PhaseGradientState
import re
import cirq


@pytest.mark.intensive
@pytest.mark.parametrize(
    "N, R, C, B, x_o_coeffs, precision",
    [
        (2, 2, 1, 1, np.array([0.35, 0.15, 0.25, 0.25]), 1e-2),
        (2, 3, 1, 1, np.array([0.35, 0.15, 0.2, 0.1, 0.2]), 2e-2),  # b_k1 = 4: 28 qubits, not 30
    
    ]
)
def test_outer_prepare(N, R, C, B, x_o_coeffs, precision):
    b_k1 = sub_bit_prec_from_epsilon(N+R*C, precision) #size if the keep register
    alt_table, keep_table, _ = preprocess_probabilities_for_reversible_sampling(
                unnormalized_probabilities=x_o_coeffs, epsilon=precision
            )
    # s = 1 keeps the phase gradient at 1 qubit; reading succ = 1 only is exact at any s
    bloq = OuterPrepare(
        alt=alt_table, keep=keep_table, b_k1=b_k1, log_block_size=1, s=1
    )
    # phase_grad holds a phase-gradient state, for the two rotations of Uniform
    bb = BloqBuilder()
    soqs = {r.name: bb.add_register(r) for r in bloq.signature.lefts() if r.name != 'phase_grad'}
    soqs['phase_grad'] = bb.add(PhaseGradientState(bloq.phase_grad_bitsize))
    out = bb.add_from(bloq.decompose_bloq(), **soqs)  # its own boxes: Uniform stays one box
    cbloq = bb.finalize(**dict(zip([r.name for r in bloq.signature.rights()], out)))

    circuit = cbloq.to_cirq_circuit()
    circuit = cirq.Circuit(circuit)
    
    if cirq.__version__ >= "1.6.1":
        # Split boxes over 13 qubits: cirq >= 1.6.1 would build their full unitary. Reuse the
        # temporary qubits of their parts, or the state vector outgrows memory.
        reuse = cirq.DecompositionContext(cirq.GreedyQubitManager('anc', maximize_reuse=True))
        circuit = cirq.Circuit(
            cirq.decompose(circuit, keep=lambda op: len(op.qubits) <= 13, context=reuse)
        )
    
    simulator = cirq.Simulator()
    result = simulator.simulate(circuit)
    sv = result.final_state_vector

    all_qubits = sorted(result.qubit_map.keys(), key=str)

    x_o_qubits = sorted(
            (q for q in all_qubits if isinstance(q, cirq.NamedQubit) and re.fullmatch(r"x_o\d", q.name)),
            key=lambda q: q.name,
        )
    

    joint_indices = [result.qubit_map[q] for q in x_o_qubits]
    n = len(all_qubits)

    print(f"x_o qubits ({len(x_o_qubits)}): {[q.name for q in x_o_qubits]}")

    succ = result.qubit_map[cirq.NamedQubit("succ_x_o")]
    p_fail = np.sum(np.abs(sv.reshape([2] * n).take(0, axis=succ)) ** 2)
    assert np.isclose(p_fail, bloq.uniform.failure_probability())

    joint_probs = {}
    for idx, amp in enumerate(sv):
        bits = format(idx, f"0{n}b")
        if abs(amp) < 1e-5 or bits[succ] == "0":
            continue

        joint_bits = "".join(bits[i] for i in joint_indices) # only considering the bits that fall into x_o register
        joint_val = int(joint_bits, 2)
        
        joint_probs[joint_val] = joint_probs.get(joint_val, 0) + abs(amp) ** 2

    print(f"joint_probs: {joint_probs}")
    x_o_state_probs = np.zeros((N+R*C,))
    for val, prob in joint_probs.items():
        x_o_state_probs[val] = prob / (1 - p_fail)

    assert np.allclose(x_o_state_probs, x_o_coeffs, rtol=0., atol=precision)
