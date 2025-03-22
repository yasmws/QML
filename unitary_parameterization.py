import pennylane as qml
import numpy as np

# =============================================================================
# 1. Simple Unitary Parameterization
# =============================================================================
def simple_unitary_block(params, wires):
    """
    Bloco simples: aplica rotações RY parametrizadas em cada qubit do bloco,
    seguido por uma porta CNOT para criar emaranhamento entre os qubits.
    """
    for i, wire in enumerate(wires):
        qml.RY(params[i], wires=wire)
    qml.CNOT(wires=wires)

# =============================================================================
# 2. General Unitary Parameterization
# =============================================================================
def param_to_unitary(params):
    """
    Implementar ainda: na teoria converte uma lista de parâmetros em uma matriz unitária 4x4.
    Na prática, pode-se usar uma decomposição (por exemplo, KAK) ou outro método para
    obter uma unidade arbitrária de dois qubits.
    """
    return np.eye(4)

def general_unitary_block(params, wires):
    U = param_to_unitary(params)
    qml.QubitUnitary(U, wires=wires)

# =============================================================================
# 3. Ancilla-based Unitary Parameterization
# =============================================================================
def param_to_three_qubit_unitary(params):
    """
    Implementar ainda: converte uma lista de parâmetros em uma matriz unitária 8x8 para três qubits.
    Substituir por uma parametrização apropriada para uma unidade arbitrária de três qubits.
    """
    return np.eye(8)

def ancilla_unitary_block(params, wires, ancilla_wire):
    """
    Bloco com ancilla: utiliza dois qubits principais (especificados em 'wires') e um qubit ancilla.
    O qubit ancilla é inicializado no estado |0⟩ e, em seguida, uma unidade de três qubits (nos
    qubits dos wires + ancilla) é aplicada.
    """
    # Garante que o qubit ancilla esteja no estado |0⟩
    qml.BasisState(np.array([0]), wires=ancilla_wire)
    U = param_to_three_qubit_unitary(params)
    qml.QubitUnitary(U, wires=wires + [ancilla_wire])