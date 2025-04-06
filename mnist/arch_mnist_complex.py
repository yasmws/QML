import pennylane as qml

def create_complex_ttn(num_qubits):
    dev = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(dev, diff_method="parameter-shift")
    def circuit(data, weights):
        # Codificação dos dados (mantemos RY, pois o encoding continua real)
        for i in range(num_qubits):
            qml.RY(data[i], wires=i)
        
        # Bloco 1 (qubits 0 e 1) – 6 parâmetros
        qml.Rot(weights[0], weights[1], weights[2], wires=0)
        qml.Rot(weights[3], weights[4], weights[5], wires=1)
        qml.CNOT(wires=[0, 1])
        
        # Bloco 2 (qubits 2 e 3) – 6 parâmetros
        qml.Rot(weights[6], weights[7], weights[8], wires=2)
        qml.Rot(weights[9], weights[10], weights[11], wires=3)
        qml.CNOT(wires=[2, 3])
        
        # Bloco 3 (segunda camada, qubits 1 e 3) – 6 parâmetros
        qml.Rot(weights[12], weights[13], weights[14], wires=1)
        qml.Rot(weights[15], weights[16], weights[17], wires=3)
        qml.CNOT(wires=[1, 3])
        
        # Medição do último qubit (qubit 3)
        return qml.expval(qml.PauliZ(wires=num_qubits - 1))
    
    return circuit
