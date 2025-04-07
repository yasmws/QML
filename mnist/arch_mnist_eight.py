import pennylane as qml

def create_simple_eight_ttn(num_qubits):
    dev = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(dev, diff_method="parameter-shift")
    def circuit(data, weights):
        #codificação dos dados (RY em cada qubit)
        for i in range(num_qubits):
            qml.RY(data[i], wires=i)
        
        #Primeira camada
        qml.RY(weights[0], wires=0)
        qml.RY(weights[1], wires=1)
        qml.CNOT(wires=[0, 1])
        
        qml.RY(weights[2], wires=2)
        qml.RY(weights[3], wires=3)
        qml.CNOT(wires=[3, 2])

        qml.RY(weights[4], wires=4)
        qml.RY(weights[5], wires=5)
        qml.CNOT(wires=[4, 5])
        
        qml.RY(weights[6], wires=6)
        qml.RY(weights[7], wires=7)
        qml.CNOT(wires=[7, 6])
        
        #Segunda camada
        qml.RY(weights[8], wires=1)
        qml.RY(weights[9], wires=2)
        qml.CNOT(wires=[1, 2])

        qml.RY(weights[10], wires=5)
        qml.RY(weights[11], wires=6)
        qml.CNOT(wires=[6, 5])

        #Terceira camada
        qml.RY(weights[12], wires=2)
        qml.RY(weights[13], wires=5)
        qml.CNOT(wires=[2, 5])

        qml.RY(weights[14], wires=5)
        
        #Medição do último qubit com uma (PauliZ)
        return qml.expval(qml.PauliZ(wires=5))
    
    return circuit