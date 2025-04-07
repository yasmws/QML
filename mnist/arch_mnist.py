import pennylane as qml

def create_simple_ttn(num_qubits):
    dev = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(dev, diff_method="parameter-shift")
    def circuit(data, weights):
        #codificação dos dados (RY em cada qubit)
        for i in range(num_qubits):
            qml.RY(data[i], wires=i)
        
        # Camadas TTN (Figura 6 do artigo para 4 qubits)
        # Primeira camada: 2 blocos (0-1 e 3-2)
        qml.RY(weights[0], wires=0)
        qml.RY(weights[1], wires=1)
        qml.CNOT(wires=[0, 1])  #Controle: 0, Alvo: 1

        qml.RY(weights[2], wires=3) 
        qml.RY(weights[3], wires=2)  
        qml.CNOT(wires=[3, 2])  #Controle: 3, Alvo: 2
        
        #segunda camada: 1 bloco (1-2)
        qml.RY(weights[4], wires=1)
        qml.RY(weights[5], wires=2)
        qml.CNOT(wires=[1, 2]) #Controle: 1, Alvo: 2

        #ultima rotação Y no qubit 2 antes da medição
        qml.RY(weights[6], wires=2)

        # Medição do último qubit com um operador PauliZ
        return qml.expval(qml.PauliZ(wires=2))
    
    return circuit