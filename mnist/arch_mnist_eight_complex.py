import pennylane as qml

def create_complex_eight_ttn(num_qubits):
    dev = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(dev, diff_method="parameter-shift")
    def circuit(data, weights):
        #codificação dos dados (RY em cada qubit)
        for i in range(num_qubits):
            qml.RY(data[i], wires=i)
        
        #Primeira camada
        qml.Rot(weights[0],weights[1],weights[2],wires=0)
        qml.Rot(weights[3],weights[4], weights[5],wires=1)
        qml.CNOT(wires=[0, 1])
        
        qml.Rot(weights[6],weights[7], weights[8], wires=2)
        qml.Rot(weights[9],weights[10], weights[11], wires=3)
        qml.CNOT(wires=[3, 2])

        qml.Rot(weights[12],weights[13], weights[14], wires=4)
        qml.Rot(weights[15],weights[16], weights[17], wires=5)
        qml.CNOT(wires=[4, 5])
        
        qml.Rot(weights[18],weights[19], weights[20], wires=6)
        qml.Rot(weights[21],weights[22], weights[23], wires=7)
        qml.CNOT(wires=[7, 6])
        
        #Segunda camada
        qml.Rot(weights[24],weights[25], weights[26], wires=1)
        qml.Rot(weights[27],weights[28], weights[29], wires=2)
        qml.CNOT(wires=[1, 2])

        qml.Rot(weights[30],weights[31], weights[32], wires=5)
        qml.Rot(weights[33],weights[34], weights[35], wires=6)
        qml.CNOT(wires=[6, 5])

        #Terceira camada
        qml.Rot(weights[36],weights[37], weights[38],wires=2)
        qml.Rot(weights[39],weights[40], weights[41], wires=5)
        qml.CNOT(wires=[2, 5])

        qml.Rot(weights[42],weights[43], weights[44], wires=5)
        
        #Medição do último qubit com uma (PauliZ)
        return qml.expval(qml.PauliZ(wires=5))
    
    return circuit