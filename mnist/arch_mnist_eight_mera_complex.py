import pennylane as qml

def create_complex_eight_mera(num_qubits):
    dev = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(dev, diff_method="parameter-shift")
    def circuit(data, weights):
        #codificação dos dados (Rot em cada qubit)
        for i in range(num_qubits):
            qml.RY(data[i], wires=i)

        #Primeira camada - MERA
        qml.Rot(weights[0],weights[1],weights[2], wires=1)
        qml.Rot(weights[3],weights[4],weights[5], wires=2)
        qml.CNOT(wires=[1, 2])

        qml.Rot(weights[6],weights[7],weights[8], wires=3)
        qml.Rot(weights[9],weights[10],weights[11], wires=4)
        qml.CNOT(wires=[3, 4])

        qml.Rot(weights[12],weights[13],weights[14], wires=5)
        qml.Rot(weights[15],weights[16],weights[17], wires=6)
        qml.CNOT(wires=[5, 6])

        
        #Primeira camada - TTN
        qml.Rot(weights[18],weights[19],weights[20], wires=0)
        qml.Rot(weights[21],weights[22],weights[23], wires=1)
        qml.CNOT(wires=[0, 1])
        
        qml.Rot(weights[24],weights[25],weights[26], wires=2)
        qml.Rot(weights[27],weights[28],weights[29], wires=3)
        qml.CNOT(wires=[3, 2])

        qml.Rot(weights[30],weights[31],weights[32], wires=4)
        qml.Rot(weights[33],weights[34],weights[35], wires=5)
        qml.CNOT(wires=[4, 5])
        
        qml.Rot(weights[36],weights[37],weights[38], wires=6)
        qml.Rot(weights[39],weights[40],weights[41], wires=7)
        qml.CNOT(wires=[7, 6])

        #Segunda camada - MERA
        qml.Rot(weights[42],weights[43],weights[44], wires=2)
        qml.Rot(weights[45],weights[46],weights[47], wires=5)
        qml.CNOT(wires=[2, 5])
   
        #Segunda camada
        qml.Rot(weights[48],weights[49],weights[50], wires=1)
        qml.Rot(weights[51],weights[52],weights[53], wires=2)
        qml.CNOT(wires=[1, 2])

        qml.Rot(weights[54],weights[55],weights[56], wires=5)
        qml.Rot(weights[57],weights[58],weights[59], wires=6)
        qml.CNOT(wires=[6, 5])

        #Terceira camada
        qml.Rot(weights[60],weights[61],weights[62], wires=2)
        qml.Rot(weights[63],weights[64],weights[65], wires=5)
        qml.CNOT(wires=[2, 5])

        qml.Rot(weights[66],weights[67],weights[68], wires=5)
        
        #Medição do último qubit com uma (PauliZ)
        return qml.expval(qml.PauliZ(wires=5))
    
    return circuit