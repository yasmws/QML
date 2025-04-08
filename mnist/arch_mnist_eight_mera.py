import pennylane as qml
import pennylane.numpy as np 

def create_simple_eight_mera(num_qubits):
    dev = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(dev, diff_method="parameter-shift")
    def circuit(data, weights):
        #codificação dos dados (RY em cada qubit)
        for i in range(num_qubits):
            qml.RY(data[i], wires=i)

        #Primeira camada - MERA
        qml.RY(weights[0], wires=1)
        qml.RY(weights[1], wires=2)
        qml.CNOT(wires=[1, 2])

        qml.RY(weights[2], wires=3)
        qml.RY(weights[3], wires=4)
        qml.CNOT(wires=[3, 4])

        qml.RY(weights[4], wires=5)
        qml.RY(weights[5], wires=6)
        qml.CNOT(wires=[5, 6])

        
        #Primeira camada - TTN
        qml.RY(weights[6], wires=0)
        qml.RY(weights[7], wires=1)
        qml.CNOT(wires=[0, 1])
        
        qml.RY(weights[8], wires=2)
        qml.RY(weights[9], wires=3)
        qml.CNOT(wires=[3, 2])

        qml.RY(weights[10], wires=4)
        qml.RY(weights[11], wires=5)
        qml.CNOT(wires=[4, 5])
        
        qml.RY(weights[12], wires=6)
        qml.RY(weights[13], wires=7)
        qml.CNOT(wires=[7, 6])

        #Segunda camada - MERA
        qml.RY(weights[14], wires=2)
        qml.RY(weights[15], wires=5)
        qml.CNOT(wires=[2, 5])
   
        #Segunda camada
        qml.RY(weights[16], wires=1)
        qml.RY(weights[17], wires=2)
        qml.CNOT(wires=[1, 2])

        qml.RY(weights[18], wires=5)
        qml.RY(weights[19], wires=6)
        qml.CNOT(wires=[6, 5])

        #Terceira camada
        qml.RY(weights[20], wires=2)
        qml.RY(weights[21], wires=5)
        qml.CNOT(wires=[2, 5])

        qml.RY(weights[22], wires=5)
        
        #Medição do último qubit com uma (PauliZ)
        return qml.expval(qml.PauliZ(wires=5))
    
    return circuit

# Cria o circuito para 8 qubits
circuit = create_simple_eight_mera(num_qubits=8)

# Define parâmetros arbitrários para desenhar o circuito
data = np.zeros(8)      # vetor com 8 elementos para os ângulos de RY na codificação
weights = np.zeros(23)  # vetor com 15 elementos, conforme utilizado no circuito

# Gera a imagem do circuito utilizando qml.draw com output "mpl"
print(qml.draw(circuit)(data, weights))



