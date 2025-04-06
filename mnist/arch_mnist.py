import pennylane as qml
from pennylane import numpy as np

def create_simple_ttn(num_qubits):
    dev = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(dev, diff_method="parameter-shift")
    def circuit(data, weights):
        #codificação dos dados (RY em cada qubit)
        for i in range(num_qubits):
            qml.RY(data[i], wires=i)
        
        # Camadas TTN (Figura 6 do artigo para 4 qubits)
        #primeira camada: 2 blocos (0-1 e 2-3)
        qml.RY(weights[0], wires=0)
        qml.RY(weights[1], wires=1)
        qml.CNOT(wires=[0, 1])
        
        qml.RY(weights[2], wires=2)
        qml.RY(weights[3], wires=3)
        qml.CNOT(wires=[2, 3])
        
        #Segunda camada: 1 bloco (1-3)
        qml.RY(weights[4], wires=1)
        qml.RY(weights[5], wires=3)
        qml.CNOT(wires=[1, 3])
        
        #Medição do último qubit com uma (PauliZ)
        return qml.expval(qml.PauliZ(wires=num_qubits - 1))
    
    return circuit

def block(weights, wires):
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.RY(weights[0], wires=wires[0])
    qml.RY(weights[1], wires=wires[1])

# Função para criar o circuito MERA
def create_simple_mera(num_qubits, num_weights):
    dev = qml.device("default.qubit", wires=num_qubits)

    # Configurações para o template MERA:
    n_block_wires = 2           # número de wires por bloco (deve ser par)
    n_params_block = 2          # cada bloco terá 2 parâmetros

    @qml.qnode(dev, diff_method="parameter-shift")
    def circuit(data, weights):
        # Codifica os dados nos qubits
        for i in range(num_qubits):
            qml.RY(data[i], wires=i)

        # Aplica o template MERA usando os argumentos na ordem correta
        qml.MERA(
            list(range(num_qubits)),
            n_block_wires,
            block,
            n_params_block,
            weights  # template_weights
        )


        # Medição: expectativa do PauliZ no último qubit
        return qml.expval(qml.PauliZ(wires=num_qubits - 1))

    return circuit