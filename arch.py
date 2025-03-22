import pennylane as qml
import numpy as np
from encoding import embedding_layer, dev, X_scaled 

# Definir um bloco unitário de dois qubits
def unitary_block(block_params, wires):
    for i, wire in enumerate(wires):
        qml.RY(block_params[i], wires=wire)
    qml.CNOT(wires=wires)

@qml.qnode(dev)
def ttn_classifier(features, params):
    # Primeira etapa: aplicar a camada de codificação importada
    embedding_layer(features)
    
    # Aplicar blocos unitários na primeira camada (por exemplo, qubits [0,1] e [2,3])
    unitary_block(params[0], wires=[0, 1])
    unitary_block(params[1], wires=[2, 3])
    
    # Segunda camada: combinar os resultados (por exemplo, utilizando os qubits 1 e 3)
    unitary_block(params[2], wires=[1, 3])
    
    # Medida final: retorno do valor de expectativa de PauliZ em um qubit
    return qml.expval(qml.PauliZ(3))

# Exemplo de parâmetros dummy para os blocos (2 parâmetros por bloco)
params = [np.array([0.1, 0.2]), np.array([0.3, 0.4]), np.array([0.5, 0.6])]

# Selecionar uma amostra do dataset já escalado
sample_features = X_scaled[0]
prediction = ttn_classifier(sample_features, params)
print("Saída do classificador (valor de expectativa):", prediction)
