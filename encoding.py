import pennylane as qml
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler

# Carregar o dataset Iris e normalizar as features para [0, π/2]
iris = datasets.load_iris()
X = iris.data
y = iris.target

scaler = MinMaxScaler(feature_range=(0, np.pi / 2))
X_scaled = scaler.fit_transform(X)

# Definir número de qubits (precisamos de um por feature)
num_qubits = X.shape[1]

# Criar o dispositivo quântico simulado
dev = qml.device("default.qubit", wires=num_qubits)

# Definir o circuito quântico de encoding
@qml.qnode(dev)
def quantum_embedding(features):
    for i in range(num_qubits):
        qml.RY(features[i], wires=i)  # Rotação Y para cada qubit com base na feature
    return qml.state()

# Testando com uma amostra do dataset
sample_data = X_scaled[0]
quantum_state = quantum_embedding(sample_data)

print("Quantum State Representation:", quantum_state)
