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

# Função de codificação: aplica as rotações de codificação em cada qubit
def embedding_layer(features):
    for i in range(num_qubits):
        qml.RY(2 * features[i], wires=i)
