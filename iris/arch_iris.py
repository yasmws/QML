import pennylane as qml
import numpy as np

num_qubits=4 #Número total de quibits utilizados no circuito (seguindo o artigo)

dev=qml.device("default.qubit",wires=num_qubits) #Criação de um simulador quântico c/ 4 quibits

@qml.qnode(dev)
def ttn_circuit(features,weights): #Def do circuito TTN, que recebe features e pesos ajustáveis
    """
    Implementação do circuito TTN (Tree Tensor Network).
    Parâmetros:
    - features: Vetor de entrada (4 valores normalizados entre 0 e π/2).
    - weights: Pesos ajustáveis aplicados às rotações Y.
    
    Retorna:
    - O valor esperado da medição no operador Pauli Z do último qubit.
    """
    for i in range(num_qubits):qml.RY(features[i],wires=i) #Codificação dos dados girando os qubits no eixo Y
    
    for i in range(num_qubits):qml.RY(weights[i],wires=i) #Aplicação das rotações Y ajustáveis para aprendizado
    
    qml.CNOT(wires=[0,1]) #Criação de emaranhamento entre os qubits 0 e 1 [Controle , Alvo]
    qml.CNOT(wires=[2,3]) #Emaranhamento entre os qubits 2 e 3
    qml.CNOT(wires=[1,2]) #Emaranhamento entre os qubits 1 e 2
   
    
    return qml.expval(qml.PauliZ(2)) #Medição da expectativa do operador Pauli (estou sando x, y e z de forma empirica a achar a melhor) no qubit de saída (2)
