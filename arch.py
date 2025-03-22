import pennylane as qml
import numpy as np
from encoding import embedding_layer, dev, X_scaled 
from unitary_parameterization import simple_unitary_block, general_unitary_block, ancilla_unitary_block

@qml.qnode(dev)
def ttn_classifier(features, params, unitary_type="simple", ancilla_wire=None):
    """
    unitary_type: define o tipo de bloco unitário a ser utilizado ("simple", "general" ou "ancilla")
    ancilla_wire: necessário se utilizar a parametrização com ancilla
    """
    # Etapa de embedding: codifica os dados no estado quântico
    embedding_layer(features)
    
    # Primeira camada: aplica blocos unitários nos pares de qubits [0,1] e [2,3]
    if unitary_type == "simple":
        simple_unitary_block(params[0], wires=[0, 1])
        simple_unitary_block(params[1], wires=[2, 3])
    elif unitary_type == "general":
        general_unitary_block(params[0], wires=[0, 1])
        general_unitary_block(params[1], wires=[2, 3])
    elif unitary_type == "ancilla":
        if ancilla_wire is None:
            raise ValueError("É necessário especificar ancilla_wire para a parametrização com ancilla")
        ancilla_unitary_block(params[0], wires=[0, 1], ancilla_wire=ancilla_wire)
        ancilla_unitary_block(params[1], wires=[2, 3], ancilla_wire=ancilla_wire)
    else:
        raise ValueError("Tipo de unidade desconhecido")
    
    # Segunda camada: combina os resultados, por exemplo, aplicando um bloco em qubits 1 e 3
    if unitary_type == "simple":
        simple_unitary_block(params[2], wires=[1, 3])
    elif unitary_type == "general":
        general_unitary_block(params[2], wires=[1, 3])
    elif unitary_type == "ancilla":
        ancilla_unitary_block(params[2], wires=[1, 3], ancilla_wire=ancilla_wire)
    
    # Medida final: retorna o valor de expectativa do operador PauliZ no qubit 3
    return qml.expval(qml.PauliZ(3))


# Exemplos de Uso:

# Para as parametrizações "simple" e "general" são 2 parâmetros por bloco
params_simple = [np.array([0.1, 0.2]), np.array([0.3, 0.4]), np.array([0.5, 0.6])]
# params_general = [np.array([0.1]*15), np.array([0.2]*15), np.array([0.3]*15)]
# # Para a parametrização com ancilla, o vetor de parâmetros pode ter tamanho diferente
# params_ancilla = [np.array([0.1]*20), np.array([0.2]*20), np.array([0.3]*20)]

sample_features = X_scaled[0]

# Executa o classificador utilizando a parametrização simples
output_simple = ttn_classifier(sample_features, params_simple, unitary_type="simple")
print("Saída (Simple):", output_simple)

# # Executa o classificador utilizando a parametrização geral
# output_general = ttn_classifier(sample_features, params_general, unitary_type="general")
# print("Saída (General):", output_general)

# # Se utilizar a parametrização com ancilla, o dispositivo tem que ter um fio extra.
# # Neste exemplo, o qubit ancilla é o fio 4.
# output_ancilla = ttn_classifier(sample_features, params_ancilla, unitary_type="ancilla", ancilla_wire=4)
# print("Saída (Ancilla):", output_ancilla)
