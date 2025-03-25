import numpy as np  #Biblioteca para operações matemáticas e arrays
from sklearn.datasets import load_iris  #Função para carregar o dataset Iris
from sklearn.model_selection import train_test_split  #Função para dividir dados em treino/teste
from sklearn.preprocessing import MinMaxScaler  #Função para normalizar dados
from pennylane.optimize import AdamOptimizer  #Otimizador Adam para ajustar os pesos
from arch_iris import ttn_circuit  #Função que define o circuito TTN (Tree Tensor Network)
import pennylane as qml  #Biblioteca para computação quântica

np.random.seed(42)  #Semente para o NumPy
qml.numpy.random.seed(42)  #Semente para o PennyLane

iris = load_iris()  #Carrega o dataset Iris
X = iris.data  #Atributos das flores
y = iris.target  #Rótulos das flores

scaler = MinMaxScaler(feature_range=(0, np.pi / 2))  #Normaliza os dados para [0, π/2]
X_scaled = scaler.fit_transform(X)  #Aplica a normalização

def encode_data():  #Função para preparar os dados para classificação binária
    def create_binary_dataset(class1, class2):  #Cria um subconjunto com classes 1 e 2
        mask = (y == class1) | (y == class2)  #Filtra as classes
        X_filtered = X_scaled[mask]  #Seleciona as features
        y_filtered = (y[mask] == class1).astype(int)  #Converte os rótulos para 0 ou 1
        return X_filtered, y_filtered

    def split_data(X, y):  #Divide dados em treino e teste
        return train_test_split(X, y, test_size=1/3, random_state=42)  #Teste: 1/3 dos dados

    X_filtered, y_filtered = create_binary_dataset(1, 2)  #Filtra apenas as classes 1 e 2
    X_train, X_test, y_train, y_test = split_data(X_filtered, y_filtered)  #Divide treino/teste
    return {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}  #Retorna os dados

def cost(weights, features, labels):  #Função de custo que calcula erro
    predictions = [ttn_circuit(f, weights) for f in features]  #Obtém previsões do circuito TTN
    predictions = qml.math.stack(predictions)  #Empilha as previsões
    labels = qml.math.convert_like(labels, predictions)  #Converte rótulos para o tipo das previsões
    return qml.math.mean((predictions - labels) ** 2)  #Erro quadrático médio

def train_model(dataset_name, X_train, X_test, y_train, y_test):  #Função de treino
    num_weights = 7  #Número de parâmetros ajustáveis
    weights = qml.numpy.random.uniform(0, 1, num_weights, requires_grad=True)  #Inicializa pesos aleatórios

    opt = AdamOptimizer(stepsize=0.1)  #Cria otimizador Adam com taxa de aprendizado 0.1

    epochs = 50  #Número de épocas de treino
    batch_size = 10  #Tamanho do lote para processamento dos dados

    for epoch in range(epochs):  #Loop de treinamento
        for i in range(0, len(X_train), batch_size):  #Processa os dados em lotes
            batch_features = X_train[i:i + batch_size]  #Seleciona lote de features
            batch_labels = y_train[i:i + batch_size]  #Seleciona rótulos correspondentes
            weights = opt.step(lambda w: cost(w, batch_features, batch_labels), weights)  #Atualiza pesos

        test_cost = cost(weights, X_test, y_test)  #Avalia o erro no conjunto de teste
        print(f"Parâmetro {dataset_name}, Época {epoch + 1}/{epochs}, Erro: {test_cost:.4f}")  #Exibe erro

    predictions = [ttn_circuit(f, weights) for f in X_test]  #Faz previsões no conjunto de teste
    accuracy = np.mean(np.round(predictions) == y_test)  #Calcula a acurácia (percentual de acertos)
    print(f"Acurácia Final para o parâmetro {dataset_name}: {accuracy * 100:.2f}%")  #Exibe acurácia
    return accuracy  #Retorna a acurácia final

if __name__ == "__main__":  #Bloco principal de execução
    data = encode_data()  #Codifica os dados para a tarefa binária

    print("Training on 1_or_2...")  #Inicia o treinamento
    acc = train_model(  #Treina o modelo e avalia a precisão
        "1_or_2",
        data["X_train"],
        data["X_test"],
        data["y_train"],
        data["y_test"]
    )
    print("Acurácia final:", acc)  #Exibe a precisão final do modelo
