import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def encode_data():
    # Carregar o dataset Iris
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Normalizar as features para [0, π/2]
    scaler = MinMaxScaler(feature_range=(0, np.pi / 2))
    X_scaled = scaler.fit_transform(X)

    # Criar subconjuntos binários
    def create_binary_dataset(class1, class2):
        mask = (y == class1) | (y == class2)
        X_filtered = X_scaled[mask]
        y_filtered = (y[mask] == class1).astype(int)
        return X_filtered, y_filtered

    # Dividir os dados em conjuntos de treino e teste
    def split_data(X, y):
        return train_test_split(X, y, test_size=1/3, random_state=42)

    # Criar os três conjuntos binários
    datasets = {}
    for classes in [(0, 1), (1, 2), (0, 2)]:
        X_filtered, y_filtered = create_binary_dataset(*classes)
        X_train, X_test, y_train, y_test = split_data(X_filtered, y_filtered)
        datasets[f"{classes[0]}_or_{classes[1]}"] = {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test
        }

    return datasets