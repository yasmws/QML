import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split

def load_mnist_data():
    (x_train_val, y_train_val), (x_test, y_test) = mnist.load_data()

    #binariza rótulos (1 para >4, 0 para <=4)
    y_train_val = (y_train_val > 4).astype(float) 
    y_test = (y_test > 4).astype(float)
    
    #separa treino (55k), validação (5k) e teste (10k)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val, y_train_val, test_size=5000, random_state=42
    )
    
    #redimensiona para 784 dimensões - imagens do mnist tem 28x28
    x_train = x_train.reshape(-1, 784)
    x_val = x_val.reshape(-1, 784)
    x_test = x_test.reshape(-1, 784)
    
    #normaliza para [0, π/2]
    scaler = MinMaxScaler(feature_range=(0, np.pi/2))
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)
    
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

def reduce_dimensions(x_train, x_val, x_test, n_components=4):
    pca = PCA(n_components=n_components)
    x_train_reduced = pca.fit_transform(x_train)
    x_val_reduced = pca.transform(x_val)
    x_test_reduced = pca.transform(x_test)
    return x_train_reduced, x_val_reduced, x_test_reduced