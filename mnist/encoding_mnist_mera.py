import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split

def load_mnist_data(task):
    (x_train_val, y_train_val), (x_test, y_test) = mnist.load_data()
    
    if task == 'gt4':
        #binariza rótulos (1 para >4, 0 para <=4)
        y_train_val = (y_train_val > 4).astype(float) 
        y_test = (y_test > 4).astype(float)
    elif task == 'even':
        #binariza rótulos (1 para par, 0 para ímpar)
        y_train_val = (y_train_val % 2 == 0).astype(float) 
        y_test = (y_test % 2 == 0).astype(float)
    elif task == '0or1':
        # Seleciona apenas os dígitos 0 e 1
        train_filter = (y_train_val == 0) | (y_train_val == 1)
        test_filter = (y_test == 0) | (y_test == 1)
        x_train_val, y_train_val = x_train_val[train_filter], y_train_val[train_filter]
        x_test, y_test = x_test[test_filter], y_test[test_filter]
        
        # Binariza os rótulos: 1 para dígito 1, 0 para dígito 0.
        y_train_val = (y_train_val == 1).astype(float)
        y_test = (y_test == 1).astype(float)
        
    elif task == '2or7':
        # Seleciona apenas os dígitos 2 e 7
        train_filter = (y_train_val == 2) | (y_train_val == 7)
        test_filter = (y_test == 2) | (y_test == 7)
        x_train_val, y_train_val = x_train_val[train_filter], y_train_val[train_filter]
        x_test, y_test = x_test[test_filter], y_test[test_filter]
        
        # Binariza os rótulos: 1 para dígito 2, 0 para dígito 7.
        y_train_val = (y_train_val == 2).astype(float)
        y_test = (y_test == 2).astype(float)
    else:
        raise ValueError("Task not recognized. Use 'gt4', 'even', '0or1', or '2or7'.")
    
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