# silencia avisos do TensorFlow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings("ignore")

import pennylane as qml
import pennylane.numpy as np  # Use a versão diferenciável do numpy
from encoding_mnist import load_mnist_data, reduce_dimensions
from arch_mnist import create_simple_ttn

def train_and_evaluate():
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_mnist_data()
    x_train, x_val, x_test = reduce_dimensions(x_train, x_val, x_test, n_components=4)
    
    # Inicializa o circuito
    num_qubits = 4
    num_weights = 7
    weights = np.random.rand(num_weights)  # Agora weights é um array diferenciável
    ttn = create_simple_ttn(num_qubits)
    
    def cost(weights, features, labels):
        predictions = [ttn(f, weights) for f in features]
        return np.mean((np.array(predictions) - labels) ** 2)
    
    # Otimização com Adam e Early Stopping
    opt = qml.AdamOptimizer(stepsize=0.1)
    best_weights = None
    best_val_acc = 0
    patience = 30
    no_improvement = 0
    
    for epoch in range(100):  # Testando com 100 épocas
        batch_idx = np.random.choice(len(x_train), 20)
        x_batch, y_batch = x_train[batch_idx], y_train[batch_idx]
        
        weights, current_cost = opt.step_and_cost(
            lambda w: cost(w, x_batch, y_batch), 
            weights
        )
        
        if epoch % 10 == 0:  # Avalia validação a cada 10 épocas

            print(f"Época {epoch}: pesos = {weights}")
            
            val_predictions = [1 if ttn(x, weights) >= 0 else 0 for x in x_val]
            val_acc = np.mean(np.array(val_predictions) == y_val)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_weights = weights.copy()
                no_improvement = 0
            else:
                no_improvement += 1
            
            if no_improvement >= patience:
                print("Early stopping!")
                break
    
    # Avaliação final 
    test_predictions = [1 if ttn(x, best_weights) >= 0 else 0 for x in x_test]
    return np.mean(np.array(test_predictions) == y_test)

if __name__ == "__main__":
    np.random.seed(42)  # Sentido da Vida, do Universo e Tudo Mais
    accuracies = []
    
    for run in range(5):  # 5 inicializações aleatórias
        print(f"\nExecução {run+1}/5 ")
        acc = train_and_evaluate()
        accuracies.append(acc)
        print(f"Acurácia: {acc:.2%}")
    
    print("\n=== Resultados Finais ===")
    print(f"Média: {np.mean(accuracies):.2%}")
    print(f"Desvio padrão: {np.std(accuracies):.2%}")
