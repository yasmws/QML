#silencia avisos do TensorFlow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pennylane as qml
from encoding_mnist_mera import load_mnist_data, reduce_dimensions
from arch_mnist import create_simple_ttn, create_simple_mera
import pandas as pd
import matplotlib.pyplot as plt

def train_and_evaluate(task):
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_mnist_data(task)
    x_train, x_val, x_test = reduce_dimensions(x_train, x_val, x_test, n_components=4)
    
    #Inicializa o circuito
    num_qubits = 4
    #Para MERA, o número de blocos (num_weights) deve ser 5.
    num_weights = 5  
    #Cada bloco recebe 2 parâmetros, logo, os pesos terão shape (5,2)
    weights = np.random.rand(num_weights, 2)
    
    #Use a arquitetura MERA
    mera = create_simple_mera(num_qubits, num_weights)
    
    def cost(weights, features, labels):
        predictions = [mera(f, weights) for f in features]
        return np.mean((predictions - labels) ** 2)
    
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
            val_predictions = [1 if mera(x, weights) >= 0 else 0 for x in x_val]
            val_acc = np.mean(val_predictions == y_val) 
            
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
    test_predictions = [1 if mera(x, best_weights) >= 0 else 0 for x in x_test]
    return np.mean(test_predictions == y_test)

if __name__ == "__main__":
    np.random.seed(42)  # Sentido da Vida, do Universo e Tudo Mais
    accuracies = []
    tasks = ['gt4', 'even', '0or1', '2or7']
    means = []
    stds = []
    for task in tasks:
        print(f"\n=== Treinamento e Avaliação para a tarefa: {task} ===")
    
        for run in range(5):  # 5 inicializações aleatórias
            print(f"\nExecução {run+1}/5 ")
            acc = train_and_evaluate(task)
            accuracies.append(acc)
            print(f"Acurácia: {acc:.2%}")
            
        print(f"\n=== Resultados para a tarefa: {task} ===")
        print(f"Média: {np.mean(accuracies):.2%}")
        print(f"Desvio padrão: {np.std(accuracies):.2%}")
        
        means.append(np.mean(accuracies))
        stds.append(np.std(accuracies))
        accuracies = []
        
    # Criar DataFrame com os resultados
    results_df = pd.DataFrame({
        "Tarefa": tasks,
        "Média": means,
        "Desvio Padrão": stds
    })

    # Criar tabela como imagem
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=results_df.values, colLabels=results_df.columns, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(results_df.columns))))

    # Salvar a tabela como imagem
    plt.savefig("resultados_tabela.png", bbox_inches='tight')
    plt.show()