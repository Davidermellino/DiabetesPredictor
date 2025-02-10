from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_validate
from sklearn.neural_network import MLPClassifier
import numpy as np
import matplotlib.pyplot as plt

class ArtificialNeuralNetworkSklearn:
    def __init__(self, hidden_layer_sizes=(32,32),max_iter=100,alpha=1e-3,solver="sgd", learning_rate_init=0.2, early_stopping=True, random_state=0):
        
        self.name = "ArtificialNeuralNetwork"
        
        self.model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=max_iter,
            alpha=alpha,
            solver=solver,
            verbose=False,
            random_state=random_state,
            learning_rate_init=learning_rate_init,
            early_stopping=early_stopping
        )
        
        

    def fit(self, features, labels):
        tr_x = features.values
        tr_y = labels.values
        self.model.fit(tr_x, tr_y)
        return self

    def predict(self, test_x):
        
        ts_x = test_x.values
        
        return self.model.predict(ts_x)
    
    # Ciclo for per provare diversi numeri di hidden layers
    def tuning_hidden_layers(self, features, labels, neurons_per_layer=32):
        best_accuracy = 0
        best_layers = None
        results = {}
        num_layers_values = []
        acc_train = []
        acc_val = []
        
        # Test da 1 a 5 layer nascosti
        for num_layers in range(1, 6):
            # Creiamo una tupla con lo stesso numero di neuroni per ogni layer
            hidden_layers = tuple([neurons_per_layer] * num_layers)  
            print(f"Training with {num_layers} hidden layers: {hidden_layers}...")
            
            # Creiamo il modello
            mlp = MLPClassifier(
                hidden_layer_sizes=hidden_layers, 
                max_iter=500, 
                random_state=42
            )
            
            # Eseguiamo 5-fold cross-validation
            scores = cross_validate(
                mlp, 
                features, 
                labels, 
                cv=5, 
                return_train_score=True, 
                n_jobs=-1
            )
            
            # Calcoliamo la media degli score
            score_train = np.mean(scores['train_score'])
            score_val = np.mean(scores['test_score'])
            
            # Stampiamo i risultati
            print(f"Number of hidden layers: {num_layers}")
            print(f"Architecture: {hidden_layers}")
            print(f"Train accuracy: {score_train:.4f}")
            print(f"Validation accuracy: {score_val:.4f}\n")
            
            # Aggiungiamo i risultati alle liste
            num_layers_values.append(num_layers)
            acc_train.append(score_train)
            acc_val.append(score_val)
            results[num_layers] = score_val
            
            # Se è la miglior accuratezza finora, aggiorniamo il best
            if score_val > best_accuracy:
                best_accuracy = score_val
                best_layers = num_layers
        
        print(f"\nMiglior numero di hidden layers: {best_layers}")
        print(f"Miglior validation accuracy: {best_accuracy:.4f}")
        
        return best_layers 