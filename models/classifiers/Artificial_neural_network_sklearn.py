from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

class ArtificialNeuralNetworkSklearn:
    def __init__(
        self,
        hidden_layer_sizes=(32),
        max_iter=100,
        alpha=1e-3,
        solver="sgd",
        learning_rate_init=0.2,
        early_stopping=True,
        random_state=0
    ):
        
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
    # def testHiddenLayers(self, features, labels):
    #     best_accuracy = 0
    #     best_layers = None
    #     results = {}
    #     for num_layers in range(5, 51, 5):
    #         hidden_layers = (num_layers,)  # la tupla per il numero di livelli
    #         print(f"Training with {num_layers} hidden layers...")
    #         pred_y = ann(train_x, test_x, train_y, hidden_layers)
    #         accuracy = accuracy_score(test_y, pred_y)
    #         results[num_layers] = accuracy
    #         print(f"Accuracy for {num_layers} layers: {accuracy:.4f}")

    #         # Se è la miglior accuratezza finora, aggiorna
    #         if accuracy > best_accuracy or (accuracy == best_accuracy and (best_layers is None or num_layers < best_layers)):
    #             best_accuracy = accuracy
    #             best_layers = num_layers

    #     # Stampa il miglior risultato
    #     print(f"\nMiglior numero di livelli: {best_layers}")
    #     print(f"Accuratezza corrispondente: {best_accuracy:.4f}")