from sklearn.metrics import euclidean_distances
import numpy as np
from imblearn.under_sampling import NearMiss
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Per il tuning
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate
from sklearn.base import BaseEstimator, ClassifierMixin

class KnnCustom(BaseEstimator, ClassifierMixin):
    #Le due superclassi sono utilizzae in fase di tuning per utilizzare la funzione cross_validate di sklearn

    def __init__(self, k = 17, weights = False):
        #usiamo di default k = 17 e weights = False perche ci fa ottenere il risultato migliore nel tuning 
        self.name = "KNN"
        self.features = [] #vettore di features che verrà caricato nel fit
        self.labels = [] #vettore di labels
        while k % 2 == 0 or k % 3 == 0: #mi assicuro che k non sia pari e non sia multiplo di 3
            k += 1
        self.k = k
        self.weights = weights
    
    def fit(self, features, labels):
        self.features=  features
        self.labels = labels
        return self
    
    def predict(self, test_x):
        # Lista per le etichette finali
        labels_pred = [self._predict_record(test_x.iloc[i,:]) for i in range(len(test_x))]
        
        return labels_pred
          
    def _predict_record(self, record): #metodo restituisce la classe predetta per il record preso in input

        record = np.expand_dims(record, axis=0) #lo devo fare solo perchè sto usando un solo record

        distances = euclidean_distances(self.features, record) #calcola le distanze fra il record e il le features   
        index_dist_sorted = np.argsort(distances,axis = 0) # ottiene gli indici ordinati in base alla distanza
    
        
        k_index = index_dist_sorted[:self.k].flatten()  # appiattisce l'array in 1D, e prende i primi k indici (in pos 0 ci sarà il più piccolo e così via)
        near_labels = self.labels.iloc[k_index].values.astype(int)  # Trova le k label più vicine, converto in int perchè arg max deve accetta solo interi
        
        if (self.weights): #se vengono utilizzati i pesi

            k_distances = distances[k_index] #vengono salvate le k distanze più piccole

            #calcolo dei pesi attraverso 1/d    
            weights = np.where(k_distances > 0, 1 / (k_distances + 1e-9), 1) # la uso per evitare divisione per 0
            weights = weights.flatten()

            weighted_counts = np.bincount(near_labels, weights=weights)  # somma pesata per ogni label

            prediction = np.argmax(weighted_counts)  # trova la label più dopo averlo pesato

        else:
            prediction = np.bincount(near_labels).argmax()
        

        return prediction
    
    def tuning_k_weights(self, X, y , weights):

        k_values = [] 
        acc_train = [] # Accuracy sul training set 
        acc_val = [] # Accuracy sul validation set
        
        for k in range(1, 50):
            if k % 2 == 0 or k % 3 == 0:
                continue
            
            knn = KnnCustom(k=k, weights=weights)
            
            # Esegue 10-fold cross-validation
            scores = cross_validate( estimator=knn, X=X, y=y, cv=5, n_jobs=-1, return_train_score=True)
            
            # Calcolo le medie degli score
            score_train = scores['train_score'].mean()
            score_val = scores['test_score'].mean()
            
            print(f"k={k}, Train: {score_train:.4f}, Val: {score_val:.4f}")
            
            # Aggiungo i valori alle liste
            k_values.append(k)
            acc_train.append(score_train)
            acc_val.append(score_val)
        

        # Trovo il miglior k basato sulla validation accuracy
        best_k_idx = np.argmax(acc_val)
        best_k = k_values[best_k_idx]
        best_val_acc = acc_val[best_k_idx]
        
        print(f"\nMiglior k: {best_k}")
        print(f"Miglior validation accuracy: {best_val_acc:.4f}")
        
        return best_k




