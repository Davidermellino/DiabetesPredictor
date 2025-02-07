from sklearn.metrics.pairwise import euclidean_distances
import numpy as np

class KnnCustom:

    def __init__(self):
        self.features = []
        self.labels = []
        self.k = 5
    
    def fit(self, features, labels):
        self.features=  features
        self.labels = labels
    
    def predict(self, records):
       

        self.k = 4  # Numero di vicini da considerare

        # Calcola la distanza Euclidea tra X_subset (train) e X_test
        distances = euclidean_distances(self.features, records)  

        # Ottiene gli indici ordinati in base alla distanza
        index_dist_sorted = np.argsort(distances, axis=0)  

        # Prende i primi k indici più vicini per ogni colonna (record test)
        k_index = index_dist_sorted[:self.k, :]  

        # Lista per le etichette finali
        labels = []

        for column in range(len(records)): 
            near_labels = self.labels.iloc[k_index[:, column]].values  # Trova le k label più vicine
            near_labels = near_labels.astype(int)  # Converte in interi
            #trasforma in nparray 1D
            near_labels = near_labels.flatten()
            most_frequent = np.bincount(near_labels).argmax()  # Trova la label più frequente
            #    most_frequent = Counter(near_labels).most_common(1)[0][0]  # Trova la label più frequente
            labels.append(most_frequent)  # Aggiunge la label più frequente alla lista finale
    
        return labels
    

