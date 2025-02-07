from sklearn.metrics import euclidean_distances
import numpy as np
class KnnCustom:

    def __init__(self, k, weights = False):
        
        self.name = "KNN"
        self.features = []
        self.labels = []
        while k % 2 == 0 or k % 3 == 0: #mi assicuro che k non sia pari e non sia multiplo di 3
            k += 1
        self.k = k
        self.w = weights
    
    def fit(self, features, labels):
        self.features=  features
        self.labels = labels
    
    
    def predict(self, test_x):
        # Lista per le etichette finali
        labels_pred = [self._predict_record(test_x.iloc[i,:]) for i in range(len(test_x))]
        
        return labels_pred
          
    def _predict_record(self, record):

        record = np.expand_dims(record, axis=0) #lo devo fare solo perchè sto usando un solo record

        distances = euclidean_distances(self.features, record) #calcola le distanze fra il record e il le features   
        index_dist_sorted = np.argsort(distances,axis = 0) # ottiene gli indici ordinati in base alla distanza
    
        
        k_index = index_dist_sorted[:self.k].flatten()  # appiattisce l'array in 1D, e prende i primi k indici (in pos 0 ci sarà il più piccolo e così via)
        near_labels = self.labels.iloc[k_index].values.astype(int)  # Trova le k label più vicine, converto in int perchè arg max deve accetta solo interi
        
        if (self.w):
            k_distances = distances[k_index]

              # la uso per evitare divisione per 0
            weights = np.where(k_distances > 0, 1 / k_distances, 1)
            weights = weights.flatten()

            weighted_counts = np.bincount(near_labels, weights=weights)  # somma pesata per ogni label

            prediction = np.argmax(weighted_counts)  # trova la label più dopo averlo pesato

        else:
            prediction = np.bincount(near_labels).argmax()
        

        return prediction