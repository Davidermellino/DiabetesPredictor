from sklearn.metrics import euclidean_distances
import numpy as np
from imblearn.under_sampling import NearMiss
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class KnnCustom:

    def __init__(self, k = 17, weights = False):
        #usiamo di default k = 17 e weights = False perche ci fa ottenere il risultato migliore nel tuning 
        self.name = "KNN"
        self.features = [] #vettore di features che verrà caricato nel fit
        self.labels = [] #vettore di labels
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
          
    def _predict_record(self, record): #metodo restituisce la classe predetta per il record preso in input

        record = np.expand_dims(record, axis=0) #lo devo fare solo perchè sto usando un solo record

        distances = euclidean_distances(self.features, record) #calcola le distanze fra il record e il le features   
        index_dist_sorted = np.argsort(distances,axis = 0) # ottiene gli indici ordinati in base alla distanza
    
        
        k_index = index_dist_sorted[:self.k].flatten()  # appiattisce l'array in 1D, e prende i primi k indici (in pos 0 ci sarà il più piccolo e così via)
        near_labels = self.labels.iloc[k_index].values.astype(int)  # Trova le k label più vicine, converto in int perchè arg max deve accetta solo interi
        
        if (self.w): #se vengono utilizzati i pesi

            k_distances = distances[k_index] #vengono salvate le k distanze più piccole

            #calcolo dei pesi attraverso 1/d    
            weights = np.where(k_distances > 0, 1 / (k_distances + 1e-9), 1) # la uso per evitare divisione per 0
            weights = weights.flatten()

            weighted_counts = np.bincount(near_labels, weights=weights)  # somma pesata per ogni label

            prediction = np.argmax(weighted_counts)  # trova la label più dopo averlo pesato

        else:

           # unique, counts = np.unique(near_labels, return_counts=True) #conta i voti
            #index = counts.argmax() # estrae il voto massimo

            prediction = np.bincount(near_labels).argmax()
        

        return prediction
    
    def tuning_k_weights(self,k,weights,stratified = True): #impostare k e se si vogliono utilizzare i pesi (weights = True/false)
        # Applichiamo undersampling
       
        rus = NearMiss()
        X, y = rus.fit_resample(self.features, self.labels)
        print("num records", X.shape[0])

        # Train-test split
        if stratified: #SPLIT STRATIFICATO
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,stratify= y)
        else: #SPLIT NON STRATIFICATO
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Lista per memorizzare i valori di k e le rispettive accuracy
        k_values = []
        accuracies = []

        for k in range(1, 50):
            if k % 2 == 0 or k % 3 == 0:
                continue
            
            knn = KnnCustom(k=k, weights=weights)
            knn.fit(X_train, y_train)
            acc = accuracy_score(y_test, knn.predict(X_test))
            
            k_values.append(k)
            accuracies.append(acc)
            
            print("k=", k, "Accuracy:", acc)

        # Plot dell'accuracy in funzione di k
        plt.figure(figsize=(10, 5))
        plt.plot(k_values, accuracies, marker='o', linestyle='dashed', color='b')
        plt.xlabel('Valore di k')
        plt.ylabel('Accuracy')
        plt.title('Andamento dell\'Accuracy al variare di k')
        plt.grid()
        plt.show()

