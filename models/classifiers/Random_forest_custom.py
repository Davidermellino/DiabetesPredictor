from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

class RandomForestCustom:
    def __init__(self, n_estimators=10, random_state=42):
        
        self.name = "RandomForest"
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.forest = []
        
    def fit(self, train_x, train_y):

        for _ in range(10):
        
            dtree = DecisionTreeClassifier(random_state=42)
            
            
            #-----PRENDO COLONNE
            columns = train_x.columns.to_numpy()
            num_records = train_x.shape[0]
        
            #---PRENDO I CAMPIONI----#
            feature_sample = np.random.choice(columns, 6, replace=False)
            
            
            #----Salvo gli indici dei record selezionati dalla funzione sample-----#
            sample_indices = train_x.sample(n=num_records, replace=False).index

            # Seleziono le righe corrispondenti per features e labels
            X = train_x.loc[sample_indices]
            y = train_y.loc[sample_indices]            
            
            X = X.loc[:, feature_sample] # prendo solo le colonne selezionate
            tree_train_x, _, tree_train_y, _ = train_test_split(X, y, random_state=0, test_size=0.25)

            #-----PREDICO-----#
            tree = dtree.fit(tree_train_x, tree_train_y)
                    
            self.forest.append((tree, feature_sample))

    
    def predict(self, test_x):
        pred_y = []
        
        for index, record in test_x.iterrows():
            votes = []
            for tree, features in self.forest:
                # Converti la riga in DataFrame mantenendo i feature names
                record_subset = pd.DataFrame([record[features]], columns=features) # <--- Modifica chiave
                votes.append(tree.predict(record_subset)[0])
            
            majority_vote = np.bincount(votes).argmax()
            pred_y.append(majority_vote)
            
        return pred_y



