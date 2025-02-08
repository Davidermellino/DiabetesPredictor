from sklearn.feature_selection import SelectKBest, chi2
import pandas as pd

class FeatureSelection:
    
    def __init__(self, train_x, train_y):
        
        self.train_x = train_x
        self.train_y = train_y
        
        
    def featureSelection_Chi2(self, train_x, train_y):
    # Creiamo il selettore
        selector = SelectKBest(chi2, k=10)
        
        # Facciamo il fit e transform
        reduced_x = selector.fit_transform(train_x, train_y)
        
        # Otteniamo le feature selezionate
        selected_features = train_x.columns[selector.get_support()].tolist()
        
        # Convertiamo reduced_x in DataFrame mantenendo i nomi delle colonne selezionate
        reduced_x_df = pd.DataFrame(reduced_x, columns=selected_features, index=train_x.index)
        
        return reduced_x_df