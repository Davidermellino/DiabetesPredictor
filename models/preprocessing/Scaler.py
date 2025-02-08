from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd

class Scaler:
    
    def __init__(self, train_x):
        self.train_x = train_x
                
    def StandardScale(self, train_x):
        scaler = StandardScaler()

        # Seleziona le colonne non binarie
        non_binary_cols = ["BMI", "GenHlth", "MentHlth", "PhysHlth", "Age", "Education", "Income"]  # Sostituisci con i nomi esatti

        # Mantieni inalterati gli attributi binari
        train_x_bin = train_x.drop(columns=non_binary_cols)

        # Applica StandardScaler solo agli attributi non binari
        train_x_non_bin = pd.DataFrame(scaler.fit_transform(train_x[non_binary_cols]), columns=non_binary_cols)

        # Ricostruisci i DataFrame unendo le colonne binarie e quelle standardizzate
        train_x_scaled = pd.concat([train_x_bin.reset_index(drop=True), train_x_non_bin], axis=1)
        
        return train_x_scaled
        
    # def StandardScale(self, train_x):
    #     scaler = StandardScaler()

    #     # Seleziona le colonne non binarie
    #     non_binary_cols = ["BMI", "GenHlth", "MentHlth", "PhysHlth", "Age", "Education", "Income"]  # Sostituisci con i nomi esatti

    #     # Mantieni inalterati gli attributi binari
    #     train_x_bin = self.train_x.drop(columns=non_binary_cols)
    #     test_x_bin = self.test_x.drop(columns=non_binary_cols)

    #     # Applica StandardScaler solo agli attributi non binari
    #     train_x_non_bin = pd.DataFrame(scaler.fit_transform(self.train_x[non_binary_cols]), columns=non_binary_cols)
    #     test_x_non_bin = pd.DataFrame(scaler.transform(self.test_x[non_binary_cols]), columns=non_binary_cols)

    #     # Ricostruisci i DataFrame unendo le colonne binarie e quelle standardizzate
    #     self.train_x_scaled = pd.concat([train_x_bin.reset_index(drop=True), train_x_non_bin], axis=1)
    #     self.test_x_scaled = pd.concat([test_x_bin.reset_index(drop=True), test_x_non_bin], axis=1)
        
    #     return self.train_x_scaled, self.test_x_scaled
    
    
    def MinMaxScale(self, train_x):
        scaler = MinMaxScaler()
        train_x_scaled = pd.DataFrame(scaler.fit_transform(train_x), columns=train_x.columns)

        return train_x_scaled
    
    # trasformazione MinMax
    # def MinMaxScale(self):
    #     scaler = MinMaxScaler()
    #     self.train_x_scaled = pd.DataFrame(scaler.fit_transform(self.train_x), columns=self.train_x.columns)
    #     self.test_x_scaled = pd.DataFrame(scaler.transform(self.test_x), columns=self.test_x.columns)

    #     return self.train_x_scaled, self.test_x_scaled