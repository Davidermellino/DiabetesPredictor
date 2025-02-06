import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class DatasetAnalyzer:
    def __init__(self):
        self.dataset = self._load_dataset() #Carica il dataset
        self.features = self.dataset.iloc[:, 1:] #Salva una matrix con tutti gli attributi
        self.label = self.dataset.iloc[:, 0] #Salva un array con le Labeò
    
    def _load_dataset(self):
        df = pd.read_csv("diabetes_012_health_indicators_BRFSS2015.csv", encoding="UTF-8")
        
        return self._recast_features(df)
        
            
    def _recast_features(self, df):
        
        df.GenHlth = df.GenHlth.astype(str)
        df.Sex = df.Sex.astype(str)

        df.loc[df.GenHlth == 5, "GenHlth"] = "poor"
        df.loc[df.GenHlth == 4, "GenHlth"] = "fair"
        df.loc[df.GenHlth == 3, "GenHlth"] = "good"
        df.loc[df.GenHlth == 2, "GenHlth"] = "veryGood"
        df.loc[df.GenHlth == 1, "GenHlth"] = "excellent"

        df.loc[df.Sex == 0, "Sex"] = "F"
        df.loc[df.Sex == 1, "Sex"] = "M"


        df.Diabetes_012 = df.Diabetes_012.astype(int)
        df.HighBP = df.HighBP.astype(int)
        df.HighChol = df.HighChol.astype(int)
        df.CholCheck = df.CholCheck.astype(int)
        df.Smoker = df.Smoker.astype(int)
        df.Stroke = df.Stroke.astype(int)
        df.HeartDiseaseorAttack = df.HeartDiseaseorAttack.astype(int)
        df.PhysActivity = df.PhysActivity.astype(int)
        df.Fruits = df.Fruits.astype(int)
        df.Veggies = df.Veggies.astype(int)
        df.HvyAlcoholConsump = df.HvyAlcoholConsump.astype(int)
        df.AnyHealthcare = df.AnyHealthcare.astype(int)
        df.NoDocbcCost = df.NoDocbcCost.astype(int)
        df.DiffWalk = df.DiffWalk.astype(int)
        
        return df
    
    def _compute_shape(self):
        
        #Seleziona le righe in base alla classe
        c0 = self.label[self.label == 0]
        c1 = self.label[self.label == 1]
        c2 = self.label[self.label == 2]
    
        #Calcolo il numero di elementi per classe
        num_c0 = c0.shape[0]
        num_c1 = c1.shape[0]
        num_c2 = c2.shape[0]
        
        return num_c0, num_c1, num_c2  # È meglio usare una tupla per mantenere l'ordine
    
    def compute_statistics(self):
        
        #Inizalizzo variabili
        numc0, numc1, numc2 = self._compute_shape()
        num_records = self.dataset.shape[0]
        num_features = self.features.shape[1]
        
        return {
            "records count": num_records,
            "features count": num_features,
            "numbers of peaple without diabetes": numc0,
            "number of peaple with prediabetes": numc1,
            "numbero of peaple with diabetes": numc2
        }
    
    def compute_feature_data(self, feature):
    
        return self.dataset[feature]
        
    
    def compute_correlation_matrix(self):
       
        return self.features.corr()

                
    
    