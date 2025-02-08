import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from shared.utils import load_dataset, recast_features
from models.preprocessing.Sampler import Sampler
from models.preprocessing.Scaler import Scaler
from models.preprocessing.FeatureSelection import FeatureSelection

class DatasetAnalyzer:
    def __init__(self):
        self.dataset = load_dataset() #Carica il dataset
        self.features = self.dataset.iloc[:, 1:] #Salva una matrix con tutti gli attributi
        self.label = self.dataset.iloc[:, 0] #Salva un array con le Labeò
        
        sampler = Sampler(self.features, self.label) #Applica il Random Undersampling
        self.features, self.label = sampler.randomSampler(self.features, self.label)
        
        self.dataset = recast_features(pd.concat([self.label, self.features], axis=1))

        # scaler = Scaler(self.features)
        # self.features = scaler.MinMaxScale(self.features)
        
        # feature_selection = FeatureSelection(self.features, self.label)
        # self.features = feature_selection.featureSelection_Chi2(self.features, self.label)
    
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
        num_records = self.features.shape[0]
        num_features = self.features.shape[1]
        
        return {
            "records count": num_records,
            "features count": num_features,
            "numbers of peaple without diabetes": numc0,
            "number of peaple with prediabetes": numc1,
            "numbero of peaple with diabetes": numc2
        }
    
    def _compute_missing_values_feature(self, feature):
        
        return self.dataset[feature].isna().sum()
    
    def compute_feature_statistics(self, feature):
        dic = {}
        
        for key, element in self.dataset[feature].describe().items():
            dic[key] = element

        dic["missing values"] = self._compute_missing_values_feature(feature)

        return dic

    def compute_missing_values_dataset(self):
       
        missing_values = self.dataset.isna().sum().sum()
        return missing_values
    
    
    def compute_feature_data(self, feature):
    
        return self.dataset[feature]
        
    
    def compute_correlation_matrix(self):
       
        #TRASFOMO I DATI IN NUMERI
        le_sex = LabelEncoder()
        le_genhlth = LabelEncoder()

        dataset_copy = self.dataset.copy()
        
        dataset_copy["Sex"] = le_sex.fit_transform(dataset_copy["Sex"])  # 'F' -> 0, 'M' -> 1
        dataset_copy["GenHlth"] = le_genhlth.fit_transform(dataset_copy["GenHlth"])  # Ordina da 'excellent' a 'poor'

        return dataset_copy.iloc[:, 1:].corr()

                
    
    