from sklearn.model_selection import train_test_split
import numpy as np

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler, InstanceHardnessThreshold, NearMiss, ClusterCentroids
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN 
from sklearn.feature_selection import SelectKBest, chi2
from combination_under_over import under_50, over_50, modelsAccuracy
from shared.utils import load_dataset
#from getDataset import getDataset
import pandas as pd   # importiamo pandas 

class PreProcessing():
    def __init__(self, train_x, test_x, train_y, test_y):
        
        
        self.train_x = train_x
        self.test_x = test_x
        self.train_y = train_y
        self.test_y = test_y
        
        self.train_x_scaled = None
        self.train_y_scaled = None
        
        #random undersampling
        self.train_x_rus = None
        self.train_y_rus = None
        
        #random oversampling
        self.train_x_ros = None
        self.train_y_ros = None
        
        #nearmiss versinoe 1
        self.train_x_nm1 = None
        self.train_y_nm1 = None
        
        #nearmiss versione 2
        self.train_x_nm2 = None
        self.train_y_nm2 = None
        
        #smote
        self.train_x_smote = None
        self.train_y_smote = None
        
        #adasyn
        self.train_x_adasyn = None
        self.train_y_adasyn  = None

    #Campionamento
    #split tramite stratificazione 
    def stratifiedSplit(self, data_x, labels_y):
        train_x, test_x, train_y, test_y = train_test_split(data_x, labels_y, random_state=0, test_size=0.3, stratify=labels_y)
        return train_x, test_x, train_y, test_y
       
    #undersampling tramite eliminazione randomica RandomUnderSampler(...).
    def randomSampler(self):
        rus = RandomUnderSampler()
        self.train_x_rus, self.train_y_rus = rus.fit_resample(self.train_x, self.train_y)
        
        return self.train_x_rus, self.train_y_rus

    #undersampling tramite distanze che preservano i più vicini ai più vicini NearMiss(version=1).
    def nearMissSampler(self):
        nm1 = NearMiss(version=1)
        self.train_x_nm1, self.train_y_nm1 = nm1.fit_resample(self.train_x, self.train_y)
        
        return self.train_x_nm1, self.train_y_nm1
    
    #undersampling tramite distanze che preservano i più vicini ai più lontani NearMiss(version=2).
    def nearMissSampler2(self):
        nm2 = NearMiss(version=2)
        self.train_x_nm2, self.train_y_nm2 = nm2.fit_resample(self.train_x, self.train_y)
        
        return self.train_x_nm2, self.train_y_nm2
    
    #random oversampling tramite RandomOverSampler(...)
    def randomOverSampler(self):
        ros = RandomOverSampler()
        self.train_x_ros, self.train_y_ros = ros.fit_resample(self.train_x, self.train_y)
        
        return self.train_x_ros, self.train_y_ros

    
    #oversampling tramite SMOTE(...)
    def smoteSampler(self):
        smote = SMOTE()
        self.train_x_smote, self.train_y_smote = smote.fit_resample(self.train_x, self.train_y)
        
        return self.train_x_smote, self.train_y_smote


    #oversampling tramite ADASYN(...) , solo sui dati di training 
    def adasynSampler(self):
        adasyn = ADASYN()
        self.train_x_adasyn, self.train_y_adasyn = adasyn.fit_resample(self.train_x, self.train_y)
        
        return self.train_x_adasyn, self.train_y_adasyn

    #feature selection tramite chi2
    def featureSelection_Chi2(self):
        reduced_x  = SelectKBest(chi2, k=10).fit_transform(self.train_x, self.train_y)
        return reduced_x

    # Standardizzazione (è una funzione di scikit learn che si basa sullo Zscore)
    def StandardScale(self):
        scaler = StandardScaler()

        # Seleziona le colonne non binarie
        non_binary_cols = ["BMI", "GenHlth", "MentHlth", "PhysHlth", "Age", "Education", "Income"]  # Sostituisci con i nomi esatti

        # Mantieni inalterati gli attributi binari
        train_x_bin = self.train_x.drop(columns=non_binary_cols)
        test_x_bin = self.test_x.drop(columns=non_binary_cols)

        # Applica StandardScaler solo agli attributi non binari
        train_x_non_bin = pd.DataFrame(scaler.fit_transform(self.train_x[non_binary_cols]), columns=non_binary_cols)
        test_x_non_bin = pd.DataFrame(scaler.transform(self.test_x[non_binary_cols]), columns=non_binary_cols)

        # Ricostruisci i DataFrame unendo le colonne binarie e quelle standardizzate
        self.train_x_scaled = pd.concat([train_x_bin.reset_index(drop=True), train_x_non_bin], axis=1)
        self.test_x_scaled = pd.concat([test_x_bin.reset_index(drop=True), test_x_non_bin], axis=1)
        
        return self.train_x_scaled, self.test_x_scaled
    
    # trasformazione MinMax
    def MinMaxScale(self):
        scaler = MinMaxScaler()
        self.train_x_scaled = pd.DataFrame(scaler.fit_transform(self.train_x), columns=self.train_x.columns)
        self.test_x_scaled = pd.DataFrame(scaler.transform(self.test_x), columns=self.test_x.columns)

        return self.train_x_scaled, self.test_x_scaled

# Funzione che combina tutte le tecniche di oversampling e undersampling e trova la combinazione migliore per il singolo modello passato come input
def combination(data_x, labels_y, input):
    tecniche_undersampling = ["RandomUnderSampling",
                                "NearMiss",
                                "NearMiss2"]
    tecniche_oversampling = ["RandomOverSampling",
                                "SMOTE",
                                "ADASYN"]
  
    results_models = []

    for under in tecniche_undersampling:
        for over in tecniche_oversampling:
            under_processed_x, under_processed_y, under_test_x, under_test_y = under_50(under, data_x, labels_y)
            over_processed_x, over_processed_y, over_test_x, over_test_y = over_50(over, under_processed_x, under_processed_y, under_test_x, under_test_y)
            
            accuracy = modelsAccuracy(input, over_processed_x, over_processed_y, over_test_x, over_test_y)
            
            if input == "DecisionalTree":
                results_models.append([input,accuracy,under,over, over_test_y.value_counts().values.tolist()])

            elif input == "NaiveBayes":
                results_models.append([input, accuracy,under,over, over_test_y.value_counts().values.tolist()])

            elif input == "ANN":
                results_models.append([input, accuracy,under,over, over_test_y.value_counts().values.tolist()])
            
            #DA RIVEDERE I CUSTOM
            elif input == "KNN_Custom":
                results_models.append([input, accuracy,under,over, over_test_y.value_counts().values.tolist()])
            
            elif input == "RandomForest_Custom":
                results_models.append([input, accuracy,under,over, over_test_y.value_counts().values.tolist()])

    return results_models