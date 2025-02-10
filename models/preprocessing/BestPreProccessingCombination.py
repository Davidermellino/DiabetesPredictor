from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from models.preprocessing.Sampler import Sampler

from models.preprocessing.FeatureSelection import FeatureSelection

from models.preprocessing.Scaler import Scaler

from models.preprocessing.BestCombUnderOver import CombinationUnderOver

from shared.constants import PREPROCESSING_NAME

import itertools
from models.classifiers.Decision_tree_sklearn import DecisionTreeSklearn
from models.classifiers.Naive_bayes_sklearn import NaiveBayesSklearn
from models.classifiers.Artificial_neural_network_sklearn import ArtificialNeuralNetworkSklearn
from models.classifiers.Random_forest_custom import RandomForestCustom
from models.classifiers.Knn_Custom import KnnCustom

class BestPreProcComb ():
    def __init__(self, classifier, features, labels):
        self.classifier = classifier
        self.features = features
        self.labels = labels

    def preproc_best(self):
        best_combo, X_preprocessed, y_preprocessed = self.find_best()
        
        print("MIGLIORE COMBINAZIONE" , best_combo)

        return X_preprocessed, y_preprocessed
    
    def find_best(self):     
        
        preprocessing_methods = PREPROCESSING_NAME[1:-1] #['Balancing', 'Feature Selection', 'Scaling', 'Best_Under_Over', 'Best Combination'] tolgo il balancing perchè uso il best Under_over

        # crea tutte le combinazioni di 1, 2 e 3 tecniche
        all_combinations = []
        best_combination_accuracy = 0
        best_combination = None
        best_X, best_y = self.features, self.labels

        for i in range(1, len(preprocessing_methods) + 1):
            all_combinations.extend(itertools.combinations(preprocessing_methods, i))
    

        for combination in all_combinations: #per ogni combinazione
            print("sto provando ", combination)
            X_preprocessed, y_preprocessed = self.features, self.labels  #copia dei dati originali
    
            for method in combination: #per ogni tecnica nella combinazione
                X_preprocessed, y_preprocessed = self.preprocess(method, X_preprocessed, y_preprocessed)  

            
            train_x, test_x, train_y, test_y = train_test_split(X_preprocessed, y_preprocessed, random_state=0, test_size=0.3, stratify= y_preprocessed)

            accuracy = self.modelsAccuracy(train_x, train_y, test_x, test_y)
               
            if accuracy > best_combination_accuracy:
                best_combination_accuracy = accuracy
                best_combination = combination
                best_X = X_preprocessed
                best_y = y_preprocessed
                
        return best_combination, best_X, best_y

    def preprocess(self, preprocessor_choice, data_x, labels):
      
        if preprocessor_choice == "Balancing":

            under_processer = Sampler(data_x, labels)
            under_x, under_y = under_processer.randomSampler(data_x, labels)
            return under_x, under_y

           
        elif preprocessor_choice == "Best_Under_Over":
            balancer = CombinationUnderOver(data_x,labels, model = self.classifier)
           
            #utilizzo questi perchè sono per tutti i migliori
            balanced_x, balanced_y = balancer.under_50("NearMiss2",data_x, labels)
            balanced_x, balanced_y = balancer.over_50("RandomOverSampling",balanced_x, balanced_y)
            return balanced_x, balanced_y


        elif preprocessor_choice == "Feature Selection":

            feature_selector = FeatureSelection(data_x, labels)
            selected_x = feature_selector.featureSelection_Chi2(data_x, labels)
            return selected_x, labels

 
        elif preprocessor_choice == "Scaling":
            scaler = Scaler(data_x)
            scaled_x = scaler.MinMaxScale(data_x)
            return scaled_x, labels

   

    def stratified_sampling(self):

        sampler = Sampler(self.features, self.label)
        train_x, test_x, train_y, test_y = sampler.stratifiedSplit(self.features, self.label)
        return train_x, test_x, train_y, test_y
    


    def modelsAccuracy(self, train_x, train_y, test_x, test_y):
        
            if self.classifier == "DecisionTree":
                dt = DecisionTreeSklearn()
                dt.fit(train_x, train_y)
                pred_y = dt.predict(test_x)
        
            elif self.classifier == "GaussianNB":
                nb = NaiveBayesSklearn()
                nb.fit(train_x, train_y)
                pred_y = nb.predict(test_x)
    
            elif self.classifier == "ArtificialNeuralNetwork":
                aan = ArtificialNeuralNetworkSklearn()
                aan.fit(train_x, train_y)
                pred_y = aan.predict(test_x)
                
            elif self.classifier == "KNN":
                knn_custom = KnnCustom()
                knn_custom.fit(train_x, train_y)
                pred_y = knn_custom.predict(test_x)
        
            elif self.classifier == "RandomForest":
                random_forest_custom = RandomForestCustom()
                random_forest_custom.fit(train_x, train_y)
                pred_y = random_forest_custom.predict(test_x)
    
            return accuracy_score(test_y,pred_y)
