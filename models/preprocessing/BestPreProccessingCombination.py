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
        #----------------SE SI VUOLE USARE DINAMICO----------------- -> si va a calcolare ogni volta la migliore combinazione

        # best_combo, X_preprocessed, y_preprocessed = self.find_best() # Trovo la miglior combinazione e la salvo, salvando anche i dati preprocessati che hanno dato migliori risulati 
        # #i dati vengono direttamente salvati per non dover nuovamente applicare le tecniche di preprocessing, avendolo già fatto durante il calcolo
        # print("MIGLIORE COMBINAZIONE" , best_combo)

        X_preprocessed, y_preprocessed = self.features, self.labels

        #----------------SE SI VUOLE USARE STATICO----------------- ->  usa le migliori combinazioni in base all'acuracy trovate
        if self.classifier == "DecisionTree": # ('Feature Selection', 'Scaling')
            combination = ('Feature Selection', 'Scaling')
        
        elif self.classifier == "GaussianNB": #('Feature Selection',)
           combination = ('Feature Selection',)
        elif self.classifier == "ArtificialNeuralNetwork": #('Scaling',)
           combination = ('Scaling',)
            
        elif self.classifier == "KNN": # ('Feature Selection', 'Best_Under_Over')
           combination = ('Feature Selection', 'Best_Under_Over')
    
        elif self.classifier == "RandomForest": #('Feature Selection', 'Scaling', 'Best_Under_Over')
            combination = ('Feature Selection', 'Scaling', 'Best_Under_Over')            

        for method in combination: #Ciclo per ogni tecnica nella combinazione
                X_preprocessed, y_preprocessed = self.preprocess(method, X_preprocessed, y_preprocessed)  
        
        print("The best combination is", combination)
        return X_preprocessed, y_preprocessed
    
    def find_best(self):     
        
        preprocessing_methods = PREPROCESSING_NAME[1:-1] #['Balancing', 'Feature Selection', 'Scaling', 'Best_Under_Over', 'Best Combination'] tolgo il balancing perchè uso il best Under_over

        all_combinations = [] # Array con tuple  
        best_combination_accuracy = 0
        best_combination = None
        best_X, best_y = self.features, self.labels

        # Crea tutte le combinazioni con 1, 2 e 3 tecniche
        for i in range(1, len(preprocessing_methods) + 1):
            all_combinations.extend(itertools.combinations(preprocessing_methods, i)) 
    

        for combination in all_combinations: # Ciclo per ogni combinazione
            
            print("sto provando ", combination)
            X_preprocessed, y_preprocessed = self.features, self.labels  # Copia dei dati originali
    
            for method in combination: #Ciclo per ogni tecnica nella combinazione
                X_preprocessed, y_preprocessed = self.preprocess(method, X_preprocessed, y_preprocessed)  

            # Split del dataset per andare a calcolare l'accuracy
            train_x, test_x, train_y, test_y = train_test_split(X_preprocessed, y_preprocessed, random_state=0, test_size=0.3, stratify= y_preprocessed)

            # Calcolo dell'accuratezza
            accuracy = self.compute_model_accuracy(train_x, train_y, test_x, test_y)

            # Ricerca dell'accuratezza migliore e viene salvata insieme alla migliore combinazione, e il dataset preprocessato
            if accuracy > best_combination_accuracy:
                best_combination_accuracy = accuracy
                best_combination = combination
                best_X = X_preprocessed
                best_y = y_preprocessed
                
        # Restituisce la migliore combinazione di tecniche di preprocessing con i rispettivi dati 
        return best_combination, best_X, best_y

    def preprocess(self, preprocessor_choice, data_x, labels): # Preprocessa il dataset con il metodo scelto
           
        if preprocessor_choice == "Best_Under_Over": # è stato messo statico, dopo aver controllato per tutti i classificatori quale sia la loro migliore combinazione di under e oversampling per diminuire il tempo computazionale e non dover ricalcolare ogni volta questa combinazione
           
            balancer = CombinationUnderOver(data_x,labels, model = self.classifier)
           
            if self.classifier == "RandomForest": # Per la random forest la miglior combinazione prevede nearMiss1 e SMOTE
                balanced_x, balanced_y = balancer.under_50("NearMiss",data_x, labels)
                balanced_x, balanced_y = balancer.over_50("SMOTE",balanced_x, balanced_y)

            else: #La combinazione migliore per tutti tranne la ranodm forest prevede NearMiss2 e RandomOversampling
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

    def compute_model_accuracy(self, train_x, train_y, test_x, test_y):# In base al classificatore selezionato restituisce l'accuratezza, viene usata per controllare l'accuratezza dopo il preprocessamento
        
            if self.classifier == "DecisionTree": # ('Feature Selection', 'Scaling')
                dt = DecisionTreeSklearn()
                dt.fit(train_x, train_y)
                pred_y = dt.predict(test_x)
        
            elif self.classifier == "GaussianNB": #('Feature Selection',)
                nb = NaiveBayesSklearn()
                nb.fit(train_x, train_y)
                pred_y = nb.predict(test_x)
    
            elif self.classifier == "ArtificialNeuralNetwork": #('Scaling',)
                aan = ArtificialNeuralNetworkSklearn()
                aan.fit(train_x, train_y)
                pred_y = aan.predict(test_x)
                
            elif self.classifier == "KNN": # ('Feature Selection', 'Best_Under_Over')
                knn_custom = KnnCustom()
                knn_custom.fit(train_x, train_y)
                pred_y = knn_custom.predict(test_x)
        
            elif self.classifier == "RandomForest": #('Feature Selection', 'Scaling', 'Best_Under_Over')
                random_forest_custom = RandomForestCustom()
                random_forest_custom.fit(train_x, train_y)
                pred_y = random_forest_custom.predict(test_x)
    
            return accuracy_score(test_y,pred_y)
