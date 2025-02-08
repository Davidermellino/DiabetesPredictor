from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from models.preprocessing.Sampler import Sampler
from models.classifiers.Decision_tree_sklearn import DecisionTreeSklearn
from models.classifiers.Naive_bayes_sklearn import NaiveBayesSklearn
from models.classifiers.Artificial_neural_network_sklearn import ArtificialNeuralNetworkSklearn
from models.classifiers.Random_forest_custom import RandomForestCustom
from models.classifiers.Knn_Custom import KnnCustom
 
 

class CombinationUnderOver:
   
    def __init__(self, data_x, labels_y, model):
        self.data_x = data_x
        self.labels_y = labels_y
        self.model = model
       
           
    def combination_Under_Over(self):
        #trovo i migliori metodi di under e over sampling
        best_under , best_over = self.best_combination()
        print(f"UNDER {best_under},OVER {best_over}")

        under_processed_x, under_processed_y = self.under_50(best_under, self.data_x, self.labels_y)
        over_processed_x, over_processed_y = self.over_50(best_over, under_processed_x, under_processed_y)
               
        return over_processed_x, over_processed_y
        
   
    def under_50(self, method, data_x, labels_y):
        
        sampler = Sampler(data_x, labels_y)
        
        if method == "RandomUnderSampling":
            processed_x, processed_y = sampler.randomSampler(data_x, labels_y)
       
        elif method == "NearMiss":
            processed_x, processed_y = sampler.nearMissSampler(data_x, labels_y)
 
        elif method == "NearMiss2":
            processed_x, processed_y = sampler.nearMissSampler2(data_x, labels_y)
 
        return processed_x, processed_y
 
    def over_50(self, method, data_x, labels_y):
        
        sampler = Sampler(data_x, labels_y)
 
        if method == "RandomOverSampling":
            processed_x, processed_y = sampler.randomOverSampler(data_x, labels_y)
       
        elif method == "SMOTE":
            processed_x, processed_y = sampler.smoteSampler(data_x, labels_y)
 
        elif method == "ADASYN":
            processed_x, processed_y  = sampler.adasynSampler(data_x, labels_y)
 
        return processed_x, processed_y
 
 


    def best_combination(self):
        tecniche_undersampling = ["RandomUnderSampling",
                                    "NearMiss",
                                    "NearMiss2"]
        tecniche_oversampling = ["RandomOverSampling",
                                    "SMOTE",
                                    "ADASYN"]
   
        best_combination_accuracy  = 0
        best_combination_over = ""
        best_combination_under = ""
 
        for under in tecniche_undersampling:
            for over in tecniche_oversampling:
                under_processed_x, under_processed_y = self.under_50(under, self.data_x, self.labels_y)
                over_processed_x, over_processed_y = self.over_50(over, under_processed_x, under_processed_y)
               
                train_x, test_x, train_y, test_y = train_test_split(over_processed_x, over_processed_y, random_state=0, test_size=0.25, stratify=over_processed_y)
 
                accuracy = self.modelsAccuracy(train_x, train_y, test_x, test_y)
               
                if accuracy > best_combination_accuracy:
                    best_combination_accuracy = accuracy
                    best_combination_over = over
                    best_combination_under = under
       

        return best_combination_under, best_combination_over 
        




    def modelsAccuracy(self, train_x, train_y, test_x, test_y):
        
            if self.model == "DecisionTree":
                dt = DecisionTreeSklearn()
                dt.fit(train_x, train_y)
                pred_y = dt.predict(test_x)
        
            elif self.model == "GaussianNB":
                nb = NaiveBayesSklearn()
                nb.fit(train_x, train_y)
                pred_y = nb.predict(test_x)
    
            elif self.model == "ArtificialNeuralNetwork":
                aan = ArtificialNeuralNetworkSklearn()
                aan.fit(train_x, train_y)
                pred_y = aan.predict(test_x)
                
            #DA RIVEDERE I CUSTOM
            elif self.model == "KNN":
                knn_custom = KnnCustom()
                knn_custom.fit(train_x, train_y)
                pred_y = knn_custom.predict(test_x)
        
            elif self.model == "RandomForest":
                random_forest_custom = RandomForestCustom()
                random_forest_custom.fit(train_x, train_y)
                pred_y = random_forest_custom.predict(test_x)
    
            return accuracy_score(test_y,pred_y)
