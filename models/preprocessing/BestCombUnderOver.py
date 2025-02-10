from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from models.preprocessing.Sampler import Sampler
from models.classifiers.Decision_tree_sklearn import DecisionTreeSklearn
from models.classifiers.Naive_bayes_sklearn import NaiveBayesSklearn
from models.classifiers.Artificial_neural_network_sklearn import ArtificialNeuralNetworkSklearn
from models.classifiers.Random_forest_custom import RandomForestCustom
from models.classifiers.Knn_Custom import KnnCustom
 
 
# Classe utilizzata per cercare la miglior combinazione tra undersampling e oversampling per il classificatore passato come parametro 
class CombinationUnderOver: 
   
    def __init__(self, data_x, labels_y, model):
        self.data_x = data_x
        self.labels_y = labels_y
        self.model = model #classificatore per il quale si deve cercare la miglior combinazione
       
           
    def combination_Under_Over(self):
        #----------------SE SI VUOLE USARE DINAMICO----------------- -> si va a calcolare ogni volta la migliore combinazione
        # #trovo i migliori metodi di under e over sampling
        # best_under , best_over = self.best_combination()
        # print(f"UNDER {best_under},OVER {best_over}")

        # #lo preprocesso con la combinazione con prestazioni migliori 
        # under_processed_x, under_processed_y = self.under_50(best_under, self.data_x, self.labels_y)
        # over_processed_x, over_processed_y = self.over_50(best_over, under_processed_x, under_processed_y)
        
        over_processed_x, over_processed_y = self.data_x, self.labels_y

        #----------------SE SI VUOLE USARE STATICO----------------- ->  usa le migliori combinazioni in base all'acuracy trovate
        if self.model == "RandomForest": # Per la random forest la miglior combinazione prevede nearMiss1 e SMOTE
                balanced_x, balanced_y = self.under_50("NearMiss",self.data_x, self.labels_y)
                balanced_x, balanced_y = self.over_50("SMOTE",balanced_x, balanced_y)
                print("The best combination is: Undersamplig NearMiss, Oversampling SMOTE")

        else: #La combinazione migliore per tutti tranne la ranodm forest prevede NearMiss2 e RandomOversampling
                balanced_x, balanced_y = self.under_50("NearMiss2",self.data_x, self.labels_y)
                balanced_x, balanced_y = self.over_50("RandomOverSampling",balanced_x, balanced_y)
                print("The best combination is: Undersamplig NearMiss2, Oversampling RandomOverSampling")
        
        return over_processed_x, over_processed_y
        
   
    def under_50(self, method, data_x, labels_y): # Preprocessa il dataset con il metodo di undersampling scelto
        
        sampler = Sampler(data_x, labels_y)
        
        if method == "RandomUnderSampling":
            processed_x, processed_y = sampler.randomSampler(data_x, labels_y)
       
        elif method == "NearMiss":
            processed_x, processed_y = sampler.nearMissSampler(data_x, labels_y)
 
        elif method == "NearMiss2":
            processed_x, processed_y = sampler.nearMissSampler2(data_x, labels_y)
 
        return processed_x, processed_y
 
    def over_50(self, method, data_x, labels_y): # Preprocessa il dataset con il metodo di oversampling scelto
        
        sampler = Sampler(data_x, labels_y)
 
        if method == "RandomOverSampling":
            processed_x, processed_y = sampler.randomOverSampler(data_x, labels_y)
       
        elif method == "SMOTE":
            processed_x, processed_y = sampler.smoteSampler(data_x, labels_y)
 
        elif method == "ADASYN":
            processed_x, processed_y  = sampler.adasynSampler(data_x, labels_y)
 
        return processed_x, processed_y
 
 
    def best_combination(self): # Cerca la migliore combinazione
      
        methods_undersampling = ["RandomUnderSampling",
                                    "NearMiss",
                                    "NearMiss2"]
        methods_oversampling = ["RandomOverSampling",
                                    "SMOTE",
                                    "ADASYN"]
   
        best_combination_accuracy  = 0
        best_combination_over = ""
        best_combination_under = ""
 

        for under in methods_undersampling: # Per ogni metodo di undersampling
            for over in methods_oversampling: # Per ogni metodo di oversampling
                under_processed_x, under_processed_y = self.under_50(under, self.data_x, self.labels_y) # Preprocesso il dataset con undersampling
                over_processed_x, over_processed_y = self.over_50(over, under_processed_x, under_processed_y) # Preprocesso il dataset con oversampling
               
                # Split del dataset
                train_x, test_x, train_y, test_y = train_test_split(over_processed_x, over_processed_y, random_state=0, test_size=0.25, stratify=over_processed_y)

                # Calcolo dell'accuratezza
                accuracy = self.compute_model_accuracy(train_x, train_y, test_x, test_y) 
               
                # Ricerca dell'accuratezza migliore
                if accuracy > best_combination_accuracy: 
                    best_combination_accuracy = accuracy
                    best_combination_over = over
                    best_combination_under = under
       

        return best_combination_under, best_combination_over # Restiusce la migliore tecnica di under e di over

    def compute_model_accuracy(self, train_x, train_y, test_x, test_y): # In base al classificatore selezionato restituisce l'accuratezza, viene usata per controllare l'accuratezza dopo il preprocessamento
        
            if self.model == "DecisionTree":#best UNDER NearMiss2, OVER RandomOverSampling
                dt = DecisionTreeSklearn()
                dt.fit(train_x, train_y)
                pred_y = dt.predict(test_x)
        
            elif self.model == "GaussianNB": #UNDER NearMiss2,OVER RandomOverSampling
                nb = NaiveBayesSklearn()
                nb.fit(train_x, train_y)
                pred_y = nb.predict(test_x)
    
            elif self.model == "ArtificialNeuralNetwork": #UNDER NearMiss2,OVER RandomOverSampling
                aan = ArtificialNeuralNetworkSklearn()
                aan.fit(train_x, train_y)
                pred_y = aan.predict(test_x)
                
            elif self.model == "KNN": #best UNDER NearMiss2, OVER RandomOverSampling
                knn_custom = KnnCustom()
                knn_custom.fit(train_x, train_y)
                pred_y = knn_custom.predict(test_x)
        
            elif self.model == "RandomForest":# best UNDER NearmMiss, OVER SMOTE 
                random_forest_custom = RandomForestCustom()
                random_forest_custom.fit(train_x, train_y)
                pred_y = random_forest_custom.predict(test_x)
    
            return accuracy_score(test_y,pred_y)
