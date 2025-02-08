from shared.utils import load_dataset

from sklearn.model_selection import train_test_split

from models.preprocessing.Sampler import Sampler

from models.preprocessing.FeatureSelection import FeatureSelection

from models.preprocessing.Scaler import Scaler

from models.preprocessing.BestCombUnderOver import CombinationUnderOver

class TrainClassifier:

    def __init__(self, classifier, preprocessing=None):

        self.classifier = classifier
        self.preprocessing = preprocessing

        #Inizializzo il dataset
        self.dataset = load_dataset() #Carica il dataset       
        dim = 10000 if classifier.name == "KNN" or classifier.name == "RandomForest" else self.dataset.shape[0]
        self.train_x, self.test_x, self.train_y, self.test_y = None,None,None,None
     
        #divido labels e features
        self.features = self.dataset.iloc[:dim, 1:] #Salva una matrix con tutti gli attributi
        self.label = self.dataset.iloc[:dim, 0] #Salva un array con le Label

 #---------------SPLIT DEL DATASET-------------------

        #se vengono selezionati metodi di preprocessing
        if self.preprocessing is not None:
            self.features, self.label = self.preprocess(self.preprocessing, self.features, self.label)
            self.train_x, self.test_x, self.train_y, self.test_y = self.stratified_sampling()

 
        #NO PREPROCESSING
        elif self.preprocessing is None:
            self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(self.features, self.label, random_state=0, test_size=0.25)

       

    def train(self):
        self.classifier.fit(self.train_x, self.train_y)
        return self.classifier.predict(self.test_x)

       

    def preprocess(self, preprocessor_choice, data_x, labels):
      
        if preprocessor_choice == "Balancing":

            under_processer = Sampler(data_x, labels)
            under_x, under_y = under_processer.randomSampler(data_x, labels)
            return under_x, under_y

           
        elif preprocessor_choice == "Best_Under_Over":
            balancer = CombinationUnderOver(data_x,labels, model = self.classifier.name)
            balanced_x, balanced_y = balancer.combination_Under_Over()
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
    



