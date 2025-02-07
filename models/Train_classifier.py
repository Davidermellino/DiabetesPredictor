from shared.utils import load_dataset
from sklearn.model_selection import train_test_split


from models.classifiers.Knn_Custom import KnnCustom

class TrainClassifier:
    def __init__(self, classifier):
        self.classifier = classifier
        #Inizializzo il dataset
        self.dataset = load_dataset() #Carica il dataset        
        dim = 10000 if classifier.name == "KNN" or classifier.name == "RandomForest" else self.dataset.shape[0]
        self.features = self.dataset.iloc[:dim, 1:] #Salva una matrix con tutti gli attributi
        self.label = self.dataset.iloc[:dim, 0] #Salva un array con le Labeò
        
        #Divido il dataset in training e test
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(self.features, self.label, random_state=0, test_size=0.25)
        

    def train(self):
        self.classifier.fit(self.train_x, self.train_y)
        return self.classifier.predict(self.test_x)
        
   