from shared.utils import load_dataset
from sklearn.model_selection import train_test_split


from models.classifiers.Knn_Custom import KnnCustom

class KnnTrain:
    def __init__(self):
        self.knnClassifier = KnnCustom()
        
        #Inizializzo il dataset
        self.dataset = load_dataset() #Carica il dataset
        
        self.features = self.dataset.iloc[:10000, 1:] #Salva una matrix con tutti gli attributi
        self.label = self.dataset.iloc[:10000, 0] #Salva un array con le Labeò
        
        #Divido il dataset in training e test
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(self.features, self.label, random_state=0, test_size=0.25)
        

    def train_knn(self):
        self.knnClassifier.fit(self.train_x, self.train_y)
        return self.knnClassifier.predict(self.test_x)
        
   