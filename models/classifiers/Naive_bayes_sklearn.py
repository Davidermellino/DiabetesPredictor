
from sklearn.naive_bayes import GaussianNB

class NaiveBayesSklearn:
    def __init__(self):
        self.name = "GaussianNB"
        self.model = GaussianNB()

    def fit(self, features, labels):
        self.model.fit(features, labels)
        return self

    def predict(self, test_x):
        return self.model.predict(test_x)