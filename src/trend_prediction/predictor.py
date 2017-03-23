import pickle

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.metrics import r2_score

__all__ = ['TrendPredictor']


class NeoralTrendPredictor(object):

    def __init__(self):
        self.model = Sequential()
        self.model.add(LSTM(4, input_dim=6))
        self.model.add(Dense(1))
        self.model.compile(loss='mean_squared_error', optimizer='adam')

    def fit(self, X, y):
        self.model.fit(X, y, nb_epoch=100, batch_size=1, verbose=2)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def mean_squared_error(self, X, y):
        mean_squared_error(y, self.predict(X))


class TrendPredictor(object):

    def __init__(self):
        self.model = SVR(cache_size=5000)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def mean_squared_error(self, X, y):
        return mean_squared_error(y, self.predict(X))

    def r2_score(self, X, y):
        return r2_score(y, self.predict(X))

    def __str__(self):
        return str(self.model)

    def save(self):
        pickle.dump(self.model, open('../models/trend-predictor.p', 'wb'))

    def load(self):
        self.model = pickle.load(open('../models/trend-predictor.p', 'rb'))
