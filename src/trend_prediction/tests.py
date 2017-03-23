import unittest

from .predictor import TrendPredictor
from models import Business
from services import DataReader


class TestTrendPrediction(unittest.TestCase):

    def setUp(self):
        self.predictor = TrendPredictor()

        self.X_train, self.X_test, self.y_train, self.y_test = \
            Business.train_test_set(DataReader().sample_businesses())

        self.predictor.fit(self.X_train, self.y_train)

    def test_accurcy(self):
        print(self.predictor.mean_squared_error(self.X_test, self.y_test))
        print(self.predictor.r2_score(self.X_test, self.y_test))
