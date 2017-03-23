import unittest

from sklearn.datasets import fetch_20newsgroups

from .predictor import TopicPredictor


class TestTopicPredictor(unittest.TestCase):

    def setUp(self):
        self.n_samples = 2000
        self.model = TopicPredictor()

    def test_fit_transform(self):
        dataset = fetch_20newsgroups(shuffle=True, random_state=1,
                                     remove=('headers', 'footers', 'quotes'))
        X = dataset.data[:self.n_samples]
        topics = self.model.fit_predict(X)

        print(topics)

        self.assertNotEqual(len(topics), 0)
