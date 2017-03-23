from typing import List
import pickle
import logging

from models.review import Review
# from .nltk_preprocessor import NLTKPreprocessor

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
# from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsOneClassifier

LOG_FILENAME = '../logs/star-predictor.log'
logging.basicConfig(filename=LOG_FILENAME, level=logging.DEBUG,)


class StarPredictor(object):

    def __init__(self):
        self._model = object()
        self._pipe = Pipeline([
            # ('preprocessor', NLTKPreprocessor()),
            ('vect', CountVectorizer(ngram_range=(1, 2))),
            ('tfidf', TfidfTransformer()),
            # ('clf', MultinomialNB())
            ('clf', OneVsOneClassifier(SGDClassifier(n_jobs=-1), n_jobs=-1))
        ])

    def fit(self, reviews: List[Review]):
        '''
        Fits model
        '''
        df_reviews = Review.to_pandas(reviews)
        self._model = self._pipe.fit(
            df_reviews['text'].values, df_reviews['class'].values)

    def fit_accuracy_score(self, reviews: List[Review]):
        '''
        Fits model and logs performance of model
        review: List[Review]
        '''
        df_reviews = Review.to_pandas(reviews)

        X_train, X_test, y_train, y_test = train_test_split(
            df_reviews['text'].values, df_reviews['class'].values)
        self._model = self._pipe.fit(X_train, y_train)

        test_accuracy = self._model.score(X_train, y_train)
        train_accuracy = self._model.score(X_test, y_test)
        clf_report = classification_report(y_test, self._model.predict(X_test))
        self._log_model(test_accuracy, train_accuracy, clf_report)

    def _log_model(self, test_accuracy, train_accuracy, clf_report):
        pipe_str = ["{0}: {1}\n".format(step_name, step)
                    for step_name, step in self._pipe.steps]
        logging.info('pipe= \n' + ''.join(pipe_str))
        logging.info('test accuracy= ' + str(test_accuracy))
        logging.info('train accuracy= ' + str(train_accuracy))
        logging.info('classification report= \n' + clf_report)

    def predict(self, review: Review):
        '''
        Predicts star for given review and return star
        review: Review object
        '''
        return self._model.predict([review.text])

    def save(self):
        pickle.dump(self._model, open('../models/star-predictor.p', 'wb'))

    def load(self):
        self._model = pickle.load(open('../models/star-predictor.p', 'rb'))
