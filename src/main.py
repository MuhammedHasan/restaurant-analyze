import logging

import click

from models.review import Review
from star_prediction.predictor import StarPredictor
import matplotlib.pyplot as plt
from services import DataReader
from models import Business
from trend_prediction import TrendPredictor
from topics import TopicPredictor


@click.group()
def cli():
    pass


@cli.command()
def fit_star_predictor():
    reviews = DataReader().reviews()
    sp = StarPredictor()
    sp.fit_accuracy_score(reviews)
    sp.save()


@cli.command()
def draw_restaurants_trand():
    for b in DataReader().businesses():
        if len(b.reviews) > 500:
            b.fit_rolling_time_series()
            b.rolling_time_series_.plot(ylim=(1, 5))
            plt.show()


@cli.command()
def svr_trend_prediction():

    logger = logging.getLogger('trend-prediction')
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.FileHandler('../logs/trend_prediction.log'))

    predictor = TrendPredictor()

    X_train, X_test, y_train, y_test = \
        Business.train_test_set(DataReader().businesses())

    predictor.fit(X_train, y_train)
    predictor.save()

    logger.info(predictor)
    logger.info('mean squared error: %s ' %
                predictor.mean_squared_error(X_test, y_test))
    logger.info('r2 score: %s ' % predictor.r2_score(X_test, y_test))

if __name__ == '__main__':
    cli()
