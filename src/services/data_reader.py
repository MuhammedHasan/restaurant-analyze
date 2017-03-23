from models import Review


class DataReader(object):

    def __init__(self):
        self.path = '../data/yelp_academic_dataset_review.json'

    def reviews(self):
        with open(self.path) as f:
            return [Review.str_to_review(i) for i in f]

    def sample_reviews(self, sample_size=200000):
        with open(self.path) as f:
            return [Review.str_to_review(f.readline()) for i in range(200000)]

    def businesses(self):
        return list(Review.convert_to_business(self.reviews()))

    def sample_businesses(self, sample_size=2000):
        return list(Review.convert_to_business(
            self.sample_reviews(sample_size)))
