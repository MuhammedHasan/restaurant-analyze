import time
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


class TopicPredictor(object):

    def __init__(self):
        self.n_topics = 10
        self.n_top_words = 3

    def top_words(self):
        for topic_idx, topic in enumerate(self.model.components_):
            yield " ".join([
                self.feature_names[i]
                for i in topic.argsort()[:-self.n_top_words - 1:-1]
            ])

    def fit(self, X):

        tf_vectorizer = CountVectorizer(
            max_df=0.95, min_df=2, stop_words='english')

        tf = tf_vectorizer.fit_transform(X)

        lda = LatentDirichletAllocation(n_topics=self.n_topics, max_iter=5,
                                        learning_method='online',
                                        learning_offset=50.,
                                        random_state=0)
        self.model = lda.fit(tf)

        self.feature_names = tf_vectorizer.get_feature_names()

        return self

    def fit_predict(self, X):
        return list(self.fit(X).top_words())

    def topics(self, reviews):
        return self.fit_predict([i.text for i in reviews])

    def topics_by_date(self, reviews, start_date, end_date):
        return self.fit_predict([
            i.text for i in reviews
            if self.date_compare(start_date, i.date, end_date)
        ])

    def date_compare(self, start, date, end):
        return start < date < end

    def to_date(self, date_str):

        return time.strptime(date_str, '%Y-%m-%d')
