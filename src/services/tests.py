import unittest

from .data_reader import DataReader


class TestDataReader(unittest.TestCase):

    def setUp(self):
        self.service = DataReader()

    @unittest.skip('too long test')
    def test_reviews(self):
        self.assertNotEqual(len(self.service.reviews()), 0)

    def test_sample_reviews(self):
        self.assertNotEqual(len(self.service.sample_reviews(20000)), 0)

    @unittest.skip('too long test')
    def test_businesses(self):
        self.assertNotEqual(len(self.service.businesses()), 0)

    def test_sample_businesses(self):
        self.assertNotEqual(len(self.service.sample_businesses()), 0)
