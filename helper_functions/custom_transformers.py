from sklearn.base import TransformerMixin, BaseEstimator


class FactorExctractor(TransformerMixin, BaseEstimator):
    """Custom Transformer that extracts specified column."""
    def __init__(self, factor):
        self.factor = factor

    def transform(self, dataset):
        return dataset[self.factor]

    def fit(self, *_):
        return self