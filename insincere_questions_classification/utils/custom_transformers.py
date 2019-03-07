from sklearn.base import TransformerMixin, BaseEstimator


class FeatureExctractor(BaseEstimator, TransformerMixin):
    """Custom Transformer that extracts specified column(s).
    :param feature: str or list od string that specifies which features will be extracted.

    Note: FeatureExctractor is not designed to handle data grouped by sample. 
    (e.g. a list of dicts).  If your data is structured this way, consider a
    transformer along the lines of `sklearn.feature_extraction.DictVectorizer`.
    """
    def __init__(self, feature):
        self.feature = feature

    def transform(self, dataset):
        return dataset[self.feature]

    def fit(self, *_):
        return self