from collections import Counter
from string import punctuation

import nltk
import pandas as pd
from nltk.corpus import stopwords
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureExctractor(BaseEstimator, TransformerMixin):
    """Custom Transformer that extracts specified column(s).
    :param feature: str or list od string that specifies which features will be extracted.

    Note: FeatureExctractor is not designed to handle data grouped by sample. 
    (e.g. a list of dicts).  If your data is structured this way, consider a
    transformer along the lines of `sklearn.feature_extraction.DictVectorizer`.
    """
    def __init__(self, feature):
        self.feature = feature

    def transform(self, data):
        return data[self.feature]

    def fit(self, *_):
        return self


class DataPrepperator(BaseEstimator, TransformerMixin):
    """Custom Transformer that prepares  the data.
    Note: The class expects pd.Series object.
    """
    def __init__(self):
        self.stop_words = stopwords.words('english')

    def find_types_of_sents_in_text(self, text):
        """ Tokenizes the question text into sentences and finds the different types of sentences.
        :param text: Text that will be processed.
        """
        return dict(Counter(map(lambda x: x[-1], nltk.sent_tokenize(text))))

    def clean_raw_data(self, text):
        """ Cleans the raw text from stop_words and punctuation.
        :param text: Variable that contains text.
        """
        return [token.lower() for token in nltk.word_tokenize(text)
                if token not in self.stop_words and token not in punctuation]

    def transform(self, data):
        """ Prepped the data by removing stop_words and punctuation and creating
        to additional features 'tokens_len' and 'number_of_questions_in_text'.
        :param data: pd.Series containing the question_text.
        """
        tokens = data.apply(self.clean_raw_data)

        tokens_len = tokens.apply(len)
        unique_sents_type = data.apply(self.find_types_of_sents_in_text)
        number_of_questions_in_text = unique_sents_type.apply(lambda x: x.get('?', 0))
        clean_text = tokens.apply(' '.join)

        return pd.concat([tokens_len, number_of_questions_in_text, clean_text], 
                          keys=['tokens_len', 'number_of_questions_in_text', data.name],
                          axis=1)

    def fit(self, *_):
        return self
