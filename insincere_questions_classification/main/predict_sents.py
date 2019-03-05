import os
import pickle
from collections import Counter
from string import punctuation

import nltk
import pandas as pd
from nltk.corpus import stopwords

from helper_functions import FactorExctractor, load_model


def find_types_of_sents_in_text(text):
    """ Tokenizes the question text into sentences and finds the different types of sentences.
    :param text: Text that will be processed.
    """
    doc = nltk.sent_tokenize(text)
    return dict(Counter(map(lambda x: x[-1], doc)))


def clean_raw_data(text):
    """ Cleans the raw text from stop_words and punctuation.
    :param text: Variable that contains text.
    """
    clean_text = nltk.word_tokenize(text)

    stop_words = stopwords.words('english')

    tokens = [token for token in clean_text
              if token not in stop_words and token not in punctuation]
    return tokens


def data_prep(df:pd.DataFrame):
    """ Prepped the data by removing stop_words and punctuation and creating
    to additional features 'tokens_len' and 'number_of_questions_in_text'.
    :param df: Dataset that will be prepped.
    """
    df['tokens'] = df['question_text'].apply(clean_raw_data)
    df['tokens_len'] = df['tokens'].apply(len)
    df['clean_text'] = df['tokens'].apply(' '.join)

    df['unique_sents_type'] = df['question_text'].apply(find_types_of_sents_in_text)
    df['number_of_questions_in_text'] = df['unique_sents_type'].apply(lambda x: x.get('?', 0))

    return df[['tokens_len', 'number_of_questions_in_text', 'clean_text']]


def make_prediction(text):
    """ Makes prediction on a given text.
    :param text: text that will be evaluated if it's Insincere or not. 
    """
    file_path = os.path.abspath(__file__)
    package_dir = os.path.dirname(os.path.dirname(file_path))

    path = os.path.join(package_dir, "models", "sgdclassifier_adv.pickle")
    model = load_model(path)

    df = pd.DataFrame([text], columns=['question_text'])
    prepped_df = data_prep(df)

    return model.predict(prepped_df)[0]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Script uses model to evaluate if text is insincere or not.')

    parser.add_argument('--text', required=True,
        help='Text that will be evaluated.')

    args = parser.parse_args()

    print("The text was evaluated with the value: ", make_prediction(args.text))
