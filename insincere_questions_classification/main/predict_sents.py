import os
import pickle

import pandas as pd

from insincere_questions_classification import load_model
from insincere_questions_classification.utils.custom_transformers import DataPreparator


def make_prediction(text):
    """ Makes prediction on a given text.
    :param text: text that will be evaluated if it's Insincere or not. 
    """
    file_path = os.path.abspath(__file__)
    package_dir = os.path.dirname(os.path.dirname(file_path))

    path = os.path.join(package_dir, "models", "sgdclassifier_adv.pickle")
    model = load_model(path)

    prepped_df = DataPreparator().transform(pd.DataFrame([text], columns=['question_text']))
    return model.predict(prepped_df['question_text'])[0]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Script uses model to evaluate if text is insincere or not.')

    parser.add_argument('--text', required=True, type=str, help='Text that will be evaluated.')

    args = parser.parse_args()

    print("The text was evaluated with the value: ", make_prediction(args.text))
