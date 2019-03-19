import pytest
import pandas as pd
from insincere_questions_classification.utils.custom_transformers import (FeatureExctractor, DataPreparator)


@pytest.fixture(scope='module')
def test_df():
    data = [["Test column. Test?", 65],
            ["How is this possible?", None],
            [None, 21],
            ["", 89]]
    return pd.DataFrame(data, columns=['question_text', 'num_col'])


@pytest.mark.feature_extractor
class TestFeatureExctractor():
    def test_if_transformer_works_correctly_for_text_column(self, test_df):
        result = FeatureExctractor(feature='question_text').transform(test_df)

        assert test_df['question_text'].equals(result)

    def test_if_transformer_works_correctly_for_numeric_column(self, test_df):
        result = FeatureExctractor(feature='num_col').transform(test_df)

        assert test_df['num_col'].equals(result)


@pytest.mark.data_preparator
class TestDataPreparator():
    def check_if_error_is_raised(self, test_df):
        expected_msg = 'Incorrect type. The argument should be pandas.Series!'

        with pytest.raises(ValueError) as excinfo:
            result = DataPreparator().transform(test_df)

        assert expected_msg == str(excinfo.value)

    def test_if_transformer_works_correctly(self, test_df):
        expected_result = pd.Series(['test column test',
                                     'how possible',
                                     None,
                                     ''], name='question_text')
        result = DataPreparator().transform(test_df['question_text'].dropna())

        assert expected_result.dropna().equals(result['question_text'])