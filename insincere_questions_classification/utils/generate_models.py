import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from insincere_questions_classification.utils.custom_transformers import \
    FeatureExctractor, DataPreparator
from insincere_questions_classification.utils.data_sampling import *
from insincere_questions_classification.utils.functions import *


def train_export_pipeline(pipeline, classifiers, data, postfix='basic'):
    """Tries different classifiers in a specified pipeline on a dataset and after that it saves the model.
    :param pipeline: sklearn Pipeline
    :param classifiers: Classifiers that will be used in the particular pipeline.
        Note classifiers should be list of dictionaries with 
        key: name of the classifier and values: Classifier object.
    :param data: Dataset that the classifiers will be used on.
    :param postfix='basic':  Postfix that will be used for generating the name of the saved model.
        Note: Name of the model file is generated line: key_of_classifier + '_' + postfix.pickle
    """
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(project_dir, "models")

    for name, classifier in classifiers.items():
        export_path = os.path.join(models_dir, f"{name}_{postfix}.pickle")
        pipeline.set_params(clf=classifier)
        model_sgd = train_model(pipeline, data, input_cols='question_text', plot=False,
                                export_path=export_path)
        print(f"Successfully save the file in {export_path}")


def train_basic_pipeline(classifiers, data):
    """Method that creates, trains and exports basic Pipeline.
    :param classifiers: Classifiers that will be applied to the pipeline
    :param data: Dataset that will be used to train and evaluate model performance.
    """
    base_pipeline = Pipeline([
        ('data_prep', DataPreparator()),
        ('feature_extract', FeatureExctractor('question_text')),
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', SGDClassifier(loss='perceptron', max_iter=100, random_state=42)),
    ])

    train_export_pipeline(base_pipeline, classifiers, data)


def train_adv_pipeline(classifiers, data):
    """Method that creates, trains and exports complex Pipeline.
    :param classifiers: Classifiers that will be applied to the pipeline
    :param data: Dataset that will be used to train and evaluate model performance.
    """
    adv_pipeline = Pipeline([
        ('data_prep', DataPreparator()),
        ('union', FeatureUnion(
            transformer_list=[
                ('numeric_features', Pipeline([
                    ('selector', FeatureExctractor(
                        ['tokens_len', 'number_of_questions_in_text']))
                ])),
                ('text_features', Pipeline([
                    ('selector', FeatureExctractor('question_text')),
                    ('vect', CountVectorizer()),
                    ('tfidf', TfidfTransformer()),
                ]))
            ]
        )),
        ('clf', SGDClassifier(loss='perceptron', max_iter=100, random_state=42)),
    ])

    train_export_pipeline(adv_pipeline, classifiers, data, postfix='adv')


if __name__ == "__main__":
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(project_dir, "input", "train.csv")

    train_df = pd.read_csv(data_path)

    train_downsampled = downsampling(train_df[train_df.target==0], 
                                     train_df[train_df.target==1])

    classifiers = {
        'sgdclassifier': SGDClassifier(loss='perceptron', max_iter=100, random_state=42),
        'linearSVC': LinearSVC(),
        'logisticregression': LogisticRegression(random_state=0, solver='lbfgs', multi_class='ovr'),
        'decisiontree': DecisionTreeClassifier(max_depth=30)
    }

    train_basic_pipeline(classifiers, train_downsampled)

    train_adv_pipeline(classifiers, train_downsampled)
