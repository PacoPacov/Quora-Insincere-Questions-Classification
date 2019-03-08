import pandas as pd
from insincere_questions_classification.utils.functions import *
from insincere_questions_classification.utils.data_sampling import *
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from insincere_questions_classification.utils.custom_transformers import FeatureExctractor

train_df = pd.read_csv("../input/train.csv")

train_downsampled = downsampling(train_df[train_df.target==0], 
                                 train_df[train_df.target==1])

classifiers = {
    'sgdclassifier': SGDClassifier(loss='perceptron', max_iter=100, random_state=42),
    'linearSVC': LinearSVC(),
    'logisticregression': LogisticRegression(random_state=0, solver='lbfgs', multi_class='ovr'),
    'decisiontree': DecisionTreeClassifier(max_depth=30)
}

base_pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier(loss='perceptron', max_iter=100, random_state=42)),
])

for name, classifier in classifiers.items():
    export_path = os.path.abspath(f"../models/{name}_basic.pickle")
    base_pipeline.set_params(clf=classifier)
    model_sgd = train_model(base_pipeline, train_downsampled, 
                            input_cols='question_text', plot=False,
                            export_path=export_path)

adv_pipeline_= Pipeline([
    ('union', FeatureUnion(
        transformer_list  = [
            ('numeric_features', Pipeline([
                ('selector', FeatureExctractor(['tokens_len', 'number_of_questions_in_text']))
            ])),
            ('text_features', Pipeline([
                ('selector', FeatureExctractor('clean_text')),
                ('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
        ]))
        ]
    )),
    ('clf', SGDClassifier(loss='perceptron', max_iter=100, random_state=42)),
])

for name, classifier in classifiers.items():
    export_path = os.path.abspath(f"../models/{name}_adv.pickle")
    base_pipeline.set_params(clf=classifier)
    model_sgd = train_model(base_pipeline, train_downsampled,
                            input_cols='question_text', plot=False,
                            export_path=export_path)
