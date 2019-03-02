import os
import pickle
import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def downsampling(majority: pd.DataFrame, minority: pd.DataFrame, replace: bool = True) -> pd.DataFrame:
    majority_downsampled = resample(majority,
                                    replace=replace,
                                    n_samples=len(minority),
                                    random_state=123)
    return pd.concat([majority_downsampled, minority])


def upsampling(majority: pd.DataFrame, minority: pd.DataFrame, replace: bool = True) -> pd.DataFrame:
    majority_upsampled = resample(minority,
                                  replace=replace,
                                  n_samples=len(majority),
                                  random_state=123)
    return pd.concat([majority_upsampled, majority])


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    import itertools

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


def train_model(pipeline: Pipeline, train_set: pd.DataFrame, export_path=None):
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import f1_score, confusion_matrix

    X_train, X_test, y_train, y_test = train_test_split(train_set['tokenized'],
                                                        train_set['target'], random_state=0)

    model = pipeline.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    print("F1 score: ", f1_score(y_test, y_predict))

    if export_path:
        export_model(model, export_path)
        log_model(y_test, y_predict, export_path)

    conf_matrix = confusion_matrix(y_predict, y_test)

    # Compute confusion matrix
    cnf_matrix = conf_matrix
    class_names = ['0', '1']
    np.set_printoptions(precision=2)

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                          title='Normalized confusion matrix')

    plt.show()

    return model


def log_model(y_true: pd.Series, y_pred: pd.Series, export_path: str)->None:
    PATH = "../models/"
    model = os.path.basename(export_path).split('.')[0]


    if not os.path.exists(os.path.abspath(PATH)):
        os.makedirs(os.path.abspath(PATH))

    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    log_file_path = os.path.join(PATH, 'log_models.csv')

    if os.path.isfile(log_file_path):
        with open(log_file_path, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(
                [model, accuracy, f1, precision, recall, export_path])
    else:
        with open(log_file_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerows([
                ['model_name', 'accuracy', 'f1',
                    'precision', 'recall', 'location_model'],
                [model, accuracy, f1, precision, recall, export_path]])


def export_model(model, path):
    """Exports the model to a pickle file.
    :param model: model that will b exported.
    :param path: path where the model will be stored.
    """
    if not os.path.basename(path).endswith('.pickle'):
        raise ValueError(
            "Need to specify the pickle file what the model will be saved in.")

    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))

    with open(path, 'wb') as f:
        pickle.dump(model, f)


def load_model(path):
    """Loades model from a pickle file.
    :param path: Path to existing file.
    """
    if not os.path.exists(path):
        raise ValueError("Incorrect path")

    with open(path, 'rb') as f:
        model = pickle.load(f)

    return model


def make_prediction(model):
    test_df = pd.read_csv("../input/test.csv")
    test_df['prediction'] = model.predict(test_df['question_text'])
    test_df[['qid', 'prediction']].to_csv("submission.csv", index=False)
