import csv
import datetime
import itertools
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from .custom_transformers import FactorExctractor


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', 
                          cmap=plt.cm.Blues):
    """ This function prints and plots the confusion matrix.
    :param cm: Confusion Matrix object
    :param classes: list with the classed
    :param normalize=True: Applies normalization by setting  it to True.
    :param title="Confusion matrix": Title of the plot.
    :param cmap=plt.cm.Blues: Color map of the plot.
    """

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


def train_model(pipeline, train_set,
                input_cols=['question_text', 'tokens_len'], target_col='target',
                export_path=None):
    """ Helper Function for easier training, evaluating and exporting of a model.
    :param pipeline: Pipeline object that will be trained.
    :param train_set: Dataset that will be used to train the model.
    :param input_cols=['question_text', 'tokens_len']: string or list of strings that specifies 
        which columns to be used for training the model.
    :param target_col='target': name of the column that contains the result.
    :param export_path=None: Optional parameter that specifies where the model should be saved.
        Note that if you set this argument the script will create a file to log the models that
         are exported.
    """
    X_train, X_test, y_train, y_test = train_test_split(train_set[input_cols],
                                                        train_set[target_col],
                                                        test_size=0.30,
                                                        random_state=72)
    start_training = datetime.datetime.now()
    model = pipeline.fit(X_train, y_train)
    training_time = datetime.datetime.now() - start_training

    y_predict = model.predict(X_test)
    print("F1 score: ", f1_score(y_test, y_predict))

    conf_matrix = confusion_matrix(y_predict, y_test)

    # Compute confusion matrix
    cnf_matrix = conf_matrix
    class_names = y_test.unique().tolist()
    np.set_printoptions(precision=2)

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                          title='Normalized confusion matrix')

    if export_path:
        log_model(y_test, y_predict, export_path,
                  training_time=training_time, training_size=X_train.shape[0])
        export_model(model, export_path)
        dir_name, file_name = os.path.split(export_path)
        output_name = file_name.split('.')[0]
        # saves the plot
        plt.savefig(os.path.join(dir_name, output_name + "_confusion_matrix.png"))

    plt.show()

    return model


def log_model(y_true, y_pred, export_path, training_time=None, training_size=None):
    """
    Helper function that saves the information of each model that is being exported.
    The csv file will contains following columns:
        ['model_name', 'accuracy', 'f1', 'precision', 'recall', 
         'training_time', 'creation_time', 'training_size', 'location_model'],

    :param y_true: targeted column
    :param y_pred: model prediction
    :param export_path: Specifies where to save the model. By default it will save it "../models/"
    :param training_time: Time needed for the model to be trained.
    :param training_size: Size of the training data

    Example:
    If export_path = "../proj_dir/exported_models/awesome_model.pkl"
    The location of the log file will be "../proj_dir/models/log_models.csv"
    """
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
                [model, accuracy, f1, precision, recall, 
                 training_time, datetime.datetime.now(), training_size, export_path])
    else:
        with open(log_file_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerows([
                ['model_name', 'accuracy', 'f1', 'precision', 'recall', 
                 'training_time', 'creation_time', 'training_size', 'location_model'],
                [model, accuracy, f1, precision, recall, 
                 training_time, datetime.datetime.now(), training_size, export_path]])


def export_model(model, path):
    """Exports the model to a pickle file.
    :param model: model that will b exported.
    :param path: path where the model will be stored.
    """
    file_name = os.path.basename(path)
    if not (file_name.endswith('.pickle') and file_name.endswith('.pkl')):
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
