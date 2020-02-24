"""
Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.
"""

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, GRU, Dropout, Dense, Input, Activation
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import model_from_json
from keras.preprocessing import sequence
from keras import metrics
from load_dataset import load_dataset
from write_results import write_results_to_be_plotted, write_scores, prepare_df, plot_precision_recall_curve, plot_auroc_curve
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from keras import backend as K
import sys
import os
import json
from math import sqrt
from explainable import compute_shap_values, calculate_histogram_for_shap_values, compute_shap_values_for_running_cases, find_explanations_for_running_cases

def clean_name(filename, pred_column, n_neurons, n_layers):
    row_process_name = filename.replace('_anonimyzed', '').replace('.csv', '').replace('_features', '').replace('data/', '')
    row_process_name = row_process_name + "_" + pred_column
    file_trained = "model/model_" + row_process_name + "_" + str(n_neurons) + "_" + str(n_layers) + ".json"
    file_trained = file_trained.replace('_running', '')
    if 'running' in row_process_name:
        running = 1
    else:
        running = 0
    row_process_name = row_process_name.replace('_running', '')
    return row_process_name, file_trained, running

def cast_predictions_to_days_hours(df):
    days = [int(str(x).split('.')[0]) for x in df['Predictions']]
    hours = [round((24 / 100 * int(str(x).split('.')[1][:2]))) for x in df['Predictions']]
    # increment day by 1 if hours are 24 and set hours to 0
    days = [i + 1 if j == 24 else i for i, j in zip(days, hours)]
    hours = [0 if j == 24 else j for i, j in zip(days, hours)]
    hours = [str(x) + 'h' if x != 0 else '' for x in hours]
    days = [str(x) + 'd' for x in days]
    res = [i + ' ' + j if j != '' else i for i, j in zip(days, hours)]
    df['Predictions'] = res
    return df

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """
        Only computes a batch-wise average of recall.
        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.
        Only computes a batch-wise average of precision.
        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def train_model(X_train, y_train, n_neurons, n_layers, class_weights, event_level, column_type, row_process_name, file_trained):
    # create the model
    model = Sequential()
    if n_layers == 1:
        model.add(LSTM(n_neurons, implementation=2, input_shape=(X_train.shape[1], X_train.shape[2]),
                       recurrent_dropout=0.2))
        model.add(BatchNormalization())
    else:
        for i in range(n_layers - 1):
            model.add(LSTM(n_neurons, implementation=2, input_shape=(X_train.shape[1], X_train.shape[2]),
                           recurrent_dropout=0.2, return_sequences=True))
            model.add(BatchNormalization())
        model.add(LSTM(n_neurons, implementation=2, recurrent_dropout=0.2))
        model.add(BatchNormalization())

    if isinstance(y_train, pd.Series):
        # add output layer (regression)
        model.add(Dense(1))
    else:
        # every neuron predicts one element of the vector
        model.add(Dense(y_train.shape[1]))
        if event_level == 0 and column_type == 'Categorical':
            # only one class is one
            model.add(Activation('softmax'))
        else:
            # more than one class can be 1 (event level)
            model.add(Activation('sigmoid'))

    # compiling the model, creating the callbacks
    if event_level == 0 and column_type == 'Categorical':
        model.compile(loss='categorical_crossentropy', optimizer='Nadam',
                      metrics=['accuracy', 'categorical_accuracy', f1])
    elif event_level == 1 and column_type == 'Categorical':
        model.compile(loss='binary_crossentropy', optimizer='Nadam', metrics=['accuracy', 'binary_accuracy', f1])
    else:
        model.compile(loss='mae', optimizer='Nadam', metrics=['mean_squared_error', 'mae', 'mape'])
    print(model.summary())
    early_stopping = EarlyStopping(patience=42)
    model_checkpoint = ModelCheckpoint(
        "model/model_" + row_process_name + "_" + str(n_neurons) + "_" + str(n_layers) + "_weights_best.h5",
        monitor='val_loss', verbose=0, save_best_only=True, mode='auto')
    lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=0, mode='auto',
                                   epsilon=0.0001, cooldown=0, min_lr=0)

    # train the model (maxlen) - adjust weights if needed and categorical column
    if class_weights is not None:
        model.fit(X_train, y_train, validation_split=0.2, callbacks=[early_stopping, model_checkpoint, lr_reducer],
                  epochs=500, batch_size=maxlen, class_weight=class_weights)
    else:
        model.fit(X_train, y_train, validation_split=0.2, callbacks=[early_stopping, model_checkpoint, lr_reducer],
                  epochs=500, batch_size=maxlen)
    # saving model shape to file
    model_json = model.to_json()
    with open(file_trained, "w") as json_file:
        json_file.write(model_json)
    with open("model/" + row_process_name + "_train_shape.json", "w") as json_file:
        shape = {'shape': X_train.shape}
        json.dump(shape, json_file)
    print("Created model and saved weights")
    return model

def prepare_data_to_be_returned(df):
    # we have to return only the latest or the max prediction for every case (seconds if remaining time)
    # df = df.groupby('CASE ID').agg(['last'])
    # df = df.groupby('CASE ID').max()
    results = []
    # return structure like [{'caseid' : 0, 'predictions': 100, 'test' = 200}, {..}, ...]
    for i in range(df.shape[0]):
        columns = {}
        for column in df.columns:
            columns[column] = df.loc[i, column]
        results.append(columns)
    return results



if len(sys.argv) < 7:
    sys.exit("python LSTM_sequence.py n_neurons n_layers filename case_id_position start_date_position date_format "
             "kafka pred_column (end_position)")
n_neurons = int(sys.argv[1])
n_layers = int(sys.argv[2])
filename = sys.argv[3]
case_id_position = int(sys.argv[4])
start_date_position = int(sys.argv[5])
date_column_format = sys.argv[6]
kafka = int(sys.argv[7])
pred_column = sys.argv[8] # remaining_time if you want to predict that
end_date_position = None
# if also end date is passed
if len(sys.argv) == 10:
    end_date_position = int(sys.argv[9])

# fix random seed for reproducibility
np.random.seed(7)

#TODO: alcuni parametri non saranno più da mettere in input / non usare più il nome ma il modello in input per capire che modello ricaricare per predire
row_process_name, file_trained, running = clean_name(filename, pred_column, n_neurons, n_layers)


# limit the quantity of memory you wanna use in your gpu
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

#train model (if it's already trained and it's not the running csv just reshow the results)
if not os.path.isfile(file_trained) or running == 0:
    X_train, y_train, X_test, y_test, test_case_ids, target_column_name, event_level, column_type, class_weights, feature_columns = load_dataset(filename, file_trained, row_process_name,
                                                                                       case_id_position, start_date_position,
                                                                                       date_column_format, kafka, end_date_position, pred_column, running)
    X_train = sequence.pad_sequences(X_train, dtype="float32")
    print("DEBUG: training shape", X_train.shape)
    maxlen = X_train.shape[1]
    X_test = sequence.pad_sequences(X_test, maxlen=X_train.shape[1], dtype="float32")
    print("DEBUG: test shape", X_test.shape)

    #here we have to train the model or load it in order to produce the same results as before
    if not os.path.isfile(file_trained):
        model = train_model(X_train, y_train, n_neurons, n_layers, class_weights, event_level, column_type, row_process_name, file_trained)
        df = prepare_df(model, X_test, y_test, test_case_ids, target_column_name, pred_column, running)
        write_results_to_be_plotted(df, y_test, row_process_name, n_neurons, n_layers)
        scores = model.evaluate(X_test, y_test, verbose=0)
        write_scores(scores, row_process_name, n_neurons, n_layers, pred_column, column_type, event_level)
    else:
        #TODO tieni questo pezzo di codice solo per provare l'istogramma degli shap values
        model = model_from_json(open(file_trained).read())
        # load saved weigths to the test model
        model.load_weights("model/model_" + row_process_name + "_" + str(n_neurons) + "_" + str(n_layers) + "_weights_best.h5")
        print("Loaded model and weights from file")
        df = prepare_df(model, X_test, y_test, test_case_ids, target_column_name, pred_column, running)

    # TODO: nei write_scores scriviamo anche AUC-precision-recall quando abbiamo da predire un attributo categorico
    if column_type == "Categorical":
        predictions_names = target_column_name
        target_column_name = ['TEST_' + x for x in target_column_name]
        plot_auroc_curve(df, predictions_names, target_column_name, row_process_name)
        plot_precision_recall_curve(df, predictions_names, target_column_name, row_process_name)
    #with front end enabled
    #results = prepare_data_to_be_returned(df)
    shapley_test = compute_shap_values(row_process_name, X_train, X_test, model, column_type)
    explanation_histogram = calculate_histogram_for_shap_values(df, target_column_name, column_type, X_test, shapley_test, feature_columns, row_process_name)

#TODO be sure also that if works if someone gives you a csv with the columns shifted
#model already trained (here is the case when you have the true test - no response variable)
elif os.path.isfile(file_trained):
    X_test, test_case_ids, target_column_name, feature_columns = load_dataset(filename, file_trained, row_process_name, case_id_position, start_date_position,
                                                                   date_column_format, kafka, end_date_position, pred_column, running)
    # shape needed to fit test data in the already trained model
    shape = json.load(open("model/" + row_process_name + "_train_shape.json"))['shape']
    X_test = sequence.pad_sequences(X_test, maxlen=shape[1], dtype="float32")
    print("DEBUG: test shape", X_test.shape)

    model = model_from_json(open(file_trained).read())
    # load saved weigths to the test model
    model.load_weights("model/model_" + row_process_name + "_" + str(n_neurons) + "_" + str(n_layers) + "_weights_best.h5")
    print("Loaded model and weights from file")
    y_test = None
    df = prepare_df(model, X_test, y_test, test_case_ids, target_column_name, pred_column, running)
    # if you are predicting remaining time you should cast results to days and hours
    if pred_column == 'remaining_time':
        df = cast_predictions_to_days_hours(df)
    #results = prepare_data_to_be_returned(df)
    # TODO: the user wants to know only one activity, so discard other predictions (for example with bac take ACTIVITY=11)
    shapley_test = compute_shap_values_for_running_cases(row_process_name, X_test, model)
    df = find_explanations_for_running_cases(shapley_test, X_test, df, feature_columns)
    df.to_csv("results/results_running_" + row_process_name + "_" + str(n_neurons) + "_" + str(n_layers) + ".csv", index=False)


else:
    print("You are requiring predictions for running cases but you haven't trained the model yet!")
