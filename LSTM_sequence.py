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
from keras.layers import LSTM, Dense, Activation
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, Callback
from keras.models import model_from_json
from keras.preprocessing import sequence
from load_dataset import prepare_dataset
from write_results import write_results_to_be_plotted, write_scores, prepare_df, plot_precision_recall_curve, plot_auroc_curve
from keras import backend as K
import json, sys, os
from explainable import compute_shap_values, compute_shap_values_for_running_cases, find_explanations_for_running_cases
from azure_facilities import store_blobs, remove_files_in_experiment


class LogRunMetrics(Callback):
    def __init__(self, run, column_type):
        self.run = run
        self.column_type = column_type
    # callback at the end of every epoch
    def on_epoch_end(self, epoch, log):
        # log on azure console the metrics defined in keras
        if self.column_type == 'Categorical':
            self.run.log('Loss', log['loss'])
            self.run.log('Accuracy', log['accuracy'])
            self.run.log('F1', log['f1'])
        else:
            self.run.log('Loss', log['loss'])
            self.run.log('Mean_squared_error', log['mean_squared_error'])

def cast_predictions_to_days_hours(df, pred_column):
    days = [int(str(x).split('.')[0]) for x in df[pred_column]]
    hours = [round((24 / 100 * int(str(x).split('.')[1][:2]))) for x in df[pred_column]]
    # increment day by 1 if hours are 24 and set hours to 0
    days = [i + 1 if j == 24 else i for i, j in zip(days, hours)]
    hours = [0 if j == 24 else j for i, j in zip(days, hours)]
    hours = [str(x) + 'h' if x != 0 else '' for x in hours]
    days = [str(x) + 'd' for x in days]
    res = [i + ' ' + j if j != '' else i for i, j in zip(days, hours)]
    #delete 0d becuase it makes no sense
    res = [x.replace('0d ', '') if '0d ' in x else x for x in res]
    df[pred_column] = res
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

def compile_model(model, event_level, column_type):
    if event_level == 0 and column_type == 'Categorical':
        model.compile(loss='categorical_crossentropy', optimizer='Nadam',
                      metrics=['accuracy', 'categorical_accuracy', f1])
    elif event_level == 1 and column_type == 'Categorical':
        model.compile(loss='binary_crossentropy', optimizer='Nadam', metrics=['accuracy', 'binary_accuracy', f1])
    else:
        model.compile(loss='mae', optimizer='Nadam', metrics=['mean_squared_error', 'mae', 'mape'])
    return model

def train_model(X_train, y_train, n_neurons, n_layers, class_weights, event_level, column_type, experiment_name, maxlen, num_epochs, run=None):
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

    if column_type == 'Numeric':
        # add output layer (regression)
        model.add(Dense(1))
    else:
        # every neuron predicts one element of the vector
        model.add(Dense(y_train.shape[1]))
        if event_level == 0:
            # only one class is one
            model.add(Activation('softmax'))
        else:
            # more than one class can be 1 (event level)
            model.add(Activation('sigmoid'))

    # compiling the model, creating the callbacks
    model = compile_model(model, event_level, column_type)

    print(model.summary())
    early_stopping = EarlyStopping(patience=42)
    model_checkpoint = ModelCheckpoint(
        experiment_name + "/model/model_" + str(n_neurons) + "_" + str(n_layers) + "_weights_best.h5",
        monitor='val_loss', verbose=0, save_best_only=True, mode='auto')
    lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=0, mode='auto',
                                   epsilon=0.0001, cooldown=0, min_lr=0)

    callbacks = [early_stopping, model_checkpoint, lr_reducer]
    if run is not None:
        #we are in Azure, log metrics in Azure console
        callbacks.append(LogRunMetrics(run, column_type))

    # train the model (maxlen) - adjust weights if needed and categorical column (more than one class to be predicted)
    if class_weights is not None and (len(class_weights.keys()) != 1):
        model.fit(X_train, y_train, validation_split=0.2, callbacks=callbacks,
                  epochs=num_epochs, batch_size=maxlen, class_weight=class_weights)
    else:
        model.fit(X_train, y_train, validation_split=0.2, callbacks=callbacks,
                  epochs=num_epochs, batch_size=maxlen)
    # saving model shape to file
    model_json = model.to_json()
    with open(experiment_name + "/model/model_" + str(n_neurons) + "_" + str(n_layers) + ".json", "w") as json_file:
        json_file.write(model_json)
    print("Created model and saved weights")
    return model


def prepare_data_predict_and_obtain_explanations(experiment_name, df, case_id_position, start_date_position, date_format,
                                    end_date_position, pred_column, mode, pred_attributes, n_neurons, n_layers,
                                    shap_calculation, override, num_epochs=500, run=None, storage_dir=None):
    # fix random seed for reproducibility
    np.random.seed(7)
    # train model
    if mode == "train":
        X_train, y_train, X_test, y_test, test_case_ids, target_column_name, event_level, column_type, class_weights, feature_columns = prepare_dataset(df, experiment_name,
                                                                                           case_id_position, start_date_position, date_format, end_date_position,
                                                                                           pred_column, mode, override, pred_attributes)
        if column_type != "Numeric":
            if pred_attributes is None:
                sys.exit("you must define for which categorical attributes you want to obtain predictions/explanations!")
        X_train = sequence.pad_sequences(X_train, dtype="float32")
        print("DEBUG: training shape", X_train.shape)
        X_test = sequence.pad_sequences(X_test, maxlen=X_train.shape[1], dtype="float32")
        print("DEBUG: test shape", X_test.shape)
        info = json.load(open(experiment_name + "/model/data_info.json"))
        info['column_type'] = column_type
        info['shape'] = X_train.shape
        with open(experiment_name + "/model/data_info.json", "w") as json_file:
            json.dump(info, json_file)

        if override is True or (override is False and not os.path.exists(experiment_name + "/model/model_100_8.json")):
            model = train_model(X_train, y_train, n_neurons, n_layers, class_weights, event_level, column_type, experiment_name, X_train.shape[1], num_epochs, run)
            df = prepare_df(model, X_test, y_test, test_case_ids, target_column_name, pred_column, mode, column_type)
            write_results_to_be_plotted(df, experiment_name, n_neurons, n_layers)
            scores = model.evaluate(X_test, y_test, verbose=0)
            write_scores(scores, experiment_name, n_neurons, n_layers, pred_column, column_type, event_level, target_column_name, df)
        else:
            #if override is False you just reload the model and calculate the shapley values
            model = model_from_json(open(experiment_name + "/model/model_100_8.json").read())
            model.load_weights(experiment_name + "/model/model_" + str(n_neurons) + "_" + str(n_layers) + "_weights_best.h5")
            print("Reloaded model and weights")
            df = prepare_df(model, X_test, y_test, test_case_ids, target_column_name, pred_column, mode, column_type)
            write_results_to_be_plotted(df, experiment_name, n_neurons, n_layers)

        if column_type == "Categorical":
            predictions_names = target_column_name
            target_column_name = ['TEST_' + x for x in target_column_name]
            plot_auroc_curve(df, predictions_names, target_column_name, experiment_name)
            plot_precision_recall_curve(df, predictions_names, target_column_name, experiment_name)
        if run is not None:
            #we copy only the model and the shap folder (results and plots can be seen in the experiment)
            store_blobs(storage_dir, ['model'])
            remove_files_in_experiment(['model'])
        if shap_calculation is True:
            indexes_to_plot = []
            attributes_to_plot = []
            # if column of type numeric you have only one shap histogram to plot
            if column_type != "Numeric":
                attributes_to_plot = pred_attributes
                for attribute in attributes_to_plot:
                    indexes_to_plot.append(target_column_name.index('TEST_'+attribute))
            compute_shap_values(df, experiment_name, X_train, X_test, model, column_type,
                                 feature_columns, indexes_to_plot, attributes_to_plot, pred_column)
            print("Generated histogram containing the explanations on the test set")
            if run is not None:
                store_blobs(storage_dir, ['shap'])
                remove_files_in_experiment(['shap'])
    #model already trained (here is the case when you have the true test - no response variable)
    elif mode == "predict":
        X_test, test_case_ids, target_column_name, feature_columns, num_events = prepare_dataset(df, experiment_name, case_id_position, start_date_position,
                                                                       date_format, end_date_position, pred_column, mode, override)
        # shape needed to fit test data in the already trained model
        shape = json.load(open(experiment_name + "/model/data_info.json"))['shape']
        column_type = json.load(open(experiment_name + "/model/data_info.json"))['column_type']
        X_test = sequence.pad_sequences(X_test, maxlen=shape[1], dtype="float32")
        print("DEBUG: test shape", X_test.shape)

        model = model_from_json(open(experiment_name + "/model/model_100_8.json").read())
        # load saved weigths to the test model
        model.load_weights(experiment_name + "/model/model_" + str(n_neurons) + "_" + str(n_layers) + "_weights_best.h5")
        print("Reloaded model and weights")
        y_test = None
        df = prepare_df(model, X_test, y_test, test_case_ids, target_column_name, pred_column, mode, column_type)
        # if you are predicting remaining time you should cast results to days and hours
        #if pred_column == 'remaining_time':
            #df = cast_predictions_to_days_hours(df, pred_column)
        #if pred_attributes are not specified means that it is a numerical column
        if pred_attributes is None:
            pred_attributes = [pred_column]
        shapley_test = compute_shap_values_for_running_cases(experiment_name, X_test, model)
        df = find_explanations_for_running_cases(shapley_test, X_test, df, feature_columns, pred_attributes, column_type, num_events, experiment_name)
        print("Generated predictions for running cases along with explanations")
        #resp = []
        # resp.append(df.columns.to_list())
        # for i in range(df.shape[0]):
        #     resp.append(df.iloc[i].to_list())
        # for i in range(df.shape[0]):
        #     row = {}
        #     values = df.loc[i, :].to_list()
        #     for j in range(len(values)):
        #         row[df.columns[j]] = values[j]
        #     resp.append(row)

        resp = {}
        for column in df.columns:
            resp[column] = df[column].to_list()

        if os.path.isdir("./outputs"):
            #we are in the cloud (write in output folder)
            experiment_name = "./outputs"
        df.to_csv(experiment_name + "/results/results_running_" + str(n_neurons) + "_" + str(n_layers) + ".csv", index=False)
        #write the results as json ready to be returned
        with open(experiment_name + "/results/results_running_" + str(n_neurons) + "_" + str(n_layers) + ".json", "w") as json_file:
            json.dump(resp, json_file)
        if run is not None:
            store_blobs(storage_dir, ['results'])
            remove_files_in_experiment(['results'])
