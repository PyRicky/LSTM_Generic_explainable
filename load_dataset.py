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
import sys

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from statistics import median
#from consumer import create_consumer, read_messages_and_create_df
import json
import math
#from producer import generate_data


def calculateTimeFromMidnight(actual_datetime):
    midnight = actual_datetime.replace(hour=0, minute=0, second=0, microsecond=0)
    timesincemidnight = (actual_datetime - midnight).total_seconds()
    return timesincemidnight

def createActivityFeatures(line, starttime, lastevtime, caseID, current_activity_end_date):
    activityTimestamp = line[1]
    activity = []
    activity.append(caseID)
    for feature in line[2:]:
        activity.append(feature)

    #add features: time from trace start, time from last_startdate_event, time from midnight, weekday+1
    activity.append((activityTimestamp - starttime).total_seconds())
    activity.append((activityTimestamp - lastevtime).total_seconds())
    activity.append(calculateTimeFromMidnight(activityTimestamp))
    activity.append(activityTimestamp.weekday() + 1)
    # if there is also end_date add features time from last_enddate_event and event_duration
    if current_activity_end_date is not None:
        #this feature has been removed since sometimes could be negative if the previous activity finishes later
        #activity.append((activityTimestamp - previous_activity_end_date).total_seconds())
        activity.append((current_activity_end_date - activityTimestamp).total_seconds())
        # add timestamp end or start to calculate remaining time later
        activity.append(current_activity_end_date)
    else:
        activity.append(activityTimestamp)
    return activity


def move_essential_columns(df, case_id_position, start_date_position):
    columns = df.columns.to_list()
    # move case_id column and start_date column to always know their position
    case = columns[case_id_position]
    start = columns[start_date_position]
    columns.pop(columns.index(case))
    columns.pop(columns.index(start))
    df = df[[case, start] + columns]
    return df


def one_hot_encoding(df):
    #case id not encoded
    for column in df.columns[1:]:
        # if column don't numbers encode
        if not np.issubdtype(df[column], np.number):
            # TODO è giusto encodare le colonne di tipo data? O facciamo la diff da 1970 in secondi e poi normalizziamo
            # Get one hot encoding (convert to str eventual 0 to avoid strange duplicate columns)
            one_hot = pd.get_dummies(df[column].apply(str), prefix=column)
            print("Encoded column:{} - Different keys: {}".format(column, one_hot.shape[1]))
            # Drop column as it is now encoded
            df = df.drop(column, axis=1)
            # Join the encoded df
            df = df.join(one_hot)
    print("Categorical columns encoded")
    return df

def convert_strings_to_datetime(df, date_column_format, end_date_position):
    # convert string columns that contain datetime to datetime
    for column in df.columns:
        try:
            #if a number do nothing
            if np.issubdtype(df[column], np.number):
                continue
            df[column] = pd.to_datetime(df[column], format=date_column_format)
        # exception means it is really a string
        except (ValueError, TypeError, OverflowError):
            pass
    return df


def find_case_finish_time(trace, num_activities):
    # we find the max finishtime for the actual case
    for i in range(num_activities):
        if i == 0:
            finishtime = trace[-i-1][-1]
        else:
            if trace[-i-1][-1] > finishtime:
                finishtime = trace[-i-1][-1]
    return finishtime

def calculate_remaining_time_for_actual_case(traces, num_activities):
    finishtime = find_case_finish_time(traces, num_activities)
    for i in range(num_activities):
        # calculate remaining time to finish the case for every activity in the actual case
        traces[-(i + 1)][-1] = (finishtime - traces[-(i + 1)][-1]).total_seconds()
    return traces

def fill_missing_end_dates(df, start_date_position, end_date_position):
    df[df.columns[end_date_position]] = df.apply(lambda row: row[start_date_position]
                  if row[end_date_position] == 0 else row[end_date_position], axis=1)
    return df

def convert_datetime_columns_to_seconds(df):
    for column in df.columns:
        try:
            if np.issubdtype(df[column], np.number):
                continue
            df[column] = pd.to_datetime(df[column])
            df[column] = (df[column] - pd.to_datetime('1970-01-01 00:00:00')).dt.total_seconds()
        except (ValueError, TypeError, OverflowError):
            pass
    return df

def add_features(df, filename, end_date_position):
    dataset = df.values
    traces = []
    # analyze first dataset line
    caseID = dataset[0][0]
    activityTimestamp = dataset[0][1]
    starttime = activityTimestamp
    lastevtime = activityTimestamp
    current_activity_end_date = None
    line = dataset[0]
    if end_date_position is not None:
        # at the begin previous and current end time are the same
        current_activity_end_date = dataset[0][end_date_position]
        line = np.delete(line, end_date_position)
    num_activities = 1
    activity = createActivityFeatures(line, starttime, lastevtime, caseID, current_activity_end_date)
    traces.append(activity)

    for line in dataset[1:, :]:
        case = line[0]
        if case == caseID:
            # continues the current case
            activityTimestamp = line[1]
            if end_date_position is not None:
                current_activity_end_date = line[end_date_position]
                line = np.delete(line, end_date_position)
            activity = createActivityFeatures(line, starttime, lastevtime, caseID, current_activity_end_date)

            # lasteventtimes become the actual
            lastevtime = activityTimestamp
            # if end_date_position is not None:
            #     previous_activity_end_date = current_activity_end_date
            traces.append(activity)
            num_activities += 1
        else:
            caseID = case
            traces = calculate_remaining_time_for_actual_case(traces, num_activities)

            activityTimestamp = line[1]
            starttime = activityTimestamp
            lastevtime = activityTimestamp
            if end_date_position is not None:
                current_activity_end_date = line[end_date_position]
                line = np.delete(line, end_date_position)
            activity = createActivityFeatures(line, starttime, lastevtime, caseID, current_activity_end_date)
            traces.append(activity)
            num_activities = 1

    # last case
    traces = calculate_remaining_time_for_actual_case(traces, num_activities)
    # construct df again with new features
    columns = df.columns
    if end_date_position is not None:
        columns = columns.delete(end_date_position)
    columns = columns.delete(1)
    columns = columns.to_list()
    if end_date_position is not None:
        columns.extend(["time_from_start", "time_from_previous_event(start)", "time_from_midnight",
                        "weekday+1", "event_duration", "remaining_time"])
    else:
        columns.extend(["time_from_start", "time_from_previous_event(start)", "time_from_midnight",
                        "weekday+1", "remaining_time"])
    df = pd.DataFrame(traces, columns=columns)
    print("Features added")
    filename = filename.replace(".csv", "_features.csv")
    df.to_csv(filename, index=False)
    return df

def generate_sequences_for_lstm(dataset, model):
    """
    Input: dataset
    Output: sequences of partial traces and remaining times to feed LSTM
    1 list of traces, each containing a list of events, each containing a list of attributes (features)
    """
    # if we are in real test we just have to generate the single sequence of all events, not 1 sequence per event
    data = []
    trace = []
    caseID = dataset[0][0]
    trace.append(dataset[0][1:].tolist())
    for line in dataset[1:, :]:
        case = line[0]
        if case == caseID:
            trace.append(line[1:].tolist())
        else:
            caseID = case
            if model is None:
                for j in range(1, len(trace) + 1):
                    data.append(trace[:j])
            else:
                data.append(trace[:len(trace)])
            trace = []
            trace.append(line[1:].tolist())
    # last case
    if model is None:
        for i in range(1, len(trace) + 1):
            data.append(trace[:i])
    else:
        data.append(trace[:len(trace)])
    return data

def pad_columns_in_real_data(df, case_ids, row_process_name):
    #fill real test df with empty missing columns (one-hot encoding can generate much more columns in train)
    train_columns = json.load(open("model/" + row_process_name + "_column_names.json"))['columns']
    # if some columns never seen in train set, now we just drop them
    # we should retrain the model with also this new columns (that will be 0 in the original train)
    columns_not_in_test = [x for x in train_columns if x not in df.columns]
    df2 = pd.DataFrame(columns=columns_not_in_test)
    #enrich test df with missing columns seen in train
    df = pd.concat([df, df2], axis=1)
    df = df.fillna(0)
    #reorder data as in train
    df = df[train_columns]
    #add again the case id column
    df = pd.concat([case_ids, df], axis=1)
    return df

def sort_df(df):
    df.sort_values([df.columns[0], df.columns[1]], axis=0, ascending=True, inplace=True, kind='quicksort', na_position='last')
    return df

def clean_df(df, case_id_position):
    #clean rows where case id = null
    df = df.drop(df[df[df.columns[case_id_position]].isnull()].index).reset_index(drop=True)
    return df


def prepare_data_and_add_features(df, filename, case_id_position, start_date_position, date_column_format, end_date_position):
    df = clean_df(df, case_id_position)
    df = df.fillna(0)
    if end_date_position is not None:
        df = fill_missing_end_dates(df, start_date_position, end_date_position)
    df = convert_strings_to_datetime(df, date_column_format, end_date_position)
    df = move_essential_columns(df, case_id_position, start_date_position)
    df = sort_df(df)
    df = add_features(df, filename, end_date_position)
    return df

def normalize_data(df, target_column, target_column_name, model):
    # normalize data (except for case id column and test column)
    case_column = df.iloc[:, 0].reset_index(drop=True)
    case_column_name = df.columns[0]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(df.iloc[:, 1:].values)
    columns = [x for x in df.columns[1:]]
    df = pd.DataFrame(scaled, columns=columns)
    # reinsert case id column and target column
    df.insert(0, case_column_name, case_column)
    # if train phase and only one test column (numeric) reattach test column
    if model is None:
        #if it is a string the target column is one (otherwise it would cycle on the letters of the column)
        if not type(target_column_name) == np.str:
            for i, name in enumerate(target_column_name):
                df[name] = target_column[target_column_name[i]]
        else:
            df[target_column_name] = target_column
    print("Normalized Data")
    return df

def detect_case_level_attribute(df, pred_column):
    # take some sample cases and understand if the attribute to be predicted is event-level or case-level
    # we assume that the column is numeric or a string
    event_level = 0
    case_ids = df[df.columns[0]].unique()
    case_ids = case_ids[:int((len(case_ids) / 100))]
    for case in case_ids:
        df_reduced = df[df[df.columns[0]] == case]
        if len(df_reduced[pred_column].unique()) == 1:
            continue
        else:
            event_level = 1
            break
    return event_level


def save_column_information_for_real_predictions(dfTrain, row_process_name, event_level, target_column_name):
    # save columns names (except for case id column and test columns) and event/case level test attribute in a file
    # (you need that for the shape when you have to predict real data)
    with open("model/" + row_process_name + "_column_names.json", "w") as json_file:
        if event_level == 0:
            if not type(target_column_name) == np.str:
                json.dump({'columns': dfTrain.iloc[:, 1:-len(list(target_column_name))].columns.to_list(), 'test': 'case', 'y_columns': list(target_column_name)}, json_file)
            else:
                json.dump({'columns': dfTrain.iloc[:, 1:-1].columns.to_list(), 'test': 'case', 'y_columns': [target_column_name]}, json_file)
        else:
            if not type(target_column_name) == np.str:
                json.dump({'columns': dfTrain.iloc[:, 1:-len(list(target_column_name))].columns.to_list(), 'test': 'event', 'y_columns': list(target_column_name)}, json_file)
            else:
                json.dump({'columns': dfTrain.iloc[:, 1:-1].columns.to_list(), 'test': 'event', 'y_columns': [target_column_name]}, json_file)

def create_class_weights(dfTrain, target_column_name):
    #for every class calculate number of instances
    class_weights = dict()
    #you must call the classes with the integer number and not the name (the order is the same)
    for i, column in enumerate(target_column_name):
        if ((dfTrain[dfTrain[column] == 1].shape[0] / dfTrain.shape[0]) * 100) < 5:
            class_weights[i] = 4
        else:
            class_weights[i] = 1
    return class_weights

def generate_train_and_test_sets(df, test_case_ids, target_column_name, event_level, column_type, row_process_name, running):
    second_quartile = median(np.unique(df.iloc[:, 0]))
    third_quartile = median(np.unique(df[df.iloc[:, 0] > second_quartile].iloc[:, 0]))
    dfTrain = df[df.iloc[:, 0] < ((second_quartile + third_quartile) / 2)]
    dfTest = df[df.iloc[:, 0] >= ((second_quartile + third_quartile) / 2)]
    if column_type == 'Categorical':
        class_weights = create_class_weights(dfTrain, target_column_name)
    else:
        class_weights = None
    # separate target variable (if a string there is only one y_column)
    if not type(target_column_name) == np.str:
        y_train = dfTrain.iloc[:, -len(list(target_column_name)):]
        y_test = dfTest.iloc[:, -len(list(target_column_name)):]
    else:
        y_train = dfTrain.iloc[:, -1]
        y_test = dfTest.iloc[:, -1]
    # retrieve test case id to later show predictions
    test_case_ids = test_case_ids[dfTrain.shape[0]:]
    save_column_information_for_real_predictions(dfTrain, row_process_name, event_level, target_column_name)

    # create sequences (only for lstm)
    if not type(target_column_name) == np.str:
        feature_columns = dfTrain.columns[1:-len(list(target_column_name))]
        X_train = generate_sequences_for_lstm(dfTrain.iloc[:, :-len(list(target_column_name))].values, running)
        X_test = generate_sequences_for_lstm(dfTest.iloc[:, :-len(list(target_column_name))].values, running)
    else:
        feature_columns = dfTrain.columns[1:-1]
        X_train = generate_sequences_for_lstm(dfTrain.iloc[:, :-1].values, running)
        X_test = generate_sequences_for_lstm(dfTest.iloc[:, :-1].values, running)
    # at every sequence must correspond a value (or array of values) to be predicted
    assert (len(y_train) == len(X_train))
    assert (len(y_test) == len(X_test))
    print("Generated train and test sets")
    #return feature columns without case id for later shapley values
    return X_train, y_train, X_test, y_test, test_case_ids, class_weights, feature_columns

def label_target_columns(df, case_ids, pred_column):
    for case in case_ids:
        # values that will be encountered during the case and will be predicted (except for the first)
        case_pred_values = df[df[df.columns[0]] == case][pred_column][1:]
        # indexes for later accessing those rows
        indexes = df[df[df.columns[0]] == case].index
        # iterate over rows - except last row since it will be removed (end case) - it's important only its value
        for actual_activity_index in range(len(indexes) - 1):
            # put next event-level values to be predicted (even the last) in the corresponding columns of the row
            actual_activity_start = df.loc[indexes[actual_activity_index], 'time_from_start']
            index = actual_activity_index
            # if some activity is in parallel (same start time) it shouldn't be predicted (since it is already happening)
            for i in range(actual_activity_index, (len(indexes)-1)):
                if df.loc[indexes[i+1], 'time_from_start'] != actual_activity_start:
                    break
                else:
                    index = i+1
            #put 1 in the target columns from the first row that has a different start time (happens later in the case)
            df.loc[indexes[actual_activity_index], case_pred_values[index:]] = 1
    return df


def load_dataset(filename, row_process_name, case_id_position, start_date_position, date_column_format,
                 end_date_position, pred_column, model):
    df = pd.read_csv(filename, header=0)
    df = prepare_data_and_add_features(df, filename, case_id_position, start_date_position, date_column_format, end_date_position)

    df = convert_datetime_columns_to_seconds(df)
    # if target column != remaining time exclude target column
    if pred_column != 'remaining_time' and model is None:
        event_level = detect_case_level_attribute(df, pred_column)
        if event_level == 0:
            # case level - test column as is
            if np.issubdtype(df[pred_column], np.number):
                # take target numeric column as is
                column_type = 'Numeric'
                target_column = df[pred_column].reset_index(drop=True)
                target_column_name = pred_column
                # add temporary column to know which rows delete later (remaining_time=0)
                target_column = pd.concat([target_column, df['remaining_time']], axis=1)
                del df[pred_column]
            else:
                # case level string (you want to discover if a client recess will be performed)
                column_type = 'Categorical'
                pred_values = df[pred_column].unique()
                # add one more test column for every value to be predicted
                for value in pred_values:
                    df[value] = 0
                # assign 1 to the column corresponding to that value
                for value in pred_values:
                    df.loc[df[pred_column] == value, value] = 1
                #eliminate old test column and take columns that are one-hot encoded test
                del df[pred_column]
                target_column = df[pred_values]
                target_column_name = pred_values
                df.drop(pred_values, axis=1, inplace=True)
                target_column = target_column.join(df['remaining_time'])
        else:
            # event level - in test column you have the last numeric value or vector for string
            if np.issubdtype(df[pred_column], np.number):
                column_type = 'Numeric'
                # if a number you want to discover the final value of the attribute (ex invoice amount)
                df_last_attribute = df.groupby(df.columns[0]).agg(['last'])[pred_column].reset_index()
                target_column = df[df.columns[0]].map(df_last_attribute.set_index(df.columns[0])['last'])
                #if you don't add y to the name you already have the same column in x, when you add the y-column after normalizing
                target_column_name = 'Y_COLUMN_' + pred_column
                target_column = pd.concat([target_column, df['remaining_time']], axis=1)
            else:
                # you want to discover all possible values (ex if a certain activity will be performed)
                column_type = 'Categorical'
                pred_values = df[pred_column].unique()
                # add one more test column for every value to be predicted
                for value in pred_values:
                    df[value] = 0
                # assign 1 to the columns that will be encountered during the case (ex: different activities)
                case_ids = df[df.columns[0]].unique()
                df = label_target_columns(df, case_ids, pred_column)
                target_column = df[pred_values]
                target_column_name = pred_values
                df.drop(pred_values, axis=1, inplace=True)
                target_column = target_column.join(df['remaining_time'])

    elif pred_column == 'remaining_time' and model is None:
        column_type = 'Numeric'
        event_level = 1
        df = df[df.loc[:, 'remaining_time'] != 0].reset_index(drop=True)
        target_column = df.loc[:, 'remaining_time'].reset_index(drop=True)
        target_column_name = 'remaining_time'
        del df[target_column_name]
    else:
        # case when you have true data to be predicted (running cases)
        #TODO assicurati di avere tutte le colonne e le train shape per tutti i dataset
        type_test = json.load(open("model/" + row_process_name + "_column_names.json"))['test']
        target_column_name = json.load(open("model/" + row_process_name + "_column_names.json"))['y_columns']
        #clean name (you have this when you save event-level column)
        cleaned_names = []
        for name in target_column_name:
            cleaned_names.append(name.replace('Y_COLUMN_', ''))
        target_column_name = cleaned_names
        #we don't have target column in true test
        target_column = None
        #if attribute is of type case remove it from the dataset (in the train it hasn't been seen, but here you have it)
        if type_test == 'case':
            del df[pred_column]
        del df['remaining_time']

    #eliminate rows where remaining time = 0 (nothing to predict) - only in train
    if model is None and pred_column != 'remaining_time':
        df = df[df.loc[:, 'remaining_time'] != 0].reset_index(drop=True)
        target_column = target_column[target_column.loc[:, 'remaining_time'] != 0].reset_index(drop=True)
        del df['remaining_time']
        del target_column['remaining_time']

    # one-hot encoding(except for case id column and remaining time column)
    df = one_hot_encoding(df)
    df = normalize_data(df, target_column, target_column_name, model)

    # around 2/3 cases are the training set, 1/3 is the test set
    # use median since you have no more cases with only one event (distribution changed)
    #convert case id series to string and cut spaces if we have dirty data (both int and string)
    df.iloc[:, 0] = df.iloc[:, 0].apply(str)
    df.iloc[:, 0] = df.iloc[:, 0].apply(lambda x: x.strip())
    test_case_ids = df.iloc[:, 0]
    # convert back to int in order to use median (if it is a string convert to categoric int values)
    try:
        df.iloc[:, 0] = pd.to_numeric(df.iloc[:, 0])
    except ValueError:
        df.iloc[:, 0] = pd.Series(df.iloc[:, 0]).astype('category').cat.codes.values
    if model is None:
        X_train, y_train, X_test, y_test, test_case_ids, class_weights, feature_columns = generate_train_and_test_sets(df, test_case_ids, target_column_name, event_level, column_type, row_process_name, model)
        return X_train, y_train, X_test, y_test, test_case_ids, target_column_name, event_level, column_type, class_weights, feature_columns
    else:
        df = pad_columns_in_real_data(df.iloc[:, 1:], df.iloc[:, 0], row_process_name)
        #here we have only the true running cases to predict
        feature_columns = df.columns[1:]
        X_test = generate_sequences_for_lstm(df.values, model)
        return X_test, test_case_ids, target_column_name, feature_columns
