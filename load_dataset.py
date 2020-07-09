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
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from statistics import median
import os, json, random


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

    #add features: time from trace start, time from last_startdate_event, time from midnight, weekday
    activity.append((activityTimestamp - starttime).total_seconds())
    activity.append((activityTimestamp - lastevtime).total_seconds())
    activity.append(calculateTimeFromMidnight(activityTimestamp))
    activity.append(activityTimestamp.weekday())
    # if there is also end_date add features time from last_enddate_event and event_duration
    if current_activity_end_date is not None:
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
            # Possibile modifica: encodare le colonne di tipo data, o facciamo la diff da 1970 in secondi e poi normalizziamo
            # One hot encoding - eventual categorical nans will be ignored
            one_hot = pd.get_dummies(df[column], prefix=column, prefix_sep='=')
            print("Encoded column:{} - Different keys: {}".format(column, one_hot.shape[1]))
            # Drop column as it is now encoded
            df = df.drop(column, axis=1)
            # Join the encoded df
            df = df.join(one_hot)
    print("Categorical columns encoded")
    return df

def convert_strings_to_datetime(df, date_format):
    # convert string columns that contain datetime to datetime
    for column in df.columns:
        try:
            #if a number do nothing
            if np.issubdtype(df[column], np.number):
                continue
            df[column] = pd.to_datetime(df[column], format=date_format)
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

def add_features(df, end_date_position):
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
                        "weekday", "event_duration", "remaining_time"])
    else:
        columns.extend(["time_from_start", "time_from_previous_event(start)", "time_from_midnight",
                        "weekday", "remaining_time"])
    df = pd.DataFrame(traces, columns=columns)
    print("Features added")
    return df

def generate_sequences_for_lstm(dataset, mode):
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
            if mode == "train":
                for j in range(1, len(trace) + 1):
                    data.append(trace[:j])
            else:
                data.append(trace[:len(trace)])
            trace = []
            trace.append(line[1:].tolist())
    # last case
    if mode == "train":
        for i in range(1, len(trace) + 1):
            data.append(trace[:i])
    else:
        data.append(trace[:len(trace)])
    return data

def pad_columns_in_real_data(df, case_ids, experiment_name):
    #fill real test df with empty missing columns (one-hot encoding can generate much more columns in train)
    train_columns = json.load(open(experiment_name + "/model/data_info.json"))['columns']
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

def fillna(df, date_format):
    for i, column in enumerate(df.columns):
        if df[column].dtype != 'object':
            df[column] = df[column].fillna(0)
        else:
            try:
                #datetime columns have null encoded as 0
                pd.to_datetime(df[column], format=date_format)
                df[column] = df[column].fillna(0)
            # exception means it is a string
            except (ValueError, TypeError, OverflowError):
                pass
                #categorical missing values have no sense encoded as 0
                #df[column] = df[column].fillna("Not present")
    return df

def prepare_data_and_add_features(df, case_id_position, start_date_position, date_format, end_date_position):
    df = fillna(df, date_format)
    if end_date_position is not None:
        df = fill_missing_end_dates(df, start_date_position, end_date_position)
    df = convert_strings_to_datetime(df, date_format)
    df = move_essential_columns(df, case_id_position, start_date_position)
    df = sort_df(df)
    df = add_features(df, end_date_position)
    df["weekday"].replace({0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday", 4: "Friday", 5: "Saturday", 6: "Sunday"}, inplace=True)
    return df

def normalize_data(df):
    # normalize data (except for case id column and test column)
    case_column = df.iloc[:, 0].reset_index(drop=True)
    case_column_name = df.columns[0]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(df.iloc[:, 1:].values)
    columns = [x for x in df.columns[1:]]
    df = pd.DataFrame(scaled, columns=columns)
    # reinsert case id column and target column
    df.insert(0, case_column_name, case_column)
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

def save_case_event_column_type(X_train, df, info, target_column_name):
    #function to understand if a particular column is of type case/event (needed for explanations later - if case no timestep info)
    if not type(target_column_name) == np.str:
        feature_columns = df.iloc[:, 1:-len(list(target_column_name))]
    else:
        feature_columns = df.iloc[:, 1:-1]
    column_types = {}
    for column in feature_columns:
        event_level = 0
        case_ids = df[df.columns[0]].unique()
        case_ids = case_ids[:int((len(case_ids) / 100))]
        for case in case_ids:
            df_reduced = df[df[df.columns[0]] == case]
            if len(df_reduced[column].unique()) == 1:
                continue
            else:
                event_level = 1
                break
        if event_level == 0:
            column_types[column] = "case"
        else:
            column_types[column] = "event"
    ohe_types = {}
    for column in X_train.columns[1:]:
        for key in column_types:
            if key in column:
                ohe_types[column] = column_types[key]
        pass
    info['columns_info'] = ohe_types
    return info


def save_column_information_for_real_predictions(X_train, dfTrain, experiment_name, event_level, target_column_name):
    # save columns names (except for case id column and test columns) and event/case information the attributes in a file
    # you need this for the shape when you have to predict real data and for the explanations
    # save also case indexes you have randomly taken (in case you want just to reload the model and work on the explanation part)
    info = {}
    with open(experiment_name + "/model/data_info.json", "w") as json_file:
        info["case_indexes"] = X_train[X_train.columns[0]].unique().tolist()
        if event_level == 0:
            if not type(target_column_name) == np.str:
                info['columns'] = X_train.iloc[:, 1:].columns.to_list()
                info['test'] = 'case'
                info['y_columns'] = list(target_column_name)
                #json.dump({'columns': dfTrain.iloc[:, 1:-len(list(target_column_name))].columns.to_list(), 'test': 'case', 'y_columns': list(target_column_name)}, json_file)
            else:
                info['columns'] = X_train.iloc[:, 1:].columns.to_list()
                info['test'] = 'case'
                info['y_columns'] = [target_column_name]
                #json.dump({'columns': dfTrain.iloc[:, 1:-1].columns.to_list(), 'test': 'case', 'y_columns': [target_column_name]}, json_file)
        else:
            if not type(target_column_name) == np.str:
                info['columns'] = X_train.iloc[:, 1:].columns.to_list()
                info['test'] = 'event'
                info['y_columns'] = list(target_column_name)
                #json.dump({'columns': dfTrain.iloc[:, 1:-len(list(target_column_name))].columns.to_list(), 'test': 'event', 'y_columns': list(target_column_name)}, json_file)
            else:
                info['columns'] = X_train.iloc[:, 1:].columns.to_list()
                info['test'] = 'event'
                info['y_columns'] = [target_column_name]
                #json.dump({'columns': dfTrain.iloc[:, 1:-1].columns.to_list(), 'test': 'event', 'y_columns': [target_column_name]}, json_file)
        info = save_case_event_column_type(X_train, dfTrain, info, target_column_name)
        json.dump(info, json_file)


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

def generate_train_and_test_sets(df, target_column, target_column_name, event_level, column_type, experiment_name, mode, override):
    #reattach predict column before splitting
    # if it is a string the target column is one (otherwise it would cycle on the letters of the column)
    if not type(target_column_name) == np.str:
        for i, name in enumerate(target_column_name):
            df[name] = target_column[target_column_name[i]]
    else:
        df[target_column_name] = target_column

    # around 2/3 cases are the training set, 1/3 is the test set
    # use median since you have no more cases with only one event (distribution changed)
    second_quartile = median(np.unique(df.iloc[:, 0]))
    third_quartile = median(np.unique(df[df.iloc[:, 0] > second_quartile].iloc[:, 0]))

    #if trained model exists just pick the previously chosen case indexes
    if os.path.exists(experiment_name + "/model/model_100_8.json") and override is False:
        case_indexes = json.load(open(experiment_name + "/model/data_info.json"))["case_indexes"]
        print("Reloaded train cases")
        dfTrain = df[df[df.columns[0]].isin(case_indexes)]
        dfTest = df[~df[df.columns[0]].isin(case_indexes)]
    else:
        #take cases for training in random order
        cases = df[df.columns[0]].unique()
        number_train_cases = len(df[df.iloc[:, 0] < ((second_quartile + third_quartile) / 2)][df.columns[0]].unique())
        train_cases = random.sample(cases.tolist(), number_train_cases)
        dfTrain = df[df[df.columns[0]].isin(train_cases)]
        dfTest = df[~df[df.columns[0]].isin(train_cases)]

    # dfTrain = df[df.iloc[:, 0] < ((second_quartile + third_quartile) / 2)]
    # dfTest = df[df.iloc[:, 0] >= ((second_quartile + third_quartile) / 2)]

    if column_type == 'Categorical':
        class_weights = create_class_weights(dfTrain, target_column_name)
    else:
        class_weights = None

    # separate target variable (if a string there is only one y_column)
    if not type(target_column_name) == np.str:
        y_train = dfTrain.iloc[:, -len(list(target_column_name)):]
        y_test = dfTest.iloc[:, -len(list(target_column_name)):]
        X_train = dfTrain.iloc[:, :-len(list(target_column_name))]
        X_test = dfTest.iloc[:, :-len(list(target_column_name))]
    else:
        y_train = dfTrain.iloc[:, -1]
        y_test = dfTest.iloc[:, -1]
        X_train = dfTrain.iloc[:, :-1]
        X_test = dfTest.iloc[:, :-1]
    # retrieve test case id to later show predictions
    test_case_ids = dfTest.iloc[:, 0]

    X_train = one_hot_encoding(X_train)
    X_train = normalize_data(X_train)
    X_test = one_hot_encoding(X_test)
    X_test = normalize_data(X_test)

    #test columns should be the same as train
    columns_not_in_test = [x for x in X_train.columns if x not in X_test.columns]
    df2 = pd.DataFrame(columns=columns_not_in_test)
    # enrich test df with missing columns seen in train
    X_test = pd.concat([X_test, df2], axis=1)
    X_test = X_test.fillna(0)
    # reorder data as in train
    X_test = X_test[X_train.columns]

    if not os.path.exists(experiment_name + "/model/model_100_8.json") or override is True:
        save_column_information_for_real_predictions(X_train, dfTrain, experiment_name, event_level, target_column_name)

    # create sequences (only for lstm)
    if not type(target_column_name) == np.str:
        feature_columns = X_train.columns[1:]
        X_train = generate_sequences_for_lstm(X_train.values, mode)
        X_test = generate_sequences_for_lstm(X_test.values, mode)
        # X_train = generate_sequences_for_lstm(dfTrain.iloc[:, :-len(list(target_column_name))].values, mode)
        # X_test = generate_sequences_for_lstm(dfTest.iloc[:, :-len(list(target_column_name))].values, mode)
    else:
        feature_columns = X_train.columns[1:]
        X_train = generate_sequences_for_lstm(X_train.values, mode)
        X_test = generate_sequences_for_lstm(X_test.values, mode)
        # X_train = generate_sequences_for_lstm(dfTrain.iloc[:, :-1].values, mode)
        # X_test = generate_sequences_for_lstm(dfTest.iloc[:, :-1].values, mode)
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


def prepare_dataset(df, experiment_name, case_id_position, start_date_position, date_format, end_date_position,
                    pred_column, mode, override=False, pred_attributes=None):
    df = prepare_data_and_add_features(df, case_id_position, start_date_position, date_format, end_date_position)
    df = convert_datetime_columns_to_seconds(df)
    # if target column != remaining time exclude target column
    if pred_column != 'remaining_time' and mode == "train":
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
                # add one more test column for every value to be predicted
                for value in pred_attributes:
                    df[value] = 0
                # assign 1 to the column corresponding to that value
                for value in pred_attributes:
                    df.loc[df[pred_column] == value, value] = 1
                #eliminate old test column and take columns that are one-hot encoded test
                del df[pred_column]
                target_column = df[pred_attributes]
                target_column_name = pred_attributes
                df.drop(pred_attributes, axis=1, inplace=True)
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
                # we take only the columns we are interested in
                target_column = df[pred_attributes]
                target_column_name = pred_attributes
                df.drop(pred_values, axis=1, inplace=True)
                target_column = target_column.join(df['remaining_time'])

    elif pred_column == 'remaining_time' and mode == "train":
        column_type = 'Numeric'
        event_level = 1
        df = df[df.loc[:, 'remaining_time'] != 0].reset_index(drop=True)
        target_column = df.loc[:, 'remaining_time'].reset_index(drop=True)
        target_column_name = 'remaining_time'
        del df[target_column_name]
    else:
        # case when you have true data to be predicted (running cases)
        type_test = json.load(open(experiment_name + "/model/data_info.json"))['test']
        target_column_name = json.load(open(experiment_name + "/model/data_info.json"))['y_columns']
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
    if mode == "train" and pred_column != 'remaining_time':
        df = df[df.loc[:, 'remaining_time'] != 0].reset_index(drop=True)
        target_column = target_column[target_column.loc[:, 'remaining_time'] != 0].reset_index(drop=True)
        del df['remaining_time']
        del target_column['remaining_time']

    # one-hot encoding(except for case id column and remaining time column) - for train we need to do that separately for train and test
    if mode == "predict":
        df = one_hot_encoding(df)
        df = normalize_data(df)

    #convert case id series to string and cut spaces if we have dirty data (both int and string)
    df.iloc[:, 0] = df.iloc[:, 0].apply(str)
    df.iloc[:, 0] = df.iloc[:, 0].apply(lambda x: x.strip())
    test_case_ids = df.iloc[:, 0]
    # convert back to int in order to use median (if it is a string convert to categoric int values)
    try:
        df.iloc[:, 0] = pd.to_numeric(df.iloc[:, 0])
    except ValueError:
        df.iloc[:, 0] = pd.Series(df.iloc[:, 0]).astype('category').cat.codes.values
    if mode == "train":
        X_train, y_train, X_test, y_test, test_case_ids, class_weights, feature_columns = generate_train_and_test_sets(df, target_column, target_column_name,
                                                                                                    event_level, column_type, experiment_name, mode, override)
        return X_train, y_train, X_test, y_test, test_case_ids, target_column_name, event_level, column_type, class_weights, feature_columns
    else:
        df = pad_columns_in_real_data(df.iloc[:, 1:], df.iloc[:, 0], experiment_name)
        #here we have only the true running cases to predict
        feature_columns = df.columns[1:]
        #for each running case understand how many events you have (with two events you can consider also the explanations from the previous timestep)
        num_events = (df.groupby(df.columns[0]).count()["time_from_start"]).reset_index(drop=True)
        X_test = generate_sequences_for_lstm(df.values, mode)
        return X_test, test_case_ids, target_column_name, feature_columns, num_events
