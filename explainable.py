import numpy as np
import pandas as pd
import shap
import os, json
import matplotlib.pyplot as plt
import seaborn as sns

def plot_histogram(explanation_histogram, experiment_name, index_name=None):
    # Fixing random state for reproducibility
    plt.rcdefaults()
    fig, ax = plt.subplots(figsize=(15, 15))

    # Example data
    shap_columns = explanation_histogram.keys()
    y_pos = np.arange(len(shap_columns))
    values = np.array(list(explanation_histogram.values()))
    error = np.random.rand(len(shap_columns))

    ix_pos = values > 0

    ax.barh(y_pos[ix_pos], values[ix_pos], xerr=error[ix_pos], align='center', color='r')
    ax.barh(y_pos[~ix_pos], values[~ix_pos], xerr=error[~ix_pos], align='center', color='b')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(shap_columns)
    ax.invert_yaxis()  # labels read top-to-bottom
    plt.setp(ax.get_yticklabels(), fontsize=12, fontweight="bold")
    plt.setp(ax.get_xticklabels(), fontsize=12, fontweight="bold")
    ax.set_title('How are predictions going?', fontsize=22)
    if index_name is None:
        plt.savefig(experiment_name + "/plots/shap_histogram.png", dpi=300, bbox_inches="tight")
    else:
        plt.savefig(experiment_name + f"/plots/shap_histogram_{index_name}.png", dpi=300, bbox_inches="tight")


def plot_heatmap(explanation_histogram, experiment_name, index_name=None):
    df = pd.DataFrame(explanation_histogram)
    df = df.set_index(["feature", "ts"]).unstack()
    df["sort"] = df.abs().max(axis=1)
    #bigger values are on top
    df = df.sort_values("sort", ascending=False).drop("sort", axis=1)
    df.columns.rename(["#", "Timesteps"], inplace=True)
    df = df.iloc[:15]
    fig, ax = plt.subplots(figsize=(40, 15))
    heatmap = sns.heatmap(df, cbar=True, cmap="RdBu_r", center=0, robust=True, annot=True, fmt='g', ax=ax, annot_kws={"fontsize": 13})
    heatmap.set_yticklabels(heatmap.get_ymajorticklabels(), fontsize=13)
    heatmap.set_xticklabels(heatmap.get_xmajorticklabels(), fontsize=13)

    if index_name is None:
        plt.savefig(experiment_name + "/plots/shap_heatmap.png", dpi=300, bbox_inches="tight")
    else:
        plt.savefig(experiment_name + f"/plots/shap_heatmap_{index_name}.png", dpi=300, bbox_inches="tight")

def refine_explanation_name(x_test_instance, explanation_index, explanation_name, timestep):
    if x_test_instance[timestep][explanation_index] == 0:
        if "=" in explanation_name:
            # if column=something it remains as it is
            explanation_name = explanation_name.replace("=", "!=")
    if "=" not in explanation_name:
        #numerical explanation
        if x_test_instance[timestep][explanation_index] <= 0.5:
            explanation_name = "Low value of " + explanation_name
        else:
            explanation_name = "High value of " + explanation_name
    return explanation_name

def add_explanation_to_histogram(x_test_instance, shap_values_instance, feature_columns, shapley_value, i, explanation_histogram, num_events, columns_info, timestep_histogram):
    # take the column name for every explanation
    explanation_index = np.where(shap_values_instance == shapley_value)[1][0]
    timestep = np.where(shap_values_instance == shapley_value)[0][0]
    explanation_name = feature_columns[explanation_index]
    explanation = {}
    # add timestep information to the timestep - only if it is not of type case - check before name refinement
    if (num_events - (timestep + 1)) != 0 and (columns_info[explanation_name] == "event"):
        explanation_name = refine_explanation_name(x_test_instance, explanation_index, explanation_name, timestep)
        print("Instance {} Timestep: {}/{} --> {} : {}".format(i + 1, timestep + 1, num_events, explanation_name, shapley_value))
        explanation["feature"] = explanation_name
        explanation["ts"] = (num_events - (timestep + 1))
        if f"#-{num_events - (timestep + 1)}" not in timestep_histogram:
            timestep_histogram[f"#-{num_events - (timestep + 1)}"] = 1
        else:
            timestep_histogram[f"#-{num_events - (timestep + 1)}"] += 1
    else:
        explanation_name = refine_explanation_name(x_test_instance, explanation_index, explanation_name, timestep)
        explanation["feature"] = explanation_name
        explanation["ts"] = 0
        print("Instance {} --> {} : {}".format(i + 1, explanation_name, shapley_value))
        if "#0" not in timestep_histogram:
            timestep_histogram["#0"] = 1
        else:
            timestep_histogram["#0"] += 1

    # if the categorical feature has a 1 means that the particular instance happened (see the contribution of the shapley values)
    found = 0
    for i, explanation_dict in enumerate(explanation_histogram):
        if (explanation_dict["feature"] == explanation_name) and (explanation_dict["ts"] == explanation["ts"]):
            if shapley_value > 0:
                explanation["#"] = explanation_dict["#"] + 1
            else:
                explanation["#"] = explanation_dict["#"] - 1
            explanation_histogram[i] = explanation
            found = 1
            break
    if found == 0:
        if shapley_value > 0:
            explanation["#"] = 1
        else:
            explanation["#"] = -1
        explanation_histogram.append(explanation)

    return explanation_histogram, timestep_histogram


def calculate_histogram_for_shap_chunk(X_test, shapley_chunk, index_name, index, current_case_id, num_events,
                                       feature_columns, explanation_histograms, df, columns_info, timestep_histograms):
    # if we haven't still created an histogram with that index name create it, otherwise update it from where you left
    if index_name not in explanation_histograms:
        explanation_histogram = []
        timestep_histogram = {}
    else:
        explanation_histogram = explanation_histograms[index_name]
        timestep_histogram = timestep_histograms[index_name]
    for i in range(shapley_chunk[index].shape[0]):
        if df.loc[i, 'CASE ID'] != current_case_id:
            current_case_id = df.loc[i, 'CASE ID']
            num_events = 1
        else:
            num_events += 1
        x_test_instance, shap_values_instance, explanation_values = find_instance_explanation_values(X_test, shapley_chunk, i, num_events, index)

        for shapley_value in explanation_values:
            explanation_histogram, timestep_histogram = add_explanation_to_histogram(x_test_instance, shap_values_instance, feature_columns,
                                                                 shapley_value, i, explanation_histogram, num_events, columns_info, timestep_histogram)
    # you have updated the histogram with the correspondent index name, put it back where it was
    explanation_histograms[index_name] = explanation_histogram
    timestep_histograms[index_name] = timestep_histogram
    #current case id and num_events must be returned because you could be in the middle of a case when you do another chunk
    return explanation_histograms, current_case_id, num_events, timestep_histograms

def find_instance_explanation_values(X_test, shapley_test, i, num_events, prediction_index=0, mode="train"):
    #mean and std dev are now done on a matrix of all timesteps (up to n events of that case)
    x_test_instance = X_test[i][-num_events:]
    shap_values_instance = shapley_test[prediction_index][i][-num_events:]
    mean = shap_values_instance[np.nonzero(shap_values_instance)].mean()
    sd = shap_values_instance[np.nonzero(shap_values_instance)].std()

    if mode == "train":
        upper_threshold = mean + 3 * sd
        lower_threshold = mean - 3 * sd
    else:
        upper_threshold = mean + 4 * sd
        lower_threshold = mean - 4 * sd

    # calculate most important values for explanation (+-3sd from the mean)
    explanation_values = shap_values_instance[(shap_values_instance > upper_threshold) | (shap_values_instance < lower_threshold)]
    # order shapley values for decrescent importance
    explanation_values = sorted(explanation_values, key=abs, reverse=True)
    return x_test_instance, shap_values_instance, explanation_values

def reduce_instances_for_shap(X_test, df):
    partition = int(len(X_test) / 8)
    for i in range(8):
        if i == 0:
            X_test_reduced = X_test[:5000]
            df_reduced = df.iloc[:5000, :]
        else:
            X_test_reduced = np.concatenate((X_test_reduced, X_test[partition * i:(partition * i + 5000)]))
            df_reduced = pd.concat([df_reduced, df.iloc[partition * i:(partition * i + 5000):, :]], ignore_index=True)
    return X_test_reduced, df_reduced

def compute_shap_values(df, experiment_name, X_train, X_test, model, column_type,
                         feature_columns, indexes_to_plot, activities_to_plot, pred_column):
    #X_test_reduced, df_reduced = reduce_instances_for_shap(X_test, df)
    X_test_reduced = X_test
    df_reduced = df
    if column_type == 'Numeric':
        # first run you compute the values and then save them
        if not os.path.isfile(experiment_name + "/shap/background.npy"):
            background = X_train[:100]
            explainer = shap.DeepExplainer(model, background)
            print("prepared explainer")
            # take only 40000 instances in 8 different chunks for performance reasons
            shapley_test = explainer.shap_values(X_test_reduced)
            np.save(experiment_name + "/shap/shapley_values_test.npy", shapley_test)
            np.save(experiment_name + "/shap/background", background)
            calculate_histogram_for_shap_values(df_reduced, column_type, X_test_reduced, shapley_test,
                                                feature_columns, experiment_name, pred_column)
        else:
            shapley_test = np.load(experiment_name + "/shap/shapley_values_test.npy", allow_pickle=True)
            calculate_histogram_for_shap_values(df_reduced, column_type, X_test_reduced, shapley_test,
                                                feature_columns, experiment_name, pred_column)
    else:
        # column of type categorical (multioutput) need data to be splitted (huge amount of memory)
        partition = int(len(X_test) / 5)
        explanation_histograms = {}
        timestep_histograms = {}
        if not os.path.isfile(experiment_name + "/shap/background.npy"):
            background = X_train[:100]
            explainer = shap.DeepExplainer(model, background)
            print("prepared explainer")

        current_case_id = 0
        num_events = 0
        columns_info = json.load(open(experiment_name + "/model/data_info.json"))["columns_info"]

        for i in range(5):
            if not os.path.isfile(experiment_name + "/shap/background.npy"):
                shapley_chunk = explainer.shap_values(X_test_reduced[partition * i:partition * (i + 1)])
                np.save(experiment_name + "/shap/shapley_values_test_" + str(i) + ".npy", shapley_chunk)
            else:
                shapley_chunk = np.load(experiment_name + "/shap/shapley_values_test_" + str(i) + ".npy", allow_pickle=True)

            # extract only the shapley values for the attributes that you want to plot
            for j, index in enumerate(indexes_to_plot):
                index_name = activities_to_plot[j]
                explanation_histograms, current_case_id, num_events, timestep_histograms = calculate_histogram_for_shap_chunk(X_test_reduced[partition * i:partition * (i + 1)],
                                                                                shapley_chunk, index_name, index, current_case_id, num_events,
                                                                                feature_columns, explanation_histograms,
                                                                                df_reduced.loc[partition * i:partition * (i + 1), :].reset_index(drop=True), columns_info, timestep_histograms)
        if not os.path.isfile(experiment_name + "/shap/background.npy"):
            np.save(experiment_name + "/shap/background", background)
        # order results of the histograms by value and plot them
        for index_name in activities_to_plot:
            explanation_histogram = explanation_histograms[index_name]
            timestep_histogram = timestep_histograms[index_name]
            timestep_histogram = {k: v for k, v in sorted(timestep_histogram.items(), key=lambda item: item[0])}
            # explanation_histogram = {k: v for k, v in sorted(explanation_histogram.items(), key=lambda item: abs(item[1]),
            #                                 reverse=True)}
            # essential_histogram = {}
            # for i, key in enumerate(explanation_histogram.keys()):
            #     if i == 30:
            #         break
            #     else:
            #         essential_histogram[key] = explanation_histogram[key]
            with open(experiment_name + f"/shap/timestep_histogram_{index_name}.json", "w") as json_file:
                json.dump(timestep_histogram, json_file)
            with open(experiment_name + f"/shap/shap_heatmap_{index_name}.json", "w") as json_file:
                json.dump(explanation_histogram, json_file, default=convert)
            plot_heatmap(explanation_histogram, experiment_name, index_name)
            #plot_histogram(essential_histogram, experiment_name, index_name)

def convert(obj):
    """function needed because numpy int is not recognized by the serializer"""
    if isinstance(obj, np.int64): return int(obj)
    raise TypeError

def calculate_histogram_for_shap_values(df, column_type, X_test, shapley_test, feature_columns,
                                        experiment_name, pred_column):
    timestep_histogram = {}
    explanation_histogram = []
    # if column numeric if the predictions goes too far from the avg real data discard the prediction from the shap graph
    if column_type == 'Numeric':
        true_mean = df['TEST'].mean()
        prediction_out_of_range_high = df.loc[df[pred_column] > (df['TEST'] + true_mean), :].index
        prediction_out_of_range_low = df.loc[df[pred_column] < (df['TEST'] - true_mean), :].index
    #now the df has random cases for test you need to realign indexes
    df.reset_index(drop=True, inplace=True)
    current_case_id = 0
    num_events = 0
    columns_info = json.load(open(experiment_name + "/model/data_info.json"))["columns_info"]
    for i in range(len(shapley_test[0])):
        # if the prediction is wrong don't consider the explanation
        if column_type == 'Numeric':
            if i in prediction_out_of_range_high or i in prediction_out_of_range_low:
                continue

        #we need to keep track of how many events have occurred in order the exact number of timesteps
        if df.loc[i, 'CASE ID'] != current_case_id:
            current_case_id = df.loc[i, 'CASE ID']
            num_events = 1
        else:
            num_events += 1

        x_test_instance, shap_values_instance, explanation_values = find_instance_explanation_values(X_test, shapley_test, i, num_events)
        for shapley_value in explanation_values:
            explanation_histogram, timestep_histogram = add_explanation_to_histogram(x_test_instance, shap_values_instance, feature_columns,
                                                                 shapley_value, i, explanation_histogram, num_events, columns_info, timestep_histogram)

    # order results of the histogram by value (python >= 3.6) and take only the first 15
    timestep_histogram = {k: v for k, v in sorted(timestep_histogram.items(), key=lambda item: item[0])}
    #explanation_histogram = {k: v for k, v in
                             #sorted(explanation_histogram.items(), key=lambda item: abs(item[1]), reverse=True)}
    # essential_histogram = {}
    # for i, key in enumerate(explanation_histogram.keys()):
    #     if i == 30:
    #         break
    #     else:
    #         essential_histogram[key] = explanation_histogram[key]
    with open(experiment_name + "/shap/shap_heatmap.json", "w") as json_file:
        json.dump(explanation_histogram, json_file, default=convert)
    with open(experiment_name + "/shap/timestep_histogram.json", "w") as json_file:
        json.dump(timestep_histogram, json_file)
    plot_heatmap(explanation_histogram, experiment_name)
    # plot_histogram(essential_histogram, experiment_name)


def compute_shap_values_for_running_cases(experiment_name, X_test, model):
    # you should already have the background and you shouldn't need no partition
    background = np.load(experiment_name + "/shap/background.npy", allow_pickle=True)
    explainer = shap.DeepExplainer(model, background)
    # save shap just for faster debugging
    if not os.path.isfile(experiment_name + "/shap/shapley_values_running.npy"):
        shapley_test = explainer.shap_values(X_test)
        np.save(experiment_name + "/shap/shapley_values_running.npy", shapley_test)
    else:
        shapley_test = np.load(experiment_name + "/shap/shapley_values_running.npy")
    return shapley_test

def add_explanation(explainable_response, explanation_name):
    if explainable_response == "":
        explainable_response = explanation_name
    else:
        explainable_response += " AND " + explanation_name
    return explainable_response

def find_explanation_for_categorical_prediction(df, explanation_name, explainable_response_positive,
                                                explainable_response_negative, pred_attribute,
                                                shapley_value, i):
    # in categorical case attribute prediction can be 0 or 1 (attribute will be performed or not)
    attribute_prediction = df.loc[i, pred_attribute]
    if attribute_prediction == 1:
        # makes sense to focus to keep only positive shap values (those who say attribute will be performed)
        if shapley_value > 0:
            explainable_response_positive = add_explanation(explainable_response_positive, explanation_name)
    else:
        if shapley_value < 0:
            explainable_response_negative = add_explanation(explainable_response_negative, explanation_name)
    return explainable_response_positive, explainable_response_negative


def find_explanations_for_running_cases(shapley_test, X_test, df, feature_columns, pred_attributes, column_type, num_events, experiment_name, mode):
    if column_type == 'Categorical':
        # round results between 0 and 1 (activity will be performed or not)
        df.loc[:, pred_attributes] = df.loc[:, pred_attributes].round(0)
    #reload case/event column information for explanations
    columns_info = json.load(open(experiment_name + "/model/data_info.json"))["columns_info"]
    #for every attribute to be predicted find its explanations
    for prediction_index in range(len(pred_attributes)):
        pred_attribute = pred_attributes[prediction_index]
        if column_type != 'Categorical':
            df['Explanations for increasing ' + pred_attribute] = ''
            df['Explanations for decreasing ' + pred_attribute] = ''
        else:
            df[f'Explanations for {pred_attribute} happening'] = ''
            df[f'Explanations for {pred_attribute} not happening'] = ''
        for i in range(len(shapley_test[prediction_index])):
            x_test_instance, shap_values_instance, explanation_values = find_instance_explanation_values(X_test, shapley_test, i, num_events.iloc[i], prediction_index, mode)
            explainable_response_positive = ""
            explainable_response_negative = ""
            for shapley_value in explanation_values:
                # take the column name for every explanation
                explanation_index = np.where(shap_values_instance == shapley_value)[1][0]
                timestep = np.where(shap_values_instance == shapley_value)[0][0]
                explanation_name = feature_columns[explanation_index]

                if columns_info[explanation_name] == "case":
                    explanation_name = refine_explanation_name(x_test_instance, explanation_index, explanation_name, timestep)
                    # could be that an attribute of type case influenced also in the past, don't insert duplicates
                    if (explanation_name in explainable_response_positive) or (explanation_name in explainable_response_negative):
                        continue
                elif (num_events.iloc[i] - (timestep + 1)) == 0:
                    explanation_name = refine_explanation_name(x_test_instance, explanation_index, explanation_name, timestep)
                else:
                    explanation_name = refine_explanation_name(x_test_instance, explanation_index, explanation_name, timestep)
                    #add timestep information to the timestep - only if type event
                    explanation_name = explanation_name + f" (-{num_events.iloc[i] - (timestep + 1)})"


                if column_type != 'Categorical':
                    if shapley_value >= 0:
                        explainable_response_positive = add_explanation(explainable_response_positive, explanation_name)
                    else:
                        explainable_response_negative = add_explanation(explainable_response_negative, explanation_name)

                else:
                    explainable_response_positive, explainable_response_negative = \
                        find_explanation_for_categorical_prediction(df, explanation_name, explainable_response_positive,
                                                                    explainable_response_negative, pred_attribute, shapley_value, i)

            if column_type != 'Categorical':
                df.loc[i, 'Explanations for increasing ' + pred_attribute] = explainable_response_positive
                df.loc[i, 'Explanations for decreasing ' + pred_attribute] = explainable_response_negative
            else:
                df.loc[i, f'Explanations for {pred_attribute} happening'] = explainable_response_positive
                df.loc[i, f'Explanations for {pred_attribute} not happening'] = explainable_response_negative
    return df
