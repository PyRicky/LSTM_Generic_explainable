import numpy as np
import shap
import os, json
import matplotlib.pyplot as plt


def plot_histogram(explanation_histogram, row_process_name):
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
    plt.savefig("plots/shap_histogram_"+row_process_name+".png", dpi=300, bbox_inches="tight")

def refine_explanation_name(x_test_instance, explanation_index, explanation_name):
    if x_test_instance[explanation_index] == 0:
        if "=" in explanation_name:
            # if column=something it remains as it is
            explanation_name = explanation_name.replace("=", "!=")
    if "=" not in explanation_name:
        #numerical explanation
        if x_test_instance[explanation_index] <= 0.5:
            explanation_name = "Low value of " + explanation_name
        else:
            explanation_name = "High value of " + explanation_name
    return explanation_name

def add_explanation_to_histogram(x_test_instance, shap_values_instance, feature_columns, value, i, explanation_histogram):
    # take the column name for every explanation
    explanation_index = np.where(shap_values_instance == value)[0][0]
    explanation_name = feature_columns[explanation_index]
    explanation_name = refine_explanation_name(x_test_instance, explanation_index, explanation_name)
    print("Instance {} --> {} : {}".format(i + 1, explanation_name, value))
    # if the categorical feature has a 1 means that the particular instance happened (see the contribution of the shapley values)
    if explanation_name not in explanation_histogram:
        if value > 0:
            explanation_histogram[explanation_name] = 1
        else:
            explanation_histogram[explanation_name] = -1
    else:
        if value > 0:
            explanation_histogram[explanation_name] += 1
        else:
            explanation_histogram[explanation_name] -= 1
    return explanation_histogram


def calculate_histogram_for_shap_chunk(X_test, shapley_index_chunk, index, feature_columns, explanation_histograms):
    # if we haven't still created an histogram with that index name create it, otherwise update it from where you left
    if index not in explanation_histograms:
        explanation_histogram = {}
    else:
        explanation_histogram = explanation_histograms[index]
    for i in range(shapley_index_chunk.shape[0]):
        # TODO test needed
        x_test_instance, shap_values_instance, explanation_values = find_instance_explanation_values(X_test, shapley_index_chunk, i)

        for value in explanation_values:
            # TODO test needed
            explanation_histogram = add_explanation_to_histogram(x_test_instance, shap_values_instance, feature_columns,
                                                                 value, i, explanation_histogram)
    # you have updated the histogram, put it back where it was
    explanation_histograms[index] = explanation_histogram
    return explanation_histograms

def find_instance_explanation_values(X_test, shapley_test, i, prediction_index=0, more_explanation_values=False):
    x_test_instance = X_test[i][-1]
    shap_values_instance = shapley_test[prediction_index][i][-1]
    mean = np.mean([x for x in shap_values_instance if x != 0])
    sd = np.std([x for x in shap_values_instance if x != 0])
    upper_threshold = mean + 3 * sd
    lower_threshold = mean - 3 * sd
    # calculate most important values for explanation (+-3sd from the mean)
    explanation_values = [x for x in shap_values_instance if x > upper_threshold or x < lower_threshold]
    if len(explanation_values) == 0 or more_explanation_values is True:
        upper_threshold = mean + 2 * sd
        lower_threshold = mean - 2 * sd
        explanation_values = [x for x in shap_values_instance if x > upper_threshold or x < lower_threshold]
        if len(explanation_values) == 0 or more_explanation_values is True:
            upper_threshold = mean + sd
            lower_threshold = mean - sd
            explanation_values = [x for x in shap_values_instance if x > upper_threshold or x < lower_threshold]
    # if they are more take only the first 5
    if len(explanation_values) > 5 and more_explanation_values is False:
        explanation_values = explanation_values[:5]
    return x_test_instance, shap_values_instance, explanation_values


def compute_shap_values(df, target_column_name, row_process_name, X_train, X_test, model, column_type,
                         feature_columns, indexes_to_plot, activities_to_plot, pred_column):
    if column_type == 'Numeric':
        # first run you compute the values and then save them
        if not os.path.isfile("shap/" + row_process_name + "_background.npy"):
            background = X_train[:100]
            explainer = shap.DeepExplainer(model, background)
            print("prepared explainer")
            shapley_test = explainer.shap_values(X_test)
            np.save("shap/" + row_process_name + "_shap_values_test.npy", shapley_test)
            np.save("shap/" + row_process_name + "_background", background)
            calculate_histogram_for_shap_values(df, target_column_name, column_type, X_test, shapley_test,
                                                feature_columns, row_process_name, pred_column)
        else:
            shapley_test = np.load("shap/" + row_process_name + "_shap_values_test.npy", allow_pickle=True)
            calculate_histogram_for_shap_values(df, target_column_name, column_type, X_test, shapley_test,
                                                feature_columns, row_process_name, pred_column)
    else:
        # column of type categorical (multioutput) need data to be splitted (huge amount of memory)
        partition = int(len(X_test) / 25)
        explanation_histograms = {}
        if not os.path.isfile("shap/" + row_process_name + "_background.npy"):
            background = X_train[:100]
            explainer = shap.DeepExplainer(model, background)
            print("prepared explainer")
        for i in range(25):
            if not os.path.isfile("shap/" + row_process_name + "_background.npy"):
                if i != 24:
                    shapley_test = explainer.shap_values(X_test[partition * i:partition * (i + 1)])
                else:
                    shapley_test = explainer.shap_values(X_test[partition * i:])
                np.save("shap/" + row_process_name + "_shap_values_test_" + str(i) + ".npy", shapley_test)
            else:
                shapley_test = np.load("shap/" + row_process_name + "_shap_values_test_" + str(i) + ".npy", allow_pickle=True)

            # extract only the shapley values for the attributes that you want to plot
            for j, index in enumerate(indexes_to_plot):
                # TODO: test this line with few instances
                shapley_index_chunk = shapley_test[index, :, :, :]
                index_name = activities_to_plot[j]
                if i != 24:
                    explanation_histograms = calculate_histogram_for_shap_chunk(X_test[partition * i:partition * (i + 1)],
                                                                                shapley_index_chunk, index_name,
                                                                                feature_columns, explanation_histograms)
                else:
                    explanation_histograms = calculate_histogram_for_shap_chunk(X_test[partition * i:],
                                                                                shapley_index_chunk, index_name,
                                                                                feature_columns, explanation_histograms)
        if not os.path.isfile("shap/" + row_process_name + "_background.npy"):
            np.save("shap/" + row_process_name + "_background", background)
        # order results of the histograms by value and plot them
        for index_name in activities_to_plot:
            explanation_histogram = explanation_histograms[index_name]
            explanation_histogram = {k: v for k, v in sorted(explanation_histogram.items(), key=lambda item: abs(item[1]),
                                            reverse=True)}
            # TODO: test if the name is correct and report it on the shap files
            row_process_name = row_process_name + '_' + index_name
            essential_histogram = {}
            for i, key in enumerate(explanation_histogram.keys()):
                if i == 15:
                    break
                else:
                    essential_histogram[key] = explanation_histogram[key]
            plot_histogram(essential_histogram, row_process_name)


def calculate_histogram_for_shap_values(df, target_column_name, column_type, X_test, shapley_test, feature_columns,
                                        row_process_name, pred_column):
    explanation_histogram = {}
    # if column numeric if the predictions goes too far from the avg real data discard the prediction from the shap graph
    if column_type == 'Numeric':
        true_mean = df[target_column_name].mean()
        prediction_out_of_range_high = df.loc[df[pred_column] > (df[target_column_name] + true_mean), :].index
        prediction_out_of_range_low = df.loc[df[pred_column] < (df['remaining_time'] - true_mean), :].index
    for i in range(len(shapley_test[0])):
        # if the prediction is wrong don't consider the explanation
        if column_type == 'Numeric':
            if i in prediction_out_of_range_high or i in prediction_out_of_range_low:
                continue

        x_test_instance, shap_values_instance, explanation_values = find_instance_explanation_values(X_test, shapley_test, i)
        for value in explanation_values:
            explanation_histogram = add_explanation_to_histogram(x_test_instance, shap_values_instance, feature_columns,
                                                                 value, i, explanation_histogram)

    # order results of the histogram by value (python >= 3.6) and take only the first 15
    explanation_histogram = {k: v for k, v in
                             sorted(explanation_histogram.items(), key=lambda item: abs(item[1]), reverse=True)}
    essential_histogram = {}
    for i, key in enumerate(explanation_histogram.keys()):
        if i == 15:
            break
        else:
            essential_histogram[key] = explanation_histogram[key]
    plot_histogram(essential_histogram, row_process_name)
    return explanation_histogram


def compute_shap_values_for_running_cases(row_process_name, X_test, model):
    # you should already have the background and you shouldn't need no partition
    background = np.load("shap/" + row_process_name + "_background.npy", allow_pickle=True)
    explainer = shap.DeepExplainer(model, background)
    shapley_test = explainer.shap_values(X_test)
    return shapley_test

def find_explanation_for_categorical_prediction(df, x_test_instance, explanation_index, explanation_name,
                                                explainable_response, pred_attribute, value, i):
    # in categorical case attribute prediction can be 0 or 1 (attribute will be performed or not)
    attribute_prediction = df.loc[i, pred_attribute]
    explanation_name = refine_explanation_name(x_test_instance, explanation_index, explanation_name)
    if attribute_prediction == 1:
        # keep only positive shap values (those who say attribute will be performed)
        if value > 0:
            explainable_response += explanation_name + " has predicted that " + pred_attribute \
                                    + " will be performed; "
    else:
        if value < 0:
            explainable_response += explanation_name + " has predicted that " + pred_attribute \
                                    + " won't be performed; "
    return explainable_response


def find_explanations_for_running_cases(shapley_test, X_test, df, feature_columns, pred_attribute, prediction_index):
    not_useful_explanations = json.load(open("conf/ignore_explanation_columns.json"))['columns']
    df['Explanation'] = 'No explanation found'
    column_type = 'Numeric'
    # shape > 3 means that there is more than one value to be predicted --> categorical case
    if df.shape[1] > 3:
        column_type = 'Categorical'
        # keep only the attribute the user wants to predict and round it between 0 and 1
        df = df[['CASE ID', pred_attribute]]
        df.loc[:, pred_attribute] = df.loc[:, pred_attribute].round(0)
    for i in range(len(shapley_test[prediction_index])):
        x_test_instance, shap_values_instance, explanation_values = find_instance_explanation_values(X_test, shapley_test, i, prediction_index)
        explainable_response = ""
        for value in explanation_values:
            skip = False
            # take the column name for every explanation
            explanation_index = np.where(shap_values_instance == value)[0][0]
            explanation_name = feature_columns[explanation_index]
            for useless_explanation in not_useful_explanations:
                if useless_explanation in explanation_name:
                    skip = True
            if skip is True:
                continue
            if column_type != 'Categorical':
                explanation_name = refine_explanation_name(x_test_instance, explanation_index, explanation_name)
                if value >= 0:
                    explainable_response += explanation_name + " has increased the prediction; "
                else:
                    explainable_response += explanation_name + " has decreased the prediction; "
            else:
                explainable_response = find_explanation_for_categorical_prediction(df, x_test_instance, explanation_index, explanation_name,
                                                            explainable_response, pred_attribute, value, i)
        #if we haven't found any explanation for categorical prediction we should find more explanations
        if explainable_response == "" and column_type == 'Categorical':
            x_test_instance, shap_values_instance, explanation_values = find_instance_explanation_values(X_test, shapley_test, i, prediction_index, True)
            for value in explanation_values:
                skip = False
                # take the column name for every explanation
                explanation_index = np.where(shap_values_instance == value)[0][0]
                explanation_name = feature_columns[explanation_index]
                for useless_explanation in not_useful_explanations:
                    if useless_explanation in explanation_name:
                        skip = True
                if skip is True:
                    continue
                explainable_response = find_explanation_for_categorical_prediction(df, x_test_instance, explanation_index, explanation_name,
                                                                                   explainable_response, pred_attribute, value, i)

        #at the end put the explanation in the correspondent row of the df
        if explainable_response == "":
            df.loc[i, 'Explanation'] = "No explanation found"
        else:
            df.loc[i, 'Explanation'] = explainable_response
    return df
