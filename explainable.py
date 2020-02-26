import numpy as np
import shap
import os, json
#import matplotlib.pyplot as plt


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


def find_instance_explanation_values(X_test, shapley_test, i, more_explanation_values=False):
    x_test_instance = X_test[i][-1]
    shap_values_instance = shapley_test[0][i][-1]
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


def calculate_histogram_for_shap_values(df, target_column_name, column_type, X_test, shapley_test, feature_columns,
                                        row_process_name):
    explanation_histogram = {}
    # if column numeric if the predictions goes too far from the avg real data discard the prediction from the shap graph
    if column_type == 'Numeric':
        true_mean = df[target_column_name].mean()
        prediction_out_of_range_high = df.loc[df['Predictions'] > (df[target_column_name] + true_mean), :].index
        prediction_out_of_range_low = df.loc[df['Predictions'] < (df['remaining_time'] - true_mean), :].index
    for i in range(shapley_test.shape[1]):
        # if the prediction is wrong don't consider the explanation
        if column_type == 'Numeric':
            if i in prediction_out_of_range_high or i in prediction_out_of_range_low:
                continue

        x_test_instance, shap_values_instance, explanation_values = find_instance_explanation_values(X_test, shapley_test, i)
        for value in explanation_values:
            # take the column name for every explanation
            explanation_index = np.where(shap_values_instance == value)[0][0]
            explanation_name = feature_columns[explanation_index]
            if x_test_instance[explanation_index] == 1:
                print("Instance {} --> {} happening: {}".format(i + 1, explanation_name, value))
                # if in the test there is a 1 means that the particular instance happened (see the contribution of the shapley values)
                if (explanation_name + " happening") not in explanation_histogram:
                    explanation_histogram[(explanation_name + " happening")] = value
                else:
                    explanation_histogram[(explanation_name + " happening")] += value
            elif x_test_instance[explanation_index] == 0:
                print("Instance {} --> {} not happening: {}".format(i + 1, explanation_name, value))
                if (explanation_name + " not happening") not in explanation_histogram:
                    explanation_histogram[(explanation_name + " not happening")] = value
                else:
                    explanation_histogram[(explanation_name + " not happening")] += value
            # handle also if for some reason the numeric value has importance in the prediction
            elif x_test_instance[explanation_index] > 0 and x_test_instance[explanation_index] < 0.5:
                if (explanation_name + " low value") not in explanation_histogram:
                    explanation_histogram[(explanation_name + " low value")] = value
                else:
                    explanation_histogram[(explanation_name + " low value")] += value
            else:
                if (explanation_name + " high value") not in explanation_histogram:
                    explanation_histogram[(explanation_name + " high value")] = value
                else:
                    explanation_histogram[(explanation_name + " high value")] += value

    # order results of the histogram by value (python >= 3.6)
    explanation_histogram = {k: v for k, v in
                             sorted(explanation_histogram.items(), key=lambda item: abs(item[1]), reverse=True)}
    # plot_histogram(explanation_histogram, row_process_name)
    return explanation_histogram


def compute_shap_values(row_process_name, X_train, X_test, model, column_type):
    # first run you compute the values and then save them
    if not os.path.isfile("shap/" + row_process_name + "_background"):
        background = X_train[:100]
        np.save("shap/" + row_process_name + "_background", background)
        explainer = shap.DeepExplainer(model, background)
        print("prepared explainer")
        if column_type == 'Numeric':
            shapley_test = explainer.shap_values(X_test)
            np.save("shap/" + row_process_name + "_shap_values_test.npy", shapley_test)
        else:
            #column of type categorical (multioutput) need data to be splitted (huge amount of memory)
            partition = int(len(X_test) / 25)
            shapley_test = []
            for i in range(25):
                if i != 24:
                    if i == 0:
                        chunk = explainer.shap_values(X_test[partition * i:partition * (i + 1)])
                        shapley_test = chunk
                    else:
                        chunk = explainer.shap_values(X_test[partition * i:partition * (i + 1)])
                        shapley_test = np.append(shapley_test, chunk, 1)
                else:
                    chunk = explainer.shap_values(X_test[partition * i:])
                    shapley_test = np.append(shapley_test, chunk, 1)
                np.save("shap/" + row_process_name + "_shap_values_test_" + str(i) + ".npy", chunk)

    # already saved values
    else:
        if column_type == 'Numeric':
            shapley_test = np.load("shap/" + row_process_name + "_shap_values_test.npy", allow_pickle=True)
        else:
            shapley_test = []
            for i in range(25):
                if len(shapley_test) == 0:
                    shapley_test = np.load("shap/" + row_process_name + "_shap_values_test_" + str(i) + ".npy", allow_pickle=True)
                else:
                    chunk = np.load("shap/" + row_process_name + "_shap_values_test_" + str(i) + ".npy", allow_pickle=True)
                    shapley_test = np.append(shapley_test, chunk, 1)
    return shapley_test


def compute_shap_values_for_running_cases(row_process_name, X_test, model):
    # you should already have the background and you shouldn't need no partition
    background = np.load("shap/" + row_process_name + "_background", allow_pickle=True)
    explainer = shap.DeepExplainer(model, background)
    shapley_test = explainer.shap_values(X_test)
    return shapley_test

def find_explanation_for_categorical_prediction(df, x_test_instance, explanation_index, explanation_name,
                                                explainable_response, pred_attribute, value, i):
    # in categorical case attribute prediction can be 0 or 1 (attribute will be performed or not)
    attribute_prediction = df.loc[i, pred_attribute]
    if attribute_prediction == 1:
        # keep only positive shap values (those who say attribute will be performed)
        if value > 0:
            if x_test_instance[explanation_index] == 1:
                explainable_response += explanation_name + " happening has predicted that " + pred_attribute \
                                        + " will be performed; "
            elif x_test_instance[explanation_index] == 0:
                explainable_response += explanation_name + " not happening has predicted that " + pred_attribute \
                                        + " will be performed; "
            elif x_test_instance[explanation_index] > 0 and x_test_instance[explanation_index] < 0.5:
                explainable_response += " A low value of " + explanation_name + " has predicted that " + pred_attribute \
                                        + " will be performed; "
            else:
                explainable_response += " An high value of " + explanation_name + " has predicted that " + pred_attribute \
                                        + " will be performed; "
    else:
        if value < 0:
            if x_test_instance[explanation_index] == 1:
                explainable_response += explanation_name + " happening has predicted that " + pred_attribute \
                                        + " won't be performed; "
            elif x_test_instance[explanation_index] == 0:
                explainable_response += explanation_name + " not happening has predicted that " + pred_attribute \
                                        + " won't be performed; "
            elif x_test_instance[explanation_index] > 0 and x_test_instance[explanation_index] < 0.5:
                explainable_response += " A low value of " + explanation_name + "has predicted that " + pred_attribute \
                                        + " won't be performed; "
            else:
                explainable_response += " An high value of " + explanation_name + "has predicted that " + pred_attribute \
                                        + " won't be performed; "
    return explainable_response


def find_explanations_for_running_cases(shapley_test, X_test, df, feature_columns, pred_attribute):
    df['Explanation'] = 'No explanation found'
    column_type = 'Numeric'
    # shape > 3 means that there is more than one value to be predicted --> categorical case
    if df.shape[1] > 3:
        column_type = 'Categorical'
        # keep only the attribute the user wants to predict and round it between 0 and 1
        df = df[['CASE ID', pred_attribute]]
        df.loc[:, pred_attribute] = df.loc[:, pred_attribute].round(0)
    for i in range(len(shapley_test[0])):
        x_test_instance, shap_values_instance, explanation_values = find_instance_explanation_values(X_test, shapley_test, i)
        explainable_response = ""
        for value in explanation_values:
            # take the column name for every explanation
            explanation_index = np.where(shap_values_instance == value)[0][0]
            explanation_name = feature_columns[explanation_index]
            if column_type != 'Categorical':
                if x_test_instance[explanation_index] == 1:
                    if value >= 0:
                        explainable_response += explanation_name + " happening has increased the prediction; "
                    else:
                        explainable_response += explanation_name + " happening has decreased the prediction; "
                elif x_test_instance[explanation_index] == 0:
                    if value >= 0:
                        explainable_response += explanation_name + " not happening has increased the prediction; "
                    else:
                        explainable_response += explanation_name + " not happening has decreased the prediction; "
                # an high or low value of the numeric feature normalized can higher or lower the prediction
                elif x_test_instance[explanation_index] > 0 and x_test_instance[explanation_index] < 0.5:
                    if value >= 0:
                        explainable_response += "A low value of " + explanation_name + " has increased the prediction; "
                    else:
                        explainable_response += "A low value of " + explanation_name + " has decreased the prediction; "
                else:
                    if value >= 0:
                        explainable_response += "An high value of " + explanation_name + " has increased the prediction; "
                    else:
                        explainable_response += "An high value of " + explanation_name + " has decreased the prediction; "
            else:
                explainable_response = find_explanation_for_categorical_prediction(df, x_test_instance, explanation_index, explanation_name,
                                                            explainable_response, pred_attribute, value, i)
        #if we haven't found any explanation for categorical prediction we should find more explanations
        if explainable_response == "" and column_type == 'Categorical':
            x_test_instance, shap_values_instance, explanation_values = find_instance_explanation_values(X_test, shapley_test, i, True)
            for value in explanation_values:
                # take the column name for every explanation
                explanation_index = np.where(shap_values_instance == value)[0][0]
                explanation_name = feature_columns[explanation_index]
                explainable_response = find_explanation_for_categorical_prediction(df, x_test_instance, explanation_index, explanation_name,
                                                                                   explainable_response, pred_attribute, value, i)

        #at the end put the explanation in the correspondent row of the df
        if explainable_response == "":
            df.loc[i, 'Explanation'] = "No explanation found"
        else:
            df.loc[i, 'Explanation'] = explainable_response
    return df
