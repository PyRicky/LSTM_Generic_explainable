import numpy as np
import pandas as pd
from math import sqrt
import math, random
from itertools import cycle
from sklearn.metrics import f1_score
from scipy import interp
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from scipy import stats
import matplotlib.pyplot as plt

def prepare_df(model, X_test, y_test, test_case_ids, target_column_name, pred_column, mode, column_type):
    # calculate and reshape predictions
    predictions = model.predict(X_test)
    predictions = np.squeeze(predictions)
    # qui deve appendere o una serie o un dataframe
    # if np.issubdtype(type(predictions[0]), np.number):
    if column_type == "Numeric":
        predictions = pd.Series(predictions)
        predictions.rename(pred_column, inplace=True)
        if mode == "train":
            y_test.rename('TEST', inplace=True)
    else:
        predictions = pd.DataFrame(predictions)
        predictions.columns = target_column_name
        if mode == "train":
            target_column_name = ['TEST_' + x for x in target_column_name]
            y_test.columns = target_column_name

    # compute a df with predictions against the real values to be plotted later
    if mode == "train":
        df = pd.concat([test_case_ids.reset_index(drop=True), predictions.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)
    else:
        #if you are in real running cases you have only one prediction per case
        df = pd.concat([pd.Series(test_case_ids.unique()), predictions.reset_index(drop=True)], axis=1)
    if pred_column == 'remaining_time':
        # convert to days only if you predict remaining time and if you're not returning results to MyInvenio
        if mode == "train":
            df.iloc[:, 1:] = df.iloc[:, 1:] / (24.0 * 3600)
        else:
            #convert to milliseconds
            df.iloc[:, 1:] = df.iloc[:, 1:] * 1000
    df.rename(columns={df.columns[0]: 'CASE ID'}, inplace=True)
    return df


def write_results_to_be_plotted(df, experiment_name, n_neurons, n_layers):
    df['Events from start'] = 0
    df['Events from end'] = 0
    unique_keys = np.unique(df[df.columns[0]])
    for key in unique_keys:
        case_length = df[df[df.columns[0]] == key].shape[0]
        events_from_start = np.arange(1, case_length + 1)
        events_from_end = events_from_start[::-1]
        df.loc[df[df.columns[0]] == key, 'Events from start'] = events_from_start
        df.loc[df[df.columns[0]] == key, 'Events from end'] = events_from_end
    df.to_csv(experiment_name + "/results/results_" + str(n_neurons) + "_" + str(n_layers) + ".csv", index=False)


def write_scores(scores, experiment_name, n_neurons, n_layers, pred_column, column_type, event_level, target_column_name, df):
    with open(experiment_name + "/results/scores_" + str(n_neurons) + "_" + str(n_layers) + ".txt", "w") as file:
        if column_type == "Numeric":
            if pred_column == 'remaining_time':
                file.write("\nRoot Mean Squared Error: %.4f MAE: %.4f MAPE: %.4f%%" % (sqrt(scores[1]/((24.0*3600)**2)), scores[2]/(24.0*3600), scores[3]))
                print("Root Mean Squared Error: %.4f MAE: %.4f MAPE: %.4f%%" % (sqrt(scores[1] / ((24.0 * 3600) ** 2)), scores[2] / (24.0 * 3600), scores[3]))
            else:
                file.write("\nRoot Mean Squared Error: %.4f MAE: %.4f MAPE: %.4f%%" % (sqrt(scores[1]), scores[2], scores[3]))
                print("Root Mean Squared Error: %.4f MAE: %.4f MAPE: %.4f%%" % (sqrt(scores[1]), scores[2], scores[3]))
            errors = df["TEST"] - df[pred_column]
            mad = stats.median_absolute_deviation(errors)
            file.write(f"\nConfidence intervals: {errors.median() - 2 * mad:.2f} / {errors.median() + 2 * mad:.2f}")
            print(f"Confidence intervals: {errors.median() - 2 * mad:.2f} / {errors.median() + 2 * mad:.2f}")
        elif event_level == 0:
            df.loc[:, target_column_name] = df.loc[:, target_column_name].round(0)
            for column in target_column_name:
                print("F1_score for {}: {:0.3f}\n".format(column, f1_score(df['TEST_' + column], df[column])))
                file.write("F1_score for {}: {:0.3f}\n".format(column, f1_score(df['TEST_' + column], df[column])))
            file.write("\nAccuracy: %.4f Categorical Accuracy: %.4f F1: %.4f%%" % (scores[1], scores[2], scores[3]))
            print("Accuracy: %.4f Categorical Accuracy: %.4f F1: %.4f%%" % (scores[1], scores[2], scores[3]))
        else:
            # round results to 1 or 0 in order to use the F1 score
            df.loc[:, target_column_name] = df.loc[:, target_column_name].round(0)
            for column in target_column_name:
                print("F1_score for {}: {:0.3f}\n".format(column, f1_score(df['TEST_' + column], df[column])))
                file.write("F1_score for {}: {:0.3f}\n".format(column, f1_score(df['TEST_' + column], df[column])))
            file.write("\nAccuracy: %.4f Binary Accuracy: %.4f F1: %.4f%%" % (scores[1], scores[2], scores[3]))
            print("Accuracy: %.4f Binary Accuracy: %.4f F1: %.4f%%" % (scores[1], scores[2], scores[3]))

def plot_auroc_curve(df, predictions_names, target_column_names, experiment_name):
    false_positive_rates = dict()
    true_positive_rates = dict()
    roc_auc = dict()
    for i, predictions, target in zip(range(len(predictions_names)), predictions_names, target_column_names):
        false_positive_rate, true_positive_rate, thresholds = roc_curve(df[target], df[predictions])
        false_positive_rates[i] = false_positive_rate
        true_positive_rates[i] = true_positive_rate
        roc_auc[i] = auc(false_positive_rate, true_positive_rate)
    # First aggregate all false positive rates (not nan)
    n_classes = len(predictions_names)
    all_fpr = np.unique(np.concatenate([false_positive_rates[i] for i in range(n_classes) if
                                        not math.isnan(auc(false_positive_rates[i], true_positive_rates[i]))]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    instances_for_mean = 0
    # from mean macro exclude examples that are nan
    for i in range(n_classes):
        if not math.isnan(auc(false_positive_rates[i], true_positive_rates[i])):
            mean_tpr += interp(all_fpr, false_positive_rates[i], true_positive_rates[i])
            instances_for_mean = instances_for_mean + 1

            # Finally average it and compute AUC
    mean_tpr /= instances_for_mean

    false_positive_rates["macro"] = all_fpr
    true_positive_rates["macro"] = mean_tpr
    roc_auc["macro"] = auc(false_positive_rates["macro"], true_positive_rates["macro"])

    # Plot all ROC curves
    plt.figure()
    plt.plot(false_positive_rates["macro"], true_positive_rates["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    color = []
    for i in range(len(predictions_names)):
        r = lambda: random.randint(0, 255)
        color.append('#%02X%02X%02X' % (r(), r(), r()))
    colors = cycle(color)
    # colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(false_positive_rates[i], true_positive_rates[i], color=color, lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'.format(predictions_names[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    #plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc=(1, -.10))
    plt.savefig(experiment_name + "/plots/auroc.png", dpi=300, bbox_inches="tight")


def plot_precision_recall_curve(df, predictions_names, target_column_names, experiment_name):
    precisions = dict()
    recalls = dict()
    average_precisions = dict()
    for i, predictions, target in zip(range(len(predictions_names)), predictions_names, target_column_names):
        precision, recall, thresholds = precision_recall_curve(df[target], df[predictions])
        precisions[i] = precision
        recalls[i] = recall
        average_precisions[i] = average_precision_score(df[target], df[predictions])
        print("Target {}:\n  Precision {}\n  Recall {}\n  Thresholds {}\n".format(predictions, precision, recall,
                                                                                  thresholds))
    plt.figure(figsize=(7, 8))
    f_scores = np.linspace(0.2, 0.8, num=4)
    lines = []
    labels = []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
        plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

    lines.append(l)
    labels.append('iso-f1 curves')

    color = []
    for i in range(len(predictions_names)):
        r = lambda: random.randint(0, 255)
        color.append('#%02X%02X%02X' % (r(), r(), r()))
    colors = cycle(color)

    for i, color in zip(range(len(predictions_names)), colors):
        l, = plt.plot(recalls[i], precisions[i], color=color, lw=2)
        lines.append(l)
        labels.append('Precision-recall for class {0} (area = {1:0.2f})'
                      ''.format(predictions_names[i], average_precisions[i]))

    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    #plt.title('Extension of Precision-Recall curve to multi-class')
    plt.legend(lines, labels, loc=(1.02,+.01), prop=dict(size=14))
    plt.savefig(experiment_name + "/plots/apr_curve.png", dpi=300, bbox_inches="tight")
