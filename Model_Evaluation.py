"""\
In this .py file we will define the following function to call from the Data_Modelling script.
This .py file is for functions that evaluate the models.

    1 - evaluation_metrics: a generic function that will calculate how well accurate each model is by comparing the
            yhat generated from the X_test data to the true y. The confusion matrix plot is commented out to save disk
            space, but I find them to be very informative.

"""


# Import Packages ####
import sklearn.metrics as metrics
import os
import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt


# Define a function that records our evaluation metrics and creates a confusion matrix
def evaluation_metrics(model, output_path, multicolinearity, transformation_method, number_of_features, feature_metric,
                       best_parameters, y, y_predict, output_yhat):

    # Make a csv file for the metrics ####
    accuracy_score = metrics.accuracy_score(y, y_predict)
    jaccard_score = metrics.jaccard_score(y, y_predict)
    f1_score = metrics.f1_score(y, y_predict)

    row = dict()

    row = {'Model': model,
           'Multi-colinearity': multicolinearity,
           'Transformation Method': transformation_method,
           'Number of Features': number_of_features,
           'Feature Metric': feature_metric,
           'Best Parameters': best_parameters,
           'Accuracy Score': accuracy_score,
           'Jaccard Score': jaccard_score,
           'F1 Score': f1_score}

    row = pd.DataFrame([row])

    # If the .csv file already exists then we will open it and add the metrics from the new model
    if os.path.exists(''.join([output_path, 'Metrics.csv'])):

        df = pd.read_csv(''.join([output_path, 'Metrics.csv']))  # Opens .csv file

        df = pd.concat([df, row])  # Adds the new row to the bottom

        # If the model is already represented it will be dropped
        df.drop_duplicates(subset=['Model', 'Multi-colinearity', 'Transformation Method',
                                   'Number of Features', 'Feature Metric'],
                           keep='last', inplace=True)

        # Save without indexes
        df.to_csv(''.join([output_path, 'Metrics.csv']), index=False)

    # If the .csv file does not already exist then we can save row directly
    else:
        row.to_csv(''.join([output_path, 'Metrics.csv']), index=False)

    # Make a csv file to store all the X_test yhats ####
    # Save y_predict as a DataFrame
    y_predict = pd.DataFrame(y_predict)
    # If the .csv file already exists then we will open it and add the metrics from the new model
    if os.path.exists(''.join([output_path, 'X_Test_Y_Predictions.csv'])):

        df = pd.read_csv(''.join([output_path, 'X_Test_Y_Predictions.csv']))  # Opens .csv file

        df = pd.concat([df, y_predict], axis='columns')  # Add new column

        df.columns = list(range(0, len(df.columns)))  # Change the column name to match the model number

        # Save without indexes
        df.to_csv(''.join([output_path, 'X_Test_Y_Predictions.csv']), index=False)

    # If the .csv file does not already exist then we can save row directly
    else:
        y_predict.to_csv(''.join([output_path, 'X_Test_Y_Predictions.csv']), index=False)

    # Make a csv file to store all the Output yhat, predictions ####
    # Save output_yhat as a DataFrame
    output_yhat = pd.DataFrame(output_yhat)
    # If the .csv file already exists then we will open it and add the metrics from the new model
    if os.path.exists(''.join([output_path, 'Output_Y_Predictions.csv'])):

        df = pd.read_csv(''.join([output_path, 'Output_Y_Predictions.csv']))  # Opens .csv file

        df = pd.concat([df, output_yhat], axis='columns')  # Add new column

        df.columns = list(range(0, len(df.columns)))  # Change the column name to match the model number

        # Save without indexes
        df.to_csv(''.join([output_path, 'Output_Y_Predictions.csv']), index=False)

    # If the .csv file does not already exist then we can save row directly
    else:
        output_yhat.to_csv(''.join([output_path, 'Output_Y_Predictions.csv']), index=False)

    # Confusion Matrix Function ####
    # cm = metrics.confusion_matrix(y, y_predict)
    # plt.figure(figsize=(9, 6))
    # plot = sns.heatmap(cm, annot=True, linewidths=0.2, cmap='viridis')
    # plot.set_xlabel('Predicted labels')
    # plot.set_ylabel('True labels')
    # plot.set_title(model)
    # plot.xaxis.set_ticklabels(['Died', 'Survived'])
    # plot.yaxis.set_ticklabels(['Died', 'Survived'])
    # plot = plot.get_figure()
    # plot.savefig(''.join([output_path, model, '.png']), dpi=600, format='png')
