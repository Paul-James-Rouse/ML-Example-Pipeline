#!/usr/bin/env python # [1]
"""\
This script is intended to be a generic example for ML problems but uses the Kaggle Titanic dataset to provide evidence
of its functionality. It will score about 77%, which I am happy with given that this whole project is to be used as a
generic example for a ML project rather than a massive push on that problem.

This code will iterate though what I call the major hyperparameters
    1 - the limit to which feature can correlate with other features (multicollinearity)
    2 - the method in which the features are scaled or normalised
    3 - the percentage of features used for modelling
    4 - the uni-variate stat method for selecting which feature to use for modelling

Then it will use GridSearchCV from sklearn to optimise the model specific hyperparameters fot the classifiers:
    1 - Logistical Regression
    2 - K-Nearest Neighbour
    3 - Support Vector Machine
    4 - Decision Tree
    5 - Random Forest

Then it build simple keras sequential models optimising the model specific hyperparameters using keras-tuners:
    1 - Hyperband
    2 - RandomSearch
    3 - BayesianOptimization

Finally, it will create a simple ensemble model using a simple majority voting system on the top 10 models as determined
by the models Jaccord and F1 scores.

Note that this primarily an example and as such each model has been designed to be computationally inexpensive rather
than optimal, which hits the effectiveness of the deep learning models quite hard.


Contents page:
    # Import Packages ####
    # Hard Coded Variables #### - these are all the variables you need to manually define to run the script.
    # Read in data ####
    # Data Exploration ####
    # Define Input X and Y ####
    # Process the Test Dataset ####
    # Model: Logistical Regression ####
    # Model: K-Nearest Neighbour ####
    # Model: Support Vector Machine ####
    # Model: Decision Tree ####
    # Model: Random Forest ####
    # Model: Deep Learning ####
    # Ensemble Modeling Majority Voting ####
"""

# Import Packages ####
import os
import sys
import pandas as pd
import numpy as np
from Data_Visualisation import graphing
from Data_Visualisation import heatmap
from Data_Visualisation import univariate_feature_selection
from Data_Visualisation import model_evaluation_graphing
from Data_Cleaning import cleaning
from Data_Cleaning import process_test
from Model_Evaluation import evaluation_metrics
# Preprocessing allows us to standardise our data
from sklearn import preprocessing
# Allows us to split our data into training and testing data
from sklearn.model_selection import train_test_split
# Allow us to find the most suitable features for our ML
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
# Allows us to test parameters of classification algorithms and find the best one
from sklearn.model_selection import GridSearchCV
# Logistic Regression classification algorithm
from sklearn.linear_model import LogisticRegression
# Support Vector Machine classification algorithm
from sklearn.svm import SVC
# Decision Tree classification algorithm
from sklearn.tree import DecisionTreeClassifier
# K Nearest Neighbors classification algorithm
from sklearn.neighbors import KNeighborsClassifier
# Random Forest classification algorithm
from sklearn.ensemble import RandomForestClassifier
# Deep learning functions
from Deep_Learning import build_model
from Deep_Learning import tune_model
# Keras tuner for deep learning hyper parameters
import keras_tuner

# Hard Coded Variables ####
# Need to change path so python can find my login information in the "Scripts" directory
sys.path.append('Scripts/')
# Location of training data
Training_Data_Path = 'C:\\Users\\paulr\\PycharmProjects\\Kaggle_Titanic\\Datasets\\train.csv'
# Location of test data
Test_Data_Path = 'C:\\Users\\paulr\\PycharmProjects\\Kaggle_Titanic\\Datasets\\test.csv'
# To reduce the affects of Multi-colinearity, what are the maximum limits variables can be correlated?
MultiColinearity = [0.7, 0.8, 0.9]
# Which Normalisation/Scaler methods do you want to use?
Transformation_Methods = [
    preprocessing.StandardScaler(),
    preprocessing.MinMaxScaler(),
    preprocessing.MaxAbsScaler(),
    preprocessing.RobustScaler(quantile_range=(25, 75)),
    preprocessing.PowerTransformer(method="yeo-johnson"),
    preprocessing.QuantileTransformer(output_distribution='uniform'),
    preprocessing.QuantileTransformer(output_distribution='normal'),
    preprocessing.Normalizer()]
# How many Feature do you want to use,based on f test performance?
Percentage_of_Features = [0.2, 0.4, 0.6, 0.8, 1]
# Which metric do you want to use for picking best features?
Feature_Metric = [f_classif, mutual_info_classif]
Metric_Names = ['F1_Score', 'MI_Score']
# Where do you want to save the graphical outputs?
Graphing_Output_Path = 'C:\\Users\\paulr\\PycharmProjects\\Kaggle_Titanic\\Graphics\\'
# If the required output folder does not exist, lets make them.
if not os.path.exists(Graphing_Output_Path):
    os.mkdir(Graphing_Output_Path)
    os.mkdir(Graphing_Output_Path+'Raw\\')
    os.mkdir(Graphing_Output_Path+'Processed\\')
    os.mkdir(Graphing_Output_Path+'Model_Evaluation\\')
    os.mkdir(Graphing_Output_Path+'Feature_Selection\\')
else:
    if not os.path.exists(Graphing_Output_Path+'Raw\\'):
        os.mkdir(Graphing_Output_Path + 'Raw\\')
    if not os.path.exists(Graphing_Output_Path+'Processed\\'):
        os.mkdir(Graphing_Output_Path + 'Processed\\')
    if not os.path.exists(Graphing_Output_Path+'Model_Evaluation\\'):
        os.mkdir(Graphing_Output_Path+'Model_Evaluation\\')
    if not os.path.exists(Graphing_Output_Path+'Feature_Selection\\'):
        os.mkdir(Graphing_Output_Path+'Feature_Selection\\')

# Read in data ####
# Load in the csv file
df_raw_train = pd.read_csv(Training_Data_Path)
df_raw_test = pd.read_csv(Test_Data_Path)

# Take the passenger IDs of the test file
Output_Passenger_IDs = df_raw_test['PassengerId']

# Data_Exploration ####
# Understand the data types
dtypes = df_raw_train.dtypes

# Understand the number of columns and rows
shape = df_raw_train.shape

# Count the number of rows, this will tell me how much is missing too
row_count = df_raw_train.count()

# Check for duplicates - we have none
duplicates = df_raw_train[df_raw_train.duplicated()]

# Check for missing data - we have in Embarked, Age and Cabin
missing_data = df_raw_train.isnull().sum()

# Describe data
description = df_raw_train.describe(include='all')

# Data Processing ####
df_processed_train = cleaning(df_raw_train)
df_processed_test = cleaning(df_raw_test)

# Plot Graphs ####
graphing(input_df=df_raw_train,
         output_path=Graphing_Output_Path+'Raw\\')

graphing(input_df=df_processed_train,
         output_path=Graphing_Output_Path+'Processed\\')

# Multi-colinearity Check by calculating correlation
corr_matrix = pd.get_dummies(df_processed_train,
                             columns=['Embarked', 'Title', 'Age_Range', 'Fare_Range'],
                             prefix=['Embarked', 'Title', 'Age_Range', 'Fare_Range']).corr().abs()

# Define Input X and Y ####
# Check for Multi-colinearity
heatmap(input_df=corr_matrix,
        output_path=Graphing_Output_Path)

corr_matrix = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# loop through the Multi-colinearity limits
for i in range(0, len(MultiColinearity)):
    to_drop = []
    for col in corr_matrix.columns:
        if any(corr_matrix[col] > MultiColinearity[i]):
            to_drop.append(col)

    # One-hot encoding
    Indy_Vars = pd.get_dummies(df_processed_train,
                               columns=['Embarked', 'Title', 'Age_Range', 'Fare_Range'],
                               prefix=['Embarked', 'Title', 'Age_Range', 'Fare_Range'])

    # Drop any columns that shows Multi-colinearity
    Indy_Vars.drop(to_drop, axis='columns', inplace=True)

    # Drop the unnecessary columns for the model that cannot be transformed ####
    Indy_Vars.drop(labels=['Survived', 'Name'], axis='columns', inplace=True)

    # Take note the total independent variables in the data
    Features = Indy_Vars.columns

    # Loop through the Transformation_Methods
    for Transformation_Method in Transformation_Methods:
        transformation = Transformation_Method
        # Transform the Independent variables
        Indy_Vars = transformation.fit(Indy_Vars).transform(Indy_Vars)

        # subset out the Survived column, which represents the dependant variable
        Depy_Var = df_processed_train['Survived'].to_numpy()

        # Test, Train Split ####
        X_train, X_test, Y_train, Y_test = train_test_split(Indy_Vars, Depy_Var, test_size=0.2, random_state=1)

        # Feature Selection
        # Graph out the results from the Uni-variate feature selection
        univariate_feature_selection(x_train=X_train,
                                     y_train=Y_train,
                                     features=Features,
                                     output_path=Graphing_Output_Path,
                                     transformation=transformation)

        # loop through number of features list
        for z in range(0, len(Percentage_of_Features)):
            # Calculate Number_of_Features
            Number_of_Features = round(len(Features)*Percentage_of_Features[z])

            for q in range(0, len(Feature_Metric)):
                # Replace X_train and remove the unwanted features
                X_train_sel = SelectKBest(Feature_Metric[q], k=Number_of_Features).fit_transform(X_train, Y_train)

                # Find the list of feature that will be kept
                Features_sel = Features[SelectKBest(Feature_Metric[q],
                                                    k=Number_of_Features).fit(X_train,
                                                                              Y_train).get_support(indices=True)]
                # Subset X_test, selecting the above features
                X_test_sel = pd.DataFrame(X_test, columns=Features)
                X_test_sel = X_test_sel.loc[:, Features_sel]
                X_test_sel = X_test_sel.to_numpy()

                # Process the Test Dataset ####
                Output_Indy_Vars = process_test(input_df=df_processed_test,
                                                multicolinearity_to_drop=to_drop,
                                                transformation_method=transformation,
                                                features_sel=Features_sel)

                print('Multi-colinearity =', MultiColinearity[i], 'Transformation = ', Transformation_Method,
                      'Number of features =', Number_of_Features)

                # Model: Logistical Regression ####
                parameters = {'C': [0.01, 0.1, 1, 10, 100],
                              'penalty': ['l1', 'l2', 'elasticnet', 'None'],
                              'solver': ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']}

                lr = LogisticRegression()

                logreg_cv = GridSearchCV(estimator=lr, param_grid=parameters, cv=10, verbose=2)

                logreg_cv.fit(X_train_sel, Y_train)

                y_predict = logreg_cv.predict(X_test_sel)

                Output_yhat = logreg_cv.predict(Output_Indy_Vars)

                evaluation_metrics(model='Logistical Regression',
                                   output_path=Graphing_Output_Path+'Model_Evaluation\\',
                                   multicolinearity=MultiColinearity[i],
                                   transformation_method=Transformation_Method,
                                   number_of_features=Number_of_Features,
                                   feature_metric=Metric_Names[q],
                                   y=Y_test,
                                   y_predict=y_predict,
                                   output_yhat=Output_yhat,
                                   best_parameters=logreg_cv.best_params_)

                # Model: K-Nearest Neighbour ####
                parameters = {'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                              'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                              'p': [1, 2]}

                KNN = KNeighborsClassifier()

                KNN_cv = GridSearchCV(estimator=KNN, param_grid=parameters, cv=10, verbose=2)

                KNN_cv.fit(X_train_sel, Y_train)

                y_predict = KNN_cv.predict(X_test_sel)

                Output_yhat = logreg_cv.predict(Output_Indy_Vars)

                evaluation_metrics(model='K-Nearest Neighbour',
                                   output_path=Graphing_Output_Path+'Model_Evaluation\\',
                                   multicolinearity=MultiColinearity[i],
                                   transformation_method=Transformation_Method,
                                   number_of_features=Number_of_Features,
                                   feature_metric=Metric_Names[q],
                                   y=Y_test,
                                   y_predict=y_predict,
                                   output_yhat=Output_yhat,
                                   best_parameters=KNN_cv.best_params_)

                # Model: Support Vector Machine ####
                parameters = {'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                              'C': [0.01, 0.1, 1, 5],
                              'gamma': ['scale'],
                              'max_iter': [1000000000],
                              'cache_size': [1000]}
                svm = SVC()

                svm_cv = GridSearchCV(estimator=svm, param_grid=parameters, cv=10, verbose=2)

                svm_cv.fit(X_train_sel, Y_train)

                y_predict = svm_cv.predict(X_test_sel)

                Output_yhat = logreg_cv.predict(Output_Indy_Vars)

                evaluation_metrics(model='Support Vector Machine',
                                   output_path=Graphing_Output_Path+'Model_Evaluation\\',
                                   multicolinearity=MultiColinearity[i],
                                   transformation_method=Transformation_Method,
                                   number_of_features=Number_of_Features,
                                   feature_metric=Metric_Names[q],
                                   y=Y_test,
                                   y_predict=y_predict,
                                   output_yhat=Output_yhat,
                                   best_parameters=svm_cv.best_params_)

                # Model: Decision Tree ####
                parameters = {'criterion': ['gini', 'entropy'],
                              'splitter': ['best', 'random'],
                              'max_depth': [2, 4, 6, 8, 10],
                              'max_features': ['auto', 'sqrt'],
                              'min_samples_leaf': [1, 2, 4],
                              'min_samples_split': [2, 5, 10]}

                tree = DecisionTreeClassifier()

                tree_cv = GridSearchCV(estimator=tree, param_grid=parameters, cv=10, verbose=2)

                tree_cv.fit(X_train_sel, Y_train)

                y_predict = tree_cv.predict(X_test_sel)

                Output_yhat = logreg_cv.predict(Output_Indy_Vars)

                evaluation_metrics(model='Decision Tree',
                                   output_path=Graphing_Output_Path+'Model_Evaluation\\',
                                   multicolinearity=MultiColinearity[i],
                                   transformation_method=Transformation_Method,
                                   number_of_features=Number_of_Features,
                                   feature_metric=Metric_Names[q],
                                   y=Y_test,
                                   y_predict=y_predict,
                                   output_yhat=Output_yhat,
                                   best_parameters=tree_cv.best_params_)

                # Model: Random Forest ####
                parameters = {'criterion': ['log_loss'],
                              'n_estimators': [100],
                              'max_depth': [2, 4, 6],
                              'max_features': ['sqrt'],
                              'min_samples_leaf': [1, 2, 4],
                              'min_samples_split': [12, 14, 18]}

                Forest = RandomForestClassifier()

                Forest_cv = GridSearchCV(estimator=Forest, param_grid=parameters, cv=10, verbose=2)

                Forest_cv.fit(X_train_sel, Y_train)

                y_predict = Forest_cv.predict(X_test_sel)

                Output_yhat = logreg_cv.predict(Output_Indy_Vars)

                evaluation_metrics(model='Random Forest',
                                   output_path=Graphing_Output_Path+'Model_Evaluation\\',
                                   multicolinearity=MultiColinearity[i],
                                   transformation_method=Transformation_Method,
                                   number_of_features=Number_of_Features,
                                   feature_metric=Metric_Names[q],
                                   y=Y_test,
                                   y_predict=y_predict,
                                   output_yhat=Output_yhat,
                                   best_parameters=Forest_cv.best_params_)

                # Model: Deep Learning ####
                # Optimise keras tuner models
                # Hyperband
                Hyperband = keras_tuner.Hyperband(
                    hypermodel=build_model,
                    objective='val_accuracy',
                    max_epochs=10,
                    seed=1,
                    overwrite=True,
                    factor=3,
                    hyperband_iterations=5)

                # Random Search
                RandomSearch = keras_tuner.RandomSearch(
                    build_model,
                    objective='val_accuracy',
                    max_trials=10,
                    executions_per_trial=5,
                    overwrite=True,
                    seed=1)

                # Bayesian Optimization
                BayesianOptimization = keras_tuner.BayesianOptimization(
                    build_model,
                    objective='val_accuracy',
                    max_trials=10,
                    executions_per_trial=5,
                    overwrite=True,
                    seed=1)

                # Run each model
                model_objects = [Hyperband, RandomSearch, BayesianOptimization]
                model_names = ['Hyperband', 'Random Search', 'Bayesian Optimization']

                for x in range(0, len(model_objects)):
                    tune_model(model_object=model_objects[x],
                               model_name=model_names[x],
                               output_path=Graphing_Output_Path+'Model_Evaluation\\',
                               multicolinearity=MultiColinearity[i],
                               transformation_method=Transformation_Method,
                               number_of_features=Number_of_Features,
                               feature_metric=Metric_Names[q],
                               x_train=X_train_sel,
                               x_test=X_test_sel,
                               y_train=Y_train,
                               y_test=Y_test,
                               output=Output_Indy_Vars)

# Ensemble Modeling Majority Voting ####
Metrics = pd.read_csv(Graphing_Output_Path+'Model_Evaluation\\Metrics.csv')  # Opens Metrics.csv file
Predictions = pd.read_csv(Graphing_Output_Path+'Model_Evaluation\\Y_Predictions.csv')  # Opens Predictions.csv file
Predictions = Predictions.astype(int)  # Makes Predictions into int

# Create plots showing the model performance metrics split by the key major hyperparameters
model_evaluation_graphing(input_metrics_df=Metrics,
                          cols_to_plot=['Model', 'Multi-colinearity', 'Transformation Method',
                                        'Number of Features', 'Feature Metric'],
                          output_path=Graphing_Output_Path+'Model_Evaluation\\')

Metrics.sort_values('Jaccard Score', axis='rows', ascending=False, inplace=True)  # sort Metrics by jaccards score

Metrics = Metrics.head(n=10)  # take top 10 models

# Subset Prediction column by the Metrics index
index = [str(x) for x in list(Metrics.index)]
Predictions = Predictions[index]

# Collect the majority votes from the top ten models
Voting_yhat = []
for i in range(0, len(Predictions)):
    if Predictions.iloc[i, :].sum() > Predictions.shape[1]/2:  # If 1 is most common
        Voting_yhat.append(1)
    if Predictions.iloc[i, :].sum() < Predictions.shape[1]/2:  # If 0 is most common
        Voting_yhat.append(0)
    if Predictions.iloc[i, :].sum() == Predictions.shape[1]/2:  # If they are equally common use the top model
        Voting_yhat.append(Predictions.iloc[i, 0])

# Make Voting_yhat a DataFrame with PassengerIDs in columns 0
Voting_yhat = pd.DataFrame(Voting_yhat)
Voting_yhat = pd.concat([Output_Passenger_IDs, Voting_yhat], axis='columns')
Voting_yhat.columns = ['PassengerID', 'Survived']

Voting_yhat.to_csv(Graphing_Output_Path+'Submission.csv', index=False)
