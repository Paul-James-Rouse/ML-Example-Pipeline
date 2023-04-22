"""\
In this .py file we will define the following functions to call from the Data_Modelling script.
Each function has a pre-processing or data wrangling function.

    1- remove_special_characters: a generic function that removes special characters from an input string
    2 - cleaning: a specialised function for the dataset that performs the pre-processing for modelling on an input
            DataFrame and returns an ouput, processed DataFrame
    3 - process_test:  a generic function that applies the major hyperparameters used on the test dataset to the
            output dataset
"""

# Import Packages ####
import pandas as pd
import numpy as np


# Remove special characters ####
def remove_special_characters(input_string):
    # A list of special_characters to be removed
    special_characters = ['@', '#', '$', '*', '&', '[', ']', '(', ')']

    for i in special_characters:
        # Replace the special character with an empty string
        input_string = str(input_string).replace(i, "")

    return input_string


# Define Function ####
def cleaning(input_df):

    # Fill Missing Observations ####
    df_processed = input_df.copy()

    # Replace Age with median split by Pclass and then Sex
    replacement_ages = pd.DataFrame(df_processed.groupby(['Pclass', 'Sex'])['Age'].median()).reset_index()

    for Pclass in df_processed['Pclass'].unique():
        for Sex in df_processed['Sex'].unique():
            df_processed.loc[(df_processed['Pclass'] == Pclass) &
                             (df_processed['Sex'] == Sex) &
                             (df_processed['Age'].isnull()), 'Age'] = \
                replacement_ages.loc[(replacement_ages['Pclass'] == Pclass) &
                                     (replacement_ages['Sex'] == Sex), 'Age'].values[0]

    # Replace all missing Embarked with 'S'
    df_processed.loc[df_processed['Embarked'].isnull(), 'Embarked'] = 'S'

    # Replace Age with median split by Pclass and then Sex
    replacement_fares = pd.DataFrame(df_processed.groupby(['Pclass', 'Sex'])['Fare'].median()).reset_index()

    for Pclass in df_processed['Pclass'].unique():
        for Sex in df_processed['Sex'].unique():
            df_processed.loc[(df_processed['Pclass'] == Pclass) &
                             (df_processed['Sex'] == Sex) &
                             (df_processed['Fare'].isnull()), 'Fare'] = \
                replacement_fares.loc[(replacement_fares['Pclass'] == Pclass) &
                                      (replacement_fares['Sex'] == Sex), 'Fare'].values[0]

    # I cannot find cabin data online, so it will have to be dropped
    df_processed.drop(labels='Cabin', axis='columns', inplace=True)

    # Data_Processing ####
    # Change sex into numerical representation
    df_processed[['Sex']] = df_processed[['Sex']].replace("male", 1)  # change male to 1
    df_processed[['Sex']] = df_processed[['Sex']].replace("female", 0)  # change female to 0

    # Feature Engineering ####
    # Add total family size column
    df_processed['Family_Size'] = df_processed['SibSp'] + df_processed['Parch']

    # Add travelling alone column
    df_processed['Travelling_Alone'] = 'ERROR'
    df_processed.loc[df_processed['Family_Size'] == 0, 'Travelling_Alone'] = 1
    df_processed.loc[df_processed['Family_Size'] != 0, 'Travelling_Alone'] = 0

    # Add a title column and populate with title based on title appearing in name
    df_processed.insert(3, 'Title', 'ERROR')

    titles_list = ['Mr.', 'Miss.', 'Mrs.', 'Master.', 'Dr.', 'Rev.']  # list to iterate through

    for title in titles_list:  # for loop to replace Title where appropriate
        df_processed.loc[np.where(df_processed['Name'].str.contains(title))[0], 'Title'] = title

    # Exceptions
    # Pool all Military titles into one label
    df_processed.loc[np.where(df_processed['Name'].str.contains('Capt. | Major. | Col.'))[0], 'Title'] = 'Military'

    # Pull Men without titles in the Mr. title
    df_processed.loc[(df_processed['Sex'] == 1) &
                     (df_processed['Title'] == 'ERROR'), 'Title'] = 'Mr.'

    # Pull any females without any spouses or siblings into the Miss. title
    df_processed.loc[(df_processed['Sex'] == 0) &
                     (df_processed['SibSp'] == 0) &
                     (df_processed['Title'] == 'ERROR'), 'Title'] = 'Miss.'

    # The  Mr. and Master Titles don't match their ages.
    # Make any Master. >18 Mr.
    df_processed.loc[(df_processed['Age'] > 18) &
                     (df_processed['Title'] == 'Master.'), 'Title'] = 'Mr.'

    # Make any Mr. < 18 Masters
    df_processed.loc[(df_processed['Age'] < 18) &
                     (df_processed['Title'] == 'Mr.'), 'Title'] = 'Master.'

    # Feature Binning ####
    # Bin Age into Age_Range
    df_processed['Age_Range'] = pd.cut(df_processed['Age'],
                                       bins=[0, 12, 18, 30, 60, 80],
                                       labels=['Children', 'Teenage', 'Young Adult', 'Adult', 'Elder'])

    # Bin Fare into Age_Range
    df_processed['Fare_Range'] = pd.qcut(df_processed['Fare'], q=4,
                                         labels=['Lowest', 'Lower-Median', 'Upper-Median', 'Upper'])

    # Sort out dtypes
    df_processed['Title'] = df_processed['Title'].astype('category')
    df_processed['Embarked'] = df_processed['Embarked'].astype('category')

    # Drop Unwanted Columns ####
    # Drop the PassengerID as the heatmap shows it was a bad predictor of survived
    df_processed.drop(labels='PassengerId', axis='columns', inplace=True)

    # Drop the Ticket as I think it has no value for the analysis
    df_processed.drop(labels='Ticket', axis='columns', inplace=True)

    return df_processed


# Define a function that processes the test dataset with the same parameters as the training dataset ####
def process_test(input_df, multicolinearity_to_drop, transformation_method, features_sel):
    # One-hot encoding ####
    input_df = pd.get_dummies(input_df,
                              columns=['Embarked', 'Title', 'Age_Range', 'Fare_Range'],
                              prefix=['Embarked', 'Title', 'Age_Range', 'Fare_Range'])

    # Drop any columns that shows Multi-colinearity
    input_df.drop(multicolinearity_to_drop, axis='columns', inplace=True)

    # Drop the unnecessary columns for the model that cannot be transformed ####
    input_df.drop(labels=['Name'], axis='columns', inplace=True)

    # Apply Feature Selection
    input_df = input_df.loc[:, features_sel]

    # Apply Transformation
    transformation = transformation_method
    input_df = transformation.fit(input_df).transform(input_df)

    return input_df
