"""\
In this .py file we will define the following function to call from the Data_Modelling script.
This is the .py file that contains all the plotting functions, it is very verbose but in my experience supervisors
    will always ask to change one specific parameter on one specific plot, so it is better to make them accessible
    rather than hidden in for loops where possible.

    1 - univariate_feature_selection: plot out the uni-variate stats metrics of the relative usefulness of each feature
    2 - heatmap: plot out a heat map showing how each feature is correlated to each-other.
    3 - graphing: exploratory plots of the dataset. This function is specific for the Titanic dataset now but to make it
            unspecialised just keep the pairplots and heatmap functions.

Note that this script will call the function remove_special_characters from Data_Cleaning.py and as such you need to
    define its location in the # Hard Coded Variables #### section

"""

# Import Packages ####
import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotnine as p9
import patchworklib as pw
# Allow us to find the most suitable features for our ML
from sklearn.feature_selection import f_classif, mutual_info_classif, chi2
# Import function to remove special charters
from Data_Cleaning import remove_special_characters

# Hard Coded Variables ####
# Need to change path so python can find my login information in the "Scripts" directory
sys.path.append("Scripts/")
# Steal the colour hex from sns
colours = sns.color_palette().as_hex()


# Uni-variate feature selection ####
def univariate_feature_selection(x_train, y_train, features, output_path, transformation):
    # Create an if condition to assess is the scalar generates negative numbers
    mask = x_train > 0
    if not mask.any():

        f_test = f_classif(x_train, y_train)
        f_test /= np.max(f_test)

        mi_test = mutual_info_classif(x_train, y_train)
        mi_test /= np.max(mi_test)

        chi2_test = chi2(x_train, y_train)
        chi2_test /= np.max(chi2_test)

        a = pd.DataFrame(f_test[0])
        a['test'] = 'f_test'
        a['feature'] = features

        b = pd.DataFrame(mi_test)
        b['test'] = 'mi_test'
        b['feature'] = features

        c = pd.DataFrame(chi2_test[0])
        c['test'] = 'chi2_test'
        c['feature'] = features

        plot = pd.concat([a, b, c], axis='rows')
        plot.columns = ["Score", "Test", "Feature"]

        plot = p9.ggplot(plot, p9.aes(x='factor(Feature)', y='Score')) + \
               p9.theme_bw() + \
               p9.theme(figure_size=(25, 10), axis_text_x=p9.element_text(rotation=65)) + \
               p9.scale_fill_manual(colours) + \
               p9.geom_bar(p9.aes(fill='Test'), stat="identity", color='#000000', position='dodge') + \
               p9.labs(title="Uni-variate Feature Selection", y="Result/Max Result", x="Feature", fill="Survived")
        plot.save(''.join([output_path, 'Feature_Selection\\', remove_special_characters(transformation), '.png']),
                  dpi=600, format='png')

    else:
        f_test = f_classif(x_train, y_train)
        f_test /= np.max(f_test)

        mi_test = mutual_info_classif(x_train, y_train)
        mi_test /= np.max(mi_test)

        a = pd.DataFrame(f_test[0])
        a['test'] = 'f_test'
        a['feature'] = features

        b = pd.DataFrame(mi_test)
        b['test'] = 'mi_test'
        b['feature'] = features

        plot = pd.concat([a, b], axis='rows')
        plot.columns = ["Score", "Test", "Feature"]

        plot = p9.ggplot(plot, p9.aes(x='factor(Feature)', y='Score')) + \
               p9.theme_bw() + \
               p9.theme(figure_size=(25, 10), axis_text_x=p9.element_text(rotation=65)) + \
               p9.scale_fill_manual(colours) + \
               p9.geom_box(p9.aes(fill='Test'), stat="identity", color='#000000', position='dodge') + \
               p9.labs(title="Uni-variate Feature Selection", y="Result/Max Result", x="Feature", fill="Survived")
        plot.save(''.join([output_path, 'Feature_Selection\\', remove_special_characters(transformation), '.png']),
                  dpi=600, format='png')


# Define Heatmap only ####
def heatmap(input_df, output_path):
    # Heatmap of Correlation
    plt.figure(figsize=(20, 20))
    plot = sns.heatmap(input_df.corr(), annot=True, linewidths=0.2, cmap='viridis')
    plot.figure.tight_layout()
    fig = plot.get_figure()
    fig.savefig(''.join([output_path, 'MultiColinearity_heatmap.png']), dpi=600, format='png')


def model_evaluation_graphing(input_metrics_df, cols_to_plot, output_path):
    # Calculate plot data
    plots = []

    for i in range(0, len(cols_to_plot)):
        plot_df = pd.DataFrame(
            input_metrics_df.groupby([cols_to_plot[i]])['Accuracy Score', 'Jaccard Score', 'F1 Score'].mean()).reset_index()
        # Make a percetange of max
        for col in ['Accuracy Score', 'Jaccard Score', 'F1 Score']:
            plot_df[col] /= np.max(plot_df[col])

        plot_df.sort_values('Jaccard Score', axis='rows', ascending=False, inplace=True)

        # Rearrange for plotnine
        length = len(pd.unique(plot_df[cols_to_plot[i]]))
        score = pd.concat([plot_df['Accuracy Score'], plot_df['Jaccard Score'], plot_df['F1 Score']], axis='rows')
        model = pd.concat([plot_df[cols_to_plot[i]], plot_df[cols_to_plot[i]], plot_df[cols_to_plot[i]]], axis='rows')
        plot_df = pd.concat([model, score], axis='columns').reset_index(drop=True)
        plot_df.loc[length - length:length - 1, 'Test'] = 'Accuracy Score'
        plot_df.loc[length:length * 2 - 1, 'Test'] = 'Jaccard Score'
        plot_df.loc[length * 2:length * 3 - 1, 'Test'] = 'F1 Score'
        plot_df.columns = [cols_to_plot[i], 'Score', 'Test']
        plot_df[cols_to_plot[i]] = pd.Categorical(plot_df[cols_to_plot[i]],
                                                  categories=pd.unique(plot_df[cols_to_plot[i]]))

        # Draw and save the plot
        plots.append(pw.load_ggplot(p9.ggplot(plot_df, p9.aes(x=cols_to_plot[i], y='Score'))
                                    + p9.theme_bw()
                                    + p9.theme(axis_text_x=p9.element_text(rotation=65))
                                    + p9.scale_fill_manual(colours)
                                    + p9.geom_bar(p9.aes(fill='Test'), stat="identity", color='black', position='dodge')
                                    + p9.labs(title=cols_to_plot[i] + " Performance", y="Mean Score / Max Mean Score",
                                              x="", fill="Test")))

    # Save Patchwork
    ((plots[0] | plots[1]) / (plots[2] | plots[3] | plots[4])).savefig(
        ''.join([output_path, 'Patchwork.png']), format='png')


# Define Total Graphing Function
def graphing(input_df, output_path):
    # Heatmap of Correlation
    fig = plt.figure(figsize=(
        len(input_df.select_dtypes(exclude=['uint8']).columns), len(input_df.select_dtypes(exclude=['uint8']).columns)))
    plot_1 = sns.heatmap(input_df.select_dtypes(exclude=['uint8']).corr(), annot=True, linewidths=0.2, cmap='viridis')
    plot_1.figure.tight_layout()
    fig = plot_1.get_figure()
    fig.savefig(''.join([output_path, 'heatmap.png']), dpi=600, format='png')

    # Sex split by Survived with labels showing proportion
    # Calculate plot data
    plot_2 = pd.DataFrame(input_df.groupby(['Sex', 'Survived'])['Name'].count()).reset_index()

    # Empty lists to populate in for loop
    proportion_values = []
    initial_values = []
    output = pd.Series([])

    # Calculate Proportion
    for Sex in plot_2['Sex'].unique():
        initial_values = plot_2[(plot_2['Sex'] == Sex)][['Name']]
        proportion_values = initial_values / initial_values.sum() * 100
        output = pd.concat([output, proportion_values])

    plot_2[['Proportion']] = output[['Name']]

    # Draw and save the plot
    plot_2 = pw.load_ggplot(p9.ggplot(plot_2, p9.aes(x='factor(Sex)', y='Name'))
                            + p9.theme_bw()
                            + p9.scale_fill_manual(colours, labels=['No', 'Yes'])
                            + p9.geom_bar(p9.aes(fill='factor(Survived)'), stat="identity", color='black',
                                          position='dodge')
                            + p9.geom_label(p9.aes(label='Proportion'), format_string='{:.1f}%',
                                            position=p9.position_dodge2(width=0.9))
                            + p9.labs(title="Sex split by Survived", y="Count", x="Sex", fill="Survived"))

    # Class split by Survived
    plot_3 = pd.DataFrame(input_df.groupby(['Pclass', 'Survived'])['Name'].count()).reset_index()

    # Empty lists to populate in for loop
    proportion_values = []
    initial_values = []
    output = pd.Series([])

    # Calculate Proportion
    for Class in plot_3['Pclass'].unique():
        initial_values = plot_3[(plot_3['Pclass'] == Class)][['Name']]
        proportion_values = initial_values / initial_values.sum() * 100
        output = pd.concat([output, proportion_values])

    plot_3[['Proportion']] = output[['Name']]

    # Draw and save the plot
    plot_3 = pw.load_ggplot(p9.ggplot(plot_3, p9.aes(x='factor(Pclass)', y='Name'))
                            + p9.theme_bw()
                            + p9.scale_fill_manual(colours, labels=['No', 'Yes'])
                            + p9.geom_bar(p9.aes(fill='factor(Survived)'), stat="identity", color='black',
                                          position='dodge')
                            + p9.geom_label(p9.aes(label='Proportion'), format_string='{:.1f}%',
                                            position=p9.position_dodge2(width=0.9))
                            + p9.labs(title="Class split by Survived", y="Count", x="Class", fill="Survived"))

    # Embarked split by Survived
    plot_4 = pd.DataFrame(input_df.groupby(['Embarked', 'Survived'])['Name'].count()).reset_index()

    # Empty lists to populate in for loop
    proportion_values = []
    initial_values = []
    output = pd.Series([])

    # Calculate Proportion
    for Port in plot_4['Embarked'].unique():
        initial_values = plot_4[(plot_4['Embarked'] == Port)][['Name']]
        proportion_values = initial_values / initial_values.sum() * 100
        output = pd.concat([output, proportion_values])

    plot_4[['Proportion']] = output[['Name']]

    # Draw and save the plot
    plot_4 = pw.load_ggplot(p9.ggplot(plot_4, p9.aes(x='Embarked', y='Name'))
                            + p9.theme_bw()
                            + p9.scale_fill_manual(colours, labels=['No', 'Yes'])
                            + p9.geom_bar(p9.aes(fill='factor(Survived)'), stat="identity", color='black',
                                          position='dodge')
                            + p9.geom_label(p9.aes(label='Proportion'), format_string='{:.1f}%',
                                            position=p9.position_dodge2(width=0.9))
                            + p9.labs(title="Embarked split by Survived", y="Count", x="Embarked", fill="Survived"))

    # Class by Sex
    plot_5 = pd.DataFrame(input_df.groupby(['Sex', 'Pclass', 'Survived'])['Name'].count()).reset_index()

    # Empty lists to populate in for loop
    proportion_values = []
    initial_values = []
    output = pd.Series([])

    # Calculate Proportion
    for sex in plot_5['Sex'].unique():
        for Pclass in plot_5['Pclass'].unique():
            initial_values = plot_5[(plot_5['Sex'] == sex) & (plot_5['Pclass'] == Pclass)][['Name']]
            proportion_values = initial_values / initial_values.sum() * 100
            output = pd.concat([output, proportion_values])

    plot_5[['Proportion']] = output[['Name']]

    # Draw and save the plot
    plot_5 = pw.load_ggplot(p9.ggplot(plot_5, p9.aes(x='factor(Pclass)', y='Name'))
                            + p9.theme_bw()
                            + p9.scale_fill_manual(colours, labels=['No', 'Yes'])
                            + p9.geom_bar(p9.aes(fill='factor(Survived)'), stat="identity", color='black',
                                          position='dodge')
                            + p9.geom_label(p9.aes(label='Proportion'), format_string='{:.1f}%',
                                            position=p9.position_dodge2(width=0.9), size=6)
                            + p9.facet_wrap("Sex")
                            + p9.labs(title="Class by Sex", y="Count", x="Class", fill="Survived"))

    # Number of Siblings / Spouses
    plot_6 = pd.DataFrame(input_df.groupby(['SibSp', 'Survived'])['Name'].count()).reset_index()

    # Empty lists to populate in for loop
    proportion_values = []
    initial_values = []
    output = pd.Series([])

    # Calculate Proportion
    for SibSp in plot_6['SibSp'].unique():
        initial_values = plot_6[(plot_6['SibSp'] == SibSp)][['Name']]
        proportion_values = initial_values / initial_values.sum() * 100
        output = pd.concat([output, proportion_values])

    plot_6[['Proportion']] = output[['Name']]

    # Draw and save the plot
    plot_6 = pw.load_ggplot(p9.ggplot(plot_6, p9.aes(x='factor(SibSp)', y='Name'))
                            + p9.theme_bw()
                            + p9.scale_fill_manual(colours, labels=['No', 'Yes'])
                            + p9.geom_bar(p9.aes(fill='factor(Survived)'), stat="identity", color='black',
                                          position='dodge')
                            + p9.geom_label(p9.aes(label='Proportion'), format_string='{:.1f}%',
                                            position=p9.position_dodge2(width=0.9), size=6)
                            + p9.labs(title='Number of Siblings / Spouses', y="Count", x="Number of Siblings / Spouses",
                                      fill="Survived"))

    # Number of Parents / Children
    plot_7 = pd.DataFrame(input_df.groupby(['Parch', 'Survived'])['Name'].count()).reset_index()

    # Empty lists to populate in for loop
    proportion_values = []
    initial_values = []
    output = pd.Series([])

    # Calculate Proportion
    for SibSp in plot_7['Parch'].unique():
        initial_values = plot_7[(plot_7['Parch'] == SibSp)][['Name']]
        proportion_values = initial_values / initial_values.sum() * 100
        output = pd.concat([output, proportion_values])

    plot_7[['Proportion']] = output[['Name']]

    # Draw and save the plot
    plot_7 = pw.load_ggplot(p9.ggplot(plot_7, p9.aes(x='factor(Parch)', y='Name'))
                            + p9.theme_bw()
                            + p9.scale_fill_manual(colours, labels=['No', 'Yes'])
                            + p9.geom_bar(p9.aes(fill='factor(Survived)'), stat="identity", color='black',
                                          position='dodge')
                            + p9.geom_label(p9.aes(label='Proportion'), format_string='{:.1f}%',
                                            position=p9.position_dodge2(width=0.9), size=6)
                            + p9.labs(title="Number of Parents / Children", y="Count", x="Number of Parents / Children",
                                      fill="Survived"))

    # Fare split by Class
    plot_8 = pw.load_ggplot(p9.ggplot(input_df, p9.aes(x='Fare', fill='factor(Pclass)', color='factor(Pclass)'))
                            + p9.theme_bw()
                            + p9.scale_fill_manual(colours)
                            + p9.scale_color_manual(colours, guide=False)
                            # + p9.geom_density(aes(y=after_stat('count')), alpha=0.75)
                            + p9.geom_density(alpha=0.75)
                            + p9.labs(title="Fare split by Class", y="Count", x="Fare", fill="Class"))

    # Fare split by Survived
    plot_9 = pw.load_ggplot(p9.ggplot(input_df, p9.aes(x='Fare', fill='factor(Survived)', color='factor(Survived)'))
                            + p9.theme_bw()
                            + p9.scale_fill_manual(colours, labels=['No', 'Yes'])
                            + p9.scale_color_manual(colours, labels=['No', 'Yes'], guide=False)
                            # + p9.geom_density(aes(y=after_stat('count')), alpha=0.75)
                            + p9.geom_density(alpha=0.75)
                            + p9.labs(title="Fare", y="Count", x="Fare split by Survived", fill="Survived"))

    # Fare split by Age
    plot_10 = pw.load_ggplot(
        p9.ggplot(input_df, p9.aes(x='Fare', y='Age', fill='factor(Survived)', color='factor(Survived)'))
        + p9.theme_bw()
        + p9.scale_fill_manual(colours, labels=['No', 'Yes'])
        + p9.scale_color_manual(colours, labels=['No', 'Yes'], guide=False)
        + p9.geom_point(alpha=0.75)
        + p9.labs(title="Fare and Age", y="Age", x="Fare", fill="Survived"))

    # Save Patchwork
    ((plot_2 | plot_3 | plot_4) / (plot_5 | plot_6 | plot_7) / (plot_8 | plot_9 | plot_10)).savefig(
        ''.join([output_path, 'Patchwork.png']), format='png')

    # Title split by Survived with labels showing proportion
    # Only Processed input_dfs will contain this information
    if 'Title' in input_df.columns:

        # Calculate plot data
        plot_11 = pd.DataFrame(input_df.groupby(['Title', 'Survived'])['Name'].count()).reset_index()

        # Empty lists to populate in for loop
        proportion_values = []
        initial_values = []
        output = pd.Series([])

        # Calculate Proportion
        for Title in plot_11['Title'].unique():
            initial_values = plot_11[(plot_11['Title'] == Title)][['Name']]
            proportion_values = initial_values / initial_values.sum() * 100
            output = pd.concat([output, proportion_values])

        plot_11[['Proportion']] = output[['Name']]
        # Draw and save the plot
        plot_11 = pw.load_ggplot(p9.ggplot(plot_11, p9.aes(x='factor(Title)', y='Name'))
                                 + p9.theme_bw()
                                 + p9.scale_fill_manual(colours, labels=['No', 'Yes'])
                                 + p9.geom_bar(p9.aes(fill='factor(Survived)'), stat="identity", color='black',
                                               position='dodge')
                                 + p9.geom_label(p9.aes(label='Proportion'), format_string='{:.1f}%',
                                                 position=p9.position_dodge2(width=0.9), size=6)
                                 + p9.labs(title="Title split by Survived", y="Count", x="Title", fill="Survived"))

    # Family_Size split by Survived with labels showing proportion
    # Only Processed input_dfs will contain this information
    if 'Family_Size' in input_df.columns:

        # Title split by Survived with labels showing proportion
        # Calculate plot data
        plot_12 = pd.DataFrame(input_df.groupby(['Family_Size', 'Survived'])['Name'].count()).reset_index()

        # Empty lists to populate in for loop
        proportion_values = []
        initial_values = []
        output = pd.Series([])

        # Calculate Proportion
        for Family_Size in plot_12['Family_Size'].unique():
            initial_values = plot_12[(plot_12['Family_Size'] == Family_Size)][['Name']]
            proportion_values = initial_values / initial_values.sum() * 100
            output = pd.concat([output, proportion_values])

        plot_12[['Proportion']] = output[['Name']]

        # Draw and save the plot
        plot_12 = pw.load_ggplot(p9.ggplot(plot_12, p9.aes(x='factor(Family_Size)', y='Name'))
                                 + p9.theme_bw()
                                 + p9.scale_fill_manual(colours, labels=['No', 'Yes'])
                                 + p9.geom_bar(p9.aes(fill='factor(Survived)'), stat="identity", color='black',
                                               position='dodge')
                                 + p9.geom_label(p9.aes(label='Proportion'), format_string='{:.1f}%',
                                                 position=p9.position_dodge2(width=0.9), size=6)
                                 + p9.labs(title="Family Size split by Survived", y="Count", x="Family Size",
                                           fill="Survived"))

    # Ensure Age Range was set correctly
    # Only Processed input_dfs will contain this information
    if 'Age_Range' in input_df.columns:
        # Fare and Age marked by Age Range
        plot_13 = pw.load_ggplot(
            p9.ggplot(input_df, p9.aes(x='Fare', y='Age', fill='Age_Range', color='Age_Range'))
            + p9.theme_bw()
            + p9.scale_fill_manual(colours)
            + p9.scale_color_manual(colours, guide=False)
            + p9.geom_point(alpha=0.75)
            + p9.labs(title="Fare and Age marked by Age Range", y="Age", x="Fare", fill="Age Range"))

    # Ensure Fare Range was set correctly
    # Only Processed input_dfs will contain this information
    if 'Fare_Range' in input_df.columns:
        # Fare and Age marked by Age Range
        plot_14 = pw.load_ggplot(
            p9.ggplot(input_df, p9.aes(x='Fare', y='Age', fill='Fare_Range', color='Fare_Range'))
            + p9.theme_bw()
            + p9.scale_fill_manual(colours)
            + p9.scale_color_manual(colours, guide=False)
            + p9.geom_point(alpha=0.75)
            + p9.labs(title="Fare and Age marked by Fare Range", y="Age", x="Fare", fill="Fare Range"))

    # Save Patchwork 2
    if 'Fare_Range' in input_df.columns:
        ((plot_11 | plot_12) / (plot_13 | plot_14)).savefig(
            ''.join([output_path, 'Patchwork_2.png']), format='png')

    # Pairplots
    plot_pp = sns.pairplot(data=input_df.select_dtypes(exclude=['uint8']), hue='Survived',
                           plot_kws={'edgecolor': 'none', 'alpha': 0.4})
    plot_pp.savefig(''.join([output_path, 'pairplot.png']), dpi=600, format='png')
