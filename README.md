# ML-Example-Pipeline

This script is intended to be a generic example for ML problems but uses the Kaggle Titanic dataset to provide evidence
of its functionality. It will score about 77%, which I am happy with given that this whole project is to be used as a
generic example for a ML project rather than a massive push on that problem. 

All the code specialised for the Titanic dataset are in Data_Cleaning.py and Data_Visualisation.py and, of course, relate to 
data-specific processing and visualisation. 

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
by the model's Jaccord and F1 scores.

Note that this primarily an example and as such each model has been designed to be computationally inexpensive rather
than optimal, which hits the effectiveness of the deep learning models quite hard.

Here is the session information since I last run the script:

keras               2.12.0
keras_tuner         1.3.4
matplotlib          3.7.1
numpy               1.23.5
pandas              1.5.3
patchworklib        0.5.2
plotnine            0.10.1
seaborn             0.12.2
session_info        1.0.0 (Joel Ostblom's fanastic package to generate this session information)
sklearn             1.2.2
tensorflow          2.12.0


Paul
