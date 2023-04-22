"""\
In this .py file we will define the following function to call from the Data_Modelling script.
This is the .py file that contains all the deel learning functions.

    1 - build_model: the keras-tuner hyperparameter optimiser function defining the layout of the sequential model.
            As you can see, we will keep this very simple.
    2 - tune_model: the function to initiate the tuning of the deep learning model using keras-tuner.

Note that this script will call the function evaluation_metrics from Model_Evaluation.py and as such you need to
    define its location in the # Hard Coded Variables #### section
"""

# Import packages ####
import sys
from Model_Evaluation import evaluation_metrics
# Import Keras for the sequential deep learning model
from tensorflow import keras
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

# Hard Coded Variables ####
sys.path.append("Scripts/")  # Need to change path so python can find my login information in the "Scripts" directory


# Define the function to build a sequential model ####
def build_model(hp):
    model = keras.Sequential()

    # Tune the number of layers.
    for i in range(hp.Int("num_layers", 1, 3)):
        model.add(
            Dense(
                # Tune number of units separately.
                units=hp.Int("units_" + str(i), min_value=5, max_value=15, step=2),
                # Tune type of activation.
                activation=hp.Choice("activation_" + str(i), ["relu", "tanh"]),
            )
        )

    # Tune whether to use dropout.
    if hp.Boolean("dropout"):
        model.add(
            Dropout(
                rate=0.25
            )
        )

    # The final layer needs one unit
    model.add(
        Dense(
            units=1,
            # Tune type of activation.
            activation='sigmoid'
        )
    )

    # Tune the learning rate of the compiler
    model.compile(
        optimizer=Adam(hp.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling="log")),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


# The function to optimise, run and evaluate the models ####

def tune_model(model_object, model_name, output_path, multicolinearity, transformation_method, number_of_features,
               feature_metric, x_train, x_test, y_train, y_test, output):

    stop_early = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5)

    model_object.search(x_train, y_train,
                        epochs=10,
                        validation_data=(x_test, y_test),
                        callbacks=[stop_early],
                        verbose=0)

    best_hyperparameters = model_object.get_best_hyperparameters(1)[0]

    model = model_object.hypermodel.build(best_hyperparameters)

    model.fit(x=x_train,
              y=y_train,
              validation_data=(x_test, y_test))

    # Predict yhat for validation data
    y_predict = model.predict(x=x_test, verbose=0)
    y_predict = y_predict.round()

    # Predict yhat for output dataset
    output_yhat = model.predict(x=output, verbose=0)
    output_yhat = output_yhat.round()

    evaluation_metrics(model=model_name,
                       output_path=output_path,
                       multicolinearity=multicolinearity,
                       transformation_method=transformation_method,
                       number_of_features=number_of_features,
                       feature_metric=feature_metric,
                       best_parameters=best_hyperparameters.values,
                       y=y_test,
                       y_predict=y_predict,
                       output_yhat=output_yhat)
