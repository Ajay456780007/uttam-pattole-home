import numpy as np
import pandas as pd
from tensorflow.keras.layers import Input, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from termcolor import cprint

from Sub_Functions.Evaluate import Evaluation_Metrics1
from Sub_Functions.Evaluate import main_est_parameters


def build_bilstm_regressor(input_shape):
    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True, input_shape=input_shape, dropout=0.2)),
        Bidirectional(LSTM(32, return_sequences=False, dropout=0.2)),
        Dense(32, activation="relu"),
        Dropout(0.3),
        Dense(16, activation="relu"),
        Dense(1, activation="linear")
    ])

    model.compile(optimizer=Adam(0.001), loss="mse")
    return model


def train_three_regressors(x_train1, y_train1,
                           x_train2, y_train2,
                           x_train3, y_train3, epochs):
    # shapes must be (samples, timesteps, features)

    x_train1 = x_train1.reshape(x_train1.shape[0], x_train1.shape[1], 1)
    x_train2 = x_train2.reshape(x_train2.shape[0], x_train2.shape[1], 1)
    x_train3 = x_train3.reshape(x_train3.shape[0], x_train3.shape[1], 1)
    model1 = build_bilstm_regressor(x_train1.shape[1:])
    model2 = build_bilstm_regressor(x_train2.shape[1:])
    model3 = build_bilstm_regressor(x_train3.shape[1:])

    cprint("Running on Rainfall Dataset", color="white", on_color="on_magenta")

    model1.fit(x_train1, y_train1, epochs=epochs, batch_size=8, verbose=1)

    cprint("Running on Temperature Dataset", color="white", on_color="on_magenta")
    model2.fit(x_train2, y_train2, epochs=epochs, batch_size=8, verbose=1)

    cprint("Running on Wind Dataset", color="white", on_color="on_magenta")
    model3.fit(x_train3, y_train3, epochs=epochs, batch_size=8, verbose=1)

    return model1, model2, model3


def get_predictions(model1, model2, model3,
                    x_test1, x_test2, x_test3, y_test1, y_test2, y_test3):
    x_test1 = x_test1.reshape(x_test1.shape[0], x_test1.shape[1], 1)
    x_test2 = x_test2.reshape(x_test2.shape[0], x_test2.shape[1], 1)
    x_test3 = x_test3.reshape(x_test3.shape[0], x_test3.shape[1], 1)

    pred_rain = model1.predict(x_test1)
    pred_temp = model2.predict(x_test2)
    pred_wind = model3.predict(x_test3)

    out1 = Evaluation_Metrics1(y_test1, pred_rain)
    out2 = Evaluation_Metrics1(y_test2, pred_temp)
    out3 = Evaluation_Metrics1(y_test3, pred_wind)

    pred_rain = pred_rain.flatten()
    pred_wind = pred_wind.flatten()
    pred_temp = pred_temp.flatten()

    return pred_rain, pred_temp, pred_wind, out1, out2, out3


def build_classifier():
    model = Sequential([
        Dense(32, activation="relu", input_shape=(3,)),
        Dense(16, activation="relu"),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def train_classifier(X, y, epochs):
    clf = build_classifier()
    cprint("Running the Classification Head:", color="white", on_color="on_grey")
    clf.fit(X, y, epochs=epochs, batch_size=8, verbose=1)
    return clf


def BiLSTM_1(x_train1, x_test1, y_train1, y_test1, x_train2, x_test2, y_train2, y_test2, x_train3, x_test3, y_train3,
          y_test3, C_train, C_test, epochs):
    # 1. Train three BiLSTM regressors
    model1, model2, model3 = train_three_regressors(x_train1, y_train1, x_train2, y_train2, x_train3, y_train3,
                                                    epochs=epochs)

    pred_rain, pred_temp, pred_wind, out1, out2, out3 = get_predictions(model1, model2, model3, x_test1, x_test2,
                                                                        x_test3, y_test1,
                                                                        y_test2, y_test3)

    X = np.column_stack([pred_rain, pred_temp, pred_wind])

    y = C_train

    classifier = train_classifier(X, y, epochs)

    pred = classifier.predict(X)

    pred = np.argmax(pred, axis=1)

    c_out1 = main_est_parameters(C_test, pred)

    return [out1, out2, out3, c_out1]
