import keras.layers
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.utils import plot_model
from tensorflow.keras.layers import Input, Dense, Reshape, Bidirectional, LSTM, Dropout
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers.legacy import Adam
from termcolor import cprint

from dbn.tensorflow import SupervisedDBNRegression
from Sub_Functions.Evaluate import Evaluation_Metrics1, Evaluation_Metrics, main_est_parameters_mul
from Sub_Functions.Evaluate import main_est_parameters

# -*- coding: utf-8 -*-

from keras.layers import Conv1D, Conv1DTranspose, MaxPooling1D, BatchNormalization, Multiply, Add, Activation, \
    AveragePooling1D
import keras.backend as K


class Feature_Pyramid_Attention_1D:

    def __init__(self, layer):
        self.layer = layer
        self.layer_shape = K.int_shape(layer)

    def downsample(self):
        max_pool_1 = MaxPooling1D(pool_size=2, padding="same")(self.layer)

        conv7_1 = Conv1D(self.layer_shape[-1], 7, padding='same', kernel_initializer='he_normal')(max_pool_1)
        conv7_1 = BatchNormalization()(conv7_1)
        conv7_1 = Activation('relu')(conv7_1)

        conv7_2 = Conv1D(self.layer_shape[-1], 7, padding='same', kernel_initializer='he_normal')(conv7_1)
        conv7_2 = BatchNormalization()(conv7_2)
        conv7_2 = Activation('relu')(conv7_2)

        max_pool_2 = MaxPooling1D(pool_size=2, padding="same")(conv7_1)

        conv5_1 = Conv1D(self.layer_shape[-1], 5, padding='same', kernel_initializer='he_normal')(max_pool_2)
        conv5_1 = BatchNormalization()(conv5_1)
        conv5_1 = Activation('relu')(conv5_1)

        conv5_2 = Conv1D(self.layer_shape[-1], 5, padding='same', kernel_initializer='he_normal')(conv5_1)
        conv5_2 = BatchNormalization()(conv5_2)
        conv5_2 = Activation('relu')(conv5_2)

        max_pool_3 = MaxPooling1D(pool_size=2, padding="same")(conv5_1)

        conv3_1 = Conv1D(self.layer_shape[-1], 3, padding='same', kernel_initializer='he_normal')(max_pool_3)
        conv3_1 = BatchNormalization()(conv3_1)
        conv3_1 = Activation('relu')(conv3_1)

        conv3_2 = Conv1D(self.layer_shape[-1], 3, padding='same', kernel_initializer='he_normal')(conv3_1)
        conv3_2 = BatchNormalization()(conv3_2)
        conv3_2 = Activation('relu')(conv3_2)

        upsampled_8 = Conv1DTranspose(self.layer_shape[-1], 2, strides=2, kernel_initializer='he_normal')(conv3_2)

        added_1 = Add()([upsampled_8, conv5_2])

        upsampled_16 = Conv1DTranspose(self.layer_shape[-1], 2, strides=2, kernel_initializer='he_normal')(added_1)

        added_2 = Add()([upsampled_16, conv7_2])

        upsampled_32 = Conv1DTranspose(self.layer_shape[-1], 2, strides=2, kernel_initializer='he_normal')(added_2)

        return upsampled_32

    def direct_branch(self):
        conv1 = Conv1D(self.layer_shape[-1], 1, padding='valid', kernel_initializer='he_normal')(self.layer)
        return conv1

    def global_pooling_branch(self):
        global_pool = AveragePooling1D(pool_size=self.layer_shape[1])(self.layer)

        conv1_2 = Conv1D(self.layer_shape[-1], 1, padding='valid', kernel_initializer='he_normal')(global_pool)

        upsampled = Conv1DTranspose(self.layer_shape[-1], self.layer_shape[1], kernel_initializer='he_normal')(conv1_2)

        return upsampled

    def FPA(self):
        down_up_conved = self.downsample()
        direct_conved = self.direct_branch()
        gpb = self.global_pooling_branch()

        multiplied = Multiply()([down_up_conved, direct_conved])
        m = tf.reduce_mean(multiplied, axis=1, keepdims=True)
        added_fpa = Add()([m, gpb])
        return added_fpa


# from Proposed_model.Scaled_dot_product_1 import ScaledDotProductAttentionLayer


class CBAMBLOCK(tf.keras.layers.Layer):
    def __init__(self, ratio=8):
        super(CBAMBLOCK, self).__init__()
        self.ratio = ratio

    def build(self, input_shape):
        channel = int(input_shape[-1])

        # channel attention
        self.shared_dense_one = Dense(channel // self.ratio, activation='relu')
        self.shared_dense_two = Dense(channel)

        # spatial attention
        self.conv1 = tf.keras.layers.Conv1D(1, kernel_size=7, padding='same', activation='sigmoid')

    def call(self, x):
        # -------- Channel Attention --------
        avg_pool = tf.reduce_mean(x, axis=1, keepdims=True)  # shape (batch,1,channels)
        max_pool = tf.reduce_max(x, axis=1, keepdims=True)

        avg_out = self.shared_dense_two(self.shared_dense_one(avg_pool))
        max_out = self.shared_dense_two(self.shared_dense_one(max_pool))

        channel_att = tf.nn.sigmoid(avg_out + max_out)  # (batch,1,channels)
        x = x * channel_att  # broadcast multiply

        avg_pool_spatial = tf.reduce_mean(x, axis=-1, keepdims=True)  # (batch, timesteps, 1)
        max_pool_spatial = tf.reduce_max(x, axis=-1, keepdims=True)
        spatial = tf.concat([avg_pool_spatial, max_pool_spatial], axis=-1)  # (batch, timesteps, 2)

        spatial_att = self.conv1(spatial)  # (batch, timesteps, 1)
        x = x * spatial_att  # broadcast multiply over channels

        return x


from tensorflow.keras import layers


class MutualCrossAttention(layers.Layer):
    def __init__(self, dropout_rate=0.1, **kwargs):
        super(MutualCrossAttention, self).__init__(**kwargs)
        self.dropout = layers.Dropout(dropout_rate)

    def call(self, x1, x2, training=None):
        """
        x1: (batch, L1, d)
        x2: (batch, L2, d)
        returns: (batch, L1, d)  # same as your PyTorch example
        """
        # query = x1, key = x2
        query = x1
        key = x2
        d = tf.cast(tf.shape(query)[-1], tf.float32)

        # A: x1 attends to x2
        # scores: (batch, L1, L2)
        scores_qk = tf.matmul(query, key, transpose_b=True) / tf.math.sqrt(d)
        attn_weights_qk = tf.nn.softmax(scores_qk, axis=-1)
        attn_weights_qk = self.dropout(attn_weights_qk, training=training)
        # output_A: (batch, L1, d)
        output_A = tf.matmul(attn_weights_qk, x2)

        # B: x2 attends to x1
        # out = x1
        scores_kq = tf.matmul(key, query, transpose_b=True) / tf.math.sqrt(d)
        attn_weights_kq = tf.nn.softmax(scores_kq, axis=-1)
        attn_weights_kq = self.dropout(attn_weights_kq, training=training)
        # output_B: (batch, L2, d), then reproject to L1 if needed
        # here we mirror your PyTorch: use x1 as values to keep shape (batch, L2, d),
        # then transpose back to match L1 if L1 == L2
        output_B = tf.matmul(attn_weights_kq, x1)

        # assume L1 == L2; else you may need additional projections
        output = output_A + output_B  # (batch, L1, d)

        return output


import tensorflow as tf
import collections


class ScaledDotProductAttentionLayer(tf.keras.layers.Layer):

    def __init__(self, num_heads, hidden_size, output_size, **kwargs):
        super(ScaledDotProductAttentionLayer, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.query_map = tf.keras.layers.Dense(hidden_size * num_heads)
        self.key_map = tf.keras.layers.Dense(hidden_size * num_heads)
        self.value_map = tf.keras.layers.Dense(hidden_size * num_heads)
        self.output_map = tf.keras.layers.Dense(output_size)

    def __call__(self, queries, keys, values):
        batch_size, num_queries, sequence_length = tf.shape(queries)[0], tf.shape(queries)[1], tf.shape(values)[1]
        Q, K, V = self.query_map(queries), self.key_map(keys), self.value_map(values)
        Q = tf.transpose(tf.reshape(Q, [batch_size, num_queries, self.num_heads, self.hidden_size]), [0, 2, 1, 3])
        K = tf.transpose(tf.reshape(K, [batch_size, sequence_length, self.num_heads, self.hidden_size]), [0, 2, 1, 3])
        V = tf.transpose(tf.reshape(V, [batch_size, sequence_length, self.num_heads, self.hidden_size]), [0, 2, 1, 3])
        S = tf.matmul(tf.nn.softmax(tf.matmul(Q, tf.transpose(K, [0, 1, 3, 2])) / tf.sqrt(float(self.hidden_size))), V)
        S = tf.reshape(tf.transpose(S, [0, 2, 1, 3]), [batch_size, num_queries, self.num_heads * self.hidden_size])
        return self.output_map(S)

    @property
    def trainable_variables(self):
        layer_variables = (
                self.query_map.trainable_variables + self.key_map.trainable_variables +
                self.value_map.trainable_variables + self.output_map.trainable_variables)
        return layer_variables

    @property
    def trainable_weights(self):
        return self.trainable_variables

    @property
    def variables(self):
        layer_variables = (
                self.query_map.variables + self.key_map.variables +
                self.value_map.variables + self.output_map.variables)
        return layer_variables

    @property
    def weights(self):
        return self.variables


# -------------------- Builder: DBN + 3-step BiLSTM regressor --------------------
def build_dbn_bilstm_regressor(input_shape):
    timesteps, features = input_shape
    dbn_input_dim = timesteps * features

    dbn = SupervisedDBNRegression(
        hidden_layers_structure=[256, 128],
        learning_rate_rbm=0.01,
        learning_rate=0.01,
        n_epochs_rbm=10,
        n_iter_backprop=10,
        batch_size=12,
        activation_function='relu',
        dropout_p=0.1
    )

    inputs = Input(shape=input_shape)  # (timesteps, features)

    flat = Reshape((dbn_input_dim,))(inputs)  # shape (batch, dbn_input_dim)

    # wrapper to call dbn.transform (numpy -> numpy). tf.py_function expects numpy inputs/outputs.
    def dbn_forward(x):
        # x is a TensorFlow tensor; get numpy array
        x_np = x.numpy()
        # dbn.transform expects shape (n_samples, dbn_input_dim)
        # it returns (n_samples, dbn_output_dim)
        out = dbn.transform(x_np)  # IMPORTANT: dbn must be trained before meaningful outputs
        # ensure dtype float32 for TF
        return out.astype(np.float32)

    # Use tf.py_function to call python-side DBN transformer
    DBN_features = tf.keras.layers.Lambda(
        lambda x: tf.py_function(dbn_forward, [x], tf.float32),
        name="dbn_transform"
    )(flat)

    DBN_features.set_shape((None, 128))

    seq = tf.expand_dims(DBN_features, axis=1)

    x = Bidirectional(LSTM(64, return_sequences=True), name="bilstm_l1")(seq)
    # cbam_a = CBAMBLOCK()(x)

    x1 = Bidirectional(LSTM(64, return_sequences=True), name="bilstm_l2")(x)

    cbam_a = CBAMBLOCK()(x1)

    # SDPA = ScaledDotProductAttentionLayer(num_heads=2,
    #                                       hidden_size=64,  # 2*64 = 128 if d_model=128
    #                                       output_size=128)
    #
    # sdpa_out = SDPA(queries=x1, keys=x1, values=x1)
    # FPA = Feature_Pyramid_Attention_1D()

    fpa_out = Feature_Pyramid_Attention_1D(cbam_a).FPA()

    MCA = MutualCrossAttention()

    mca_out = MCA(x1, fpa_out)

    # attn_fused = layers.Average(name="attn_fusion")([sdpa_out, mca_out])
    x = Bidirectional(LSTM(64, return_sequences=False), name="bilstm_l3")(mca_out)

    out = Dense(32, activation="relu", name="dense_head1")(x)
    out = Dropout(0.1)(out)
    out = Dense(16, activation="relu", name="dense_head2")(out)
    output = Dense(1, activation="linear", name="regression_output")(out)

    model = Model(inputs=inputs, outputs=output, name="DBN_BiLSTM_Regressor")

    plot_model(
        model,
        to_file="DBN_BiLSTM_CBAM_Flow.png",
        show_shapes=True,
        show_layer_names=True,
        expand_nested=True
    )
    model.compile(optimizer=Adam(1e-3), loss="mse", metrics=["mae"])

    return dbn, model, inputs, output


# -------------------- Train three regressors (rain, temp, wind) --------------------
def train_three_regressors(x_train1, y_train1,
                           x_train2, y_train2,
                           x_train3, y_train3,
                           epochs,
                           ):
    """
    Trains three DBN+3-BiLSTM regressors (one per dataset). Returns trained (dbn, model) triples.
    x_train* shapes must be (samples, timesteps) or (samples, timesteps, features).
    This function will ensure they are (samples, timesteps, 1).
    """

    def prepare(X):
        # ensure shape (n, timesteps, 1)
        X = np.asarray(X)
        if X.ndim == 2:
            X = X.reshape((X.shape[0], X.shape[1], 1))
        return X

    x_train1 = prepare(x_train1)
    x_train2 = prepare(x_train2)
    x_train3 = prepare(x_train3)

    timesteps = x_train1.shape[1]
    features = x_train1.shape[2]

    timesteps1 = x_train2.shape[1]
    features1 = x_train2.shape[2]

    timesteps2 = x_train3.shape[1]
    features2 = x_train3.shape[2]

    # Build models
    cprint("Building DBN+BiLSTM regressor for dataset 1 (Rain)", color="white", on_color="on_magenta")
    dbn1, model1, m1, o1 = build_dbn_bilstm_regressor((timesteps, features), )

    cprint("Building DBN+BiLSTM regressor for dataset 2 (Temp)", color="white", on_color="on_magenta")
    dbn2, model2, m2, o2 = build_dbn_bilstm_regressor((timesteps1, features1), )

    cprint("Building DBN+BiLSTM regressor for dataset 3 (Wind)", color="white", on_color="on_magenta")
    dbn3, model3, m3, o3 = build_dbn_bilstm_regressor((timesteps2, features2), )

    X1_flat = x_train1.reshape(x_train1.shape[0], -1)
    X2_flat = x_train2.reshape(x_train2.shape[0], -1)
    X3_flat = x_train3.reshape(x_train3.shape[0], -1)

    cprint("Pretraining DBN-1 on flattened Rain data", color="white", on_color="on_magenta")
    dbn1.fit(X1_flat, y_train1)

    cprint("Pretraining DBN-2 on flattened Temp data", color="white", on_color="on_magenta")
    dbn2.fit(X2_flat, y_train2)

    cprint("Pretraining DBN-3 on flattened Wind data", color="white", on_color="on_magenta")
    dbn3.fit(X3_flat, y_train3)

    cprint("Training BiLSTM regressor for Rain", color="white", on_color="on_magenta")
    model1.fit(x_train1, y_train1, epochs=epochs, batch_size=8, verbose=1)

    cprint("Training BiLSTM regressor for Temp", color="white", on_color="on_magenta")
    model2.fit(x_train2, y_train2, epochs=epochs, batch_size=8, verbose=1)

    cprint("Training BiLSTM regressor for Wind", color="white", on_color="on_magenta")
    model3.fit(x_train3, y_train3, epochs=epochs, batch_size=8, verbose=1)

    return (dbn1, model1, m1, o1), (dbn2, model2, m2, o2), (dbn3, model3, m3, o3)


def get_predictions(triple1, triple2, triple3,
                    x_test1, x_test2, x_test3, y_test1, y_test2, y_test3):
    _, model1, _, _ = triple1
    _, model2, _, _ = triple2
    _, model3, _, _ = triple3

    def prepare(X):
        X = np.asarray(X)
        if X.ndim == 2:
            X = X.reshape((X.shape[0], X.shape[1], 1))
        return X

    x_test1 = prepare(x_test1)
    x_test2 = prepare(x_test2)
    x_test3 = prepare(x_test3)

    pred_rain = model1.predict(x_test1)
    pred_temp = model2.predict(x_test2)
    pred_wind = model3.predict(x_test3)

    out1 = Evaluation_Metrics(y_test1, pred_rain)
    out2 = Evaluation_Metrics(y_test2, pred_temp)
    out3 = Evaluation_Metrics(y_test3, pred_wind)

    pred_rain = pred_rain.flatten()
    pred_wind = pred_wind.flatten()
    pred_temp = pred_temp.flatten()

    return pred_rain, pred_temp, pred_wind, out1, out2, out3


# -------------------- Classifier (same as your original) --------------------
def build_classifier():
    model = Sequential([
        Dense(32, activation="relu", input_shape=(3,)),
        Dense(16, activation="relu"),
        Dense(8, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    plot_model(model, to_file="Architecture_classifier.jpg", show_dtype=True, show_shapes=True)
    return model


def train_classifier(X, y, epochs):
    clf = build_classifier()
    cprint("Running the Classification Head:", color="white", on_color="on_grey")
    clf.fit(X, y, epochs=epochs, batch_size=8, verbose=1)
    return clf


def Proposed_model2(x_train1, x_test1, y_train1, y_test1,
                    x_train2, x_test2, y_train2, y_test2,
                    x_train3, x_test3, y_train3, y_test3,
                    C_train, C_test, epochs, ):
    # Train regressors
    triple1, triple2, triple3 = train_three_regressors(x_train1, y_train1,
                                                       x_train2, y_train2,
                                                       x_train3, y_train3,
                                                       epochs)
    # Extract from triples (YOUR CODE - CORRECT)
    _, _model1, m1, o1 = triple1  # m1=Rain input tensor, o1=Rain output tensor
    _, _model2, m2, o2 = triple2  # m2=Temp input tensor, o2=Temp output tensor
    _, model3, m3, o3 = triple3  # m3=Wind input tensor, o3=Wind output tensor

    # ✅ FIXED: Correct concatenate + Reshape
    fused = tf.keras.layers.Concatenate(name="Feature_Fusion")([o1, o2, o3])  # (None, 3)
    fused = tf.keras.layers.Reshape((3,), name="Reshape_3_features")(fused)  # (None, 3)

    # ✅ Classifier (YOUR CODE - CORRECT)
    x1 = Dense(32, activation="relu", name="Dense32")(fused)
    x2 = Dense(16, activation="relu", name="Dense16")(x1)
    out1 = Dense(8, activation="softmax", name="Classifier_Output")(x2)

    # ✅ FIXED: SINGLE CLASSIFIER OUTPUT ONLY
    complete_model = Model(
        inputs=[m1, m2, m3],  # ✅ 3 Regressor INPUTS
        outputs=out1,  # ✅ ONLY Classifier output (8 classes)
        name="Complete_Pipeline"
    )

    # ✅ Plot
    plot_model(complete_model, to_file="arc.jpg", show_shapes=True, show_layer_names=True, expand_nested=True)

    pred_rain, pred_temp, pred_wind, out1, out2, out3 = get_predictions(triple1, triple2, triple3,
                                                                        x_test1, x_test2, x_test3,
                                                                        y_test1, y_test2, y_test3)

    X_for_clf = np.column_stack([pred_rain, pred_temp, pred_wind])
    y_clf = C_train

    classifier = train_classifier(X_for_clf, y_clf, epochs)

    pred = classifier.predict(X_for_clf)
    pred_labels = np.argmax(pred, axis=1)

    c_out1 = main_est_parameters_mul(C_test, pred_labels)

    return [out1, out2, out3, c_out1]

# give me the complete code for only plotting and saving the image , which means only compile no need other than anu=ythinhg , pass
# input like same a and run on the models and pass the output to the classifier and plot the architevture an
# and save the image give me the code for it
