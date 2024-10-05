import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from lab_2_fce.lab_2.models.enn import ENN
from lab_2_fce.lab_2.tools import data_processing as dp


class EnnModels:

    def __init__(self, input_neurons=2, hidden_layer_count=1, hidden_neurons=10, n=200, compare=True):
        self._train_d, self._train_a, self._test_d2, self._test_a2 = dp.prepared_data(-1, 1, n)

        if compare:
            self._custom_model = ENN(input_neurons, hidden_layer_count, hidden_neurons)
        self._keras_model = keras.Sequential()
        self._keras_model.add(layers.Input(shape=(None, input_neurons)))
        for _ in range(hidden_layer_count):
            self._keras_model.add(layers.SimpleRNN(hidden_neurons,
                                                   return_sequences=True))
        # self._keras_model.add(
        #     layers.SimpleRNN(hidden_neurons, return_sequences=False))
        self._keras_model.add(layers.Dense(1, activation='linear'))

        self._keras_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.05),
                                  loss='mean_squared_error',
                                  metrics=['accuracy'])

    def fit(self):
        if self._custom_model:
            self._custom_model.fit(self._train_d.reshape(-1, 1, 2), self._train_a, epochs=100)

        train_d_reshaped = self._train_d.reshape((self._train_d.shape[0], 1, self._train_d.shape[1]))
        self._keras_model.fit(train_d_reshaped, self._train_a, epochs=100)
        test_d_reshaped = self._test_d2.reshape((self._test_d2.shape[0], 1, self._test_d2.shape[1]))
        elman_loss, elman_accuracy = self._keras_model.evaluate(test_d_reshaped, self._test_a2)
        print("Elman Model - Loss:", elman_loss)
        print("Elman Model - Accuracy:", elman_accuracy)

    def predict(self):
        test_d_reshaped = self._test_d2.reshape((self._test_d2.shape[0], 1, self._test_d2.shape[1]))
        prediction_k = self._keras_model.predict(test_d_reshaped)

        if self._custom_model:
            prediction_c = [i for i in self._custom_model.predict(self._test_d2.reshape(-1, 1, 2))]
        else:
            prediction_c = np.zeros(len(prediction_k))
        print(prediction_c)
        #
        # for ans, pc, pk in zip(self._test_a2, prediction_c, prediction_k):
        #     print(f"z = {ans}; pcm = {pc[0]}; pkm = {pk[0]}")

        custom_model_mse = np.mean((self._test_a2 - prediction_c) ** 2)
        keras_model_mse = np.mean((self._test_a2 - prediction_k) ** 2)
        return f"enn custom model mse = {custom_model_mse};\n enn keras model mse = {keras_model_mse}"
