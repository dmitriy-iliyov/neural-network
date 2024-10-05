import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from lab_2_fce.lab_2.models.cnn import CNN
from lab_2_fce.lab_2.tools import data_processing as dp


class CnnModels:

    def __init__(self, input_neurons=2, hidden_layer_count=1, hidden_neurons=20, n=200, compare=True):
        self._train_d, self._train_a, self._test_d2, self._test_a2 = dp.prepared_data(-1, 1, n)

        if compare:
            self._custom_model = CNN(input_neurons, hidden_layer_count, hidden_neurons)

        inputs = keras.Input(shape=(input_neurons,))
        hidden_layer = layers.Dense(hidden_neurons, activation='relu')(inputs)
        for _ in range(1, hidden_layer_count):
            hidden_layer = layers.Dense(hidden_neurons, activation='relu')(hidden_layer)
        outputs = layers.Dense(1, activation='linear')(hidden_layer)
        self._keras_model = keras.Model(inputs=inputs, outputs=outputs)
        self._keras_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.05),
                                  loss='mean_squared_error',
                                  metrics=['mean_squared_error'])

    def fit(self):
        if self._custom_model:
            self._custom_model.fit(self._train_d, self._train_a, epochs=100)

        self._keras_model.fit(self._train_d, self._train_a, epochs=100)
        loss, mse = self._keras_model.evaluate(self._test_d2, self._test_a2)
        print("Keras Model - Loss:", loss)
        print("Keras Model - MSE:", mse)

    def predict(self):
        prediction_k = self._keras_model.predict(self._test_d2)
        if self._custom_model:
            prediction_c = [self._custom_model.predict(i) for i in self._test_d2]
        else:
            prediction_c = np.zeros(len(prediction_k))
        # for ans, pc, pk in zip(self._test_a2, prediction_c, prediction_k):
        #     print(f"z = {ans}; pcm = {pc[0]}; pkm = {pk[0]}")
        custom_model_mse = np.mean((self._test_a2 - prediction_c) ** 2)
        keras_model_mse = np.mean((self._test_a2 - prediction_k) ** 2)
        return f"cnn custom model mse = {custom_model_mse};\n cnn keras model mse = {keras_model_mse}"

