import numpy as np
import tensorflow as tf
from tensorflow import keras

from lab_2_fce.lab_2.models.fnn import FNN
from lab_2_fce.lab_2.tools import data_processing as dp, ploter, filer
from lab_2_fce.lab_2_keras.models.keras_fnn import KerasFNN


class NetworkComparor:

    def __init__(self, input_neurons, hidden_layer_count, hidden_neurons):
        self._custom_model = FNN(input_neurons, hidden_layer_count, hidden_neurons)
        self._keras_model = KerasFNN(input_neurons, hidden_layer_count, hidden_neurons)
        self._test_a2 = None
        self._test_d2 = None

    def fit(self, data_set, n=200, epochs=100):
        _train_d, _train_a, self._test_d2, self._test_a2 =
        test_size = int(n / 5)
        # self._train_d3, self._test_a3 = dp.prepared_data(-1, 1, test_size, False)

        cm_stat = self._custom_model.fit(_train_d, _train_a, epochs=epochs)
        km_stat = self._keras_model.fit(_train_d, _train_a, epochs=epochs)

    def predict(self):
        prediction_k = self._keras_model.predict(self._test_d2)
        prediction_c = [self._custom_model.predict(i) for i in self._test_d2]
        # for ans, pc, pk in zip(self._test_a2, prediction_c, prediction_k):
        #     print(f"z = {ans}; pcm = {pc[0]}; pkm = {pk[0]}")
        custom_model_mse = np.mean((self._test_a2 - prediction_c)**2)
        keras_model_mse = np.mean((self._test_a2 - prediction_k)**2)
        return f"fnn custom model mse = {custom_model_mse};\n fnn keras model mse = {keras_model_mse}"
