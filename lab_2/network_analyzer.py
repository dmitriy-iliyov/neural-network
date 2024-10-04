import numpy as np

from tools import data_processing as dp, ploter
from models import fnn, cnn, enn
from tools import filer


class NetworkAnalyzer:

    def __init__(self):
        self.td2 = None
        self.td1 = None
        self.a1 = None
        self.a2 = None
        self.network_list = [fnn.FNN(1, 10),
                             cnn.CNN(1, 20),
                             enn.ENN(1, 15)]
        self.network_list_2 = [fnn.FNN(1, 20),
                               cnn.CNN(2, 10),
                               enn.ENN(3, 5)]

    def fit_networks(self, epochs, learning_rate, batch_size):
        self.td1, self.a1 = dp.prepared_data(-1, 1, 100)
        self.td2, self.a2 = dp.prepared_data(0, 10, 20)
        self._fit_part(self.network_list, epochs, learning_rate, batch_size)
        self._fit_part(self.network_list_2, epochs, learning_rate, batch_size)

    def _fit_part(self, network_list, epochs, learning_rate, batch_size):
        test_result = {}
        accuracies = {}
        mses = {}
        for network in network_list:
            if isinstance(network, enn.ENN):
                time_series = 1
                _fit_data = network.fit(self.td1.reshape(-1, time_series, 2), self.a1, epochs, learning_rate,
                                        batch_size)
                ploter.one_fit_statistic(_fit_data)
                predictions = [i for i in network.predict(self.td2.reshape(-1, time_series, 2))]
            else:
                _fit_data = network.fit(self.td1, self.a1, epochs, learning_rate, batch_size)
                ploter.one_fit_statistic(_fit_data)
                predictions = [network.predict(i)[0] for i in self.td2]
            count = 0
            deviation = max(self.a2) / 100
            for i, p in enumerate(predictions):
                if abs(p - self.a2[i]) < deviation:
                    count += 1
            mses[_fit_data['network']] = np.mean((self.a2 - predictions) ** 2)
            test_result[_fit_data['network']] = count / len(self.a2)
            accuracies[_fit_data['network']] = (sum(_fit_data['accuracy'][-10:]) / 10)
        for key in test_result.keys():
            filer.save('tests/test_results.txt', f"{key} accuracy: {accuracies[key]:.3f}; score: {test_result[key]}\n")
            print(f"{key} accuracy: {accuracies[key]:.3f}; mse : {mses[key]:.5f}; score: {test_result[key]}")
