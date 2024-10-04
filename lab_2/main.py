import network_analyzer
from tools import data_processing

NetworkAnalyzer_ = network_analyzer.NetworkAnalyzer()
NetworkAnalyzer_.fit_networks(100, 0.01, 1)
