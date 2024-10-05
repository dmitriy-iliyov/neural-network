import time

import network_analyzer

NetworkAnalyzer_ = network_analyzer.NetworkAnalyzer()
for i in range(10):
    start = time.time()
    NetworkAnalyzer_.fit_networks(100, 0.05, 1, 100)
    print(f"executing time : {time.time() - start}")
    print("sleeping...")
    time.sleep(300)
NetworkAnalyzer_.print_resulting_plots()
