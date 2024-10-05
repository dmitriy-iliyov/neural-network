import time

import network_analyzer

NetworkAnalyzer_ = network_analyzer.NetworkAnalyzer()
for i in range(10):
    start = time.time()
    NetworkAnalyzer_.fit_networks(100, 0.05, 1, 200)
    print(f"executing time : {time.time() - start:.0f} secs")
    print("sleeping...\n")
    time.sleep(600)
NetworkAnalyzer_.print_resulting_plots(100, 40)
