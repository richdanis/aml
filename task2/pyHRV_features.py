import numpy as np
import pandas as pd
import pyhrv.time_domain as td
import pyhrv.tools as tools
from template_features import *

X = pd.read_csv("X_train.csv")
feat = dict()

for i in range(X.shape[0]):

    signal_raw = X.iloc[i].dropna().to_numpy(dtype='float32')

    templates, r_peaks, rr = pre_process(signal_raw)
    try:
        nni = tools.nn_intervals(r_peaks)
        signal_results = td.time_domain(rpeaks=r_peaks, nni=nni, plot=False, show=False)
        for key in signal_results.keys():
            if key != "sdnn_index" and key != "sdann" and key != "nni_histogram":
                if key not in feat.keys():
                    feat[key] = [signal_results[key]]
                else:
                    feat[key].append(signal_results[key])
    except:
        for key in feat.keys():
            feat[key].append(np.nan)

pd.DataFrame(feat).to_csv("hsv_features.csv")