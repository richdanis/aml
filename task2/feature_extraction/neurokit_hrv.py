import numpy as np
import pandas as pd
import neurokit2 as nk
from full_preprocess import *

X = pd.read_csv("X_test.csv")

feat = None
for i in range(X.shape[0]):

  current = None
  signal_raw = X.iloc[i].dropna().to_numpy(dtype='float32')

  templates, r_peaks, rr, signal = pre_process(signal_raw)

  try:

    hrv_time = nk.hrv_time(r_peaks, sampling_rate=300)
    hrv_frequency = nk.hrv_frequency(r_peaks, sampling_rate=300)
    hrv_nonlinear = nk.hrv_nonlinear(r_peaks, sampling_rate=300)

    current = pd.concat((hrv_time, hrv_frequency, hrv_nonlinear), axis=1)

  except:
    new_row = dict()
    for col in feat.columns:
      new_row[col] = np.nan
    current = pd.DataFrame(new_row, index=[i])

  if feat is None:
    feat = current
  else:
    feat = pd.concat((feat, current), axis=0)

feat = feat.dropna(thresh=feat.shape[0]*0.6,how='all',axis=1)

feat.to_csv("hrv_neurokit_test.csv", index=False)