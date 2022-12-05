import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import biosppy.signals.ecg as ecg
from biosppy.signals.tools import filter_signal
import scipy as sc
import neurokit2 as nk
from full_preprocess import *


def interval(a, b):
    select = ~np.logical_or(np.isnan(a), np.isnan(b))
    a = a[select]
    b = b[select]

    if not np.any(a) or not np.any(b):
        return np.array([np.nan, np.nan, np.nan, np.nan])

    c = b - a

    return np.array([np.median(c), c.max(), c.min(), c.std()])


def slope(a, b, signal):
    select = ~np.logical_or(np.isnan(a), np.isnan(b))
    a = a[select]
    b = b[select]

    if not np.any(a) or not np.any(b):
        return np.array([np.nan, np.nan, np.nan, np.nan])

    a = a.astype(int)
    b = b.astype(int)

    slopes = np.empty((a.shape[0],))
    for i in range(a.shape[0]):
        slopes[i] = (signal[b[i]] - signal[a[i]]) / (b[i] - a[i])

    return np.array([np.median(slopes), slopes.max(), slopes.min(), slopes.std()])


def amplitudes(a, signal):
    if not np.any(a):
        return np.array([np.nan, np.nan, np.nan, np.nan, np.nan])

    a = a[~np.isnan(a)]
    a = a.astype(int)
    amps = signal[a]

    return np.array([np.median(amps), amps.max(), amps.min(), amps.std(), a.std()])


def p_to_r_count(a):
    if not np.any(a):
        return np.array([np.nan])
    return np.array([(~np.isnan(a)).sum() / a.shape[0]])


X = pd.read_csv("X_test.csv")

feat = None
for i in range(X.shape[0]):

    signal_raw = X.iloc[i].dropna().to_numpy(dtype='float32')

    templates, r_peaks, rr, signal = pre_process(signal_raw)

    current = None

    try:

        _, pos = nk.ecg_delineate(signal, r_peaks, sampling_rate=300)

        if r_peaks.shape[0] > 1:
            r_peaks = r_peaks.astype(int)
            p_peaks = np.array(pos['ECG_P_Peaks'])
            q_peaks = np.array(pos['ECG_Q_Peaks'])
            s_peaks = np.array(pos['ECG_S_Peaks'])
            t_peaks = np.array(pos['ECG_T_Peaks'])

            curr = interval(p_peaks, r_peaks)
            curr = np.hstack((curr, interval(q_peaks, t_peaks)))
            curr = np.hstack((curr, interval(q_peaks, s_peaks)))
            curr = np.hstack((curr, interval(s_peaks, t_peaks)))

            curr = np.hstack((curr, slope(q_peaks, r_peaks, signal)))
            curr = np.hstack((curr, slope(r_peaks, s_peaks, signal)))
            curr = np.hstack((curr, slope(s_peaks, t_peaks, signal)))

            curr = np.hstack((curr, amplitudes(r_peaks, signal)))
            curr = np.hstack((curr, amplitudes(p_peaks, signal)))
            curr = np.hstack((curr, amplitudes(q_peaks, signal)))
            curr = np.hstack((curr, amplitudes(s_peaks, signal)))
            curr = np.hstack((curr, amplitudes(t_peaks, signal)))

            curr = np.hstack((curr, p_to_r_count(p_peaks)))

    except:

        curr = np.full((54,), np.nan)

    if feat is None:
        feat = curr
    else:
        feat = np.vstack((feat, curr))

pd.DataFrame(feat).to_csv("neurokit_morphological_test.csv")
