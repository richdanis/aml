import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import biosppy.signals.ecg as ecg
import scipy as sc
from hrvanalysis import remove_outliers, remove_ectopic_beats, interpolate_nan_values
from biosppy.signals.tools import filter_signal
from hrvanalysis import get_time_domain_features, get_geometrical_features, get_csi_cvi_features, get_poincare_plot_features, get_sampen

X = pd.read_csv("X_train.csv")

feat = dict()

for i in range(X.shape[0]):

    signal_raw = X.iloc[i].dropna().to_numpy(dtype='float32')

    order = int(0.3 * 300)
    signal, _, _ = filter_signal(signal=signal_raw,
                                 ftype='FIR',
                                 band='bandpass',
                                 order=order,
                                 frequency=[3, 45],
                                 sampling_rate=300)
    (r_peaks,) = ecg.hamilton_segmenter(signal_raw, sampling_rate=300)
    (r_peaks,) = ecg.correct_rpeaks(
        signal=signal, rpeaks=r_peaks, sampling_rate=300, tol=0.05
    )

    rr_intervals = np.diff(r_peaks) * 1000 / 300

    # This remove outliers from signal
    rr_intervals_without_outliers = remove_outliers(rr_intervals=rr_intervals,
                                                    low_rri=300, high_rri=2000, verbose=False)
    # This replace outliers nan values with linear interpolation
    interpolated_rr_intervals = interpolate_nan_values(rr_intervals=rr_intervals_without_outliers,
                                                       interpolation_method="linear")

    # This remove ectopic beats from signal
    nn_intervals_list = remove_ectopic_beats(rr_intervals=interpolated_rr_intervals, method="malik", verbose=False)
    # This replace ectopic beats nan values with linear interpolation
    nn_intervals = interpolate_nan_values(rr_intervals=nn_intervals_list)

    try:
        f1 = time_domain_features = get_time_domain_features(nn_intervals)
        f1.update(get_geometrical_features(nn_intervals))
        f1.update(get_csi_cvi_features(nn_intervals))
        f1.update(get_poincare_plot_features(nn_intervals))
        f1.update(get_sampen(nn_intervals))

        del f1['tinn']
        for key in f1.keys():
            if key not in feat.keys():
                feat[key] = [f1[key]]
            else:
                feat[key].append(f1[key])

    except:

        for key in feat.keys():
            feat[key].append(np.nan)

pd.DataFrame(feat).to_csv("hrv-analysis_features.csv")