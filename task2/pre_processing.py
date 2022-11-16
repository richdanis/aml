import numpy as np 
from biosppy.signals.tools import filter_signal as filter_biosppy
from biosppy.signals import ecg 

def filter_signal(signal, freq=300.0, bandwidth=[3, 45]):
    order = int(0.3*freq)
    signal, _, _ = filter_biosppy(signal=signal,
                                      ftype='FIR',
                                      band='bandpass',
                                      order=order,
                                      frequency=bandwidth,
                                      sampling_rate=freq)
    return signal

"""
def ecg(signal, freq=300.0):
    ECG_analysis = ecg.ecg(signal=signal, sampling_rate=freq, show=False)
    ts=ECG_analysis['ts']
    signal=ECG_analysis['filtered']
    rpeaks=ECG_analysis['rpeaks']
    templates_ts=ECG_analysis['templates_ts']
    templates=ECG_analysis['templates']
    return rpeaks
"""
    
def pre_process(signal, freq=300.0):
    ECG_analysis = ecg.ecg(signal=signal, sampling_rate=freq, show=False)
    ts=ECG_analysis['ts']
    signal_flt=ECG_analysis['filtered']
    rpeaks=ECG_analysis['rpeaks']
    templates_ts=ECG_analysis['templates_ts']
    templates=ECG_analysis['templates']
    
    # check for flipped signals
    templates_min = np.min(np.median(templates, axis=0))
    templates_max = np.max(np.median(templates, axis=0))
    if np.abs(templates_min) > np.abs(templates_max):
        signal *= -1
        signal_flt *= -1
        templates *= -1

    # normalization
    signal /= templates_max
    signal_flt /= templates_max
    templates /= templates_max
    
    return ts, signal, signal_flt, rpeaks, templates_ts, templates