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
    templates=ECG_analysis['templates']
    
    # check for flipped signals
    templates_min = np.min(np.median(templates, axis=0))
    templates_max = np.max(np.median(templates, axis=0))
    if np.abs(templates_min) > np.abs(templates_max):
        signal *= -1
        ECG_analysis = ecg.ecg(signal=signal, sampling_rate=freq, show=False)
        templates=ECG_analysis['templates']
    ts=ECG_analysis['ts']
    signal_flt=ECG_analysis['filtered']
    rpeaks=ECG_analysis['rpeaks']
    templates_ts=ECG_analysis['templates_ts']

    # normalization
    signal /= templates_max
    signal_flt /= templates_max
    templates /= templates_max
    
    return ts, signal, signal_flt, rpeaks, templates_ts, templates





def check_R(rpeaks, templates_ts, templates, corr_thresh=0.9, freq=300.0):

    # Check lengths
    assert len(rpeaks) == len(templates)
    
    r_id = int(-templates_ts[0]*freq)
    median = np.median(templates, axis=0)
    anom=set()
    
    # Loop through rpeaks
    for i in range(len(templates)):
        correlation_coefficient = np.corrcoef(median[r_id - 25:r_id + 25],templates[i, r_id - 25:r_id + 25])
        # Check correlation
        if correlation_coefficient[0, 1] < corr_thresh:
            anom.add(i)
    return anom

"""            # Compute cross correlation
            cross_correlation = signal.correlate(
                self.median_template[self.template_rpeak_sp - 25:self.template_rpeak_sp + 25],
                self.templates[self.template_rpeak_sp - 25:self.template_rpeak_sp + 25, i]
            )

            # Correct rpeak
            rpeak_corrected = \
                self.rpeaks[i] - \
                (np.argmax(cross_correlation) -
                    len(self.median_template[self.template_rpeak_sp - 25:self.template_rpeak_sp + 25]))

            # Check to see if shifting the R-Peak improved the correlation coefficient
            if self.check_improvement(rpeak_corrected, correlation_threshold):

                # Update rpeaks array
                self.rpeaks[i] = rpeak_corrected

    # Re-extract templates
    self.templates, self.rpeaks = self.extract_templates(self.rpeaks)

    # Re-compute median template
    self.median_template = np.median(self.templates, axis=1)

    # Check lengths
    assert len(self.rpeaks) == self.templates.shape[1]"""