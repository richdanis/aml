import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import biosppy.signals.ecg as ecg
from biosppy.signals.tools import filter_signal
import scipy as sc


def apply_filter(signal, filter_bandwidth=[3, 45]):
    # Apply bandpass filter to smoothen the signal

    # Calculate filter order
    order = int(0.3 * 300)

    # Filter signal
    signal, _, _ = filter_signal(signal=signal,
                                 ftype='FIR',
                                 band='bandpass',
                                 order=order,
                                 frequency=filter_bandwidth,
                                 sampling_rate=300)

    return signal


def invert(signal, templates, signal_raw):
    median_template = np.median(templates, axis=1)
    t_min = np.min(median_template)
    t_max = np.max(median_template)

    if np.abs(t_min) > np.abs(t_max):
        signal *= -1
        templates *= -1
        signal_raw *= -1

    return signal, templates, signal_raw


def normalize(signal, templates, signal_raw):
    median_template = np.median(templates, axis=1)
    templates_max = np.max(median_template)

    # Normalize ECG signals
    signal_raw /= templates_max
    signal /= templates_max
    templates /= templates_max

    return signal, templates, signal_raw


def r_peak_check(signal, templates, rpeaks, correlation_threshold=0.9):
    # Calculate correlation of each heartbeat with median heartbeat in region
    # [r_peak_pos-25, r_peak_pos+25], correct r_peak_pos if correlation improved

    # Check lengths
    assert len(rpeaks) == templates.shape[1]
    median_template = np.median(templates, axis=1)
    r_peak_pos = int(300 * 0.25)

    # Loop through rpeaks
    for template_id in range(templates.shape[1]):

        # Calculate correlation coefficient
        correlation_coefficient = np.corrcoef(
            median_template[r_peak_pos - 25:r_peak_pos + 25],
            templates[r_peak_pos - 25:r_peak_pos + 25, template_id]
        )

        # Check correlation
        if correlation_coefficient[0, 1] < correlation_threshold:

            # Compute cross correlation
            cross_correlation = sc.signal.correlate(
                median_template[r_peak_pos - 25:r_peak_pos + 25],
                templates[r_peak_pos - 25:r_peak_pos + 25, template_id]
            )

            # Correct rpeak
            rpeak_corrected = \
                rpeaks[template_id] - \
                (np.argmax(cross_correlation) -
                 len(median_template[r_peak_pos - 25:r_peak_pos + 25]))

            # Check to see if shifting the R-Peak improved the correlation coefficient
            if check_improvement(signal, templates, rpeak_corrected, correlation_threshold):
                # Update rpeaks array
                rpeaks[template_id] = rpeak_corrected

    # Re-extract templates
    templates, rpeaks = ecg.extract_heartbeats(signal, rpeaks, sampling_rate=300, before=0.25, after=0.4)
    templates = templates.T

    # Check lengths
    assert len(rpeaks) == templates.shape[1]

    return templates, rpeaks


def check_improvement(signal, templates, rpeak_corrected, correlation_threshold):
    r_peak_pos = int(300 * 0.25)
    # Before R-Peak
    a = rpeak_corrected - int(300 * 0.25)

    # After R-Peak
    b = rpeak_corrected + int(300 * 0.4)

    median_template = np.median(templates, axis=1)

    if a >= 0 and b < len(signal):

        # Update template
        template_corrected = signal[a:b]

        # Calculate correlation coefficient
        correlation_coefficient = np.corrcoef(
            median_template[r_peak_pos - 25:r_peak_pos + 25],
            template_corrected[r_peak_pos - 25:r_peak_pos + 25]
        )

        # Check new correlation
        if correlation_coefficient[0, 1] >= correlation_threshold:
            return True
        else:
            return False
    else:
        return False


def filter_rpeaks(templates, rpeaks, correlation_threshold=0.9):
    # Calculate correlation of each heartbeat with median heartbeat in region
    # [r_peak_pos-25, r_peak_pos+25], remove if correlation lower than 0.9

    # Get rpeaks is floats
    rpeaks = rpeaks.astype(float)
    median_template = np.median(templates, axis=1)
    r_peak_pos = int(300 * 0.25)

    # Loop through templates
    for template_id in range(templates.shape[1]):

        # Calculate correlation coefficient
        correlation_coefficient = np.corrcoef(
            median_template[r_peak_pos - 25:r_peak_pos + 25],
            templates[r_peak_pos - 25:r_peak_pos + 25, template_id]
        )

        # Check correlation
        if correlation_coefficient[0, 1] < correlation_threshold:
            rpeaks[template_id] = np.nan

    # Get good rpeaks
    rpeaks_good = rpeaks[np.isfinite(rpeaks)]

    # Get good and bad
    templates_good = templates[:, np.where(np.isfinite(rpeaks))[0]]

    return templates_good, rpeaks_good


def pre_process(signal_raw):
    # filter
    signal = apply_filter(signal_raw)

    (r_peaks,) = ecg.hamilton_segmenter(signal, sampling_rate=300)

    (r_peaks,) = ecg.correct_rpeaks(
        signal=signal, rpeaks=r_peaks, sampling_rate=300, tol=0.05
    )

    # rr intervals
    rr = np.diff(r_peaks) * 1 / 300

    templates, r_peaks = ecg.extract_heartbeats(signal, r_peaks, sampling_rate=300, before=0.25, after=0.4)

    # continue
    templates = templates.T
    # templates are of shape (number sample points, number beats)

    # multiply inverted signals by -1
    signal, templates, signal_raw = invert(signal, templates, signal_raw)

    # normalize
    signal, templates, signal_raw = normalize(signal, templates, signal_raw)
    # steps above are done for all features

    # correct some shifted r_peaks -> this is done for template and hrv statistics
    templates, r_peaks = r_peak_check(signal, templates, r_peaks)

    # filter noisy heartbeats based on correlation with median heartbeat
    # -> this is only done for template statistics
    templates, r_peaks = filter_rpeaks(templates, r_peaks)

    return templates, r_peaks, rr, signal


def filter_invert(signal_raw):
    signal = apply_filter(signal_raw)
    (r_peaks,) = ecg.hamilton_segmenter(signal, sampling_rate=300)
    (r_peaks,) = ecg.correct_rpeaks(signal=signal, rpeaks=r_peaks, sampling_rate=300, tol=0.05)
    templates, r_peaks = ecg.extract_heartbeats(signal, r_peaks, sampling_rate=300, before=0.25, after=0.4)
    templates = templates.T
    signal, templates, signal_raw = invert(signal, templates, signal_raw)
    templates, r_peaks = r_peak_check(signal, templates, r_peaks)
    signal, templates, signal_raw = invert(signal, templates, signal_raw)
    return signal
