import numpy as np
import scipy as sp

from scipy import signal as signal_sp
import pywt
from utils import hfd



def full_waveform_statistics(signal, freq=300):

        # Empty dictionary
        feat = dict()

        # Calculate statistics
        feat['min'] = np.min(signal)
        feat['max'] = np.max(signal)
        feat['mean'] = np.mean(signal)
        feat['median'] = np.median(signal)
        feat['std'] = np.std(signal)
        feat['skew'] = sp.stats.skew(signal)
        feat['kurtosis'] = sp.stats.kurtosis(signal)
        feat['duration'] = len(signal)/freq

        return feat
    

def decomposition_level(n_steps, level):

    remainder = n_steps % 2**level
    
    # Set starting multiplication factor
    factor = 1

    # Set updated waveform length variable
    n_steps_new = n_steps

    # Loop through multiplication factors until minimum factor found
    while remainder != 0:

        # Update multiplication factor
        factor += 1

        # Update waveform length
        n_steps_new = factor * n_steps

        # Calculate updated remainder
        remainder = n_steps_new % 2**level

    return n_steps_new



def padding(signal, n_steps_new):

    # Calculate required padding
    pad_count = int(np.abs(len(signal) - n_steps_new))

    # Calculate before waveform padding
    pad_before = int(np.floor(pad_count / 2.0))

    # Calculate after waveform padding
    pad_after = pad_count - pad_before

    # Add padding to waveform
    signal_padded = np.append(np.zeros(pad_before), np.append(signal, np.zeros(pad_after)))

    return signal_padded, pad_before, pad_after



def stationary_wavelet_transform(signal, wavelet, level):

    # Calculate waveform length
    n_steps = len(signal)

    # Calculate minimum waveform length for SWT of certain decomposition level
    n_steps_new = decomposition_level(n_steps, level)

    # Add necessary padding to waveform
    signal_padded, pad_before, pad_after = padding(signal, n_steps_new)

    # Compute stationary wavelet transform
    swt = pywt.swtn(signal_padded, wavelet=wavelet, level=level, start_level=0)

    # Loop through decomposition levels and remove padding
    for lev in range(len(swt)):
        for detail_level in ['d','a']:
            swt[lev][detail_level] = swt[lev][detail_level][pad_before:len(signal_padded)-pad_after]

    return swt



def wavelet_statistics(signal, freq=300.0):

    # Empty dictionary
    feat = dict()

    # Decomposition level
    decomp_level = 4

    # Stationary wavelet transform
    swt = stationary_wavelet_transform(signal, wavelet='db4', level=decomp_level)

    # Set frequency band
    freq_band = {
        "low": (3, 10),
        "med": (10, 30),
        "high": (30, 45)
        }

    """
    Frequency Domain
    """
    for level in range(len(swt)):
        for detail_level in ['d','a']:
            for band_level in freq_band:

  
                # Compute Welch periodogram
                fxx, pxx = signal_sp.welch(x=swt[level][detail_level], fs=freq)

                # Get frequency band
                band_index = np.logical_and(fxx >= freq_band[band_level][0], fxx < freq_band[band_level][1])
        
                # Calculate maximum power
                max_power = np.max(pxx[band_index])

                # Calculate average power
                mean_power = np.trapz(y=pxx[band_index], x=fxx[band_index])

                # Calculate max/mean power ratio
                feat[f'swt_{detail_level}_{level+1}_{band_level}_power_ratio'] = max_power/mean_power


            # Log-energy entropy
            feat[f'swt_{detail_level}_{level+1}_energy_entropy'] = np.sum(np.log10(np.power(swt[level][detail_level], 2)))

            # Higuchi_fractal
            feat[f'swt_{detail_level}_{level+1}_higuchi_fractal'] = hfd(swt[level][detail_level], k_max=10)


    return feat

