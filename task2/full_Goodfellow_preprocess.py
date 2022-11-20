import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
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


def extract_templates(signal, rpeaks, before=0.25, after=0.4):

  # Extract heartbeat templates based on hardcoded values before and after r_peak
  
  # convert delimiters to samples
  before = int(before * 300)
  after = int(after * 300)

  # Sort R-Peaks in ascending order
  rpeaks = np.sort(rpeaks)

  # Get number of sample points in waveform
  length = len(signal)

  # Create empty list for templates
  templates = []

  # Create empty list for new rpeaks that match templates dimension
  rpeaks_new = np.empty(0, dtype=int)

  # Loop through R-Peaks
  for rpeak in rpeaks:

      # Before R-Peak
      a = rpeak - before
      if a < 0:
          continue

      # After R-Peak
      b = rpeak + after
      if b > length:
          break

      # Append template list
      templates.append(signal[a:b])

      # Append new rpeaks list
      rpeaks_new = np.append(rpeaks_new, rpeak)

  # Convert list to numpy array
  templates = np.array(templates).T

  return templates, rpeaks_new

def invert(signal, templates, signal_raw):

  t_min = np.min(np.median(templates, axis=1))
  t_max = np.max(np.median(templates, axis=1))

  if np.abs(t_min) > np.abs(t_max):
    
    signal *= -1
    templates *= -1
    signal_raw *= -1

  return signal, templates, signal_raw

def normalize(signal, templates, signal_raw):

  templates_max = np.max(np.median(templates, axis=1))

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
  r_peak_pos = int(300*0.25)

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
  templates, rpeaks = extract_templates(signal, rpeaks)

  # Re-compute median template
  median_template = np.median(templates, axis=1)

  # Check lengths
  assert len(rpeaks) == templates.shape[1]

  return templates, rpeaks

def check_improvement(signal, templates, rpeak_corrected, correlation_threshold):
  
  r_peak_pos = int(300*0.25)
  # Before R-Peak
  a = rpeak_corrected - int(300*0.25)

  # After R-Peak
  b = rpeak_corrected + int(300*0.4)

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
  r_peak_pos = int(300*0.25)

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

  # Get good and bad rpeaks
  rpeaks_good = rpeaks[np.isfinite(rpeaks)]
  rpeaks_bad = rpeaks[~np.isfinite(rpeaks)]

  # Get good and bad
  templates_good = templates[:, np.where(np.isfinite(rpeaks))[0]]
  if len(np.where(~np.isfinite(rpeaks))[0]) > 0:
      templates_bad = templates[:, np.where(~np.isfinite(rpeaks))[0]]

  # Get median templates
  median_template_good = np.median(templates_good, axis=1)
  if len(np.where(~np.isfinite(rpeaks))[0]) > 0:
      median_template_bad = np.median(templates_bad, axis=1)

  return templates_good, rpeaks_good