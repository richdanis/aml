import numpy as np
import biosppy.signals.ecg as ecg
from full_Goodfellow_preprocess import *


def calculate_qrs_bounds(templates, fs=300):
    # Empty lists of QRS start and end times
    qrs_starts_sp = []
    qrs_ends_sp = []

    rpeak_pos = int(300 * 0.25)
    after_rpeak = int(300 * 0.4)

    # Loop through templates
    for template in range(templates.shape[1]):

        # Get zero crossings before the R-Peak
        pre_qrs_zero_crossings = np.where(
            np.diff(np.sign(templates[0:rpeak_pos, template]))
        )[0]

        # Check length
        if len(pre_qrs_zero_crossings) >= 2:
            # Append QRS starting index
            qrs_starts_sp = np.append(qrs_starts_sp, pre_qrs_zero_crossings[-2])

        if len(qrs_starts_sp) > 0:

            qrs_start_sp = int(np.median(qrs_starts_sp))

        else:
            qrs_start_sp = int(rpeak_pos / 2.0)

        # Get zero crossings after the R-Peak
        post_qrs_zero_crossings = np.where(
            np.diff(np.sign(templates[rpeak_pos:-1, template]))
        )[0]

        # Check length
        if len(post_qrs_zero_crossings) >= 2:
            # Append QRS ending index
            qrs_ends_sp = np.append(qrs_ends_sp, post_qrs_zero_crossings[-2])

        if len(qrs_ends_sp) > 0:

            qrs_end_sp = int(rpeak_pos + np.median(qrs_ends_sp))

        else:
            qrs_end_sp = int(rpeak_pos + after_rpeak / 2.0)

    return qrs_start_sp, qrs_end_sp


def preprocess_pqrst(templates):
    rpeak_pos = int(300 * 0.25)
    median_template = np.median(templates, axis=1)
    # Get QRS start point
    qrs_start_sp = rpeak_pos - 30

    # Get QRS end point
    qrs_end_sp = rpeak_pos + 40

    # Get QR median template
    qr_median_template = median_template[qrs_start_sp:rpeak_pos]

    # Get RS median template
    rs_median_template = median_template[rpeak_pos:qrs_end_sp]

    # Get QR templates
    qr_templates = templates[qrs_start_sp:rpeak_pos, :]

    # Get RS templates
    rs_templates = templates[rpeak_pos:qrs_end_sp, :]

    """
    R-Wave
    """
    r_amps = templates[rpeak_pos, :]
    r_amp = median_template[rpeak_pos]
    r_amp_std = np.std(r_amps)

    """
    Q-Wave
    """
    # Get array of Q-wave times (sp)
    q_times_sp = np.array(
        [qrs_start_sp + np.argmin(qr_templates[:, col]) for col in range(qr_templates.shape[1])]
    )

    # Get array of Q-wave amplitudes
    q_amps = np.array(
        [templates[q_times_sp[col], col] for col in range(templates.shape[1])]
    )

    # Get array of Q-wave times (sp)
    q_pos = qrs_start_sp + np.argmin(qr_median_template)

    # Get array of Q-wave amplitudes
    q_amp = median_template[q_pos]
    q_amp_std = np.std(q_amps)
    q_times_std = np.std(q_times_sp)

    """
    P-Wave
    """
    # Get array of Q-wave times (sp)
    p_times_sp = np.array([
        np.argmax(templates[0:q_times_sp[col], col])
        for col in range(templates.shape[1])
    ])

    # Get array of Q-wave amplitudes
    p_amps = np.array(
        [templates[p_times_sp[col], col] for col in range(templates.shape[1])]
    )

    # Get array of Q-wave times (sp)
    p_pos = np.argmax(median_template[0:q_pos])

    # Get array of Q-wave amplitudes
    p_amp = median_template[p_pos]
    p_amp_std = np.std(p_amps)
    p_times_std = np.std(p_times_sp)

    """
    S-Wave
    """
    # Get array of Q-wave times (sp)
    s_times_sp = np.array([
        rpeak_pos + np.argmin(rs_templates[:, col])
        for col in range(rs_templates.shape[1])
    ])

    # Get array of Q-wave amplitudes
    s_amps = np.array(
        [templates[s_times_sp[col], col] for col in range(templates.shape[1])]
    )

    # Get array of Q-wave times (sp)
    s_pos = rpeak_pos + np.argmin(rs_median_template)

    # Get array of Q-wave amplitudes
    s_amp = median_template[s_pos]
    s_amp_std = np.std(s_amps)
    s_times_std = np.std(s_times_sp)

    """
    T-Wave
    """
    # Get array of Q-wave times (sp)
    t_times_sp = np.array([
        s_times_sp[col] + np.argmax(templates[s_times_sp[col]:, col])
        for col in range(templates.shape[1])
    ])

    # Get array of Q-wave amplitudes
    t_amps = np.array(
        [templates[t_times_sp[col], col] for col in range(templates.shape[1])]
    )

    # Get array of Q-wave times (sp)
    t_pos = s_pos + np.argmax(median_template[s_pos:])

    # Get array of Q-wave amplitudes
    t_amp = median_template[t_pos]
    t_amp_std = np.std(t_amps)
    t_times_std = np.std(t_times_sp)

    return r_amp, q_amp, p_amp, s_amp, t_amp, rpeak_pos, q_pos, p_pos, s_pos, t_pos, r_amp_std, q_amp_std, p_amp_std, \
           s_amp_std, t_amp_std, q_times_std, p_times_std, s_times_std, t_times_std


def calculate_qrs_correlation_statistics(templates):
    if templates.shape[1] > 1:

        # Get start and end points
        rpeak_pos = int(0.25 * 300)
        start_sp = rpeak_pos - 30
        end_sp = rpeak_pos + 40

        # Calculate correlation matrix
        correlation_matrix = np.corrcoef(np.transpose(templates[start_sp:end_sp, :]))

        # Get upper triangle
        upper_triangle = np.triu(correlation_matrix, k=1).flatten()

        # Get upper triangle index where values are not zero
        upper_triangle_index = np.triu(correlation_matrix, k=1).flatten().nonzero()[0]

        # Get upper triangle values where values are not zero
        upper_triangle = upper_triangle[upper_triangle_index]

        # Calculate correlation matrix statistics
        qrs_corr_coeff_median = np.median(upper_triangle)
        qrs_corr_coeff_std = np.std(upper_triangle, ddof=1)

    else:
        qrs_corr_coeff_median = np.nan
        qrs_corr_coeff_std = np.nan

    return qrs_corr_coeff_median, qrs_corr_coeff_std


def extract_template_features(X):
    # amplitudes
    median_r_amp = np.empty((X.shape[0],))
    median_q_amp = np.empty((X.shape[0],))
    median_p_amp = np.empty((X.shape[0],))
    median_s_amp = np.empty((X.shape[0],))
    median_t_amp = np.empty((X.shape[0],))

    # amplitude stds
    r_amp_std = np.empty((X.shape[0],))
    q_amp_std = np.empty((X.shape[0],))
    p_amp_std = np.empty((X.shape[0],))
    s_amp_std = np.empty((X.shape[0],))
    t_amp_std = np.empty((X.shape[0],))

    # wave time stds

    q_time_std = np.empty((X.shape[0],))
    p_time_std = np.empty((X.shape[0],))
    s_time_std = np.empty((X.shape[0],))
    t_time_std = np.empty((X.shape[0],))

    # intervals
    median_qrs = np.empty((X.shape[0],))
    median_pr = np.empty((X.shape[0],))
    median_qt = np.empty((X.shape[0],))
    median_qs = np.empty((X.shape[0],))

    # st value
    st = np.empty((X.shape[0],))

    # qrs correlation
    qrs_corr_coeff_med = np.empty((X.shape[0],))
    qrs_corr_coeff_std = np.empty((X.shape[0],))

    for i in range(X.shape[0]):

        signal_raw = X.iloc[i].dropna().to_numpy(dtype='float32')

        ecg_processed = ecg.ecg(signal_raw, 300, show=False)

        # filter
        signal = apply_filter(signal_raw)

        # get r_peaks
        r_peaks = ecg_processed['rpeaks']
        rr = np.median(np.diff(r_peaks)) * 1000 / 300

        # using extraction method of Goodfellow, they use 0.25 seconds before r_peak,
        # as opposed to 0.2 before r_peak as in ecg.ecg
        templates, r_peaks = extract_templates(signal, r_peaks)

        # templates are of shape (number sample points, number beats), this is a bit
        # confusing, but I left it as is to make adaption of the following functions
        # easier

        # multiply inverted signals by -1
        signal, templates, signal_raw = invert(signal, templates, signal_raw)

        # normalize
        signal, templates, signal_raw = normalize(signal, templates, signal_raw)

        # steps above are done for all features

        # correct some shifted r_peaks -> this is done for template and hrv statistics
        templates, r_peaks = r_peak_check(signal, templates, r_peaks)

        # filter noisy heartbeats based on correlation with median heartbeat
        # -> this is only done for template statistics
        before = r_peaks.shape[0]
        templates, r_peaks = filter_rpeaks(templates, r_peaks)
        median_template = np.median(templates, axis=1)

        if templates.shape[1] > 0:

            qrs_start, qrs_end = calculate_qrs_bounds(templates)
            r_amp, q_amp, p_amp, s_amp, t_amp, r_pos, q_pos, p_pos, s_pos, t_pos, r_std, q_std, p_std, s_std, t_std, q_times_std, p_times_std, s_times_std, t_times_std = preprocess_pqrst(
                templates)
            qrs_corr_med, qrs_corr_std = calculate_qrs_correlation_statistics(templates)

            # amplitudes
            median_r_amp[i] = r_amp
            median_q_amp[i] = q_amp
            median_p_amp[i] = p_amp
            median_s_amp[i] = s_amp
            median_t_amp[i] = t_amp

            # amplitude stds
            r_amp_std[i] = r_std
            q_amp_std[i] = q_std
            p_amp_std[i] = p_std
            s_amp_std[i] = s_std
            t_amp_std[i] = t_std

            # wave time stds
            q_time_std[i] = q_times_std
            p_time_std[i] = p_times_std
            s_time_std[i] = s_times_std
            t_time_std[i] = t_times_std

            # intervals
            # median_rr[i] = rr
            median_qrs[i] = qrs_end - qrs_start
            median_pr[i] = r_pos - p_pos
            median_qt[i] = t_pos - q_pos
            median_qs[i] = s_pos - q_pos

            # st
            st[i] = np.abs(median_template[int(round(qrs_start))] - np.mean(
                median_template[int(round(qrs_end)):int((round(qrs_end) + 0.04 * 300))]))

            # qrs correlation
            qrs_corr_coeff_med[i] = qrs_corr_med
            qrs_corr_coeff_std[i] = qrs_corr_std

        else:

            # amplitudes
            median_r_amp[i] = np.nan
            median_q_amp[i] = np.nan
            median_p_amp[i] = np.nan
            median_s_amp[i] = np.nan
            median_t_amp[i] = np.nan

            # amplitude stds
            r_amp_std[i] = np.nan
            q_amp_std[i] = np.nan
            p_amp_std[i] = np.nan
            s_amp_std[i] = np.nan
            t_amp_std[i] = np.nan

            # wave time stds
            q_time_std[i] = np.nan
            p_time_std[i] = np.nan
            s_time_std[i] = np.nan
            t_time_std[i] = np.nan

            # intervals
            median_qrs[i] = np.nan
            median_pr[i] = np.nan
            median_qt[i] = np.nan
            median_qs[i] = np.nan

            # st
            st[i] = np.nan

            # qrs correlation
            qrs_corr_coeff_med[i] = np.nan
            qrs_corr_coeff_std[i] = np.nan

    feat = dict()
    feat["Ramp"] = median_r_amp
    feat["Qamp"] = median_q_amp
    feat["Pamp"] = median_p_amp
    feat["Samp"] = median_s_amp
    feat["Tamp"] = median_t_amp
    feat["qrs"] = median_qrs
    feat["pr"] = median_pr
    feat["qt"] = median_qt
    feat["qs"] = median_qs
    feat["st"] = st
    feat["R_amp_std"] = r_amp_std
    feat["Q_amp_std"] = q_amp_std
    feat["P_amp_std"] = p_amp_std
    feat["S_amp_std"] = s_amp_std
    feat["T_amp_std"] = t_amp_std
    feat["Q_time_std"] = q_time_std
    feat["P_time_std"] = p_time_std
    feat["S_time_std"] = s_time_std
    feat["T_time_std"] = t_time_std
    feat["qrs_corr_coeff_med"] = qrs_corr_coeff_med
    feat["qrs_corr_coeff_std"] = qrs_corr_coeff_std

    return feat
