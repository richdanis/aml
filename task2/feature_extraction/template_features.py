import numpy as np
import biosppy.signals.ecg as ecg
from full_Goodfellow_preprocess import *


def calculate_qrs_bounds(templates):
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
    r_amp_skew = sc.stats.skew(r_amps)
    r_amp_kurtosis = sc.stats.kurtosis(r_amps)

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
    p_eng = np.sum(np.power(median_template[max(p_pos - 10, 0):(p_pos + 10)], 2))

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
    t_eng = np.sum(np.power(median_template[(t_pos - 10):(t_pos + 10)], 2))

    """
    Interwave
    """
    pr_std = np.std(rpeak_pos - p_times_sp)
    qt_std = np.std(t_times_sp - q_times_sp)
    qs_std = np.std(s_times_sp - q_times_sp)

    return r_amp, q_amp, p_amp, s_amp, t_amp, rpeak_pos, q_pos, p_pos, s_pos, \
           t_pos, r_amp_std, q_amp_std, p_amp_std, s_amp_std, t_amp_std, \
           q_times_std, p_times_std, s_times_std, t_times_std, r_amp_skew, \
           r_amp_kurtosis, p_eng, t_eng, pr_std, qt_std, qs_std


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


def arhythmia_index(dRR):
    # from https://www.cinc.org/Proceedings/2002/pdf/485.pdf
    # worked nicely for https://www.cinc.org/archives/2017/pdf/342-204.pdf
    if dRR.shape[0] < 3:
        return np.nan
    cat = np.zeros((dRR.shape[0] - 2,))
    a = 0.9
    b = 0.9
    c = 1.5

    for i in range(1, dRR.shape[0] - 1):
        if dRR[i] < 0.6 and dRR[i] < dRR[i + 1]:
            cat[i - 1] = 1
            pulse = 1
            for j in range(i + 1, dRR.shape[0] - 1):
                if (dRR[j - 1] < 0.8 and dRR[j] < 0.8 and dRR[j + 1] < 0.8) or (dRR[j - 1] + dRR[j] + dRR[j + 1] < 1.8):
                    cat[j - 1] = 1
                    pulse += 1
                elif pulse < 4:
                    cat[(i - 1):j] = np.zeros((j - i + 1,))

        if dRR[i] < a * dRR[i - 1] and dRR[i - 1] < b * dRR[i + 1]:
            cat[i - 1] = 1

        if dRR[i] > c * dRR[i - 1]:
            cat[i - 1] = 1

    return np.sum(cat) / cat.shape[0]


# from https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=4360105&tag=1
def bin_count(seg):
    m = seg
    bc = 0
    pc = 0

    for i in range(-2, 3):
        b = np.sum(seg.diagonal(i) != 0)
        p = np.sum(seg.diagonal(i))
        bc += b
        pc += p
        m = m - np.diag(m.diagonal(i), i)

    return bc, pc, m


def af_evidence(rr, origin_boundary=20, bin_edge=600):
    drr = np.diff(rr) * 1000
    # need drr in ms

    # from -600 ms to 600ms -> 1200ms / 40 = 30 bins in each direction
    # 15 is middle
    bins = np.zeros((30, 30))
    # need regions 1,2,3,4 (hard), 5,6,10 (somewhat easier, especially 5,6) and 7,8,12 (somewhat easier, especially 7,8)

    lorentz = np.stack((drr[1:], drr[:-1]), axis=-1)

    originCount = np.sum(np.abs(lorentz), axis=1)
    originCount = (originCount <= origin_boundary).sum()

    # removing outliers
    cleaned = []
    for i in range(lorentz.shape[0]):
        if np.sum(np.abs(lorentz[i])) < 1500:
            cleaned.append(lorentz[i])

    lorentz = np.array(cleaned)
    if not cleaned:
        lorentz = np.array([[0, 0]])

    h, _, _ = np.histogram2d(lorentz[:, 0], lorentz[:, 1], bins=30,
                             range=[[-bin_edge, bin_edge], [-bin_edge, bin_edge]])

    # clearing the zero segment
    h[13, 14:16] = 0
    h[14:16, 13:17] = 0
    h[16, 14:16] = 0

    irregularity_evidence = np.sum(h != 0)

    # calculate bin count and point count for segments 9, 10, 11, 12
    seg_12 = h[15:, 15:]
    bc_12, pc_12, m = bin_count(seg_12)
    h[15:, 15:] = m

    seg_11 = h[15:, :15]
    seg_11 = np.fliplr(seg_11)
    bc_11, pc_11, m = bin_count(seg_11)
    h[15:, :15] = np.fliplr(m)

    seg_10 = h[:15, :15]
    bc_10, pc_10, m = bin_count(seg_10)
    h[:15, :15] = m

    seg_9 = h[:15, 15:]
    seg_9 = np.fliplr(seg_9)
    bc_9, pc_9, m = bin_count(seg_9)
    h[:15, 15:] = np.fliplr(m)

    # calculate bin count and point count for segments 5, 6, 7, 8
    seg_5 = h[:15, 13:17]
    bc_5 = np.sum(seg_5 != 0)
    pc_5 = np.sum(seg_5)

    seg_6 = h[13:17, :15]
    bc_6 = np.sum(seg_6 != 0)
    pc_6 = np.sum(seg_6)

    seg_7 = h[15:, 13:17]
    bc_7 = np.sum(seg_7 != 0)
    pc_7 = np.sum(seg_7)

    seg_8 = h[13:17, 15:]
    bc_8 = np.sum(seg_8 != 0)
    pc_8 = np.sum(seg_8)

    # clear the segments 5, 6, 7, 8 for easier calculation of 1, 2, 3, 4
    h[13:17, :] = 0
    h[:, 13:17] = 0

    # calculate bin count and point count for segments 1, 2, 3, 4
    seg_1 = h[:13, 17:]
    bc_1 = np.sum(seg_1 != 0)
    pc_1 = np.sum(seg_1)

    seg_2 = h[:13, :13]
    bc_2 = np.sum(seg_2 != 0)
    pc_2 = np.sum(seg_2)

    seg_3 = h[17:, :13]
    bc_3 = np.sum(seg_3 != 0)
    pc_3 = np.sum(seg_3)

    seg_4 = h[17:, 17:]
    bc_4 = np.sum(seg_4 != 0)
    pc_4 = np.sum(seg_4)

    pace_evidence = (pc_1 - bc_1) + (pc_2 - bc_2) + (pc_3 - bc_3) + (pc_4 - bc_4) \
                    + (pc_5 - bc_5) + (pc_6 - bc_6) + (pc_10 - bc_10) \
                    - (pc_7 - bc_7) - (pc_8 - bc_8) - (pc_12 - bc_12)

    af_evidence = irregularity_evidence - 2 * pace_evidence - originCount

    return af_evidence


def extract_template_features(X):
    # template features

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

    pr_stdv = np.empty((X.shape[0],))
    qt_stdv = np.empty((X.shape[0],))
    qs_stdv = np.empty((X.shape[0],))

    # st value
    st = np.empty((X.shape[0],))

    # qrs correlation
    qrs_corr_coeff_med = np.empty((X.shape[0],))
    qrs_corr_coeff_std = np.empty((X.shape[0],))

    # arhythmia index
    arhythm_idx = np.empty((X.shape[0],))

    # r_amp features
    r_kurtosis = np.empty((X.shape[0],))
    r_skew = np.empty((X.shape[0],))

    # energies
    p_energy = np.empty((X.shape[0],))
    t_energy = np.empty((X.shape[0],))

    for i in range(X.shape[0]):

        signal_raw = X.iloc[i].dropna().to_numpy(dtype='float32')

        templates, r_peaks, rr = pre_process(signal_raw)

        if templates.shape[1] > 0:

            median_template = np.median(templates, axis=1)

            qrs_start, qrs_end = calculate_qrs_bounds(templates)
            r_amp, q_amp, p_amp, s_amp, t_amp, r_pos, q_pos, p_pos, s_pos, t_pos, r_std, q_std, p_std, \
            s_std, t_std, q_times_std, p_times_std, s_times_std, t_times_std, r_amp_skew, r_amp_kurtosis, \
            p_eng, t_eng, pr_std, qt_std, qs_std = preprocess_pqrst(templates)

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

            pr_stdv[i] = pr_std
            qt_stdv[i] = qt_std
            qs_stdv[i] = qs_std

            # intervals
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

            # arhythmia index
            arhythm_idx[i] = arhythmia_index(rr)

            # r_amp features
            r_kurtosis[i] = r_amp_kurtosis
            r_skew[i] = r_amp_skew

            # energies
            p_energy[i] = p_eng
            t_energy[i] = t_eng


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

            pr_stdv[i] = np.nan
            qt_stdv[i] = np.nan
            qs_stdv[i] = np.nan

            # st
            st[i] = np.nan

            # qrs correlation
            qrs_corr_coeff_med[i] = np.nan
            qrs_corr_coeff_std[i] = np.nan

            # arhythmia index
            arhythm_idx[i] = np.nan

            # r_amp features
            r_kurtosis[i] = np.nan
            r_skew[i] = np.nan

            # energies
            p_energy[i] = np.nan
            t_energy[i] = np.nan

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
    feat["pr_std"] = pr_stdv
    feat["qt_std"] = qt_stdv
    feat["qs_std"] = qs_stdv
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
    feat["arhythm_index"] = arhythm_idx
    feat["r_amp_kurtosis"] = r_kurtosis
    feat["r_amp_skew"] = r_skew
    feat["p_eng"] = p_energy
    feat["t_eng"] = t_energy

    return feat
