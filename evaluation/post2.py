"""The post processing files for calculating respiratory rate using FFT.
The file also includes helper funcs such as detrend, power2db etc.
"""

import numpy as np
import scipy
import scipy.io
from scipy.signal import butter
from scipy.sparse import spdiags
from copy import deepcopy

def _next_power_of_2(x):
    """Calculate the nearest power of 2."""
    return 1 if x == 0 else 2 ** (x - 1).bit_length()

def _detrend(input_signal, lambda_value):
    """Detrend respiratory signal."""
    signal_length = input_signal.shape[0]
    # observation matrix
    H = np.identity(signal_length)
    ones = np.ones(signal_length)
    minus_twos = -2 * np.ones(signal_length)
    diags_data = np.array([ones, minus_twos, ones])
    diags_index = np.array([0, 1, 2])
    D = spdiags(diags_data, diags_index,
                (signal_length - 2), signal_length).toarray()
    detrended_signal = np.dot(
        (H - np.linalg.inv(H + (lambda_value ** 2) * np.dot(D.T, D))), input_signal)
    return detrended_signal

def power2db(mag):
    """Convert power to db."""
    return 10 * np.log10(mag)

def _calculate_fft_rr(resp_signal, fs=30, low_pass=0.1, high_pass=0.5):
    """Calculate respiratory rate based on respiratory signal using Fast Fourier transform (FFT).
    
    Args:
        resp_signal: respiratory signal
        fs: sampling frequency in Hz
        low_pass: low cutoff frequency in Hz (0.1 Hz = 6 breaths/min)
        high_pass: high cutoff frequency in Hz (0.5 Hz = 30 breaths/min)
    
    Returns:
        float: respiratory rate in breaths per minute
    """
    resp_signal = np.expand_dims(resp_signal, 0)
    N = _next_power_of_2(resp_signal.shape[1])
    f_resp, pxx_resp = scipy.signal.periodogram(resp_signal, fs=fs, nfft=N, detrend=False)
    fmask_resp = np.argwhere((f_resp >= low_pass) & (f_resp <= high_pass))
    
    # Handle case where no frequencies in the respiratory range are found
    if len(fmask_resp) == 0:
        return 15.0  # Return a default value
        
    mask_resp = np.take(f_resp, fmask_resp)
    mask_pxx = np.take(pxx_resp, fmask_resp)
    
    # Find the frequency with maximum power in the respiratory band
    fft_rr = np.take(mask_resp, np.argmax(mask_pxx, 0))[0] * 60  # Convert Hz to breaths/min
    return fft_rr

def _calculate_peak_rr(resp_signal, fs):
    """Calculate respiratory rate based on respiratory signal using peak detection."""
    resp_peaks, _ = scipy.signal.find_peaks(resp_signal, distance=fs*1.0)  # Min 1.0s between peaks (max 60 bpm)
    if len(resp_peaks) < 2:
        return 15.0  # Return a default value if not enough peaks
    rr_peak = 60 / (np.mean(np.diff(resp_peaks)) / fs)
    return rr_peak

def _compute_macc(pred_signal, gt_signal):
    """Calculate maximum amplitude of cross correlation (MACC) by computing correlation at all time lags.
        Args:
            pred_signal(np.array): predicted respiratory signal 
            gt_signal(np.array): ground truth, label respiratory signal
        Returns:
            MACC(float): Maximum Amplitude of Cross-Correlation
    """
    pred = deepcopy(pred_signal)
    gt = deepcopy(gt_signal)
    pred = np.squeeze(pred)
    gt = np.squeeze(gt)
    min_len = np.min((len(pred), len(gt)))
    pred = pred[:min_len]
    gt = gt[:min_len]
    lags = np.arange(0, len(pred)-1, 1)
    tlcc_list = []
    for lag in lags:
        cross_corr = np.abs(np.corrcoef(
            pred, np.roll(gt, lag))[0][1])
        tlcc_list.append(cross_corr)
    macc = max(tlcc_list)
    return macc

def _calculate_SNR(pred_resp_signal, rr_label, fs=30, low_pass=0.1, high_pass=0.5):
    """Calculate SNR as the ratio of the area under the curve of the frequency spectrum around the first and second harmonics 
        of the ground truth RR frequency to the area under the curve of the remainder of the frequency spectrum.

        Args:
            pred_resp_signal(np.array): predicted respiratory signal 
            rr_label(float): ground truth respiratory rate in breaths/min
            fs(int or float): sampling rate of the signal
            low_pass(float): low frequency cutoff for respiratory band in Hz
            high_pass(float): high frequency cutoff for respiratory band in Hz
        Returns:
            SNR(float): Signal-to-Noise Ratio
    """
    # Get the first and second harmonics of the ground truth RR in Hz
    first_harmonic_freq = rr_label / 60  # Convert breaths/min to Hz
    second_harmonic_freq = 2 * first_harmonic_freq
    deviation = 3 / 60  # 3 breaths/min converted to Hz

    # Calculate FFT
    pred_resp_signal = np.expand_dims(pred_resp_signal, 0)
    N = _next_power_of_2(pred_resp_signal.shape[1])
    f_resp, pxx_resp = scipy.signal.periodogram(pred_resp_signal, fs=fs, nfft=N, detrend=False)

    # Calculate the indices corresponding to the frequency ranges
    idx_harmonic1 = np.argwhere((f_resp >= (first_harmonic_freq - deviation)) & 
                                (f_resp <= (first_harmonic_freq + deviation)))
    idx_harmonic2 = np.argwhere((f_resp >= (second_harmonic_freq - deviation)) & 
                                (f_resp <= (second_harmonic_freq + deviation)))
    idx_remainder = np.argwhere((f_resp >= low_pass) & (f_resp <= high_pass) &
                               ~((f_resp >= (first_harmonic_freq - deviation)) & 
                                 (f_resp <= (first_harmonic_freq + deviation))) &
                               ~((f_resp >= (second_harmonic_freq - deviation)) & 
                                 (f_resp <= (second_harmonic_freq + deviation))))

    # Select the corresponding values from the periodogram
    pxx_resp = np.squeeze(pxx_resp)
    
    if len(idx_harmonic1) > 0:
        pxx_harmonic1 = pxx_resp[idx_harmonic1]
        signal_power_hm1 = np.sum(pxx_harmonic1**2)
    else:
        signal_power_hm1 = 0
        
    if len(idx_harmonic2) > 0:
        pxx_harmonic2 = pxx_resp[idx_harmonic2]
        signal_power_hm2 = np.sum(pxx_harmonic2**2)
    else:
        signal_power_hm2 = 0
    
    if len(idx_remainder) > 0:
        pxx_remainder = pxx_resp[idx_remainder]
        signal_power_rem = np.sum(pxx_remainder**2)
    else:
        signal_power_rem = 1e-10  # Avoid division by zero

    # Calculate the SNR as the ratio of the areas
    if not signal_power_rem == 0: # catches divide by 0 runtime warning 
        SNR = power2db((signal_power_hm1 + signal_power_hm2) / signal_power_rem)
    else:
        SNR = 0
    return SNR

def calculate_metric_per_video(predictions, labels, fs=30, diff_flag=False, use_bandpass=True, hr_method='FFT'):
    """Calculate video-level RR and SNR"""
    if diff_flag:  # if the predictions and labels are 1st derivative of respiratory signal.
        predictions = _detrend(np.cumsum(predictions), 100)
        labels = _detrend(np.cumsum(labels), 100)
    else:
        predictions = _detrend(predictions, 100)
        labels = _detrend(labels, 100)
    if use_bandpass:
        # bandpass filter between [0.1, 0.5] Hz
        # equals [6, 30] breaths per min
        [b, a] = butter(2, [0.1 / fs * 2, 0.5 / fs * 2], btype='bandpass')
        predictions = scipy.signal.filtfilt(b, a, np.double(predictions))
        labels = scipy.signal.filtfilt(b, a, np.double(labels))
    
    macc = _compute_macc(predictions, labels)

    if hr_method == 'FFT':
        rr_pred = _calculate_fft_rr(predictions, fs=fs)
        rr_label = _calculate_fft_rr(labels, fs=fs)
    elif hr_method == 'Peak':
        rr_pred = _calculate_peak_rr(predictions, fs=fs)
        rr_label = _calculate_peak_rr(labels, fs=fs)
    else:
        raise ValueError('Please use FFT or Peak to calculate your RR.')
    SNR = _calculate_SNR(predictions, rr_label, fs=fs)
    return rr_label, rr_pred, SNR, macc