import numpy as np
import pandas as pd
import scipy.stats as sp_stats
import antropy as ant
import mne_features.univariate

SR = 1000  # Hz
WINDOW_SIZE = 250  # samples (250 ms at 1000 Hz)


def extract_depth_features(epochs, subj):
    """
    Extracts features for the depth model from epoched data.

    Args:
        epochs (np.ndarray or List): 2D array-like structure (n_epochs, n_samples)
                                     containing the signal windows.
        subj (str): The subject identifier.

    Returns:
        pd.DataFrame: DataFrame with one row per epoch and one column per feature.
    """
    mobility, complexity = ant.hjorth_params(epochs, axis=1)
    feat = {
        'subj': np.full(len(epochs), subj),
        'epoch_id': np.arange(len(epochs)),
        'kurtosis': sp_stats.kurtosis(epochs, axis=1),
        'hjorth_mobility': mobility,
        'hjorth_complexity': complexity,
        'ptp_amp': np.ptp(epochs, axis=1),
        'samp_entropy': np.apply_along_axis(ant.sample_entropy, axis=1, arr=epochs)
    }

    feat = pd.DataFrame(feat)

    kaiser = mne_features.univariate.compute_teager_kaiser_energy(np.array(epochs))
    reshaped_list = np.array(kaiser).reshape(-1, 12)
    X_kaiser = pd.DataFrame(reshaped_list)
    X_kaiser.columns = [
        f'teager_kaiser_energy_{i}_mean' if j % 2 == 0 else f'teager_kaiser_energy_{i}_std'
        for i in range(6) for j in range(2)
    ]

    feat = pd.concat([feat, X_kaiser], axis=1)
    return feat


def extract_zeeg_features(epochs, subj, sr):
    """
    Extracts a comprehensive set of zEEG features from epoched data.

    Calculates features across several domains:
    - Basic statistical (mean, std, ptp, etc.)
    - Fractal / nonlinear (Higuchi, Katz, Hjorth)
    - Entropy (Sample, Spectral, SVD)
    - Power (Absolute, Normalized band power, band ratios)
    - Energy (Band energy, band ratios)
    - Wavelet energy
    - Teager-Kaiser energy

    Args:
        epochs (np.ndarray or List): 2D array-like (n_epochs, n_samples).
        subj (str): The subject identifier.
        sr (int): The sampling rate in Hz.

    Returns:
        pd.DataFrame: DataFrame with one row per epoch and one column per feature.
    """
    data = np.array(epochs)
    uni = mne_features.univariate

    bands = {
        'theta': (4, 8), 'alpha': (8, 12), 'sigma': (12, 16),
        'beta': (16, 30), 'gamma': (30, 100), 'fast': (100, 300)
    }
    band_list = ['theta', 'alpha', 'sigma', 'beta', 'gamma', 'fast']
    psd_params = {'psd_method': 'welch', 'psd_params': None}

    basic_features = {
        'ptp_amp': uni.compute_ptp_amp(data),
        'mean': uni.compute_mean(data),
        'std': uni.compute_std(data),
        'variance': uni.compute_variance(data),
        'skewness': uni.compute_skewness(data),
        'kurtosis': uni.compute_kurtosis(data),
        'quantile': uni.compute_quantile(data, q=0.75),
        'rms': uni.compute_rms(data),
        'line_length': uni.compute_line_length(data),
        'zero_crossings': uni.compute_zero_crossings(data, threshold=np.finfo(float).eps),
    }

    complexity_features = {
        'higuchi_fd': uni.compute_higuchi_fd(data, kmax=10),
        'katz_fd': uni.compute_katz_fd(data),
        'hurst_exp': uni.compute_hurst_exp(data),
        'hjorth_mobility': uni.compute_hjorth_mobility(data),
        'hjorth_complexity': uni.compute_hjorth_complexity(data),
        'hjorth_mobility_spect': uni.compute_hjorth_mobility_spect(sr, data, normalize=False, **psd_params),
        'hjorth_complexity_spect': uni.compute_hjorth_complexity_spect(sr, data, normalize=False, **psd_params),
    }

    entropy_features = {
        'app_entropy': uni.compute_app_entropy(data, emb=2, metric='chebyshev'),
        'samp_entropy': uni.compute_samp_entropy(data, emb=2, metric='chebyshev'),
        'spect_entropy': uni.compute_spect_entropy(sr, data, **psd_params),
        'svd_entropy': uni.compute_svd_entropy(data, tau=2, emb=10),
        'svd_fisher_info': uni.compute_svd_fisher_info(data, tau=2, emb=10),
        'decorr_time': uni.compute_decorr_time(sr, data),
    }

    abspow = uni.compute_pow_freq_bands(sr, data, {'total': (0.1, 500)}, False, psd_method='multitaper')

    df_basic = pd.DataFrame({**basic_features, **complexity_features, **entropy_features, 'abspow_': abspow})

    slope = uni.compute_spect_slope(sr, data, fmin=0.1, fmax=50, with_intercept=True, psd_method='welch')
    df_slope = pd.DataFrame(
        np.array(slope).reshape(-1, 4),
        columns=['spect_slope_intercept', 'spect_slope_slope', 'spect_slope_MSE', 'spect_slope_R2']
    )

    pow_bands = uni.compute_pow_freq_bands(
        data=data, sfreq=sr, freq_bands=bands, normalize=True,
        ratios=None, psd_method='multitaper', log=False
    )
    df_pow = pd.DataFrame(
        np.array(pow_bands).reshape(-1, len(bands)),
        columns=[f'pow_freq_bands_{b}' for b in band_list]
    )
    for b1 in band_list:
        for b2 in band_list:
            if b1 != b2:
                df_pow[f'pow_freq_bands_{b1}/{b2}'] = df_pow[f'pow_freq_bands_{b1}'] / df_pow[f'pow_freq_bands_{b2}']

    energy = uni.compute_energy_freq_bands(sr, data, freq_bands=bands)
    df_energy = pd.DataFrame(
        np.array(energy).reshape(-1, len(bands)),
        columns=[f'energy_freq_bands_{b}' for b in band_list]
    )

    for b1 in band_list:
        for b2 in band_list:
            if b1 != b2 and f'energy_freq_bands_{b2[0]}{b1[0]}' not in df_energy.columns:
                df_energy[f'energy_freq_bands_{b1[0]}{b2[0]}'] = (
                    df_energy[f'energy_freq_bands_{b1}'] / df_energy[f'energy_freq_bands_{b2}']
                )

    wave = uni.compute_wavelet_coef_energy(data, wavelet_name='db4')
    df_wave = pd.DataFrame(
        np.array(wave).reshape(-1, 5),
        columns=[f'wavelet_coef_energy_{i}' for i in range(5)]
    )

    kaiser = uni.compute_teager_kaiser_energy(data)
    df_kaiser = pd.DataFrame(
        np.array(kaiser).reshape(-1, 12),
        columns=[f'teager_kaiser_energy_{i}_{stat}' for i in range(6) for stat in ['mean', 'std']]
    )

    df = pd.concat([df_basic, df_slope, df_energy, df_kaiser, df_pow, df_wave], axis=1)
    df.insert(0, 'subj', subj)
    df.insert(1, 'epoch_id', np.arange(len(df)))

    return df


def raw_chan_to_feat(raw, chan, subj, depth):
    """
    Processes a single channel from a raw file, epochs it, and extracts features.

    This function normalizes the channel, segments it into 250ms windows,
    and calls the appropriate feature extraction function (zEEG or depth).
    Finally, it adds global channel-level features to all epochs.

    Args:
        raw (mne.io.Raw): The MNE raw object.
        chan (str): The name of the channel to process.
        subj (str): The subject identifier.
        depth (bool): Flag. If True, extract depth features.
                      If False, extract zEEG features.

    Returns:
        pd.DataFrame: A DataFrame of features, one row per valid epoch.
    """
    epochs = []
    chan_raw = raw.copy().pick([chan]).get_data(reject_by_annotation='NaN').flatten()

    chan_norm = (chan_raw - np.nanmean(chan_raw)) / np.nanstd(chan_raw)

    for i in range(0, len(chan_norm) - SR, WINDOW_SIZE):
        if not np.isnan(chan_norm[i: i + WINDOW_SIZE]).any():
            epochs.append(chan_norm[i: i + WINDOW_SIZE])

    if depth:
        curr_feat = extract_depth_features(epochs, subj)
    else:
        curr_feat = extract_zeeg_features(epochs, subj, raw.info['sfreq'])

    chan_feat = {
        'chan_name': chan,
        'chan_ptp': np.ptp(chan_norm[~np.isnan(chan_norm)]),
        'chan_skew': sp_stats.skew(chan_norm[~np.isnan(chan_norm)]),
        'chan_kurt': sp_stats.kurtosis(chan_norm[~np.isnan(chan_norm)]),
    }

    for feat in chan_feat.keys():
        curr_feat[feat] = chan_feat[feat]

    return curr_feat
