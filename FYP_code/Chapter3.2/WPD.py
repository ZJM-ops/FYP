import os
import numpy as np
import pandas as pd
import pywt
from scipy.signal import welch
from tqdm import tqdm
from pathlib import Path
from scipy.spatial.distance import cdist

segmented_data_root = Path('/home/student/s230005071/shu/segmented_data')
output_csv_path = Path('/home/student/s230005071/shu/features.csv')
wavelet_name = 'db4'
decomposition_level = 3
sampling_rate = 1000

group_mapping = {
    'VPA组': 'VPA_group', 'mir过表达组': 'mir_group', 'pilo组': 'pilo_group',
    'sponges组': 'sponges_group', '空载组': 'scramble_group'
}
label_mapping = {'Baseline': 'Baseline', 'Ictal': 'Ictal'}

def get_frequency_bands(wavelet, level, sfreq):
    """Generate the frequency bands corresponding to wavelet packet nodes."""
    if level <= 0:
        return []

    wp = pywt.WaveletPacket(data=None, wavelet=wavelet, mode='symmetric', maxlevel=level)
    nodes = [node.path for node in wp.get_level(level, 'natural')]
    freq_bands = []

    # 'a' = low-pass branch = 0, 'd' = high-pass = 1
    path_map = {'a': 0, 'd': 1}

    for node_path in nodes:
        freq_index = 0
        for char in reversed(node_path):
            freq_index = (freq_index << 1) | path_map[char]

        band_width = (sfreq / 2) / (2**level)
        f_low = freq_index * band_width
        f_high = (freq_index + 1) * band_width

        freq_bands.append({'node_path': node_path, 'freq_range': (f_low, f_high)})

    sorted_bands = sorted(freq_bands, key=lambda x: x['freq_range'][0], reverse=True)
    for i, band in enumerate(sorted_bands):
        band['band_name'] = f"band{i}"

    return sorted_bands


# Approximate Entropy
def approximate_entropy(signal, m=2, r=None):
    """Compute approximate entropy (ApEn). Falls back to zero when unstable."""
    if signal is None or len(signal) < m + 1:
        return 0.0

    if r is None:
        std_dev = np.std(signal)
        r = 0.2 * std_dev if std_dev > 1e-10 else 0.2

    def _phi(m_len):
        n = len(signal)
        x = np.array([signal[i:i + m_len] for i in range(n - m_len + 1)])
        dist_matrix = cdist(x, x, metric='chebyshev')
        C = np.sum(dist_matrix <= r, axis=1) / (n - m_len + 1)
        return np.mean(np.log(C + 1e-10))

    try:
        return _phi(m) - _phi(m + 1)
    except:
        return 0.0
# WPD
def wavelet_packet_decomposition(signal, wavelet='db4', level=3):
    """Run wavelet packet decomposition and return terminal node signals."""
    wp = pywt.WaveletPacket(data=signal, wavelet=wavelet, mode='symmetric', maxlevel=level)
    nodes = [node.path for node in wp.get_level(level, 'natural')]
    return {node: wp[node].data for node in nodes}


def extract_band_features(band_signal, band_name):
    """Extract mean, std, PSD mean, ApEn, and energy for a band."""
    features = {}
    if band_signal is None or len(band_signal) < 10:
        zero_vals = {
            f'{band_name}_mean': 0.0,
            f'{band_name}_std': 0.0,
            f'{band_name}_psd_mean': 0.0,
            f'{band_name}_approx_entropy': 0.0,
            f'{band_name}_energy': 0.0
        }
        features.update(zero_vals)
        return features

    features[f'{band_name}_mean'] = np.mean(band_signal)
    features[f'{band_name}_std'] = np.std(band_signal)

    # PSD mean 
    try:
        nperseg = min(256, len(band_signal) // 2)
        if nperseg > 0:
            _, psd = welch(band_signal, sampling_rate, nperseg=nperseg)
            features[f'{band_name}_psd_mean'] = np.mean(psd)
        else:
            features[f'{band_name}_psd_mean'] = 0.0
    except:
        features[f'{band_name}_psd_mean'] = 0.0

    features[f'{band_name}_approx_entropy'] = approximate_entropy(band_signal)
    features[f'{band_name}_energy'] = np.sum(np.square(band_signal))

    return features


def extract_wavelet_packet_features(segment_data, band_mapping_list):
    """Extract features for all frequency bands from a single EEG segment."""
    features = {}
    try:
        wp_coeffs = wavelet_packet_decomposition(segment_data, wavelet_name, decomposition_level)
        for band in band_mapping_list:
            band_signal = wp_coeffs.get(band['node_path'])
            features.update(extract_band_features(band_signal, band['band_name']))
    except:
        for band in band_mapping_list:
            features.update(extract_band_features(None, band['band_name']))
    return features


# Main feature extraction
sorted_frequency_bands = get_frequency_bands(wavelet_name, decomposition_level, sampling_rate)
npy_files = list(segmented_data_root.rglob('*.npy'))

all_features_list = []

for filepath in tqdm(npy_files, desc="extracting", disable=False):
    try:
        relative_path = filepath.relative_to(segmented_data_root)
        parts = relative_path.parts
        if len(parts) < 3:
            continue

        group_chinese = parts[-3]
        label = parts[-2]

        animal_id = '_'.join(filepath.name.split('_')[:2])
        data = np.load(filepath, allow_pickle=True)
        if data is None or data.size == 0:
            continue

        for i, ch_data in enumerate(data):
            feats = extract_wavelet_packet_features(ch_data, sorted_frequency_bands)
            feats['animal_id'] = animal_id
            feats['group'] = group_mapping.get(group_chinese, group_chinese)
            feats['label'] = label_mapping.get(label, label)
            feats['channel'] = i + 1
            all_features_list.append(feats)
    except:
        continue

if all_features_list:
    df = pd.DataFrame(all_features_list)
    meta_cols = ['animal_id', 'group', 'label', 'channel']
    feature_cols = sorted([c for c in df.columns if c not in meta_cols])
    df = df[meta_cols + feature_cols]
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv_path, index=False)
