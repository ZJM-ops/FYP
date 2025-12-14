import os
import numpy as np
import pywt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob
from multiprocessing import Pool, cpu_count
import concurrent.futures
import json

segmented_data_root = '/home/student/s230005071/shu/segmented_data'
output_root_dir = '/home/student/s230005071/shu/cwt_images'

wavelet_name = 'cmor1.5-1.0'
f_min = 250.0
f_max = 490.0
num_scales = 500
sampling_rate = 1000
duration = 2.0
sampling_period = 1.0 / sampling_rate

group_mapping = {
    'VPA组': 'VPA_group',
    'mir过表达组': 'mir_overexpression_group',
    'pilo组': 'pilo_group',
    'sponges组': 'sponges_group',
    '空载组': 'scramble_group'
}

label_mapping = {
    'Baseline': 'Baseline', 'Pre-30m': 'Pre-30m', 'Pre-20m': 'Pre-20m',
    'Pre-10m': 'Pre-10m', 'Ictal': 'Ictal', 'Post-10m': 'Post-10m',
    'Post-1h': 'Post-1h', 'Post-2h': 'Post-2h', 'Post-3h': 'Post-3h',
    'Chronic-1d': 'Chronic-1d', 'Chronic-3d': 'Chronic-3d',
    'Chronic-7d': 'Chronic-7d', 'Chronic-28d': 'Chronic-28d'
}

central_frequency = pywt.central_frequency(wavelet_name)
scale_min = (central_frequency * sampling_rate) / f_max
scale_max = (central_frequency * sampling_rate) / f_min
scales = np.logspace(np.log10(scale_min), np.log10(scale_max), num=num_scales)


def process_single_file(args):
    filepath, global_max_energy = args

    path_parts = filepath.split(os.sep)
    group_chinese = path_parts[-3]
    label_chinese = path_parts[-2]

    group_english = group_mapping.get(group_chinese, group_chinese)
    label_english = label_mapping.get(label_chinese, label_chinese)

    base_filename = os.path.basename(filepath).replace('.npy', '')
    seg_index_start = base_filename.rfind('_seg')
    animal_id_part = base_filename[:seg_index_start]
    segment_index = base_filename[seg_index_start+1:]

    parts_to_keep = [
        p for p in animal_id_part.split('_')
        if p not in group_mapping and p not in label_mapping
    ]
    final_animal_id = '_'.join(parts_to_keep)
    new_base_filename = f"{group_english}_{label_english}_{final_animal_id}_{segment_index}"

    segment_data = np.load(filepath)

    for i, channel_data in enumerate(segment_data):
        output_dir = os.path.join(output_root_dir, group_english, label_english)
        output_path = os.path.join(output_dir, f"{new_base_filename}_ch{i+1}.png")

        if os.path.exists(output_path):
            continue

        coefficients, _ = pywt.cwt(channel_data, scales, wavelet_name, sampling_period=sampling_period)

        fig = plt.figure(figsize=(2.24, 2.24), dpi=100)
        plt.imshow(np.abs(coefficients), cmap='jet', aspect='auto',
                   extent=(0.0, duration, f_min, f_max),
                   vmin=0, vmax=global_max_energy)
        plt.axis('off')
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)


def calculate_file_max(filepath):
    segment_data = np.load(filepath)
    coefficients, _ = pywt.cwt(segment_data, scales, wavelet_name, sampling_period=sampling_period)
    return np.max(np.abs(coefficients))


if __name__ == '__main__':
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    if not os.path.exists(output_root_dir):
        os.makedirs(output_root_dir)

    #Baseline + Ictal
    all_files = glob.glob(os.path.join(segmented_data_root, '**', '*.npy'), recursive=True)
    target_labels = {'Baseline', 'Ictal'}
    npy_files = [
        fp for fp in all_files
        if os.path.basename(os.path.dirname(fp)) in target_labels
    ]

    if not npy_files:
        exit()

    # Max value
    num_cores = cpu_count()
    max_value_file = os.path.join(output_root_dir, "global_max_energy.json")
    global_max_energy = 0.0

    if os.path.exists(max_value_file):
        with open(max_value_file, 'r') as f:
            global_max_energy = json.load(f).get("global_max_energy", 0.0)

    if global_max_energy == 0.0:
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
            future_to_file = {executor.submit(calculate_file_max, fp): fp for fp in npy_files}
            for future in tqdm(concurrent.futures.as_completed(future_to_file), total=len(npy_files)):
                file_max = future.result()
                global_max_energy = max(global_max_energy, file_max)

        with open(max_value_file, 'w') as f:
            json.dump({"global_max_energy": global_max_energy}, f, indent=4)

    if global_max_energy == 0.0:
        exit()

    #CWt
    process_args = [(fp, global_max_energy) for fp in npy_files]
    with Pool(num_cores) as pool:
        list(tqdm(pool.imap_unordered(process_single_file, process_args), total=len(process_args)))
