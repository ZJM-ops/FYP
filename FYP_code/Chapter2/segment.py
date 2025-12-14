import os
import mne
import numpy as np
import pandas as pd
from tqdm import tqdm
import glob


fif_root_dir = '/home/student/s230005071/shu/preprocessed_data'
output_root_dir = '/home/student/s230005071/shu/segmented_data'
labels_csv_path = '/home/student/s230005071/shu/labels.csv'

window_length_sec = 2
overlap_ratio = 0.5

label_mapping = {
    'Baseline': 'Baseline', 'Pre-30m': 'Pre-30m', 'Pre-20m': 'Pre-20m',
    'Pre-10m': 'Pre-10m', 'Ictal': 'Ictal', 'Post-10m': 'Post-10m',
    'Post-1h': 'Post-1h', 'Post-2h': 'Post-2h', 'Post-3h': 'Post-3h',
    'Chronic-1d': 'Chronic-1d', 'Chronic-3d': 'Chronic-3d',
    'Chronic-7d': 'Chronic-7d', 'Chronic-28d': 'Chronic-28d'
}
original_labels_to_process = list(label_mapping.keys())

os.makedirs(output_root_dir, exist_ok=True)
labels_df = pd.read_csv(labels_csv_path)


matched_metadata = []
relevant_records = labels_df[labels_df['label'].isin(original_labels_to_process)]

for _, row in tqdm(relevant_records.iterrows(), total=len(relevant_records), desc=" "):
    animal_id = str(row['animal_id']).strip()
    group = str(row['group']).strip()
    label = str(row['label']).strip()
    original_filename_base = os.path.basename(row['filepath']).replace('.edf', '')
    expected_fif_name = f"{animal_id}_{original_filename_base}-raw.fif".replace(' ', '_')
    expected_fif_path = os.path.join(fif_root_dir, group, label, expected_fif_name)

    
    matched_metadata.append({
            'fif_filepath': expected_fif_path,
            'animal_id': animal_id,
            'group': group,
            'original_label': label
        })

processed_count = 0
skipped_count = 0

for meta in tqdm(matched_metadata, desc=" "):
    fif_filepath = meta['fif_filepath']
    animal_id = meta['animal_id']
    group = meta['group']
    original_label = meta['original_label']
    
    new_label = label_mapping[original_label]
    output_dir = os.path.join(output_root_dir, group, new_label)
    

    output_file_pattern = f"{animal_id}_{group}_{new_label}_seg*.npy"
    existing_segments = glob.glob(os.path.join(output_dir, output_file_pattern))
    
    if existing_segments:
        tqdm.write(f"\n  Skip {os.path.basename(fif_filepath)}")
        skipped_count += 1
        continue
    
    try:
        os.makedirs(output_dir, exist_ok=True)
        raw = mne.io.read_raw_fif(fif_filepath, preload=True, verbose=False)
        sfreq = raw.info['sfreq']
        
        window_samples = int(window_length_sec * sfreq)
        step_samples = int(window_samples * (1 - overlap_ratio))
        

        num_segments = 0
        for start_sample in range(0, raw.n_times - window_samples + 1, step_samples):
            segment_data = raw.get_data(start=start_sample, stop=start_sample + window_samples)
            segment_filename = f"{animal_id}_{group}_{new_label}_seg{num_segments:04d}.npy"
            output_path = os.path.join(output_dir, segment_filename)
            np.save(output_path, segment_data)
            num_segments += 1
        
        processed_count += 1
        tqdm.write(f"\n  success '{os.path.basename(fif_filepath)}' save {num_segments} segment to '{output_dir}'")

    except Exception as e:
        tqdm.write(f"\n  Wrong {fif_filepath} {e}")
   
