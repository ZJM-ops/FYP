import mne
import pandas as pd
import os


labels_df = pd.read_csv("/home/student/s230005071/shu/labels.csv") 
required_cols = ['filepath', 'animal_id', 'group', 'label']

channels_with_prefix = [
    'EEG Fp1-Ref', 'EEG Fp2-Ref', 'EEG F3-Ref', 'EEG F4-Ref', 
    'EEG C3-Ref', 'EEG C4-Ref', 'EEG P3-Ref', 'EEG P4-Ref', 
    ]
channels_without_prefix = [
    'Fp1-Ref', 'Fp2-Ref', 'F3-Ref', 'F4-Ref', 
    'C3-Ref', 'C4-Ref', 'P3-Ref', 'P4-Ref', 
]

output_root_dir = '/home/student/s230005071/shu/preprocessed_data'
if not os.path.exists(output_root_dir):
    os.makedirs(output_root_dir)
failed_files = []

for index, row in labels_df.iterrows():
    filepath = row['filepath']
    animal_id = row['animal_id']
    group = row['group']
    label = row['label']
    original_filename = os.path.basename(filepath)
    print(f"Processing:{group}/{label}/{original_filename} ---")

    try:
        if not os.path.exists(filepath):
            raise FileNotFoundError("Not exist")
        target_dir = os.path.join(output_root_dir, group, label)
        os.makedirs(target_dir, exist_ok=True)
        new_filename = f"{animal_id}_{original_filename}".replace('.edf', '-raw.fif').replace(' ', '_')
        output_filepath = os.path.join(target_dir, new_filename)
        if os.path.exists(output_filepath):
            continue

        raw_info = mne.io.read_raw_edf(filepath, preload=False, verbose=False)
        
        if channels_with_prefix[0] in raw_info.ch_names:
            channels_to_use = channels_with_prefix
        else:
            channels_to_use = channels_without_prefix
        
        raw = raw_info.load_data(verbose=False)

        current_channels_to_keep = [ch for ch in channels_to_use if ch in raw.ch_names]
        raw.pick_channels(current_channels_to_keep)


        raw.filter(l_freq=1.0, h_freq=490.0, verbose=False)
        raw.notch_filter(freqs=range(50, 500, 50), verbose=False)
        raw.save(output_filepath, overwrite=True)


    except Exception as e:
        error_message = str(e).replace('\n', ' ').strip()
        print(f"{error_message}")
        failed_files.append({'filepath': filepath, 'animal_id': animal_id, 'group': group, 'label': label, 'reason': error_message})
