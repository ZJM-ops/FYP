import os
import pandas as pd
import re

def create_label_file(root_folder, output_filepath):
    
    parsed_data = []

    keywords = {
        'Baseline': {'建模前1d', '建模前1天', 'baseline','da00109a', 'da00109j', 'da0010a1'},
        'Ictal': {'止惊前10min'},
        
        'Pre-30m': {'iv级前30min', 'iv前30min', 'iv级发作前30min'},
        'Pre-20m': {'iv级前20min', '20min'},
        'Pre-10m': {'iv级前10min', 'iv前10min', 'iv级发作前10min', "iv发作前10min"},
        
        'Post-10m': {'止惊后10min','止惊后10mim'},
        'Post-1h': {'止惊后1h'},
        'Post-2h': {'止惊后2h'},
        'Post-3h': {'止惊后3h'},

        'Chronic-1d': {'建模后1d','建模后第1d','建模后第1天'},
        'Chronic-3d': {'建模后3d','建模后第3d', '建模后第3天'},
        'Chronic-7d': {'建模后7d', '建模后第7天','建模后第7d', '7d-1'},
        'Chronic-28d': {'建模后28d', '建模后第28天', '建模后第28d','止惊后28d'}
    }

    for dirpath, _, filenames in os.walk(root_folder):
        for filename in filenames:
            if not filename.endswith('.edf'):
                continue

            full_path = os.path.join(dirpath, filename)
            relative_path = os.path.relpath(full_path, root_folder)
            parts = relative_path.replace('\\', '/').split('/')

            if len(parts) < 3:
                continue

            group = parts[0]
            animal_folder_name = parts[1]

            animal_num_match = re.search(r'(\d+)', animal_folder_name)
            if animal_num_match:
                animal_id = f"{group.replace('组', '')}_{animal_num_match.group(1)}"
            else:
                animal_id = f"{group.replace('组', '')}_{animal_folder_name}"

            clean_filename = filename.lstrip('- ').strip().lower().replace('.edf', '')
            label = "To_Ignore"


            matched = False
            for lbl, kw_set in keywords.items():
                if clean_filename in kw_set:
                    label = lbl
                    matched = True
                    break

            if not matched:
                for lbl, kw_set in keywords.items():
                    if any(kw in clean_filename for kw in kw_set):
                        label = lbl
                        break

            parsed_data.append({
                'filepath': os.path.join(root_folder, relative_path),
                'animal_id': animal_id,
                'group': group,
                'label': label
            })

    df = pd.DataFrame(parsed_data)
    core_df = df[df['label'] != 'To_Ignore'].copy()
    core_df = core_df.sort_values(by=['group', 'animal_id']).reset_index(drop=True)
    core_df.to_csv(output_filepath, index=False, encoding='utf-8-sig')

    #not match
    ignored_df = df[df['label'] == 'To_Ignore']
    if not ignored_df.empty:
        ignored_output_filepath = os.path.join(os.path.dirname(output_filepath), 'ignored_labels.csv')
        ignored_df.to_csv(ignored_output_filepath, index=False, encoding='utf-8-sig')


if __name__ == '__main__':
    root_folder_path = "/home/student/s230005071/shu/脑电图数据-已提取的10min脑电图"
    output_csv_path = '/home/student/s230005071/shu/labels.csv'
    if os.path.isdir(root_folder_path):
        create_label_file(root_folder_path, output_csv_path)
