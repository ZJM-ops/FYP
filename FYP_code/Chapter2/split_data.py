import os
import pandas as pd
from sklearn.model_selection import train_test_split
import glob
from tqdm import tqdm


features_csv_path = '/home/student/s230005071/shu/features.csv'#DL
images_root_dir = "/home/student/s230005071/shu/cwt_images_250-490"#CWT
npy_root_dir ="/home/student/s230005071/shu/segmented_data"#1D
output_dir = '/home/student/s230005071/shu/split_data'#all

train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15
random_seed = 42 


if not os.path.exists(output_dir):
    os.makedirs(output_dir)


df_features = pd.read_csv(features_csv_path)
print(f"成功加载 {features_csv_path}，共 {len(df_features)} 个样本。")


all_animal_ids = df_features['animal_id'].unique()

train_ids, temp_ids = train_test_split(
    all_animal_ids,
    test_size=(val_ratio + test_ratio),
    random_state=random_seed
)

val_ids, test_ids = train_test_split(
    temp_ids,
    test_size=test_ratio / (val_ratio + test_ratio),
    random_state=random_seed
)

print(f"Total animal_id: {len(all_animal_ids)}")
print(f"Train set animal_id: {len(train_ids)}")
print(f"Val set animal_id: {len(val_ids)}")
print(f"Test set animal_id: {len(test_ids)}")

id_to_group = pd.Series(df_features.group.values, index=df_features.animal_id).to_dict()
all_groups = df_features['group'].unique()

for group in sorted(all_groups):
    group_ids = {id for id, g in id_to_group.items() if g == group}
    
    group_train_ids = sorted(list(group_ids.intersection(train_ids)))
    group_val_ids = sorted(list(group_ids.intersection(val_ids)))
    group_test_ids = sorted(list(group_ids.intersection(test_ids)))
    
    print(f"Train set({len(group_train_ids)}): {group_train_ids}")
    print(f"Val set({len(group_val_ids)}): {group_val_ids}")
    print(f"Test set({len(group_test_ids)}): {group_test_ids}")


train_df = df_features[df_features['animal_id'].isin(train_ids)]
val_df = df_features[df_features['animal_id'].isin(val_ids)]
test_df = df_features[df_features['animal_id'].isin(test_ids)]

print(f"Train set num: {len(train_df)} ({len(train_df) / len(df_features):.2%})")
print(f"Val set num: {len(val_df)} ({len(val_df) / len(df_features):.2%})")
print(f"Test set num: {len(test_df)} ({len(test_df) / len(df_features):.2%})")

train_df.to_csv(os.path.join(output_dir, 'train_data.csv'), index=False)
val_df.to_csv(os.path.join(output_dir, 'val_data.csv'), index=False)
test_df.to_csv(os.path.join(output_dir, 'test_data.csv'), index=False)


valid_folders = ['Baseline', 'Ictal']
all_image_paths = glob.glob(os.path.join(images_root_dir, '**', '*.png'), recursive=True)
print(f"\n找到 {len(all_image_paths)} 个时频图文件。")


filtered_image_paths = [path for path in all_image_paths if any(folder in path for folder in valid_folders)]
print(f"筛选后找到 {len(filtered_image_paths)} 个时频图文件。")

train_image_paths = []
val_image_paths = []
test_image_paths = []

sorted_animal_ids = sorted(all_animal_ids, key=len, reverse=True)

for img_path in tqdm(filtered_image_paths):
    filename = os.path.basename(img_path)
    current_animal_id = None
    for aid in sorted_animal_ids:
        if aid in filename:
            current_animal_id = aid
            break

    if current_animal_id:
        if current_animal_id in train_ids:
            train_image_paths.append(img_path)
        elif current_animal_id in val_ids:
            val_image_paths.append(img_path)
        elif current_animal_id in test_ids:
            test_image_paths.append(img_path)

print("2D result")
print(f"train: {len(train_image_paths)} ({len(train_image_paths) / len(filtered_image_paths):.2%})")
print(f"val: {len(val_image_paths)} ({len(val_image_paths) / len(filtered_image_paths):.2%})")
print(f"test: {len(test_image_paths)} ({len(test_image_paths) / len(filtered_image_paths):.2%})")


with open(os.path.join(output_dir, 'train_images.txt'), 'w') as f:
    f.write('\n'.join(train_image_paths))
with open(os.path.join(output_dir, 'val_images.txt'), 'w') as f:
    f.write('\n'.join(val_image_paths))
with open(os.path.join(output_dir, 'test_images.txt'), 'w') as f:
    f.write('\n'.join(test_image_paths))

#1d
valid_folders = ['Baseline', 'Ictal']
all_npy_paths = glob.glob(os.path.join(npy_root_dir, '**', '*.npy'), recursive=True)
print(f"\n {len(all_npy_paths)} 个.npy")

filtered_npy_paths = [path for path in all_npy_paths if any(folder in path for folder in valid_folders)]
print(f"筛选后找到 {len(filtered_npy_paths)} 个.npy")

train_npy_paths = []
val_npy_paths = []
test_npy_paths = []

print("(1D)")
for npy_path in tqdm(filtered_npy_paths):
    filename = os.path.basename(npy_path)
    current_animal_id = None
    for aid in sorted_animal_ids:
        if aid in filename:
            current_animal_id = aid
            break
            
    if current_animal_id:
        if current_animal_id in train_ids:
            train_npy_paths.append(npy_path)
        elif current_animal_id in val_ids:
            val_npy_paths.append(npy_path)
        elif current_animal_id in test_ids:
            test_npy_paths.append(npy_path)


print(f"train num{len(train_npy_paths)} ({len(train_npy_paths) / len(filtered_npy_paths):.2%})")
print(f"val num{len(val_npy_paths)} ({len(val_npy_paths) / len(filtered_npy_paths):.2%})")
print(f"test num{len(test_npy_paths)} ({len(test_npy_paths) / len(filtered_npy_paths):.2%})")

with open(os.path.join(output_dir, 'train_1d.txt'), 'w') as f:
    f.write('\n'.join(train_npy_paths))
with open(os.path.join(output_dir, 'val_1d.txt'), 'w') as f:
    f.write('\n'.join(val_npy_paths))
with open(os.path.join(output_dir, 'test_1d.txt'), 'w') as f:
    f.write('\n'.join(test_npy_paths))
