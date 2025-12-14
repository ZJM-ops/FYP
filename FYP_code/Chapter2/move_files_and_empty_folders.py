import os
import shutil

source_base_dir = "/home/student/s230005071/shu/脑电图数据-已提取的10min脑电图"
destination_dir ="/home/student/s230005071/shu/delete"


files_to_move = [
    ('pilo建模刚发作.edf',),
    ('pilo建模前1d.edf',),
    ('pilo建模时.edf',),
    ('建模前1d.edf',),
    ('止惊前.edf',),

    # mir过表达组
    #('mir过表达组', '24号', '止惊前10min-1.edf'),
    #('mir过表达组', '31号', '止惊前10min.edf'),
    #('mir过表达组', '33号', '建模前1d.edf'),
    #('mir过表达组', '35号', '建模前1d-1.edf'),
    #('mir过表达组', '35号', '建模前1d.edf'),
    #('mir过表达组', '35号', '止惊前10min.edf'),
    ('mir过表达组', '24号', '24号建模前1d.edf'),
    ('mir过表达组', '29号', '29号建模前1d.edf'),
    ('mir过表达组', '30号（电极问题）', '30号IV级前20min.edf'),
    ('mir过表达组', '30号（电极问题）', '30号建模前1d.edf'),
    ('mir过表达组', '31号', '31号IV级前30min.edf'),
    ('mir过表达组', '31号', '止惊后10min.edf'),
    ('mir过表达组', '35号', '28d.edf'),
    ('mir过表达组', '35号', '35号止惊后3h.edf'),

    # pilo组
    #('pilo组', 'pilo组10号', '10号建模前1d.edf'),
    #('pilo组', 'pilo组15号', '15号止惊后3h.edf'),
    #('pilo组', 'pilo组15号', '建模前1d-1.edf'),
    #('pilo组', 'pilo组1号', '建模前1d.edf'),
    #('pilo组', 'pilo组2号', '2号建模后第1天（1）.edf'),
    ('pilo组', 'pilo组15号', '15号建模后3d.edf'),
    ('pilo组', 'pilo组15号', '15号止惊后2h.edf'),
    ('pilo组', 'pilo组2号', '2号止惊后2h.edf'),
    ('pilo组', 'pilo组7号', 'pilo组7号IV级前30min.edf'),
    ('pilo组', 'pilo组7号', 'pilo组7号止惊前10min.edf'),
    ('pilo组', 'pilo组7号', 'pilo组7号建模后第3d.edf'),

    # sponges组
    #('sponges组', '41号', '止惊前10min.edf'),
    #('sponges组', '43号', 'DA00109A.edf'),
    ('sponges组', '44号', '44号IV级前30min(1).edf'),
    #('sponges组', '44号', 'DA00109J.edf'),
    #('sponges组', '45号', '止惊后10min.edf'),
    #('sponges组', '46号', '7d-1.edf'),
    #('sponges组', '46号', 'DA0010A1.edf'),
    ('sponges组', '40号', '40号建模前1d.edf'),
    ('sponges组', '40号', 'DA00108Q.edf'),
    ('sponges组', '41号', '41号建模前1d.edf'),
    ('sponges组', '41号', '41号止惊后10min.edf'),
    ('sponges组', '41号', 'DA00108V.edf'),
    ('sponges组', '43号', '43号建模前1d.edf'),
    ('sponges组', '44号', '44号IV级前10min.edf'),

    # VPA组
    ('VPA组', '18号', '18号建模前1d.edf'),
    #('VPA组', 'VPA组12号', 'IV级前20min-1.edf'),
    #('VPA组', 'VPA组12号', '建模前1d.edf'),
    #('VPA组', 'VPA组12号', '建模后1d-1.edf'),
    #('VPA组', 'VPA组12号', '建模后1d-2.edf'),
    #('VPA组', 'VPA组12号', '建模后3d-1.edf'),
    #('VPA组', 'VPA组12号', '止惊后10min-1.edf'),
    #('VPA组', 'VPA组12号', '止惊后10min-2.edf'),
    #('VPA组', 'VPA组12号', '止惊后10min-3.edf'),
    #('VPA组', 'VPA组12号', '止惊后1h-1.edf'),
    #('VPA组', 'VPA组12号', '止惊后2h-1.edf'),
    #('VPA组', 'VPA组16号', '20min.edf'),
    #('VPA组', 'VPA组16号', 'IV级前30min-1.edf'),
    #('VPA组', 'VPA组16号', 'IV级前30min-2.edf'),
    #('VPA组', 'VPA组16号', 'IV级前30min-3.edf'),
    #('VPA组', 'VPA组16号', '止惊前10min-1.edf'),
    #('VPA组', 'VPA组16号', '止惊前10min-2.edf'),
    #('VPA组', 'VPA组19号', '建模后28d-1.edf'),
    #('VPA组', 'VPA组19号', '止惊后10min-1.edf'),
    #('VPA组', 'VPA组19号', '止惊后10min-2.edf'),
    #('VPA组', 'VPA组19号', '止惊后10min-3.edf'),
    #('VPA组', 'VPA组21号', '21号止惊前10min.edf'),
    #('VPA组', 'VPA组21号', 'IV级前20min-1.edf'),
    ('VPA组', 'VPA组16号', '16号止惊后1h.edf'),
    ('VPA组', 'VPA组23号', '23号IV级前30min.edf'),
    ('VPA组', 'VPA组23号', '23号建模后28d.edf'),

    # 空载组
    #('空载组', '50号', '止惊前10min-1.edf'),
    #('空载组', '53号', 'IV级前10min-1.edf'),
    #('空载组', '53号', 'IV级前20min(1).edf'),
    #('空载组', '53号', '止惊前10min-1.edf'),
    #('空载组', '53号', '止惊前10min.edf'),
    ('空载组', '48号', '48号止惊前10min.edf'),
    ('空载组', '48号', 'IV级前30min.edf'),
    ('空载组', '48号', '建模前1d.edf'),
    ('空载组', '49号', '建模前1d.edf'),
    ('空载组', '50号', '50号止惊前10min.edf'),
    ('空载组', '50号', 'IV级前10min.edf'),
    ('空载组', '50号', '建模前1d-1.edf'),
    ('空载组', '50号', '建模前1d.edf'),
    ('空载组', '53号', 'IV级前20min-1.edf'),
    ('空载组', '53号', '建模前1d.edf'),
    ('空载组', '53号', '止惊后10min.edf'),
    ('空载组', '54号', 'IV级前20min.edf'),
    ('空载组', '54号', '建模前1d.edf'),
    ('空载组', '55号', '55号建模前1d.edf'),
    ('空载组', '55号', 'IV级前30min.edf'),
]


def move_files_and_empty_folders():
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

 
    moved_files_count = 0
    not_found_files_count = 0

    for parts in files_to_move:
        source_path = os.path.join(source_base_dir, *parts)
        destination_path = os.path.join(destination_dir, parts[-1])

        if os.path.exists(source_path):
            try:
                shutil.move(source_path, destination_path)
                print(f"成功移动文件: {source_path} -> {destination_path}")
                moved_files_count += 1
            except Exception as e:
                print(f"移动文件时出错: {source_path}. 错误: {e}")
        else:
            print(f"文件未找到，跳过: {source_path}")
            not_found_files_count += 1
    
    print(f"\n文件移动完成。成功移动 {moved_files_count} 个文件，{not_found_files_count} 个文件未找到。")

    print("\n--- 开始检查并移动空文件夹 ---")
    moved_folders_count = 0
    
    for dirpath, dirnames, filenames in os.walk(source_base_dir, topdown=False):
        if dirpath == source_base_dir:
            continue

        if not dirnames and not filenames:
            try:
                folder_name = os.path.basename(dirpath)
                destination_folder_path = os.path.join(destination_dir, folder_name)
  
                counter = 1
                original_destination_path = destination_folder_path
                while os.path.exists(destination_folder_path):
                    destination_folder_path = f"{original_destination_path}_{counter}"
                    counter += 1

                shutil.move(dirpath, destination_folder_path)
                print(f"成功移动空文件夹: {dirpath} -> {destination_folder_path}")
                moved_folders_count += 1
            except Exception as e:
                print(f"移动空文件夹时出错: {dirpath}. 错误: {e}")

    print(f"\n空文件夹处理完成。成功移动 {moved_folders_count} 个空文件夹。")

if __name__ == "__main__":
    move_files_and_empty_folders()