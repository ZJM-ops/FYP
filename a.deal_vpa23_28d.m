% 使用MATLAB/EEGLAB截取EDF文件最后10分钟并导出为EDF格式
clear; close all;
[ALLEEG, EEG, CURRENTSET, ALLCOM] = eeglab;
filepath = 'D:\FYP\shu\脑电图数据-已提取的10min脑电图\VPA组\VPA组23号\';
filename = '23号建模后28d.edf';
fullfile_path = fullfile(filepath, filename);

% 要截取的时长
duration_to_cut_sec = 600; % 10分钟
fprintf('正在加载文件: %s\n', fullfile_path);
try
    EEG = pop_biosig(fullfile_path);
        [ALLEEG, EEG, CURRENTSET] = eeg_store(ALLEEG, EEG, 0);
    fprintf('文件加载成功！\n');
catch ME
    fprintf('错误：文件加载失败！\n');
    fprintf('错误信息: %s\n', ME.message);
    return; 

total_duration_sec = EEG.xmax;
start_time_sec = total_duration_sec - duration_to_cut_sec;

if start_time_sec < 0
    fprintf('错误：文件总时长 (%.2f秒) 小于要截取的10分钟。\n', total_duration_sec);
    return; % 如果文件太短，则终止脚本
end

fprintf('文件总时长: %.2f 秒\n', total_duration_sec);
fprintf('将截取最后 %d 秒的数据...\n', duration_to_cut_sec);
fprintf('截取范围 (秒): 从 %.3f 到 %.3f\n', start_time_sec, total_duration_sec);

try
    EEG = pop_select(EEG, 'time', [start_time_sec, total_duration_sec]);
    % 再次存储和更新工作区
    [ALLEEG, EEG, CURRENTSET] = eeg_store(ALLEEG, EEG, 1);
    fprintf('数据截取成功！\n');
catch ME
    fprintf('错误：数据截取失败！\n');
    fprintf('错误信息: %s\n', ME.message);
    return;
end

output_filename = '23号建模后28d_last10min.edf';
output_filepath = filepath; 

fprintf('正在将文件导出为EDF格式...\n');
fprintf('保存路径: %s\n', fullfile(output_filepath, output_filename));

try
        pop_writeeeg(EEG, fullfile(output_filepath, output_filename), 'TYPE', 'EDF');
    fprintf('\n操作最终成功！文件已保存为EDF格式。\n');
catch ME
    fprintf('\n错误：导出为EDF格式失败！\n');
    fprintf('请确保您已经正确安装了 "BioSig" 插件并重启了MATLAB。\n');
    fprintf('错误信息: %s\n', ME.message);
    % 如果导出失败，作为备选方案，保存为.set格式以便保留工作
    output_filename_set = '23号建模后28d_last10min_matlab_backup.set';
    pop_saveset(EEG, 'filename', output_filename_set, 'filepath', output_filepath);
    fprintf('已将截取的数据保存为EEGLAB的.set格式作为备份: %s\n', output_filename_set);
end

eeglab redraw; % 刷新EEGLAB图形界面

