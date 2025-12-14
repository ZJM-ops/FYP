clear; close all;
[ALLEEG, EEG, CURRENTSET] = eeglab;

filepath = 'D:\FYP\shu\脑电图数据-已提取的10min脑电图\VPA组\VPA组23号\';
filename = '23号建模后28d.edf';
fullpath = fullfile(filepath, filename);

duration_sec = 600;  % 截取最后 10 分钟

% 加载 EDF
try
    EEG = pop_biosig(fullpath);
    [ALLEEG, EEG] = eeg_store(ALLEEG, EEG, 0);
catch ME
    fprintf('加载失败：%s\n', ME.message);
    return;
end

% 时间范围
total_sec = EEG.xmax;
start_sec = total_sec - duration_sec;
if start_sec < 0
    fprintf('文件长度不足 10 分钟。\n');
    return;
end

% 截取数据
try
    EEG = pop_select(EEG, 'time', [start_sec total_sec]);
    [ALLEEG, EEG] = eeg_store(ALLEEG, EEG, 1);
catch ME
    fprintf('截取失败：%s\n', ME.message);
    return;
end

% 导出 EDF
output_name = '23号建模后28d_last10min.edf';

try
    pop_writeeeg(EEG, fullfile(filepath, output_name), 'TYPE', 'EDF');
catch ME
    fprintf('导出EDF失败：%s\n', ME.message);
    pop_saveset(EEG, 'filename', 'backup_last10min.set', 'filepath', filepath);
end

eeglab redraw;
