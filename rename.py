import os
import shutil

def get_subdirectories(path):
    subdirectories = []
    for entry in os.scandir(path):
        if entry.is_dir():
            subdirectories.append(entry.path)
    return subdirectories

def rename_and_movefile(i,subdir):
    
    current_path = subdir
    parent_path = os.path.dirname(current_path)
    # 新建目标文件夹
    new_folder_path = os.path.join(parent_path, "dst")
    os.makedirs(new_folder_path, exist_ok=True)
    for filename in os.listdir(current_path):
        new_filename = filename[:-4] + f'{i}.bmp'
        print(new_filename)
        #os.rename(os.path.join(current_path, filename), os.path.join(current_path, new_filename))
        # 移动文件到新建的文件夹
        shutil.copy2(os.path.join(current_path, filename), new_folder_path)
        # 重命名文件
        print("ori ", os.path.join(new_folder_path, filename))
        print("new ", os.path.join(new_folder_path, new_filename))
        os.rename(os.path.join(new_folder_path, filename), os.path.join(new_folder_path, new_filename))


# 获取当前文件夹路径
current_path = os.getcwd()

# 获取当前文件夹下的子文件夹列表
subdirs = get_subdirectories(current_path)

# 打印子文件夹列表
i = 1
for subdir in subdirs:
    if subdir.endswith('dst'):
        continue
    rename_and_movefile(i,subdir)
    i = i + 1
    
