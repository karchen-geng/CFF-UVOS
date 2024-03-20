# 将objects数据集变为eval模式
import os
import shutil

def traverse_files(dir):
    print("start-----------------------------")
    for file_name in os.listdir(dir):
        file_path = os.path.join(dir, file_name)
        for file_pre in os.listdir(file_path):
            file_dir = os.path.join(file_path, file_pre)
            for file in os.listdir(file_dir):
                shutil.copy2(os.path.join(file_dir, file), file_path+'/')
                os.renames(os.path.join(file_path, file), os.path.join(file_path, file_pre + '_' + file))
    print("end---------------------------")

def del_filepath(dir):
    for file_name in os.listdir(dir):
        file_path = os.path.join(dir, file_name)
        for file in os.listdir(file_path):
            if not os.path.isfile(os.path.join(file_path, file)):
                shutil.rmtree(os.path.join(file_path, file))

fir_dir = "/media/wyx/2T/gck/YouTubeObjects_for_eval/JPEGImages"
#  修改路徑
# traverse_files(fir_dir)

#  刪除文件夾
# del_filepath(fir_dir)