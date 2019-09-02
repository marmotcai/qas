
import os
import time

def params_split(params, flag = ':'):
    return params.split(flag)

def get_code_from_filename(filepath):
    path, filename = os.path.split(filepath)
    return os.path.splitext(filename)

def get_suffix_from_filepath(filepath):
    return os.path.splitext(filepath)[1]

def check_path_exists(filepath):
    path = os.path.dirname(filepath)
    if not path_exists(path):
        return mkdir(path)
    return True

def mkdir(path):
    path = path.strip() # 去除首位空格
    path = path.rstrip("\\") # 去除尾部 \ 符号
    # 判断结果
    if not os.path.exists(path):
        os.makedirs(path) # 如果不存在则创建目录

def path_exists(path):
    return os.path.exists(path)

def getfiletime(path):
    timestamp = os.path.getmtime(path)
    return timestamp, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))

def isint(num):
    try:
        num = int(str(num))
        return isinstance(num, int)
    except:
        return False