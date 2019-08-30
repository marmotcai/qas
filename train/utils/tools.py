
import os

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
    if not os.path.isdir(path):
        path = os.path.dirname(path)
    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip("\\")
    # 判断结果
    if not os.path.exists(path):
        # 如果不存在则创建目录
        os.makedirs(path)
        return True
    else:
        return False

def path_exists(path):
    return os.path.exists(path)

def isint(num):
    try:
        num = int(str(num))
        return isinstance(num, int)
    except:
        return False