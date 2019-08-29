
import os

def mkdir(path):
    filename = os.path.basename(path)
    if len(filename) > 0:
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
    if os.path.exists(path):
        return True;
    return False

def isInt(num):
    try:
        num = int(str(num))
        return isinstance(num, int)
    except:
        return False