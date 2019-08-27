
import os
import tensorflow as tf

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

def test_gpu():
    device_name = tf.test.gpu_device_name()
    if device_name != '/device:GPU:0':
        return 'GPU device not found'
        # raise SystemError('GPU device not found')
    return 'Found GPU at: {}'.format(device_name)

def path_exists(path):
    if os.path.exists(path):
        return True;
    return False