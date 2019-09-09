
import os
import sys
import time
import platform
import datetime as dt
import multiprocessing
import subprocess
import trendanalysis as ta

def TestPlatform():
    print ("--------------Operation System-----------------------")
    #Windows will be : (32bit, WindowsPE)
    #Linux will be : (32bit, ELF)
    print(platform.architecture())

    #Windows will be : Windows-XP-5.1.2600-SP3 or Windows-post2008Server-6.1.7600
    #Linux will be : Linux-2.6.18-128.el5-i686-with-redhat-5.3-Final
    print(platform.platform())

    #Windows will be : Windows
    #Linux will be : Linux
    print(platform.system())

    print ("--------------Python Version-------------------------")
    #Windows and Linux will be : 3.1.1 or 3.1.3
    print(platform.python_version())

def UsePlatform():
    sysstr = platform.system()
    return sysstr

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

def print_sys_info():
    lists = sys.argv  # 传递给Python脚本的命令行参数列表 => python p.py -> ['p.py'] / python p.py a 1 -> ['p.py', 'a', '1'] / 程序内执行 -> ['']
    strs = sys.getdefaultencoding()  # 默认字符集名称
    strs = sys.getfilesystemencoding()  # 系统文件名字符集名称
    num = sys.getrefcount(object)  # 返回object的引用计数(比实际多1个)
    dicts = sys.modules  # 已加载的模块, 可修改, 但不能通过修改返回的字典进行修改
    lists = sys.path  # 模块搜索路径
    sys.path.append(".")  # 动态添加模块搜索路径
    strs = sys.platform  # 平台标识符(系统身份进行详细的检查,推荐使用) Linux:'linux' / Windows:'win32' / Cygwin:'cygwin' / Mac OS X:'darwin'
    strs = sys.version  # python解释器版本
    lists = sys.thread_info  # 线程信息
    num = sys.api_version  # 解释器C API版本

class Timer():
    def __init__(self):
        self.start_dt = None

    def start(self):
        self.start_dt = dt.datetime.now()

    def stop(self):
        end_dt = dt.datetime.now()
        print('Time taken: %s' % (end_dt - self.start_dt))

class process(multiprocessing.Process):
    def __init__(self, cmd, args):
        multiprocessing.Process.__init__(self)
        self.args = []
        self.args.append(cmd)
        for j in range(0, len(args)):
            self.args.append(args[j])

        print(self.args)

    def run(self):
        ta.g.log.info("start process:")
        ta.g.log.info(self.args)
        subprocess.check_call(self.args)