import os
import sys
import threading
import main
import trendanalysis as ta
from trendanalysis.core import manager as g_man
from trendanalysis.utils import daemon

class DaemonThread:
    __init = 1

    def __init__(self):
        self.__sem = threading.Semaphore(value = 1) # 初始化信号量，最大并发数
        ta.g.log.debug("Start daemon thread..")
        return

    def handle(self, params):
        #开启线程，传入参数
        _thread = threading.Thread(target = self.__run, args = (params,))
        _thread.setDaemon(True)
        _thread.start()#启动线程
        return

    def __run(self, params):
        self.__sem.acquire()#信号量减1
        ta.g.log.debug("load config and run : " + params)
        main.service(params)
        self.__sem.release()#信号量加1
        return

class TDaemon(daemon.Daemon):
    def __init__(self, *args, **kwargs):
        super(TDaemon, self).__init__(*args, **kwargs)
        # ta.g.print_current_env_nformation()

    def run(self):
        main.service()

def control_daemon(action):
    os.system(" ".join((sys.executable, __file__, action)))

def usage():
    print("usage : start, stop , restart")

if __name__ == '__main__':
    if len(sys.argv) == 1:
        usage()
    elif len(sys.argv) == 2:
        arg = sys.argv[1]
        if arg in ('start', 'stop', 'restart'):
            d = TDaemon(ta.g.pid, verbose = 0)
            getattr(d, arg)()
