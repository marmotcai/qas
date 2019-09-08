import os
import sys

import trendanalysis as ta
from trendanalysis.core import manager as g_man
from trendanalysis.utils import daemon

class TDaemon(daemon.Daemon):
    def __init__(self, *args, **kwargs):
        super(TDaemon, self).__init__(*args, **kwargs)
        ta.g.print_current_env_nformation()
        ta.g.log.debug("start daemon..")

    def run(self):
        g_man.loadconfig_and_run()

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
        if arg in ('thread'):
            dt = DaemonThread()
            dt.handle(ta.g.config_file)
