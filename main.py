import os

import sys
import getopt
import update
import trendanalysis as ta
from trendanalysis.core import manager as g_man
from trendanalysis.core import data_manager as g_dm

################################################################################

def usage():
    print("-h --help,           Display this help and exit")
    print("-v --version,        Print version infomation")
    print("-u --update,         update new ver")
    print("-i --initialize,    initialize")
    print("-d --daemon,         daemon mode")
    print("-t --test,           Test mode (e.g:-t gpu)")
    print("-d --download,       Download data (e.g:-d inx=inx_code.csv)")
    print("-m --modeling,       Training data build model")
    print("-e --evaluation,     Evaluation model")

def main(cmd, argv):
#    if len(cmd) > 0:
#        my_tools.process(cmd, argv).start()
#        return
    try:
        options, args = getopt.getopt(argv, "hvuidt:d:m:p:e:", ["help", "version", "update", "initialize", "daemon", "test=", "download=", "modeling=", "evaluation=", "predict="])
    except getopt.GetoptError:
        sys.exit()

    for name, value in options:
        if name in ("-h", "--help"):
            usage()
        if name in ("-v", "--version"):
            ta.g.print_current_env_nformation()
        if name in ("-u", "--update"):
            update.main(argv)
        if name in ("-i", "--initialize"):
            g_man.main(argv)
        if name in ("-d", "--daemon"):
            os.system(cmd + " daemon_main.py start")
            # my_tools.process(cmd, ["daemon_main.py", "start"]).start()
        if name in ("-t", "--test"):
            g_man.test(value)
        if name in ("-d", "--download"):
            g_dm.main(argv)
        if name in ("-m", "--modeling"):
            g_man.main(argv)
        if name in ("-p", "--predict"):
            g_man.main(argv)
        if name in ("-e", "--evaluation"):
            g_man.evaluation(value)

if __name__ == '__main__':
    main(sys._base_executable, sys.argv[1:])
    sys.exit()