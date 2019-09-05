import sys
import time
import getopt
import multiprocessing
import subprocess

import schedule

import update
import trendanalysis as ta
from trendanalysis import global_obj
from trendanalysis.core import manager as g_man
from trendanalysis.core import data_manager as g_dm

################################################################################

def usage():
    print("-h --help,           Display this help and exit")
    print("-v --version,        Print version infomation")
    print("-t --test,           Test mode (e.g:-t gpu)")
    print("-d --download,       Download data (e.g:-d inx=inx_code.csv)")
    print("-m --modeling,       Training data build model")
    print("-e --evaluation,     Evaluation model")

def loaddata(filename):
    data = g_dm.train_data(filename)
    print(data.df.tail(10))
    return data

def evaluation(params):
    type = ""
    if "|" in params:
        type, filepath = params.split("|")
    else:
        type = ta.g.default_model_type
        filepath = params

    if "," in filepath:
        mod_filepath, data_filepath = filepath.split(",")
    else:
        return

    if not ta.g_tools.path_exists(mod_filepath):
        mod_filepath = ta.g.stk_path + mod_filepath
    if not ta.g_tools.path_exists(mod_filepath):
        print(mod_filepath + " is not exists")
        return

    if not ta.g_tools.path_exists(data_filepath):
        data_filepath = ta.g.stk_path + data_filepath
    if not ta.g_tools.path_exists(data_filepath):
        print(data_filepath + " is not exists")
        return

    model = g_man.model(loaddata(data_filepath))
    model.modeling(type)
    model.eva(mod_filepath)

def test(type):
    if type in ("gpu"):
        print(g_man.test_gpu())

class process(multiprocessing.Process):
    def __init__(self, cmd, args):
        multiprocessing.Process.__init__(self)
        self.args = []
        self.args.append(cmd)
        for j in range(0, len(args)):
            self.args.append(args[j])

        print(self.args)

    def run(self):
        subprocess.check_call(self.args)

def schedule_analysis(at, cmd, args):
    if len(at) > 0 and at[0] in ("seconds"):
        schedule.every(int(at[1])).seconds.do(main, cmd, args)

    if len(at) > 0 and at[0] in ("minutes"):
        schedule.every(int(at[1])).minutes.do(main, cmd, args)

    if len(at) > 0 and at[0] in ("hour"):
        schedule.every(int(at[1])).hour.do(main, cmd, args)

    if len(at) > 0 and at[0] in ("day"):
        schedule.every().day.at(at[1]).do(main, cmd, args)

def loadconfig_and_run(filename):
    g = global_obj.Global(filename)
    for index in range(0, len(g.schedules)):
        item = g.schedules[index]
        at = item['at'].split(",")
        cmdlst = item['cmd'].split(",")
        if len(cmdlst) > 1:
            cmd = cmdlst[0]
            args = cmdlst[1]
        else:
            cmd = "python"
            args = cmdlst

        print('schedule', index, ":")
        print(" at:      ", at)
        print(" cmd:     ", cmd)
        print(" args:    ", args)

        schedule_analysis(at, cmd, args)

    while True:
        schedule.run_pending()
        time.sleep(5)

def main(cmd, argv):
    if len(cmd) > 0:
        process(cmd, argv).start()
        return

    try:
        options, args = getopt.getopt(argv, "hvuil:s:t:d:m:p:e:", ["help", "version", "update", "initialize", "load=", "setting=", "test=", "download=", "modeling=", "evaluation=", "predict="])
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
        if name in ("-l", "--load"):
            loadconfig_and_run(value)
        if name in ("-t", "--test"):
            test(value)
        if name in ("-d", "--download"):
            g_dm.main(argv)
        if name in ("-m", "--modeling"):
            g_man.main(argv)
        if name in ("-p", "--predict"):
            g_man.main(argv)
        if name in ("-e", "--evaluation"):
            evaluation(value)

if __name__ == '__main__':
    main("", sys.argv[1:])
    sys.exit()