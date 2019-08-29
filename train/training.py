from __future__ import print_function

import sys
import time
import getopt
import multiprocessing
import subprocess

import schedule as sc

import tensorflow as tf

from train.quant import dataobject as my_do
from train.quant import modeling as my_mo

from train.utils import params as my_params
from train.utils import tools as my_tools
from train.utils import update as my_update

################################################################################

def test_gpu():
    device_name = tf.test.gpu_device_name()
    if device_name != '/device:GPU:0':
        return 'GPU device not found'
        # raise SystemError('GPU device not found')
    return 'Found GPU at: {}'.format(device_name)

def usage():
    print("-h --help,           Display this help and exit")
    print("-v --version,        Print version infomation")
    print("-s --setting,        Set params (e.g: -s datapath=./data")
    print("-t --test,           Test mode (e.g:-t gpu)")
    print("-d --download,       Download data (e.g:-d inx=inx_code.csv)")
    print("-m --modeling,       Training data build model")
    print("-e --evaluation,     Evaluation model")

def setting(setting):
    if "=" in setting:
        item, value = setting.split("=")
    else:
        return

    if len(item) <= 0 or len(value) <= 0:
        print("setting params error!")
        return

    my_params.global_obj.set_item_value(item, value)

def loaddata(filename):
    data = my_do.train_data(filename)
    print(data.df.tail(10))
    return data

def evaluation(params):
    type = ""
    if "|" in params:
        type, filepath = params.split("|")
    else:
        type = my_params.default_model
        filepath = params

    if "," in filepath:
        mod_filepath, data_filepath = filepath.split(",")
    else:
        return

    if not my_tools.path_exists(mod_filepath):
        mod_filepath = my_params.g_config.day_path + mod_filepath
    if not my_tools.path_exists(mod_filepath):
        print(mod_filepath + " is not exists")
        return

    if not my_tools.path_exists(data_filepath):
        data_filepath = my_params.g_config.day_path + data_filepath
    if not my_tools.path_exists(data_filepath):
        print(data_filepath + " is not exists")
        return

    model = my_mo.model(loaddata(data_filepath))
    model.modeling(type)
    model.eva(mod_filepath)

def test(type):
    if type in ("gpu"):
        print(my_tools.test_gpu())

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

def loadconfig(filename):
    my_params.g_config.load_config(filename)

    for index in range(0, len(my_params.g_config.schedules)):
        cmder = my_params.g_config.schedules[index]
        at = cmder.time.split(",")
        cmdstr = cmder.cmd.split(",")

        if len(cmdstr) > 1:
            args = cmdstr[1].split(" ")
            cmdstr = cmdstr[0]
        else:
            args = cmder.cmd.split(" ")
            cmdstr = ""

        schedule(at, cmdstr, args)

def schedule(at, cmd, args):
    if len(at) > 0 and at[0] in ("seconds"):
        sc.every(int(at[1])).seconds.do(main, cmd, args)

    if len(at) > 0 and at[0] in ("minutes"):
        sc.every(int(at[1])).minutes.do(main, cmd, args)

    if len(at) > 0 and at[0] in ("hour"):
        sc.every(int(at[1])).hour.do(main, cmd, args)

    if len(at) > 0 and at[0] in ("day"):
        sc.every().day.at(at[1]).do(main, cmd, args)

    while True:
        sc.run_pending()
        time.sleep(5)

def main(cmd, argv):
    if len(cmd) > 0:
        process(cmd, argv).start()
        return

    try:
        options, args = getopt.getopt(argv, "hvuil:s:t:d:m:e:", ["help", "version", "update", "load=", "setting=", "test=", "download=", "modeling=", "evaluation="])
    except getopt.GetoptError:
        sys.exit()

    for name, value in options:
        if name in ("-h", "--help"):
            usage()

        if name in ("-v", "--version"):
            my_params.g.config.print_current_information()

        if name in ("-u", "--update"):
            my_update.main(argv)

        if name in ("-i", "--initialize"):
            usage()

        if name in ("-l", "--load"):
            loadconfig(value)
        if name in ("-s", "--setting"):
            setting(value)
        if name in ("-t", "--test"):
            test_gpu(value)
        if name in ("-d", "--download"):
            my_do.main(argv)
        if name in ("-m", "--modeling"):
            my_mo.main(argv)
        if name in ("-e", "--evaluation"):
            evaluation(value)

if __name__ == '__main__':
    main("", sys.argv[1:])
    sys.exit()
