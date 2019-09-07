
import json
import os
import datetime
import pandas as pd

import trendanalysis as ta
from trendanalysis.utils import tools as my_tools
from trendanalysis.utils import logger as my_logger

class Global:
    class cmdobj:
        def __init__(self, cmd, time):
            self.cmd = cmd
            self.time = time

    def get_config_path(self):  # config.json的绝对路径
        return os.path.join(self.get_cur_dir(), "config.json")

    def get_parent_dir(self):  # 当前文件的父目录绝对路径
        return os.path.dirname(__file__)

    def get_cur_dir(self):
        return os.path.abspath(os.curdir)

    def __init__(self, config_file="config.json"):
        # init config obj
        self.config_file = os.path.join(self.get_cur_dir(), config_file)
        self.config = json.load(open(self.config_file, 'r'))

        # init data dir
        self.data_path = os.path.join(self.get_cur_dir(), self.config['data']['base'])
        my_tools.mkdir(self.data_path)
        self.mod_path = os.path.join(self.data_path, self.config['data']['mod'])
        my_tools.mkdir(self.mod_path)
        self.stk_path = os.path.join(self.data_path, self.config['data']['stk'])
        my_tools.mkdir(self.stk_path)
        self.inx_path = os.path.join(self.data_path, self.config['data']['inx'])
        my_tools.mkdir(self.inx_path)

        # init log obj
        self.log_path = os.path.join(self.get_cur_dir(), self.config['general']['logpath'])
        my_tools.mkdir(self.log_path)
        nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
        self.log_file = os.path.join(self.log_path, self.config['general']['logfile'].replace("$TIME", nowTime))
        self.log = my_logger.logger(self.log_file, __name__)

        # init pid
        self.pid = self.data_path + ta.__name__ + '.pid'

        # init schedules
        self.schedules = self.config['general']['schedule']

        #
        pd.options.display.max_rows = int(self.config['pd']['max_rows'])
        pd.options.display.float_format = '{:.1f}'.format

        #
        # self.print_current_env_nformation()

    def print_current_env_nformation(self):
        print("-----------------------------")
        print(ta.__describe__)
        print(ta.__version__)
        print("***********")
        print("log_file:    " + self.log_file)
        print("data_path:   " + self.data_path)
        print("mod_path:   " + self.mod_path)
        print("stk_path:    " + self.stk_path)
        print("inx_path:    " + self.inx_path)

        print("schedule count: " + str(len(self.schedules)))

        for index in range(0, len(self.schedules)):
            item = self.schedules[index]
            print('schedule', index, ":")
            print(" cmd:     " + item['cmd'])
            print(" at:      " + item['at'])

        print("-----------------------------")
