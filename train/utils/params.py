

import configparser
from train.utils import tools as my_tools
from train.utils import logger as log

import pandas as pd

################################################################################

version = 'Atom Quant Analysis System, Version: 0.0.1'
app_name = 'qas'

default_datapath = './data/'
default_logpath = default_datapath + 'logs/'
default_pid = default_datapath + app_name + '.pid'

default_configfile = './config.ini'
default_section_setting = 'setting'
default_section_schedule = 'schedule_'

default_modpath = 'mod/'
default_daypath = 'day/'
default_stkpath = 'stk/'
default_inxpath = 'inx/'
default_inx_filename = 'my_inx_code.csv'
default_stk_inx_filename = 'my_stk_inx.csv'
default_stk_base_filename = 'my_stk_base.csv'
default_stk_code_filename = 'my_stk_code.csv'

default_model_type = 'rate'

################################################################################
#
ohlc_lst = ['open', 'high', 'low', 'close']
volume_lst = ['volume']
#
ohlcv_lst = ohlc_lst + ['volume']
#
ohlcd_lst = ['date'] + ohlc_lst
ohlcdv_lst = ['date'] + ohlcv_lst
#

profit_lst = ['next_profit_1', 'next_profit_2', 'next_profit_3', 'next_profit_4', 'next_profit_5', 'next_profit_6', 'next_profit_7', 'next_profit_8', 'next_profit_9', 'next_profit_10']

################################################################################

ma100_lst_var = [2, 3, 5, 10, 15, 20, 25, 30, 50, 100]
ma100_lst = ['ma_2', 'ma_3', 'ma_5', 'ma_10', 'ma_15', 'ma_20', 'ma_25', 'ma_30', 'ma_50', 'ma_100']
ma200_lst_var = [2, 3, 5, 10, 15, 20, 25, 30, 50, 100, 150, 200]
ma200_lst = ['ma_2', 'ma_3', 'ma_5', 'ma_10', 'ma_15', 'ma_20','ma_30', 'ma_50', 'ma_100', 'ma_150', 'ma_200']
ma030_lst_var = [2, 3, 5, 10, 15, 20, 25, 30]
ma030_lst = ['ma_2', 'ma_3', 'ma_5', 'ma_10', 'ma_15', 'ma_20', 'ma_25', 'ma_30']
xagv_lst = ['xavg1', 'xavg2', 'xavg3', 'xavg4', 'xavg5', 'xavg6', 'xavg7', 'xavg8', 'xavg9']
rate_lst = ['next_rate_5', 'next_rate_10']
other_lst = ['price_range', 'amp', 'amp_type']

################################################################################

class cmder:
    def __init__(self, cmd, time):
        self.cmd = cmd
        self.time = time


class Global:

    def __init__(self):

        my_tools.mkdir(default_datapath)

        self.config = config()
        self.log = log.logger(self.config.log_file, app_name)

        pd.options.display.max_rows = 10
        pd.options.display.float_format = '{:.1f}'.format

class config:
    def __init__(self, filename = default_configfile):
        self.conf = configparser.ConfigParser()

        if my_tools.path_exists(filename) == False:
            self.init_config(filename)

        self.load_config(filename)

    def clear(self):

        self.data_path = ""
        self.stk_path = ""
        self.inx_path = ""

        self.schedules = []

    def init_config(self, filename):
        self.conf.add_section(default_section_setting)
        self.conf.set(default_section_setting, 'logfile', default_logpath + app_name + '.log')
        self.conf.set(default_section_setting, 'datapath', default_datapath)
        self.conf.set(default_section_setting, 'modpath', default_datapath + default_modpath)
        self.conf.set(default_section_setting, 'daypath', default_datapath + default_daypath)
        self.conf.set(default_section_setting, 'stkpath', default_datapath + default_stkpath)
        self.conf.set(default_section_setting, 'inxpath', default_datapath + default_inxpath)

        section_schedule = default_section_schedule + "0"
        self.conf.add_section(section_schedule)
        self.conf.set(section_schedule, 'cmd', "-d " + default_inx_filename)
        self.conf.set(section_schedule, 'at', "seconds,10")

        with open(filename, 'w+') as fw:
            self.conf.write(fw)

    def load_config(self, filename):
        self.clear()
        self.conf.read(filename)  # 装载配置文件

        self.log_file = self.conf.get(default_section_setting, 'logfile')
        self.data_path = self.conf.get(default_section_setting, 'datapath')
        self.mod_path = self.conf.get(default_section_setting, 'modpath')
        self.stk_path = self.conf.get(default_section_setting, 'stkpath')
        self.inx_path = self.conf.get(default_section_setting, 'inxpath')

        self.schedules = []
        index = 0
        while True:
            section_schedule = default_section_schedule + str(index)
            try:
                cmd = self.conf.get(section_schedule, 'cmd')
                time = self.conf.get(section_schedule, 'at')

                self.schedules.append(cmder(cmd, time))
            except:
                break

            index = index + 1

        # section = self.conf.sections()[0]
        # print(section) # print(self.conf.options(section))
        # print(self.conf.items(section))

    def print_current_information(self):
        print("-----------------------------")
        print(version)
        print("***********")
        print("log_file:    " + self.log_file)
        print("data_path:   " + self.data_path)
        print("mod_path:   " + self.mod_path)
        print("stk_path:    " + self.stk_path)
        print("inx_path:    " + self.inx_path)

        print("schedule count: " +  str(len(self.schedules)))
        for index in range(0, len(self.schedules)):
            section_schedule = default_section_schedule + str(index)
            cmder = self.schedules[index]
            print(section_schedule + ":")
            print("cmd:     " + cmder.cmd)
            print("at:      " + cmder.time)

        print("-----------------------------")

g = Global()
g.config.print_current_information()

################################################################################
