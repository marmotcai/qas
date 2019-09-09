import sys
import os
import math
import time
import getopt
import arrow
import threading
import numpy as np
import pandas as pd

import tensorflow as tf
import keras
from keras.utils import plot_model
from keras.models import load_model
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

import schedule
from trendanalysis import global_obj
import trendanalysis as ta
from trendanalysis.vendor import ztools as zt
from trendanalysis.vendor import zai_keras as zks
from trendanalysis.core import data_manager as my_dm
from trendanalysis.core import evaluation as eva
from trendanalysis.core import model as my_model
from trendanalysis.utils import tools as my_tools
from trendanalysis.core.data_processor import DataLoader

################################################################################

class model():
    def __init__(self, type, datafile):
        self.do = None

        self.type = ta.g.config['model']['type']
        if len(type) > 0:
            self.type = type

        if len(datafile) > 0:
            self.setdata(datafile)

    def setdata(self, filename = ""):
        self.do = my_dm.train_data(filename)

    def setmod(self, filename):
        self.mx = load_model(filename)
        return self.mx

    def save(self, model, filename):
        if len(filename) <= 0:
            return False
        my_tools.check_path_exists(filename)
        return model.save(filename)

    def modeling(self, model_filename = ""):

        if (self.do == None):
            return

        self.do.prepared(self.type, action='modeling') # 数据预处理

        # 分离训练和测试数据
        self.df_train, self.df_test = my_dm.util.split(self.do.df, 0.6)

        # 构建训练特征数据
        other_features_lst = ta.g.config['data']['ohlcv'] + ta.g.config['data']['profit'] # + xagv_lst + ma100_lst + other_lst
        self.x_train = my_dm.util.get_features(self.df_train, other_features_lst)
        self.x_test = my_dm.util.get_features(self.df_test, other_features_lst)

        #############################################################################################################

        # 构建特征，也就是结果值Y
        if self.type == 'rate':
            self.y_train = my_dm.util.prepared_y(self.df_train, 'next_rate_10_type', 'onehot')
            self.y_test = my_dm.util.prepared_y(self.df_test, 'next_rate_10_type', 'onehot')

            y_lst = self.y_train[0]
            x_lst = other_features_lst

            num_in, num_out = len(x_lst), len(y_lst)

        if self.type == 'price':
            self.y_train = my_dm.util.prepared_y(self.df_train, 'next_profit_10')
            self.y_test = my_dm.util.prepared_y(self.df_test, 'next_profit_10')

            y_lst = 1
            x_lst = other_features_lst

            num_in, num_out = len(x_lst), y_lst

        print('\n self.df_test.tail()', self.df_test.tail())
        print('\n self.x_train.shape,', self.x_train.shape)
        print('\n type(self.x_train),', type(self.x_train))

        rxn, txn = self.x_train.shape[0], self.x_test.shape[0]
        self.x_train, self.x_test = self.x_train.reshape(rxn, num_in, -1), self.x_test.reshape(txn, num_in, -1)
        print('\n x_train.shape,', self.x_train.shape)
        print('\n type(x_train),', type(self.x_train))

        print('\n num_in, num_out:', num_in, num_out)

        if not my_tools.path_exists(model_filename):
            # mx = zks.rnn010(num_in, num_out)
            # mx = zks.lstm010(num_in, num_out)
            mx = zks.lstm020typ(num_in, num_out)
        else:
            mx = self.setmod(model_filename)

        mx.summary()
        plot_model(mx, to_file = ta.g.log_path + 'model.png')

        print('\n#4 模型训练 fit')
        tbCallBack = keras.callbacks.TensorBoard(log_dir = ta.g.log_path, write_graph = True, write_images=True)
        tn0 = arrow.now()
        mx.fit(self.x_train, self.y_train, epochs = 500, batch_size = 512, callbacks = [tbCallBack])
        tn = zt.timNSec('', tn0, True)
        self.save(mx, model_filename)

        eva_obj = eva.evaluation(self.do)
        eva_obj.predict(mx, self.df_test, self.x_test)

    def eva(self, mod_filename):
        if my_tools.path_exists(mod_filename):
            model = self.setmod(mod_filename)
        eva_obj = eva.evaluation(self.do)

        x_test = my_dm.util.get_features(self.do.df, ta.g.config['data']['ohlcv'] + ta.g.config['data']['profit'])
        eva_obj.predict(model, self.do.df, x_test)

    def predict(self, mod_filename):
        if not my_tools.path_exists(mod_filename):
            return

        if (self.do == None):
            return

        self.do.prepared(self.type, action='predict') # 数据预处理

        other_features_lst = ta.g.config['data']['ohlcv'] + ta.g.config['data']['profit'] # + xagv_lst + ma100_lst + other_lst
        x_df = my_dm.util.get_features(self.do.df.tail(5), other_features_lst)

        txn = x_df.shape[0]
        x_lst = other_features_lst
        num_in = len(x_lst)
        x_df = x_df.reshape(txn, num_in, -1)
        print(x_df)

        mo = self.setmod(mod_filename)
        y_df = mo.predict(x_df)
        print(y_df)
        return y_df

################################################################################

import csv

def init_initcodefile():
    initcode_file = os.path.join(ta.g.data_path, ta.g.config["data"]["init_codefile"])
    fieldnames = ['code', 'name', 'industry', 'area']
    rows = [['601988', '中国银行', '银行', '北京'],
            ['601398', '工商银行', '银行', '北京'],
            ['601939', '建设银行', '银行', '北京'],
            ['000821', '京山轻机', '轻工机械', '湖北'],
            ['002572', '索菲亚', '家居用品', '广东'],
            ['300096', '易联众', '软件服务', '福建']
            ]
    with open(initcode_file, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)  # 写入csv文件的表头
        writer.writerows(rows)  # 同时写入多行信息
        f.close()

def init_db():
    initcode_file = os.path.join(ta.g.data_path, ta.g.config["data"]["initcodefile"])
    data_frame = pd.read_csv(initcode_file, index_col=False, encoding='gbk')
    for index, row in data_frame.iterrows():
        # Company.objects.create(name=row['name'], stock_code=row['code'])
        print(row['code'], ':', row['name'])

def initialize(params):
    if "codefile" == params.lower():
        init_initcodefile()
    if "db" == params.lower():
        init_db()
    if "downall" == params.lower():
        my_dm.download_all()

def prepared(params):
    param_lst = []
    if "," in params: param_lst = params.split(",")
    else:
        if "|" in params: param_lst = params.split("|")
        else:
            param_lst.append(params)

    def get_param(param_lst):
        type, code, datafile, modfile, lstfile = "rate", "", "", "", ""

        if len(param_lst) <= 1:
            param = my_tools.params_split(param_lst[0])
            param0 = param[0]
            if len(param) > 1:
                if 'data' == param0:
                    datafile = param[1]
                if 'code' == param0:
                    code = param[1]
                if 'lst' == param0:
                    lstfile = param[1]
            else:
                if 'init' in param0.lower():
                    lstfile = ta.g.config['data']['init_codefile']
                else:
                    if '.csv' in param0.lower():
                        datafile = param0
                    else:
                        code = param0
        else:
            for j in range(0, len(param_lst)):
                param = my_tools.params_split(param_lst[j])
                param0 = param[0]
                if 'type' == param0:
                    type = param[1]
                if 'code' == param0:
                    code = param[1]
                if 'data' == param0:
                    datafile = param[1]
                if 'mod' == param0:
                    modfile = param[1]
                if 'lst' == param0:
                    lstfile = param[1]

        if len(code) > 0:
            code = "%06d" % int(code)
        return type, code, datafile, modfile, lstfile

    type, code, datafile, modfile, lstfile = get_param(param_lst)

    if len(code) > 0 and len(datafile) <= 0: # 有代码没数据文件则先下载
        _, datafile = my_dm.download_from_code(code, '2007-01-01')

    if len(lstfile) > 0:
        if not my_tools.path_exists(lstfile):
            lstfile = os.path.join(ta.g.data_path, lstfile)
        if not my_tools.path_exists(lstfile):
            ta.g.log.error("can't find data file: " + lstfile)
            return
    else:
        if len(datafile) > 0:
            if not my_tools.path_exists(datafile):
                datafile = os.path.join(ta.g.stk_path, datafile)
            if not my_tools.path_exists(datafile):
                ta.g.log.error("can't find data file: " + datafile)
                return

    if len(code) <= 0 and len(datafile) > 0:
        code, _ = my_tools.get_code_from_filename(datafile)

    if len(modfile) <= 0 and len(code) > 0:
        modfile = ta.g.mod_path + code + ".h5"

    return type, code, datafile, modfile, lstfile

# TODO(atoml.com): 训练入口
def train(params):
    type, code, datafile, modfile, lstfile = prepared(params)
    if len(lstfile) > 0:
        data_frame = pd.read_csv(lstfile, index_col=False, encoding='gbk')
        for index, row in data_frame.iterrows():
            code = "%06d" % int(row['code'])
            _, datafile = my_dm.download_from_code(code, '2007-01-01')
            print(row['code'], ':', row['name'])
            training(type, code, datafile)
    else:
        training(type, code, datafile)

def training(type, code, datafile):

    m = my_model.Model()
    m.build_model(ta.g.config)  # 根据配置文件新建模型

    predict = True
    split = ta.g.config['data']['train_test_split']
    if not predict:
        split = 1  # 若不评估模型准确度，则将全部历史数据用于训练

    # 从本地加载训练和测试数据
    data = DataLoader(datafile, split, ta.g.config['data']['ohlcv'] + ta.g.config['data']['profit'])

    # 训练模型：
    # out-of memory generative training
    steps_per_epoch = math.ceil(
        (data.len_train - ta.g.config['data']['sequence_length']) / ta.g.config['training']['batch_size'])
    m.train_generator(
        data_gen=data.generate_train_batch(
            seq_len=ta.g.config['data']['sequence_length'],
            batch_size=ta.g.config['training']['batch_size'],
            normalise=ta.g.config['data']['normalise']
        ),
        epochs=ta.g.config['training']['epochs'],
        batch_size=ta.g.config['training']['batch_size'],
        steps_per_epoch=steps_per_epoch,
        save_dir=ta.g.mod_path,
        save_name=code
    )

    # 预测
    if predict:
        x_test, y_test = data.get_test_data(
            seq_len=ta.g.config['data']['sequence_length'],
            normalise=ta.g.config['data']['normalise']
        )

        predictions = m.predict_sequences_multiple(x_test, ta.g.config['data']['sequence_length'],
                                                       ta.g.config['data']['sequence_length'])
        print("训练：\n", predictions)

def plot_results(predicted_data, true_data):  # predicted_data与true_data：同长度一维数组
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()


# TODO(atoml.com): 预测入口
# 对指定公司的股票进行预测
def prediction(params):
    type, code, datafile, modfile, lstfile = prepared(params)

    '''
    使用保存的模型，对输入数据进行预测
    '''
    data = DataLoader(
        datafile,  # configs['data']['filename']
        ta.g.config['data']['train_test_split'],
        ta.g.config['data']['ohlcv'] + ta.g.config['data']['profit']
    )

    file_path = modfile
    m = my_model.Model()
    keras.backend.clear_session()
    m.load_model(file_path)  # 根据配置文件新建模型

    pre_len = 30
    real = False
    # predict_length = configs['data']['sequence_length']   # 预测长度
    predict_length = pre_len
    if real:  # 用最近一个窗口的数据进行预测，没有对比数据
        win_position = -1
    else:  # 用指定位置的一个窗口数据进行预测，有对比真实数据（用于绘图对比）
        win_position = -ta.g.config['data']['sequence_length']

    x_test, y_test = data.get_test_data(
        seq_len=ta.g.config['data']['sequence_length'],
        normalise=False
    )

    x_test = x_test[win_position]
    x_test = x_test[np.newaxis, :, :]
    if not real:
        y_test_real = y_test[win_position:win_position + predict_length]

    base = x_test[0][0][0]
    print("base value:\n", base)

    x_test, y_test = data.get_test_data(
        seq_len=ta.g.config['data']['sequence_length'],
        normalise=ta.g.config['data']['normalise']
    )
    x_test = x_test[win_position]
    x_test = x_test[np.newaxis, :, :]

    # predictions = model.predict_sequences_multiple(x_test, configs['data']['sequence_length'],
    #                                                predict_length)

    predictions = m.predict_1_win_sequence(x_test, ta.g.config['data']['sequence_length'], predict_length)
    # 反归一化
    predictions_array = np.array(predictions)
    predictions_array = base * (1 + predictions_array)
    predictions = predictions_array.tolist()

    print("预测数据:\n", predictions)
    if not real:
        print("真实数据：\n", y_test_real)

    plot = False
    # plot_results_multiple(predictions, y_test, predict_length)
    if plot:
        if real:
            plot_results(predictions, [])
        else:
            plot_results(predictions, y_test_real)

    return format_predictions(predictions)

def format_predictions(predictions):    # 给预测数据添加对应日期
    date_predict = []
    cur = datetime.now()
    cur += timedelta(days=1)
    counter = 0

    while counter < len(predictions):
        if cur.isoweekday()  == 6:
            cur = cur + timedelta(days=2)
        if cur.isoweekday()  == 7:
            cur = cur + timedelta(days=1)
        date_predict.append([cur.strftime("%Y-%m-%d"),predictions[counter]])
        cur = cur + timedelta(days=1)
        counter += 1

    return date_predict

# TODO(atoml.com): 获取历史数据
# 二维数组：[[data,value],[...]]
def get_hist_data(code, recent_day=30):  # 获取某股票，指定天数的历史close数据,包含日期
    _, datafile = my_dm.download_from_code(code, '2007-01-01')

    cols = ['date', 'close']
    data_frame = pd.read_csv(datafile)
    data_frame = data_frame.sort_values('date')  # 日期排序
    close_data = data_frame.get(cols).values[-recent_day:]
    return close_data.tolist()

def modeling(params):
    type, code, datafile, modfile = prepared(params)

    mo = model(type, datafile)
    mo.modeling(modfile)

def predict(params):
    type, code, datafile, modfile = prepared(params)

    if not my_tools.path_exists(modfile):
        modeling(params)

    mo = model(type, datafile)
    y = mo.predict(modfile)
    print(y)

def schedule_analysis(at, cmd, args):
    if len(at) > 0 and at[0] in ("seconds"):
        schedule.every(int(at[1])).seconds.do(main, cmd, args)

    if len(at) > 0 and at[0] in ("minutes"):
        schedule.every(int(at[1])).minutes.do(main, cmd, args)

    if len(at) > 0 and at[0] in ("hour"):
        schedule.every(int(at[1])).hour.do(main, cmd, args)

    if len(at) > 0 and at[0] in ("day"):
        schedule.every().day.at(at[1]).do(main, cmd, args)

    ##############################################################

    if len(at) > 0 and at[0] in ("monday"):
        schedule.every().monday.at(at[1]).do(main, cmd, args)
    if len(at) > 0 and at[0] in ("tuesday"):
        schedule.every().tuesday.at(at[1]).do(main, cmd, args)
    if len(at) > 0 and at[0] in ("wednesday"):
        schedule.every().wednesday.at(at[1]).do(main, cmd, args)
    if len(at) > 0 and at[0] in ("thursday"):
        schedule.every().thursday.at(at[1]).do(main, cmd, args)
    if len(at) > 0 and at[0] in ("friday"):
        schedule.every().friday.at(at[1]).do(main, cmd, args)
    if len(at) > 0 and at[0] in ("saturday"):
        schedule.every().saturday.at(at[1]).do(main, cmd, args)
    if len(at) > 0 and at[0] in ("sunday"):
        schedule.every().sunday.at(at[1]).do(main, cmd, args)

def service(filename = ""):
    g = ta.g
    if (len(filename) > 0):
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

class DaemonThread:
    __init = 1

    def __init__(self):
        self.__sem = threading.Semaphore(value = 1)#初始化信号量，最大并发数
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
        service(params)
        self.__sem.release()#信号量加1
        return

def evaluation(params):
    print(params)

def test(type):
    if type in ("gpu"):
        print(test_gpu())
    if type in ("d"):
        service(ta.g.config_file)

def test_gpu():
    device_name = tf.test.gpu_device_name()
    if device_name != '/device:GPU:0':
        return 'GPU device not found'
        # raise SystemError('GPU device not found')
    return 'Found GPU at: {}'.format(device_name)

def main(argv):
    try:
        options, args = getopt.getopt(argv, "i:m:p:t:", ["initialize", "modeling=", "predict=", "test="])
    except getopt.GetoptError:
        sys.exit()

    for name, value in options:
        if name in ("-i", "--initialize"):
            initialize(value)
        if name in ("-m", "--modeling"):
            # modeling(value)
            train(value)
        if name in ("-p", "--predict"):
            # predict(value)
            prediction(value)
        if name in ("-t", "--test"):
            test_gpu()

if __name__ == '__main__':
    main(sys.argv[1:])
    sys.exit()

################################################################################