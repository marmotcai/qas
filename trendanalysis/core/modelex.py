# -*-coding:utf-8 -*-
import numpy as np
import pandas as pd
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, LSTM
from keras.utils import plot_model

from trendanalysis.utils.tools import Timer
from trendanalysis.core import data_manager as my_dm

import trendanalysis as ta
from trendanalysis.vendor import zsys
from trendanalysis.vendor import ztools as zt
from trendanalysis.vendor import zpd_talib as zta
from trendanalysis.vendor import ztools_data as zdat

class DataPrepared():
    def __init__(self):
        pd.set_option('display.max_rows', 10)
        pd.set_option('display.width', 450)
        pd.set_option('display.float_format', zt.xfloat3)

    def get_onehot(df, k):
        return pd.get_dummies(df[k]).values

    def get_features(df, features_lst):
        return df[features_lst].values

    # 填充前一天和后一天的值
    def prepared_pre_next(df):
        df['next_open'] = df['open'].shift(-1)  # 后一天的开盘价
        df['pre_close'] = df['close'].shift(1)  # 前一天收盘价
        return df

    # 计算均值
    def prepared_avg(df):
        ohlc = ta.g.config['data']['ohlc']
        df['ohlc_avg'] = df[ohlc].mean(axis=1).round(2)  # 当天OHLC均值
        df['dprice_max'] = df['ohlc_avg'].rolling(10).max()
        df['dprice_min'] = df['ohlc_avg'].rolling(10).min()
        df['dprice_avg'] = df['ohlc_avg'].rolling(10).mean()
        df = zdat.df_xed_nextDay(df, ksgn='ohlc_avg', newSgn='xavg', nday=10)  # 10日均值
        return df

    # 计算MA均线
    def prepared_ma(df):
        return zta.mul_talib(zta.MA, df, ksgn='ohlc_avg', vlst=zsys.ma100Lst_var)  # ma

    # 计算振幅
    def prepared_amp(df):
        df['price_range'] = df['high'].sub(df['low'])  # 当天振幅
        df['amp'] = df['price_range'].div(df['pre_close'])  # 当天振幅
        df['amp_type'] = df['amp'].apply(zt.iff3type, d0=0.03, d9=0.05, v3=3, v2=2, v1=1)  # 振幅分类器
        return df

    def prepared_next_profit(df, num=5, count=2, step=5):
        for i in range(count):
            j = num + i * step
            keyname_profit = 'next_profit_' + str(j)
            df[keyname_profit] = df['close'].shift(-1 * (j)).sub(df['close'])
        return df

    def prepared_next_rate(df, num=5, count=2, step=5):
        for i in range(count):
            j = num + i * step
            keyname_profit = 'next_profit_' + str(j)
            keyname_rate = 'next_rate_' + str(j)
            df[keyname_profit] = df['close'].shift(-1 * (j)).sub(df['close'])
            df[keyname_rate] = df[keyname_profit].div(df['close'])
        df['next_rate_10_type'] = df['next_rate_10'].apply(zt.iff3type, d0=0, d9=0.05, v3=3, v2=2, v1=1)  # 振幅分类器
        return df

    # 次日数据
    def prepared_next(df):
        df['next_ohlc_avg'] = df['ohlc_avg'].shift(-1)
        df['next_price_range'] = df['price_range'].shift(-1)
        df['next_amp'] = df['amp'].shift(-1)
        df['next_amp_type'] = df['amp_type'].shift(-1)
        return df

    # 其它处理
    def prepared_other(df):
        return df

    def prepared_clean(df, type='dropna'):
        # 清除NaN值
        if type == 'dropna':
            df = df.dropna(axis=0, how='any', inplace=False)  # 删除为空的行
        else:
            df = df.fillna(method='pad')
            df = df.fillna(method='bfill')
        #
        return df

    # 处理标签数据
    def prepared_y(df, y_key, type=''):
        # 处理标签
        df['y'] = df[y_key]  # 输出

        if (type == 'onehot'):
            # 分类模式， One-Hot
            return get_onehot(df, 'y')
        else:
            return df['y']

    def split(df, frac, random=True):
        # 训练数据和测试数据分割
        if not random:
            dnum_train = len(df.index)
            dnum_test = round(dnum_train * frac)
            return df.head(dnum_test), df.tail(dnum_train - dnum_test)
        else:
            return df.sample(frac=frac, replace=True), df.sample(frac=1 - frac, replace=True)

class DataLoaderEx():
    """A class for loading and transforming data for the lstm model"""

    def __init__(self, filename, split):
        '''
        filename:数据所在文件名， '.csv'格式文件
        split:训练与测试数据分割变量
        cols:选择data的一列或者多列进行分析，如 Close 和 Volume
        '''
        dataframe = pd.read_csv(filename)
        dataframe = self.model_rate(dataframe)
        self.features_lst = dataframe.columns.values

        i_split = int(len(dataframe) * split)
        self.data_train = dataframe.get(self.features_lst)[:i_split]  # 选择指定的列 进行分割 得到 未处理的训练数据
        self.data_test = dataframe.get(self.features_lst)[i_split:]
        self.len_train = len(self.data_train)
        self.len_test = len(self.data_test)
        self.len_train_windows = None

    def model_rate(self, df):
        df = df.sort_values('date')  # 日期排序
        df = DataPrepared.prepared_pre_next(df)  # 前一天收盘价和后一天的开盘价
        df = DataPrepared.prepared_next_profit(df, 1, 10, 1)  # 计算第1天到第10天的收益值，从第1天开始，计算10次，步长为1天\
        df = DataPrepared.prepared_next_rate(df, 5, 2, 5)  # 计算第5天和第10天的收益率，从第5天开始，计算2次，步长为5天
        df = DataPrepared.prepared_avg(df)  # 填充均值
        df = DataPrepared.prepared_ma(df)  # 填充MA均线
        df = DataPrepared.prepared_amp(df)  # 填充最大振幅
        df = DataPrepared.prepared_next(df)  # 填充次日数据
        df = DataPrepared.prepared_other(df)  # 填充其它swi
        return df

    def get_test_data(self, seq_len, normalise):
        '''
        Create x, y test data windows
        Warning: batch method, not generative, make sure you have enough memory to
        load data, otherwise reduce size of the training split.
        '''
        data_windows = []
        for i in range(self.len_test - seq_len):
            data_windows.append(self.data_test[i:i + seq_len])  # 每一个元素是长度为seq_len的 list即一个window

        data_windows = np.array(data_windows).astype(float)
        data_windows = self.normalise_windows(data_windows, single_window=False) if normalise else data_windows

        x = data_windows[:, :-1]  # 获取每个数据窗口的前除最后一组数据的其它全部数据
        y = data_windows[:, -1, [0]]  # 获取每个数据窗口最后一组数据其中的开盘价

        return x, y

    def get_train_data(self, seq_len, normalise):
        '''
        Create x, y train data windows
        Warning: batch method, not generative, make sure you have enough memory to
        load data, otherwise use generate_training_window() method.
        '''
        data_x = []
        data_y = []
        for i in range(self.len_train - seq_len):
            x, y = self._next_window(i, seq_len, normalise)
            data_x.append(x)
            data_y.append(y)

        return np.array(data_x), np.array(data_y)

    def generate_train_batch(self, seq_len, batch_size, normalise):
        '''Yield a generator of training data from filename on given list of cols split for train/test'''
        i = 0
        while i < (self.len_train - seq_len):
            x_batch = []
            y_batch = []
            for b in range(batch_size):
                if i >= (self.len_train - seq_len):
                    # stop-condition for a smaller final batch if data doesn't divide evenly
                    yield np.array(x_batch), np.array(y_batch)
                x, y = self._next_window(i, seq_len, normalise)
                x_batch.append(x)
                y_batch.append(y)
                i += 1
            yield np.array(x_batch), np.array(y_batch)

    def _next_window(self, i, seq_len, normalise):
        '''Generates the next data window from the given index location i'''
        window = self.data_train[i: i + seq_len]
        window = self.normalise_windows(window, single_window=True)[0] if normalise else window
        x = window[: -1]
        y = window[-1, [0]]  # 最后一行的 0个元素 组成array类型，若是[0,2]则取第0个和第2个元素组成array，[-1, 0]：则是取最后一行第0个元素，
        # 只返回该元素的值[]和()用于索引都是切片操作，所以这里的y即label是 第一列Close列
        return x, y

    def normalise_windows(self, window_data, single_window=False):
        '''Normalise window with a base value of zero'''
        normalised_data = []
        window_data = [window_data] if single_window else window_data
        for window in window_data:
            normalised_window = []
            for col_i in range(window.shape[1]):
                normalised_col = [((float(p) / float((window[0, col_i]) + 0.001)) - 1) for p in window[:, col_i]]
                normalised_window.append(normalised_col)
            normalised_window = np.array(
                normalised_window).T  # reshape and transpose array back into original multidimensional format
            normalised_data.append(normalised_window)
        return np.array(normalised_data)

class ModelEx():

    def __init__(self):
        self.model = Sequential()

    def load_model(self, filepath):
        '''
        从本地保存的模型参数来加载模型
        filepath: .h5 格式文件
        '''
        print('[Model] Loading model from file %s' % filepath)
        self.model = load_model(filepath)

    def build_model(self, configs):
        """
        新建一个模型
        configs:配置文件
        """
        timer = Timer()
        timer.start()

        for layer in configs['model']['layers']:
            neurons = layer['neurons'] if 'neurons' in layer else None
            dropout_rate = layer['rate'] if 'rate' in layer else None
            activation = layer['activation'] if 'activation' in layer else None
            return_seq = layer['return_seq'] if 'return_seq' in layer else None
            input_timesteps = layer['input_timesteps'] if 'input_timesteps' in layer else None
            input_dim = layer['input_dim'] if 'input_dim' in layer else None

            if layer['type'] == 'dense':
                self.model.add(Dense(neurons, activation=activation))
            if layer['type'] == 'lstm':
                self.model.add(LSTM(neurons, input_shape=(input_timesteps, input_dim), return_sequences=return_seq))
            if layer['type'] == 'dropout':
                self.model.add(Dropout(dropout_rate))

        self.model.compile(loss=configs['model']['loss'], optimizer=configs['model']['optimizer'])

        print('[Model] Model Compiled')
        timer.stop()  # 输出构建一个模型耗时

        self.model.summary()
        plot_model(self.model, to_file=ta.g.log_path + 'model.png')


def training(code, datafile):
    m = ModelEx()
    m.build_model(ta.g.config)  # 根据配置文件新建模型

    evaluation = ta.g.config['data']['evaluation']
    split = ta.g.config['data']['train_test_split']
    if not evaluation:
        split = 1  # 若不评估模型准确度，则将全部历史数据用于训练

    # features_lst = ta.g.config['data']['ohlcv'] + ta.g.config['data']['profit']
    # 从本地加载训练和测试数据
    data = DataLoaderEx(datafile, split)
    x_train = my_dm.util.get_features(data.data_train, data.features_lst)
    x_test = my_dm.util.get_features(data.data_test, data.features_lst)

    y_train = my_dm.util.prepared_y(data.data_train, 'next_rate_10_type', 'onehot')
    y_test = my_dm.util.prepared_y(data.data_test, 'next_rate_10_type', 'onehot')

    y_lst = y_train[0]
    x_lst = data.features_lst

    num_in, num_out = len(x_lst), len(y_lst)

    print('\n self.df_test.tail()', data.data_test.tail())
    print('\n self.x_train.shape,', x_train.shape)
    print('\n type(self.x_train),', type(x_train))

    rxn, txn = x_train.shape[0], x_test.shape[0]
    x_train, x_test = x_train.reshape(rxn, num_in, -1), x_test.reshape(txn, num_in, -1)
    print('\n x_train.shape,', x_train.shape)
    print('\n type(x_train),', type(x_train))

    print('\n num_in, num_out:', num_in, num_out)
