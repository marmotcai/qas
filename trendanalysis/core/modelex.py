# -*-coding:utf-8 -*-

import arrow
import numpy as np
import pandas as pd

import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, LSTM
from keras.utils import plot_model
import pydot_ng as pydot

print(pydot.find_graphviz())

from trendanalysis.utils.tools import Timer
from trendanalysis.core import data_manager as my_dm

import trendanalysis as ta
from trendanalysis.vendor import ztools as zt
from trendanalysis.utils import tools as my_tools
from trendanalysis.core import evaluation as eva
from trendanalysis.vendor import zai_keras as zks

class DataPrepared():
    def __init__(self):
        pd.set_option('display.max_rows', 10)
        pd.set_option('display.width', 450)
        pd.set_option('display.float_format', zt.xfloat3)

    def prepared(df, featureslist):
        x_columns = []
        df = df.sort_values('date')  # 日期排序

        for features in featureslist:
            if features == 'ohlcv':
                for i in ta.g.config["data"]["ohlcv"]:
                    x_columns.append(i)

            if features == 'avg':
                df, columns = DataPrepared.prepared_avg(df)  # 填充均值
                for i in columns: x_columns.append(i)

            if features == 'ma':
                df, columns = DataPrepared.prepared_ma(df)  # 填充MA均线
                for i in columns: x_columns.append(i)

            if features == 'pre_next':
                df, columns = DataPrepared.prepared_pre_next(df)  # 前一天收盘价和后一天的开盘价
                for i in columns: x_columns.append(i)

            if features == 'next_profit':
                df, columns = DataPrepared.prepared_next_profit(df, 5, 2, 5)  # 计算第5天到第10天的收益值，从第5天开始，计算2次，步长为5天
                for i in columns: x_columns.append(i)

            if features == 'next_rate':
                df, columns = DataPrepared.prepared_next_rate(df, 5, 2, 5)  # 计算第5天和第10天的收益率，从第5天开始，计算2次，步长为5天
                for i in columns: x_columns.append(i)

            if features == 'amp':
                df, columns = DataPrepared.prepared_amp(df)  # 填充最大振幅
                for i in columns: x_columns.append(i)

        df = DataPrepared.prepared_other(df)  # 填充其它swi
        return df, x_columns

    def get_onehot(df, k):
        return pd.get_dummies(df[k]).values

    def get_features(df, features_lst):
        return df[features_lst].values

    # 填充前一天和后一天的值
    def prepared_pre_next(df):
        columns = []
        df['next_open'] = df['open'].shift(-1)  # 后一天的开盘价
        columns.append('next_open')
        df['pre_close'] = df['close'].shift(1)  # 前一天收盘价
        columns.append('pre_close')
        return df, columns

    # 计算均值
    def prepared_avg(df):
        columns = []

        ohlc = ta.g.config['data']['ohlc']

        df['ohlc_avg'] = df[ohlc].mean(axis=1).round(2)  # 当天OHLC均值
        columns.append('ohlc_avg')
        df['ohlc10_max'] = df['ohlc_avg'].rolling(10).max()
        columns.append('ohlc10_max')
        df['ohlc10_min'] = df['ohlc_avg'].rolling(10).min()
        columns.append('ohlc10_min')
        df['ohlc10_avg'] = df['ohlc_avg'].rolling(10).mean()
        columns.append('ohlc10_avg')
        df['next_ohlc_avg'] = df['ohlc_avg'].shift(-1)
        columns.append('next_ohlc_avg')

        for i in range(-5, 5):
            ksgn = 'ohlc_avg_' + str(i)
            columns.append(ksgn)
            df[ksgn] = df['ohlc_avg'].shift(-i)

        df = DataPrepared.prepared_clean(df)  # 删除有空值的行

        return df, columns

    def ma(df, n, ksgn='close'):
        xnam = 'ma_{n}'.format(n=n)
        ds2 = pd.Series(df[ksgn], name=xnam, index=df.index);
        ds5 = ds2.rolling(center=False, window=n).mean()
        df = df.join(ds5)
        return df, xnam

    # 计算MA均线
    def prepared_ma(df):
        columns = []
        vlst = ta.g.config["data"]["ma100"]
        for xd in vlst:
            df, xnam = DataPrepared.ma(df, xd, 'ohlc_avg')
            columns.append(xnam)
        return df, columns

    # 计算振幅
    def prepared_amp(df):
        columns = []

        df['price_range'] = df['high'].sub(df['low'])  # 当天振幅
        columns.append('price_range')
        df['next_price_range'] = df['price_range'].shift(-1)
        columns.append('next_price_range')

        df['amp'] = df['price_range'].div(df['pre_close'])  # 当天振幅
        columns.append('amp')
        df['amp_type'] = df['amp'].apply(zt.iff3type, d0=0.03, d9=0.05, v3=3, v2=2, v1=1)  # 振幅分类器
        columns.append('amp_type')
        df['next_amp'] = df['amp'].shift(-1)
        columns.append('next_amp')
        df['next_amp_type'] = df['amp_type'].shift(-1)
        columns.append('next_amp_type')

        return df, columns

    def prepared_next_profit(df, num=5, count=2, step=5):
        columns = []
        for i in range(count):
            j = num + i * step
            ksgn = 'next_profit_' + str(j)
            df[ksgn] = df['close'].shift(-1 * (j)).sub(df['close'])
            columns.append(ksgn)
        return df, columns

    def prepared_next_rate(df, num=5, count=2, step=5):
        columns = []
        for i in range(count):
            j = num + i * step

            ksgn_profit = 'next_profit_' + str(j)
            ksgn_rate = 'next_rate_' + str(j)

            df[ksgn_profit] = df['close'].shift(-1 * (j)).sub(df['close'])
            df[ksgn_rate] = df[ksgn_profit].div(df['close'])

            columns.append(ksgn_profit)
            columns.append(ksgn_rate)

        df['next_rate_10_type'] = df['next_rate_10'].apply(zt.iff3type, d0=0, d9=0.05, v3=3, v2=2, v1=1)  # 振幅分类器
        columns.append('next_rate_10_type')

        return df, columns

    # 其它处理
    def prepared_other(df):
        return df

    def prepared_clean(df, type='dropna'):
        # 清除NaN值
        if type == 'dropna':
            df = df.dropna(axis=0, how='any', inplace=False)  # 删除为空的行
        if type == 'pad':
            df = df.fillna(method='pad')
        if type == 'bfill':
            df = df.fillna(method='bfill')

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

    def __init__(self, filename, featureslist, split):
        '''
        filename:数据所在文件名， '.csv'格式文件
        split:训练与测试数据分割变量
        cols:选择data的一列或者多列进行分析，如 Close 和 Volume
        '''
        self.dataframe = pd.read_csv(filename)
        self.dataframe, self.x_columns = DataPrepared.prepared(self.dataframe, featureslist)

        i_split = int(len(self.dataframe) * split)
        self.data_train = self.dataframe.get(self.x_columns)[:i_split]  # 选择指定的列 进行分割 得到 未处理的训练数据
        self.data_test = self.dataframe.get(self.x_columns)[i_split:]
        self.len_train = len(self.data_train)
        self.len_test = len(self.data_test)
        self.len_train_windows = None

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

    def save_model(self, filename):
        if len(filename) <= 0:
            return False
        my_tools.check_path_exists(filename)
        return self.model.save(filename)

    def build_model(self, configs, num_in = 10, num_out = 1):
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

            if layer['type'] == 'lstm':

                if isinstance(neurons, str) and neurons[0] == "x":
                    neurons_in = num_in * int(neurons.replace("x", ""))
                else:
                    neurons_in = neurons

                if isinstance(input_timesteps, str) and input_timesteps[0] == "x":
                    input_timesteps_in = num_in * int(input_timesteps.replace("x", ""))
                else:
                    input_timesteps_in = input_timesteps

                if isinstance(input_dim, str) and input_dim[0] == "x":
                    input_dim_in = num_in * int(input_dim.replace("x", ""))
                else:
                    input_dim_in = input_dim

                self.model.add(LSTM(neurons_in, input_shape=(input_timesteps_in, input_dim_in), return_sequences=return_seq))
            if layer['type'] == 'dropout':
                self.model.add(Dropout(dropout_rate))
            if layer['type'] == 'dense':
                if isinstance(neurons, str) and neurons[0] == "x":
                    neurons_out = num_out * int(neurons.replace("x", ""))
                else:
                    neurons_out = neurons
                self.model.add(Dense(neurons_out, activation=activation))

        self.model.compile(loss=configs['model']['loss'], optimizer=configs['model']['optimizer'],
                           metrics=configs['model']['acc'])

        print('[Model] Model Compiled')
        timer.stop()  # 输出构建一个模型耗时


# TODO 训练入口
def training(code, datafile, modelfile):
    evaluation = ta.g.config['data']['evaluation']  # 是否评估
    split = ta.g.config['data']['train_test_split']
    if not evaluation:
        split = 1  # 若不评估模型准确度，则将全部历史数据用于训练

    # 从本地加载训练和测试数据
    data = DataLoaderEx(datafile, ta.g.config['model']['xfeatures'], split)


    x_train = my_dm.util.get_features(data.data_train, data.x_columns)
    x_test = my_dm.util.get_features(data.data_test, data.x_columns)

    y_train = my_dm.util.prepared_y(data.data_train, 'next_rate_10_type', 'onehot')
    y_test = my_dm.util.prepared_y(data.data_test, 'next_rate_10_type', 'onehot')

    y_lst = y_train[0]
    x_lst = x_train[0]

    num_in, num_out = len(x_lst), len(y_lst)

    print('\n self.df_test.tail()', data.data_test.tail())
    print('\n self.x_train.shape,', x_train.shape)
    print('\n type(self.x_train),', type(x_train))

    rxn = x_train.shape[0]
    x_train = x_train.reshape(rxn, num_in, -1)
    txn = x_test.shape[0]
    x_test = x_test.reshape(txn, num_in, -1)

    print('\n x_train.shape,', x_train.shape)
    print('\n type(x_train),', type(x_train))

    print('\n num_in, num_out:', num_in, num_out)

    # TODO 开始建模
    m = ModelEx()
    m.build_model(ta.g.config, num_in, num_out)  # 根据配置文件新建模型
    # m.model = zks.lstm020typ(num_in, num_out)

    m.model.summary()
    plot_model(m.model, to_file=ta.g.log_path + 'model.png')

    print('\n#4 模型训练 fit')
    tbCallBack = keras.callbacks.TensorBoard(log_dir=ta.g.log_path, write_graph=True, write_images=True)
    tn0 = arrow.now()
    m.model.fit(x_train, y_train, epochs=500, batch_size=512, callbacks=[tbCallBack])
    tn = zt.timNSec('', tn0, True)

    m.save_model(modelfile)

    eva_obj = eva.evaluation(data)
    eva_obj.predict(m.model, data.data_test, x_test)
