import os
import sys
import getopt

import pickle
import pandas as pd

from trendanalysis.vendor import zsys
from trendanalysis.vendor import zpd_talib as zta
from trendanalysis.vendor import ztools as zt
from trendanalysis.vendor import ztools_data as zdat
from trendanalysis.vendor import ztools_datadown as zddown

import trendanalysis as ta
from trendanalysis.utils import tools as my_tools
from trendanalysis.core import downloader as dler

################################################################################

class download():
    def __init__(self):
        ta.g.log.info("start download ...")

    def checkdir(self, path): # 创建目录
        if my_tools.path_exists(path) == False:
            my_tools.mkdir(path)

    def download_all(self, tim0 = '1994-01-01'):
        ta.g.log.info("download all data from " + tim0)

        self.checkdir(ta.g.data_path)
        dler.down_stk_base(ta.g.data_path)
        dler.down_stk_pool(ta.g.stk_path, ta.g.data_path + ta.g.config['data']['stk_base_filename'], xtyp = 'D')

    def download_code(self, downpath, code, tim0):
        filename = downpath + code + '.csv'

        self.checkdir(downpath)
        return zddown.down_stk010(filename, code, 'D', tim0);

    def download_inx(self, downpath = ta.g.data_path, filename = "inx_code.csv"):
        if my_tools.path_exists(filename) == False:
            ta.g.log.error("inx file is not exists and exit")
            return False

        self.checkdir(downpath)

        zddown.down_stk_inx(downpath, filename);
        return True

    def downlaod_stk(self, downpath = ta.g.data_path, filename = "stk_code.csv"):
        if my_tools.path_exists(filename) == False:
            ta.g.log.error("stk file is not exists and exit")
            return False

        if my_tools.path_exists(downpath) == False:
            my_tools.mkdir(downpath)

        xtyp = 'D' # xtyp = '5'
        zddown.down_stk_all(downpath, filename, xtyp)

################################################################################

def download_all(tim0 = '2007-01-01'):
    down_obj = download()
    down_obj.download_all(tim0)

def download_from_path(path):
    for root, dirs, files in os.walk(path):
        for file in files:
            suffix = os.path.splitext(file)[1]
            if '.csv' == suffix.lower():
                download_from_inxfile(os.path.join(root, file))

def download_from_inxfile(filepath):
    type = ""
    if "=" in filepath:
        type, filename = filepath.split("=")
    else:
        if "inx" in filepath:
            type = "inx"
        if "stk" in filepath:
            type = "stk"
        filename = filepath

    if len(filename) <= 0:
        ta.g.log.error("download params error!")
        return

    if not my_tools.path_exists(filename):
        filename = ta.g.data_path + filename
    if not my_tools.path_exists(filename):
        ta.g.log.error(filename + " is not exists")
        return

    down_obj = download()
    if type == "inx":
        down_obj.download_inx(ta.g.config.inx_path, filename)
    if type == "stk":
        down_obj.downlaod_stk(ta.g.stk_path, filename)

def download_from_code(code, tim0 = '2007-01-01'):
    down_obj = download()
    return down_obj.download_code(ta.g.stk_path, code, tim0)

def main(argv):
    try:
        options, args = getopt.getopt(argv, "d:", ["download="])
    except getopt.GetoptError:
        sys.exit()

    for name, value in options:
        if name in ("-d", "--download"):
            if value == "all":
                download_all()
            if my_tools.isint(value):
                download_from_code(value)
            if os.path.isdir(value):
                download_from_path(value)
            if os.path.isfile(value):
                download_from_inxfile(value)

if __name__ == '__main__':
    main(sys.argv[1:])
    sys.exit()

################################################################################

class util():
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
        df['ohlc_avg'] = df[ohlc].mean(axis = 1).round(2) # 当天OHLC均值
        df['dprice_max'] = df['ohlc_avg'].rolling(10).max()
        df['dprice_min'] = df['ohlc_avg'].rolling(10).min()
        df['dprice_avg'] = df['ohlc_avg'].rolling(10).mean()
        df = zdat.df_xed_nextDay(df, ksgn = 'ohlc_avg', newSgn = 'xavg', nday = 10) #10日均值
        return df

    # 计算MA均线
    def prepared_ma(df):
        return zta.mul_talib(zta.MA, df, ksgn = 'ohlc_avg', vlst = zsys.ma100Lst_var) # ma

    # 计算振幅
    def prepared_amp(df):
        df['price_range'] = df['high'].sub(df['low'])  # 当天振幅
        df['amp'] = df['price_range'].div(df['pre_close'])  # 当天振幅
        df['amp_type'] = df['amp'].apply(zt.iff3type, d0 = 0.03, d9 = 0.05, v3=3, v2=2, v1=1)  # 振幅分类器
        return df

    def prepared_next_profit(df, num = 5, count = 2, step = 5):
        for i in range(count):
            j = num + i * step
            keyname_profit = 'next_profit_' + str(j)
            df[keyname_profit] = df['close'].shift(-1 * (j)).sub(df['close'])
        return df

    def prepared_next_rate(df, num = 5, count = 2, step = 5):
        for i in range(count):
            j = num + i * step
            keyname_profit = 'next_profit_' + str(j)
            keyname_rate = 'next_rate_' + str(j)
            df[keyname_profit] = df['close'].shift(-1 * (j)).sub(df['close'])
            df[keyname_rate] = df[keyname_profit].div(df['close'])
        df['next_rate_10_type'] = df['next_rate_10'].apply(zt.iff3type,  d0 = 0, d9 = 0.05, v3 = 3, v2 = 2, v1 = 1)  # 振幅分类器
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

    def prepared_clean(df, type = 'dropna'):
        # 清除NaN值
        if type == 'dropna':
            df = df.dropna(axis=0, how='any', inplace=False)  # 删除为空的行
        else:
            df = df.fillna(method='pad')
            df = df.fillna(method='bfill')
        #
        return df

    # 处理标签数据
    def prepared_y(df, y_key, type = ''):
        # 处理标签
        df['y'] = df[y_key] # 输出

        if (type == 'onehot'):
            # 分类模式， One-Hot
            return util.get_onehot(df, 'y')
        else:
            return df['y']

    def split(df, frac, random = True):
        # 训练数据和测试数据分割
        if not random:
            dnum_train = len(df.index)
            dnum_test = round(dnum_train * frac)
            return df.head(dnum_test), df.tail(dnum_train - dnum_test)
        else:
            return df.sample(frac = frac, replace = True), df.sample(frac = 1 - frac, replace = True)

################################################################################

class  train_data():

    def __init__(self, data_file = ""):

        self.model_type = {'rate': self.model_rate, 'price': self.model_price, 'amp': self.model_amp}  # 定义策略执行函数

        if len(data_file) > 0:
            self.df = pd.read_csv(data_file, index_col = 0)

    def load(self, filepath):
        f = open(filepath, 'rb')
        x = pickle.load(f)
        f.close()
        return x

    def save(self, filepath, df):
        f = open(filepath, 'wb')
        pickle.dump(df, f)
        f.close()

    def training(self, df):
        return df

    # def plot(self):

    ################################################################################

    def model_rate(self, df):
        df = df.sort_values('date')  # 日期排序
        df = util.prepared_pre_next(df)  # 前一天收盘价和后一天的开盘价
        df = util.prepared_next_profit(df, 1, 10, 1)  # 计算第1天到第10天的收益值，从第1天开始，计算10次，步长为1天\
        df = util.prepared_next_rate(df, 5, 2, 5)  # 计算第5天和第10天的收益率，从第5天开始，计算2次，步长为5天
        df = util.prepared_avg(df)  # 填充均值
        df = util.prepared_ma(df)  # 填充MA均线
        df = util.prepared_amp(df)  # 填充最大振幅
        df = util.prepared_next(df)  # 填充次日数据
        df = util.prepared_other(df)  # 填充其它swi
        return df

    def model_price(self, df):
        df = df.sort_values('date')  # 日期排序
        df = util.prepared_pre_next(df)  # 前一天收盘价和后一天的开盘价
        df = util.prepared_next_profit(df, 1, 10, 1)  # 计算第1天到第10天的收益值，从第1天开始，计算10次，步长为1天\
        df = util.prepared_next_rate(df, 5, 2, 5)  # 计算第5天和第10天的收益率，从第5天开始，计算2次，步长为5天
        df = util.prepared_avg(df)  # 填充均值
        df = util.prepared_ma(df)  # 填充MA均线
        df = util.prepared_amp(df)  # 填充最大振幅
        df = util.prepared_next(df)  # 填充次日数据
        df = util.prepared_other(df)  # 填充其它swi
        return df

    def model_amp(self, df):
        df = df.sort_values('date')  # 日期排序
        return df

    ################################################################################

    def prepared(self, model_type = 'rate', action = 'modeling'):
        model_execution = self.model_type[model_type]
        df = model_execution(self. df)
        if action == 'modeling':
            df = util.prepared_clean(df) # 清理数据
        self.df = df
        print(df.tail(5))

################################################################################

