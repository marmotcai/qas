import os

import tushare as ts
import numpy as np
import pandas as pd

ohlcLst = ['open', 'high', 'low', 'close']
ohlcVLst = ohlcLst+['volume']
ohlcDVLst = ['date'] + ohlcLst + ['volume']

def df_readcsv_start(filename, by, start_d):
    df = pd.read_csv(filename, index_col = False, encoding = 'gbk')
    if (len(df) > 0):
        df = df.sort_values([by], ascending = True)
        xc = df.index[-1]
        _xt = df[by][xc]
        s2 = str(_xt)
        if s2 != 'nan':
            start_d = s2.split(" ")[0]
        #
    return df, start_d

def xappend(df, df0, by, num_round = 3):
    if (len(df0) > 0):
        df2 = df0.append(df)
        df2 = df2.sort_values([by], ascending=True)
        df2.drop_duplicates(subset = by, keep = 'last', inplace = True)
        df = df2
    #
    df = df.sort_values([by], ascending = False)
    df = np.round(df, num_round)
    #
    return df

class databar():
    def __init__(self, code, start_d = '2001-01-01'):
        print("init databar ...")
        self.index_code = '000300'
        self.stock_code = code
        self.start_d = start_d
        self.end_d = None

        self.cons = ts.get_apis()

    def get_index(self, code): # 获取指数数据
        df = ts.bar(code, conn = self.cons, asset = 'INDEX', start_date = self.start_d, end_date = self.end_d)

        # df['ohlc_avg'] = df[['open', 'high', 'low', 'close']].mean(axis=1).round(2)
        # df = df.drop(['code', 'open', 'high', 'low', 'close', 'amount'], axis = 1)
        # df = df.rename(columns={'vol':'index_vol', 'p_change':'index_p_change'})
        # df['index_reserve'] = df['index_vol'].div(df['index_ohlc_avg'])
        #
        # print(df.head(10))
        return df

    def get_stock(self): # 获取股票数据
        df = ts.bar(self.stock_code, conn = self.cons, start_date = self.start_d, end_date = self.end_d, ma=[5, 10, 20], factors=['vr', 'tor'])

        df['ohlc_avg'] = df[['open', 'high', 'low', 'close']].mean(axis=1).round(2)
        df = df.drop(['code'], axis = 1)

        # df = df.drop(['open', 'high', 'low', 'close', 'amount'], axis = 1)

        # print(df)
        # df['reserve'] = df['amount'].div(df['vol'])
        #
        # print(df.head(5))
        return df

    def get(self):
        df_index = self.get_index()
        df_stock = self.get_stock()
        # df_stock = pd.merge(df_stock, df_index, left_index=True, right_index=True)
        df_stock['y'] = df_stock['close'].shift(+5)

        df_stock = df_stock.dropna(axis=0, how='any', inplace=False)  # 删除为空的行
        print(df_stock.head(20))
        return df_stock

class download():
    def __init__(self, dir):
        print("init download ...")
        self.checkdir(dir)
        self.down_dir = dir

    def mkdir(self, dir):
        dir = dir.strip() # 去除首位空格
        dir = dir.rstrip("\\") # 去除尾部 \ 符号
        # 判断结果
        if not os.path.exists(dir):
            os.makedirs(dir) # 如果不存在则创建目录

    def checkdir(self, path): # 创建目录
        if os.path.exists(path) == False:
            self.mkdir(path)

    def down_stk_inx(self, dir, code, start_d = '1994-01-01'):
        ''' 下载大盘指数数据,简版股票数据，可下载到1994年股市开市起
        【输入】
            code:指数代码
            dir,数据文件目录
            start_d,数据起始时间
        '''
        df = []
        df0 = []
        filename = dir + code + '.csv'
        if os.path.exists(filename):
            df0, start_d = df_readcsv_start(filename, 'date', start_d)

        try:
            # df = ts.get_h_data(code, start = start_d, index = True, end = None, retry_count = 5, pause = 1) # Day9
            df = ts.get_k_data(code, index = True, start = start_d, end = None)
            df = df.drop(['code'], axis = 1)
            if len(df) > 0:
                if (len(df0) > 0):
                    df = xappend(df, df0, 'date')

                df = df.sort_values(['date'], ascending = False)
                print(df.head(1))
                print("download to file : ", filename)
                df.to_csv(filename, index = False, encoding = 'gbk')
        except IOError: 
            pass #skip, error

        return df    

    def download_inx(self, file):
        df = pd.read_csv(file, encoding = 'gbk')
        n = len(df['code'])
        for i in range(n):
            dfi = df.iloc[i]
            tim0 = dfi['tim0']
            code = "%06d" %dfi['code']
            print("\n", i + 1, "/", n, "code,", code, tim0)

            self.down_stk_inx(self.down_dir, code, tim0)

            #filename = self.down_dir + code + '.csv'

            #databar_obj = databar(code)
            #df_inx = databar_obj.get_index(code)
            #df_inx.to_csv(filename, index = False, encoding = 'gbk')
        return None

    def download_stock(self, code, start_d = '2001-01-01'):
        filename = self.down_dir + code + '.csv'

        databar_obj = databar(code, start_d)
        df = databar_obj.get()
        df.to_csv(filename, index = False, encoding = 'gbk')
        return df

    def download_industry_classified(self):
        filename = self.down_dir + 'industry.csv'

        df = ts.get_industry_classified()
        df.to_csv(filename, index = False, encoding = 'gbk')
        return df

    def read_by_file(self, filename, debug = True):
        df = pd.read_csv(filename, nrows = 500 if debug else None)
        print(df.tail(5))

        y_columns = df.columns[df.columns.size - 1]
        print("y:", y_columns)

        X = df.loc[:, [x for x in df.columns.tolist() if (x != 'date') and (x != y_columns)]].as_matrix()
        y = np.array(df[y_columns])

        return X, y

###############################################################


# 填充前一天和后一天的值
def prepared_pre_next(df):
    df['next_open'] = df['open'].shift(-1)  # 后一天的开盘价
    df['pre_close'] = df['close'].shift(1)  # 前一天收盘价
    return df, ['next_open', 'pre_close']

def df_xed_nextDay(df, ksgn='avg', newSgn='xavg', nday=10):
    colx = []
    for i in range(1, nday):
        xss = newSgn + str(i)
        df[xss] = df[ksgn].shift(-i)
        colx.append(xss)
    #
    return df, colx

# 计算均值
def prepared_avg(df):
    df['ohlc_avg'] = df[ohlcLst].mean(axis=1).round(2)  # 当天OHLC均值
    df['dprice_max'] = df['ohlc_avg'].rolling(10).max()
    df['dprice_min'] = df['ohlc_avg'].rolling(10).min()
    df['dprice_avg'] = df['ohlc_avg'].rolling(10).mean()
    #
    return df, ['ohlc_avg', 'dprice_max', 'dprice_min', 'dprice_avg']

def prepared_avgx(df):
    df, colx = df_xed_nextDay(df, ksgn='ohlc_avg', newSgn='xavg', nday=10)  # 10日均值
    #
    return df, colx

def mul_talib(xfun, df, ksgn='close', vlst=[5, 10, 15, 30, 50, 100]):
    for xd in vlst:
        df = xfun(df, xd, ksgn)
        # print(df.head())
    return df

def MA(df, n, ksgn='close'):
    '''
    def MA(df, n,ksgn='close'):  
    #Moving Average  
    MA是简单平均线，也就是平常说的均线
    【输入】
        df, pd.dataframe格式数据源
        n，时间长度
        ksgn，列名，一般是：close收盘价
    【输出】    
        df, pd.dataframe格式数据源,
        增加了一栏：ma_{n}，均线数据
    '''
    xnam = 'ma_{n}'.format(n=n)
    ds2 = pd.Series(df[ksgn], name = xnam, index = df.index);
    ds5 = ds2.rolling(center = False, window = n).mean()
    # print(ds5.head()); print(df.head())
    #
    df = df.join(ds5)
    #
    return df

# 计算MA均线
def prepared_ma(df):
    ma100Lst_var = [2, 3, 5, 10, 15, 20, 25, 30, 50, 100]
    ma100Lst = ['ma_2', 'ma_3', 'ma_5', 'ma_10', 'ma_15', 'ma_20', 'ma_25', 'ma_30', 'ma_50', 'ma_100']
    # ma
    return mul_talib(MA, df, ksgn = 'ohlc_avg', vlst = ma100Lst_var), ma100Lst

def iff3type(x, d0 = 95, d9 = 105, v3 = 3, v2 = 2, v1 = 1):
    if x > d9: 
        return v3
    elif x < d0: 
        return v1
    else: 
        return v2

# 计算振幅
def prepared_amp(df):
    df['price_range'] = df['high'].sub(df['low'])  # 当天振幅
    df['amp'] = df['price_range'].div(df['pre_close'])  # 当天振幅
    df['amp_type'] = df['amp'].apply(iff3type, d0 = 0.03, d9 = 0.05, v3 = 3, v2 = 2, v1 = 1)  # 振幅分类器
    #
    return df, ['price_range', 'amp', 'amp_type']


def prepared_clean(df, type='dropna'):
    # 清除NaN值
    if type == 'dropna':
        df = df.dropna(axis=0, how='any', inplace=False)  # 删除为空的行
    else:
        df = df.fillna(method='pad')
        df = df.fillna(method='bfill')
    #
    return df


def down_stk(down_filepath, xcod, xtyp='D', tim0='1994-01-01', tim1=None, xinx=False):
    ''' 中国A股数据下载子程序
    【输入】
        xcod:股票代码
        rdat,数据文件目录
        xtyp (str)：k线数据模式，默认为D，日线
            D=日 W=周 M=月 ；5=5分钟 15=15分钟 ，30=30分钟 60=60分钟

    '''

    xd = []
    xd0 = []

    # if os.path.exists(down_filepath): # 读取已经下载的数据，避免重复下载
    #     xd0, tim0 = df_rdcsv_tim0(down_filepath, 'date', tim0)

    try:
        xdk = ts.get_k_data(xcod, start = tim0, end = tim1, ktype = xtyp, index = xinx)
        xdk = xdk.sort_values('date')  # 日期排序
        xdk, pre_next_col = prepared_pre_next(xdk)
        xdk, avg_col = prepared_avg(xdk)
        xdk, avgx_col = prepared_avgx(xdk)
        xdk, ma_col = prepared_ma(xdk)
        xdk, amp_col = prepared_amp(xdk)
        xdk = prepared_clean(xdk, 'dropna')
        print(xdk.columns)
        print(xdk.head())
        # -------------
        if len(xdk) > 0:
            xd = xdk[ohlcDVLst + pre_next_col + avg_col + avgx_col + ma_col + amp_col]
            if (len(xd0) > 0):
                xd = df_xappend(xd, xd0, 'date')
            #
            xd = xd.sort_values(['date'], ascending=False)
            xd.to_csv(down_filepath, index=False, encoding='gbk')
    except IOError:
        pass  # skip,error

    return xd, down_filepath

###############################################################

def down_data(dir, code, debug=True):
    """
    date：日期
    open：开盘价
    high：最高价
    close：收盘价
    low：最低价
    volume：成交量
    price_change：价格变动
    p_change：涨跌幅
    ma5：5日均价
    ma10：10日均价
    ma20:20日均价
    v_ma5:5日均量
    v_ma10:10日均量
    v_ma20:20日均量
    turnover:换手率
    """
    print("==> start download data : ", code)

    down_stk(dir + code + ".csv", code, tim0='2001-01-01', tim1='2017-12-01')

    df = ts.get_index()
    print(df.head())

    # df = ts.get_hist_data(code)
    # df.to_csv(dir + code + ".csv",
    # columns=["open", "high", "low", "close", "volume", "price_change", "p_change", "ma5", "v_ma5"])


def read_data(input_path, debug=True):
    """
    Read stocks data.

    Args:
        input_path (str): directory to nasdaq dataset.

    Returns:
        X (np.ndarray): features.
        y (np.ndarray): ground truth.

    """
    df = pd.read_csv(input_path, nrows=500 if debug else None)
    print(df.tail(5))
    y_columns = df.columns[df.columns.size - 1]
    print(df.tail(5), "/n", y_columns)
    X = df.loc[:, [x for x in df.columns.tolist() if (x != 'date') and (x != y_columns)]].as_matrix()
    y = np.array(df[y_columns])

    return X, y
