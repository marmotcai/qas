import os

import tushare as ts
import numpy as np
import pandas as pd

ohlcLst = ['open', 'high', 'low', 'close']
ohlcVLst = ohlcLst+['volume']
ohlcDVLst = ['date'] + ohlcLst + ['volume']



def df_xappend(df, df0, ksgn, num_round=3, vlst=ohlcDVLst):
    if (len(df0) > 0):
        df2 = df0.append(df)
        df2 = df2.sort_values([ksgn], ascending=True)
        df2.drop_duplicates(subset=ksgn, keep='last', inplace=True)
        # xd2.index = pd.to_datetime(xd2.index); xd = xd2
        df = df2
    #
    df = df.sort_values([ksgn], ascending=False)
    df = np.round(df, num_round)
    df2 = df[vlst]
    #
    return df2


def df_rdcsv_tim0(fss, ksgn, tim0):
    xd0 = pd.read_csv(fss, index_col=False, encoding='gbk')
    # print('\nxd0\n', xd0.head())
    if (len(xd0) > 0):
        # xd0 = xd0.sort_index(ascending = False);
        # xd0 = xd0.sort_values(['date'],ascending = False);
        xd0 = xd0.sort_values([ksgn], ascending=True)
        # print('\nxd0\n', xd0)
        xc = xd0.index[-1]
        _xt = xd0[ksgn][xc]  # xc = xd0.index[-1]
        s2 = str(_xt)
        # print('\nxc,', xc, _xt, 's2,', s2)
        if s2 != 'nan':
            tim0 = s2.split(" ")[0]
    #
    return xd0, tim0

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
        xdk = ts.get_k_data(xcod, start=tim0, end=tim1, ktype=xtyp, index=xinx)
        xdk, pre_next_col = prepared_pre_next(xdk)
        xdk, avg_col = prepared_avg(xdk)
        xdk, avgx_col = prepared_avgx(xdk)
        xdk, ma_col = prepared_ma(xdk)
        xdk, amp_col = prepared_amp(xdk)
        xdk = prepared_clean(xdk, 'dropna')
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


def download(dir, code, debug=True):
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
    X = df.loc[:, [x for x in df.columns.tolist() if x != 'NDX']].as_matrix()
    y = np.array(df.NDX)

    return X, y
