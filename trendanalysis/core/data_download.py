import os
import tushare as ts
import pandas as pd
import numpy as np
import trendanalysis as ta
from trendanalysis.utils import tools as my_tools

def down_stk_base(downpath, inxfile, basefile, codefile):
    '''
    下载时基本参数数据时，有时会出现错误提升：
          timeout: timed out
          属于正常现象，是因为网络问题，等几分钟，再次运行几次
    '''
    rss = downpath
    #
    fss = rss + inxfile

    dat = ts.get_index()
    dat.to_csv(fss, index = False, encoding = 'gbk', date_format = 'str');

    # =========
    fss = rss + basefile
    print(fss);
    dat = ts.get_stock_basics();
    dat.to_csv(fss, encoding = 'gbk', date_format = 'str');

    c20 = ['code', 'name', 'industry', 'area'];
    d20 = dat.loc[:, c20]
    d20['code'] = d20.index;

    fss = rss + codefile;
    print(fss);
    d20 = d20.sort_index()  # values(['date'], ascending = False);
    d20.to_csv(fss, index = False, encoding = 'gbk', date_format = 'str');

    '''
    # sz50,上证50；hs300,沪深300；zz500，中证500
    fss = rss + 'stk_sz50.csv';
    print(fss);
    dat = ts.get_sz50s();
    if len(dat) > 3:
        dat.to_csv(fss, index = False, encoding = 'gbk', date_format = 'str');

    fss = rss + 'stk_hs300.csv';
    print(fss);
    dat = ts.get_hs300s();
    if len(dat) > 3:
        dat.to_csv(fss, index = False, encoding = 'gbk', date_format = 'str');

    fss = rss + 'stk_zz500.csv';
    print(fss);
    dat = ts.get_zz500s();
    if len(dat) > 3:
        dat.to_csv(fss, index = False, encoding = 'gbk', date_format = 'str');
    '''
def get_stkDir(xtyp):
    fgInx, xsub = False, ta.global_obj.config.stk_path
    if xtyp == 'index_cn': fgInx, xsub = True, 'inx/'
    elif xtyp == 'bond_cn': xsub = 'bond/'
    elif xtyp == 'etf_cn': xsub = 'etf/'
    elif xtyp == 'stock_cn':xsub = ta.global_obj.config.stk_path
    else: xsub = ''

    return fgInx, xsub

def df_rdcsv_tim0(fss, ksgn, tim0):
    # xd0 = pd.read_csv(fss, index_col = False, encoding = 'gbk')
    xd0 = pd.read_csv(fss, index_col = False, encoding = 'utf8')
    # print('\nxd0\n', xd0.head())
    if (len(xd0) > 0):
        # xd0 = xd0.sort_index(ascending = False);
        # xd0 = xd0.sort_values(['date'], ascending = False);
        xd0 = xd0.sort_values([ksgn], ascending = True);
        # print('\nxd0\n', xd0)
        xc = xd0.index[-1];  ###
        _xt = xd0[ksgn][xc];  # xc = xd0.index[-1];###
        s2 = str(_xt);
        # print('\nxc,', xc, _xt, 's2,', s2)
        if s2 != 'nan':
            tim0 = s2.split(" ")[0]
    return xd0, tim0

def df_xappend(df, df0, ksgn, num_round, vlst):
    if (len(df0) > 0):
        df2 = df0.append(df)
        df2 = df2.sort_values([ksgn], ascending = True);
        df2.drop_duplicates(subset = ksgn, keep = 'last', inplace = True);
        # xd2.index = pd.to_datetime(xd2.index); xd = xd2
        df = df2
    #
    df = df.sort_values([ksgn], ascending = False);
    df = np.round(df, num_round);
    df2 = df[vlst]
    #
    return df2

def down_stk_code(rdat, xcod, xtyp = 'D', fgInx = False):
    ''' 中国A股数据下载子程序
    【输入】
        xcod:股票代码
        rdat,数据文件目录
        xtyp (str)：k线数据模式，默认为D，日线
            D=日 W=周 M=月 ；5=5分钟 15=15分钟 ，30=30分钟 60=60分钟
        fgInx,指数模式，默认为：False
    '''

    tim0, fss = '1994-01-01', rdat + xcod + '.csv'
    xd0, xd = [], []
    xfg = os.path.exists(fss) and (os.path.getsize(fss) > 0)
    if xfg:
        xd0, tim0 = df_rdcsv_tim0(fss, 'date', tim0)

    print('\t', xfg, xtyp, fss, ",", tim0)
    # -----------
    try:
        xdk = ts.get_k_data(xcod, index = fgInx, start = tim0, end = None, ktype = xtyp);
        xd = xdk
        # -------------
        if len(xd) > 0:
            # print(xdk)

            xd = xdk[ta.g.config['data']['dohlcv']]
            xd = df_xappend(xd, xd0, 'date', 3, ta.g.config['data']['dohlcv'])
            #
            xd = xd.sort_values(['date'], ascending = False);
            xd.to_csv(fss, index=False, encoding = 'utf8')

    except IOError:
        pass  # skip,error

    return xd, fss

def down_stk_pool(down_path, finx, xn0 = 0, xn9 = 100, xtyp = 'D'):
    '''
    根据finx股票列表文件，下载所有，或追加日线数据
    自动去重，排序
    【输入】
        rdat,数据文件目录
        finx:股票代码文件
        xtyp (str)：k线数据模式，默认为D，日线
            D=日 W=周 M=月 ；5=5分钟 15=15分钟 ，30=30分钟 60=60分钟
    '''
    stkPool = pd.read_csv(finx, encoding = 'gbk'); print(finx);
    xn100 = len(stkPool['code']);
    xn9 = max(xn100, xn9)
    for i, rx in stkPool.iterrows():
        if xn0 <= i <= xn9:
            code = "%06d" % rx.code

            my_tools.check_path_exists(down_path)

            print("\n", i, "/", xn9, '@', code, rx['name'], ",@", down_path)

            down_stk_code(down_path, code, xtyp, fgInx = False);
