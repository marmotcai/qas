import arrow
import numpy as np

from train.vendor import ztools as zt
from train.vendor import ztools_data as zdat
from train.vendor import ztools_tq as ztq
from train.utils import params as p

class evaluation():

    def __init__(self, do):
        self.do = do # 数据对象

    def predict(self, mo, df_x, x):
        print('\n#5 模型预测 predict')
        tn0 = arrow.now()
        y_pred0 = mo.predict(x)
        tn = zt.timNSec('', tn0, True)
        y_pred = np.argmax(y_pred0, axis = 1) + 1
        #
        df_x['y_pred'] = zdat.ds4x(y_pred, df_x.index, True)
        df_x.to_csv(p.default_datapath + 'my.csv', index = False)
        print('NaN的数量:', df_x.isnull().sum().sum())

        print('\n#6 acc准确度分析')
        print('\nky0=10')

        dacc, dfx, a10 = ztq.ai_acc_xed2ext(df_x.y, df_x.y_pred, ky0 = 3, fgDebug = True)

        x1, x2 = df_x['y'].value_counts(), df_x['y_pred'].value_counts()
        zt.prx('x1', x1);
        zt.prx('x2', x2)

        return df_x
