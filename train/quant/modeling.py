import sys, os
import getopt
import arrow
import keras
from keras.utils import plot_model
from keras.models import load_model

from train.vendor import ztools as zt
from train.vendor import zai_keras as zks
from train.quant import dataobject as my_do
from train.quant import evaluation as eva
from train.utils import params as my_params
from train.utils import tools as my_tools

################################################################################

class model():
    def __init__(self, type, datafile):
        self.do = None

        self.type = my_params.default_model_type
        if len(type) > 0:
            self.type = type

        if len(datafile) > 0:
            self.setdata(datafile)

    def setdata(self, filename = ""):
        self.do = my_do.train_data(filename)

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

        self.do.prepared(self.type) # 数据预处理

        # 分离训练和测试数据
        self.df_train, self.df_test = my_do.util.split(self.do.df, 0.6)

        # 构建训练特征数据
        other_features_lst = my_params.ohlc_lst + my_params.volume_lst + my_params.profit_lst # + xagv_lst + ma100_lst + other_lst
        self.x_train = my_do.util.get_features(self.df_train, other_features_lst)
        self.x_test = my_do.util.get_features(self.df_test, other_features_lst)

        #############################################################################################################

        # 构建特征，也就是结果值Y
        self.y_train = my_do.util.prepared_y(self.df_train, 'next_rate_10_type')
        self.y_test = my_do.util.prepared_y(self.df_test, 'next_rate_10_type')

        y_lst = self.y_train[0]
        x_lst = other_features_lst

        num_in, num_out = len(x_lst), len(y_lst)

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
        plot_model(mx, to_file = my_params.default_logpath + 'model.png')

        print('\n#4 模型训练 fit')
        tbCallBack = keras.callbacks.TensorBoard(log_dir = my_params.default_logpath, write_graph = True, write_images=True)
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

        x_test = my_do.util.get_features(self.do.df, my_params.ohlc_lst + my_params.volume_lst + my_params.profit_lst)
        eva_obj.predict(model, self.do.df, x_test)

    def predict(self, mod_filename):
        if not my_tools.path_exists(mod_filename):
            return

        if (self.do == None):
            return

        self.do.prepared(self.type)  # 数据预处理

        other_features_lst = my_params.ohlc_lst + my_params.volume_lst + my_params.profit_lst # + xagv_lst + ma100_lst + other_lst
        x_df = my_do.util.get_features(self.do.df.tail(5), other_features_lst)

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

def initialize(type = "all"):
    if type == "all":
        my_do.download_all()

def prepared(params):
    type = "rate"
    code = ""
    datafile = ""
    modfile = ""

    param_lst = []
    if "," in params:
        param_lst = params.split(",")
    if "|" in params:
        param_lst = params.split("|")

    def get_param(param):
        type = "rate"
        code = ""
        datafile = ""
        modfile = ""
        if 'type' == param[0]:
            type = param[1]
        if 'code' == param[0]:
            code = param[1]
        if 'data' == param[0]:
            datafile = param[1]
        if 'mod' == param[0]:
            modfile = param[1]

        return type, code, datafile, modfile

    if len(param_lst) < 1:
        param = my_tools.params_split(params)
        type, code, datafile, modfile = get_param(param)
    else:
        for j in range(0, len(param_lst)):
            param = my_tools.params_split(param_lst[j])
            type, code, datafile, modfile = get_param(param)

    if len(code) > 0 and len(datafile) <= 0: # 有代码没数据文件则先下载
        _, datafile = my_do.download_from_code(code, '2007-01-01')

    if len(datafile) > 0:
        if not my_tools.path_exists(datafile):
            datafile = os.path.join(my_params.g.config.stk_path, datafile)
    if not my_tools.path_exists(datafile):
        my_params.g.log.error("can't find data file: " + datafile)
        return

    if len(code) <= 0 and len(datafile) > 0:
        code, _ = my_tools.get_code_from_filename(datafile)
    if len(modfile) <= 0 and len(code) > 0:
        modfile = my_params.g.config.mod_path + code + ".h5"

    return type, code, datafile, modfile

def modeling(params):
    type, code, datafile, modfile = prepared(params)

    mo = model(type, datafile)
    mo.modeling(modfile)

def predict(params):
    type, code, datafile, modfile = prepared(params)

    if not my_tools.path_exists(modfile):
        return False

    mo = model(type, datafile)
    mo.predict(modfile)

################################################################################

def main(argv):
    try:
        options, args = getopt.getopt(argv, "im:p:", ["initialize", "modeling=", "predict="])

    except getopt.GetoptError:
        sys.exit()

    for name, value in options:
        if name in ("-i", "--initialize"):
            initialize("all")
        if name in ("-m", "--modeling"):
            modeling(value)
        if name in ("-p", "--predict"):
            predict(value)

if __name__ == '__main__':
    main(sys.argv[1:])
    sys.exit()

################################################################################