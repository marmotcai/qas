import sys
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
    def __init__(self, type, code, datafile, modfile):
        self.do = None

        self.type = my_params.default_model_type
        if len(type) > 0:
            self.type = type

        if len(code) > 0 and len(datafile) <= 0:
            self.code = code
            _, datafile = my_do.download_from_code(code)

        if len(datafile) > 0:
            self.setdata(datafile)
            self.modeling(self.type) # 数据预处理

        if len(modfile) > 0:
            self.setmod(modfile)

    def setdata(self, filename = ""):
        self.do = my_do.train_data(filename)

    def setmod(self, filename):
        self.mx = load_model(filename)
        return self.mx

    def modeling(self, type = 'rate'): # 建模过程
        if (self.do != None):
            self.do.prepared(type)

    def save(self, model, filename):
        if len(filename) > 0:
            model.save(filename)

    def building(self, model_filename = ""):

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
            mx.summary()
            plot_model(mx, to_file = my_params.default_logpath + 'model.png')

            print('\n#4 模型训练 fit')
            tbCallBack = keras.callbacks.TensorBoard(log_dir = my_params.default_logpath, write_graph = True, write_images=True)
            tn0 = arrow.now()
            mx.fit(self.x_train, self.y_train, epochs = 500, batch_size = 512, callbacks = [tbCallBack])
            tn = zt.timNSec('', tn0, True)

            self.save(mx, model_filename)
        else:
            mx = self.setmod(model_filename)

        eva_obj = eva.evaluation(self.do)
        eva_obj.predict(mx, self.df_test, self.x_test)

    def eva(self, mode_filename):
        if my_tools.path_exists(mode_filename):
            model = self.setmod(mode_filename)
        eva_obj = eva.evaluation(self.do)

        x_test = my_do.util.get_features(self.do.df, my_params.ohlc_lst + my_params.volume_lst + my_params.profit_lst)
        eva_obj.predict(model, self.do.df, x_test)

################################################################################

def modeling(params):
    param_lst = []
    if "," in params:
        param_lst = params.split(",")
    if "|" in params:
        param_lst = params.split("|")

    if len(param_lst) < 1:
        my_params.g.log.error("modeling params is error!")
        return

    type = "rate"
    code = ""
    datafile = ""
    modfile = ""
    for j in range(0, len(param_lst)):
        param = param_lst[j]
        if 'type' in param.lower():
            type = param.split(":")[1]
        if 'code' in param.lower():
            code = param.split(":")[1]
        if 'data' in param.lower():
            datafile = param.split(":")[1]
        if 'mod' in param.lower():
            modfile = param.split(":")[1]

    mo = model(type, code, datafile, modfile)
    mo.building()

#    if "," in params:
#        model_lst = params.split(",")
#    else:
#        model_lst.append(my_params.default_model)
#        model_lst.append(params)
#
#    mod_type = my_params.default_model
#    data_filepath = ""
#    mod_filepath = ""
#
#    for j in range(0, len(model_lst)):
#        param = model_lst[j]
#        suffix = os.path.splitext(param)[1]
#        if len(suffix) <= 0:
#            mod_type = param
#        else:
#            filepath = param
#
#            if '.csv' == suffix.lower():
#                if not my_tools.path_exists(filepath):
#                    filepath = my_params.g_config.day_path + filepath
#                if not my_tools.path_exists(filepath):
#                    print(filepath + " is not exists")
#                    return
#
#                data_filepath = filepath
#
#            if '.mod' == suffix.lower():
#                mod_filepath = filepath
#
#    mo = model()
#    if len(data_filepath) > 0:
#        mo.setdata(data_filepath)
#
#    if len(mod_filepath) <= 0:
#        mod_filepath = filepath + ".mod"
#
#    mo.modeling(mod_type)
#    mo.building(mod_filepath)

################################################################################

def main(argv):
    try:
        options, args = getopt.getopt(argv, "m:", ["modeling="])

    except getopt.GetoptError:
        sys.exit()

    for name, value in options:
        if name in ("-m", "--modeling"):
            modeling(value)

if __name__ == '__main__':
    main(sys.argv[1:])
    sys.exit()

################################################################################