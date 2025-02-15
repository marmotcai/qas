# -*- coding: utf-8 -*- 
'''
TopQuant-简称TQ极宽智能量化回溯分析系统，培训课件-配套教学python程序

Top极宽量化(原zw量化)，Python量化第一品牌 
by Top极宽·量化开源团队 2017.10.1 首发

网站： www.TopQuant.vip      www.ziwang.com
QQ群: Top极宽量化1群，124134140
      Top极宽量化2群，650924099
      Top极宽量化3群，450853713
  

  
文件名:zai_ks.py
默认缩写：import zai_keras as zks
简介：Top极宽量化·keras神经网络、深度学习工具箱
 

'''
#

#import talib as ta

#

#import multiprocessing
#
#import sklearn
#from sklearn import metrics

from keras import initializers, layers
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, SimpleRNN
from keras.optimizers import RMSprop


#
# import tensorlayer as tl

#


#-------------------
#
#
#-------------------

#------misc

#------model

#-------------MLP
def mlp01():
    model = Sequential()
    model.add(Dense(1, name='mlp01',input_dim=1)) 
    #
    #model.compile('adam', 'mse', metrics=['acc'])
    #
    return model



def mlp010(num_in=10,num_out=1):
    model = Sequential()
    #
    model.add(Dense(num_in*4, input_dim=num_in, activation='relu'))
    model.add(Dense(num_out))
    #
    # mean_squared_error
    model.compile('adam', 'mse', metrics=['acc'])
    #
    return model


def mlp020(num_in=10,num_out=1):
    model = Sequential()
    #
    model.add(Dense(num_in*4, input_dim=num_in, kernel_initializer='normal', activation='relu'))
    model.add(Dense(num_out, kernel_initializer='normal'))
    #
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    #--------------
    #
    return model


#----------rnn
    
def rnn010(num_in,num_out):
    model = Sequential()
    #
    model.add(SimpleRNN(num_in*4,
                    kernel_initializer=initializers.RandomNormal(stddev=0.001),
                    recurrent_initializer=initializers.Identity(gain=1.0),
                    activation='relu',
                    input_shape=(num_in,1)
                    ))
                    
    #
    model.add(Dense(num_out,activation='softmax'))
    #
    rmsprop = RMSprop(lr=1e-6)
    model.compile(loss='categorical_crossentropy',optimizer=rmsprop,metrics=['accuracy'])
    #
    return model

#    

def rnn020(num_in,num_out):
    model = Sequential()
    #
    model.add(SimpleRNN(num_in*8, input_shape=(num_in,1)))
    model.add(Dense(num_out, activation='softmax'))
        
    #
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    #
    return model    

#----------lstm
#   lstm.num    
def lstm010(num_in,num_out=1):    
    model = Sequential()
    #
    model.add(LSTM(num_in*8, input_shape=(num_in,1)))
    model.add(layers.Dense(1))
    #
    model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])
    #
    return model



def lstm020typ(num_in=10,num_out=1):
    model = Sequential()
    #

    model.add(LSTM(num_in*8, return_sequences=True,input_shape=(num_in, 1)))
    model.add(Dropout(0.2))
    #
    model.add(LSTM(num_in*4))
    model.add(Dropout(0.2))
    #
    model.add(Dense(num_out, activation='softmax'))
    #

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    #
    return model

