# -*- coding: utf-8 -*-

"""
"""

# !/usr/bin/env python
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

import os
from subprocess import check_output

#import missingno as msno

from pandas import datetime
from pandas import DataFrame
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

from math import sqrt
from pandas import datetime

# 是否加载已经训练好的模型
IS_LOAD_MODEL = False


# convert time series into supervised learning problem
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	targets = df.shift(0)
	drop_list = [i for i in range(n_vars -1)]
	targets.drop(targets.columns[drop_list], axis=1, inplace=True)
	cols.append(targets)
	names += ['var%d' % (n_out -1)]
	#print(names)
	#print(cols)
	# put it all together
	agg = pd.concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg


# convert time series into supervised learning problem
def series_to_supervised_1(sales, n_in=1, n_out=1, n_vars=2, test_ratio=0.2):
	columns = ['vendibility','quantity']
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	names += ['var%d' % (n_out -1)]
	#empty dataframe
	supervised_values_all_dc = pd.DataFrame(columns=names)
	supervised_values_all_dc_train = pd.DataFrame(columns=names)
	supervised_values_all_dc_test= pd.DataFrame(columns=names)
	# print(supervised_values_all_dc)
	dcs = sales['dc_id'].unique()
	for dc in dcs:
		sales_single_sku = sales[sales['dc_id'] == dc].sort_values(['dc_id','datetime'])
		supervised_values = series_to_supervised(sales_single_sku[columns].values, n_in, n_out)
		n_test = int(supervised_values.shape[0] * test_ratio)
		train, test = supervised_values[0:-n_test], supervised_values[-n_test:]
		supervised_values_all_dc = supervised_values_all_dc.append(supervised_values)
		supervised_values_all_dc_train = supervised_values_all_dc_train.append(train)
		supervised_values_all_dc_test = supervised_values_all_dc_test.append(test)
	return supervised_values_all_dc,supervised_values_all_dc_train,supervised_values_all_dc_test

# fit an LSTM network to training data
def fit_lstm(train,n_seq, n_feature, n_batch, n_epoch, n_neurons):
	# reshape training into [samples, timesteps, features]
	X, y = train[:, :-1], train[:, -1]
	print(X.shape)
	X = X.reshape(X.shape[0], n_seq, n_feature)
	# design network
	model = Sequential()
	model.add(LSTM(n_neurons, batch_input_shape=(n_batch, X.shape[1], X.shape[2]), stateful=True))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer='adam')
	# fit network
	for i in range(n_epoch):
		print("epoch:" + str(i) + "/" + str(n_epoch))
		model.fit(X, y, epochs=1, batch_size=n_batch, verbose=1, shuffle=False)
		model.reset_states()
	return model

# evaluate_lstm
def eval_lstm(model, test, n_seq, n_feature, n_batch=1):
    test_values = test.values
    test_X, test_y = test_values[:, :-1], test_values[:, -1]
    test_X = test_X.reshape((test_X.shape[0], n_seq, n_feature))

    # make a prediction
    forecasts = list()
    for i in range(len(test_X)):
        X = test_X[i]
        X = X.reshape(1, n_seq, n_feature)
        # make forecast
        forecast = forecast_lstm(model, X, n_batch)
        # store the forecast
        forecasts.append(forecast)

    # calculate RMSE
    rmse = sqrt(mean_squared_error(test_y, forecasts))
    print('Test RMSE: %.3f' % rmse)

# make one forecast with an LSTM,
def forecast_lstm(model, X, n_batch):
	# make forecast
	forecast = model.predict(X, batch_size=n_batch)
	# convert to array
	return [x for x in forecast[0, :]]


def main():
    """
        主函数
    """
    # 加载数据
    data_dir = './jdata/'
    sales = pd.read_csv(os.path.join(data_dir, 'sku_sales.csv'))
    items = pd.read_csv(os.path.join(data_dir, 'sku_info.csv'))
    attr = pd.read_csv(os.path.join(data_dir, 'sku_attr.csv'))
    promo = pd.read_csv(os.path.join(data_dir, 'sku_prom.csv'))
    quantile = pd.read_csv(os.path.join(data_dir, 'sku_quantile.csv'))
    promo_test = pd.read_csv(os.path.join(data_dir, 'sku_prom_testing_2018Jan.csv'))

    sales['datetime'] = sales['date'].map(lambda x: datetime.strptime(x, '%Y-%m-%d'))

    # LSTM 参数
    n_seq = 8
    n_feature = 2
    n_batch = 1
    n_epochs = 100
    n_neurons = 3

    # 所有的item_sku_id
    columns = ['vendibility', 'quantity']
    features = []
    result = []

    all_off_sets = np.arange(1, 32)
    # all_off_sets = np.arange(1, 3)
    all_item_sku_ids = items['item_sku_id'].unique()
    # all_item_sku_ids=np.arange(100, 102)
    for off_set in all_off_sets:
        all_train = []
        all_test = []
        # 调用方法训练，预测
        for item_sku_id in all_item_sku_ids:
            # 准备时序数据
            sales_single_sku = sales[sales['item_sku_id'] == item_sku_id].sort_values(['dc_id', 'datetime'])
            supervised_values, train, test = series_to_supervised_1(sales_single_sku, n_seq, off_set)
            if(len(all_train) == 0):
                all_train = train
                all_test = test
            else:
                all_train = all_train.append(train)
                all_test = all_test.append(test)

        print('训练： off_set={}'.format(off_set))
        model = fit_lstm(train.values, n_seq, n_feature, n_batch, n_epochs, n_neurons)
        #eval_lstm(model, test, n_seq, n_feature, n_batch)


        # 调用方法训练，预测
        for item_sku_id in all_item_sku_ids:
            # for submit
            # 获取6个不同的dc
            dcs = sales[sales['item_sku_id'] == item_sku_id]['dc_id'].unique()
            for dc in dcs:
                feature = {}
                sales_single_sku = sales[sales['item_sku_id'] == item_sku_id][sales['dc_id'] == dc].sort_values(
                    ['dc_id', 'datetime'])
                sales_single_sku_submit = sales_single_sku.tail(n_seq)
                feature['date'] = off_set
                feature['dc_id'] = dc
                feature['item_sku_id'] = item_sku_id
                if(len(sales_single_sku) == 0):
                    feature['quantity'] = 0

                if (len(sales_single_sku_submit) < n_seq and len(sales_single_sku_submit) > 0):
                    feature['quantity'] = sales_single_sku.tail(1)['quantity'].unique()[0]

                if (len(sales_single_sku_submit) == n_seq ):
                    sales_single_sku_submit_feature = sales_single_sku_submit[columns].values.reshape(1, n_seq,
                                                                                                      n_feature)
                    quantity = forecast_lstm(model, sales_single_sku_submit_feature, n_batch)[0]
                    feature['quantity'] = quantity

                features.append(feature)
            print('预测-save：item_sku_id={}, off_set={}'.format(item_sku_id, off_set))
            result = pd.DataFrame(features)

        print('预测-save：item_sku_id={}, all offset'.format(item_sku_id))
        result = pd.DataFrame(features)
        result.to_csv('./jdata_out/submit.csv', index=False)
    result = pd.DataFrame(features)
    print('预测-save：all')
    result.to_csv('./jdata_out/submit.csv', index=False)


if __name__ == '__main__':
    main()
