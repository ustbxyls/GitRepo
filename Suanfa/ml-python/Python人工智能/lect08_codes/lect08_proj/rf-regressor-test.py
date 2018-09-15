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

from sklearn.ensemble import RandomForestRegressor

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
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    targets = df.shift(0)
    drop_list = [i for i in range(n_vars - 1)]
    targets.drop(targets.columns[drop_list], axis=1, inplace=True)
    cols.append(targets)
    names += ['var%d' % (n_out - 1)]
    # print(names)
    # print(cols)
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    # print("series_to_supervised shape:" + str(agg.shape))
    return agg


# convert time series into supervised learning problem
def series_to_supervised_1(sales, n_in=1, n_out=1, n_vars=2, test_ratio=0.2):
    columns = ['vendibility', 'quantity']
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    names += ['var%d' % (n_out - 1)]
    # empty dataframe
    supervised_values_all_dc = pd.DataFrame(columns=names)
    supervised_values_all_dc_train = pd.DataFrame(columns=names)
    supervised_values_all_dc_test = pd.DataFrame(columns=names)
    # print(supervised_values_all_dc)
    dcs = sales['dc_id'].unique()
    sku_id = sales.head(1)['item_sku_id'].unique()[0]
    item_first_cate_cd = sales.head(1)['item_first_cate_cd'].unique()[0]
    item_second_cate_cd = sales.head(1)['item_second_cate_cd'].unique()[0]
    item_third_cate_cd = sales.head(1)['item_third_cate_cd'].unique()[0]
    brand_code = sales.head(1)['brand_code'].unique()[0]

    sku_item_info_columns = ['item_first_cate_cd', 'item_second_cate_cd', 'item_third_cate_cd', 'brand_code']
    for dc in dcs:
        sales_single_sku = sales[sales['dc_id'] == dc].sort_values(['dc_id', 'datetime'])
        supervised_values = series_to_supervised(sales_single_sku[columns].values, n_in, n_out)
        # 增加dc和商品信息
        supervised_values['dc_id'] = dc
        supervised_values['item_first_cate_cd'] = item_first_cate_cd
        supervised_values['item_second_cate_cd'] = item_second_cate_cd
        supervised_values['item_third_cate_cd'] = item_third_cate_cd
        supervised_values['brand_code'] = brand_code
        n_test = int(supervised_values.shape[0] * test_ratio)
        train, test = supervised_values[0:-n_test], supervised_values[-n_test:]
        supervised_values_all_dc = supervised_values_all_dc.append(supervised_values)
        supervised_values_all_dc_train = supervised_values_all_dc_train.append(train)
        supervised_values_all_dc_test = supervised_values_all_dc_test.append(test)
    return supervised_values_all_dc, supervised_values_all_dc_train, supervised_values_all_dc_test

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

    # left join items
    sales_items = pd.merge(sales, items, how='left', on='item_sku_id')
    sales = sales_items

    n_seq = 8
    n_feature = 2
    n_batch = 1
    n_epochs = 50
    n_neurons = 3
    all_off_sets = np.arange(1, 32)
    #all_off_sets = np.arange(1, 2)
    all_item_sku_ids = items['item_sku_id'].unique()
    #all_item_sku_ids=np.arange(1, 2)
    result_train = []
    result_test = []
    print("预处理：")
    for off_set in all_off_sets:
        for item_sku_id in all_item_sku_ids:
            sales_single_sku = sales[sales['item_sku_id'] == item_sku_id].sort_values(['dc_id', 'datetime'])
            supervised_values, train, test = series_to_supervised_1(sales_single_sku, n_seq, off_set)
            if (len(result_train)) == 0:
                result_train = train
                result_test = test
            else:
                result_train = pd.concat([result_train, train])
                result_test = pd.concat([result_test, test])

    all_train = result_train.copy
    y_train = result_train.pop('var0')
    x_train = result_train
    all_test = result_test.copy
    y_test = result_test.pop('var0')
    x_test = result_test

    #训练
    clf = RandomForestRegressor(n_estimators=100, criterion='mae', verbose=1)  # 这里使用100个决策树
    # clf.fit(x_train.head(10000),y_train.head(10000))
    clf.fit(x_train, y_train)
    score = clf.score(x_test, y_test)
    result = clf.predict(x_test)
    # calculate MSE
    mse = mean_squared_error(y_test, result)
    print('Test MSE: %.3f' % mse)

    # submit
    columns = ['vendibility', 'quantity']
    features = []
    result = []
    for off_set in all_off_sets:
        for item_sku_id in all_item_sku_ids:
            dcs = sales[sales['item_sku_id'] == item_sku_id]['dc_id'].unique()
            for dc in dcs:
                feature = {}
                sales_single_sku_submit = sales[sales['item_sku_id'] == item_sku_id][sales['dc_id'] == dc].sort_values(
                    ['dc_id', 'datetime']).tail(n_seq)
                sales_single_sku_submit_feature = sales_single_sku_submit[columns].values.reshape(1, n_seq, n_feature)
                item_first_cate_cd = sales.head(1)['item_first_cate_cd'].unique()[0]
                item_second_cate_cd = sales.head(1)['item_second_cate_cd'].unique()[0]
                item_third_cate_cd = sales.head(1)['item_third_cate_cd'].unique()[0]
                brand_code = sales.head(1)['brand_code'].unique()[0]
                item_phase = np.array([brand_code, dc, item_first_cate_cd, item_second_cate_cd, item_third_cate_cd])
                sales_single_sku_submit_feature = np.append(item_phase, sales_single_sku_submit_feature)
                quantity = clf.predict(sales_single_sku_submit_feature)
                feature['date'] = off_set
                feature['dc_id'] = dc
                feature['item_sku_id'] = item_sku_id
                feature['quantity'] = quantity
                features.append(feature)


    result = pd.DataFrame(features)
    print('预测-save：all')
    result.to_csv('./jdata_out/submit_rf_20180724.csv', index=False)


if __name__ == '__main__':
    main()
