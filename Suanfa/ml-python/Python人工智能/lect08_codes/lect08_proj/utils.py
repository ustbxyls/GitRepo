# -*- coding: utf-8 -*-

"""
    作者:     Robin
    版本:     1.0
    日期:     2018/03
    文件名:    config.py
    功能：     配置文件

    声明：小象学院拥有完全知识产权的权利；只限于善意学习者在本课程使用，
         不得在课程范围外向任何第三方散播。任何其他人或机构不得盗版、复制、仿造其中的创意，
         我们将保留一切通过法律手段追究违反者的权利
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

import config


def load_data():
    """
        加载数据集，进行按天重采样，并且将时间设置为索引

        返回：
            - proc_daily_df：处理后的数据集
    """
    data_df = pd.read_csv(config.data_file)
    data_df['date'] = pd.to_datetime(data_df['Timestamp'], unit='s').dt.date
    data_df.set_index('date', inplace=True)
    data_df.index = pd.to_datetime(data_df.index)

    # 确保按时间排序
    data_df.sort_index(inplace=True)

    # 按天重采样
    daily_data_df = data_df.resample('D').mean()
    daily_data_df.dropna(inplace=True)

    # 选取使用的列
    proc_daily_df = daily_data_df[config.stats_cols + [config.raw_label_col]]

    print(proc_daily_df.head())

    return proc_daily_df


def make_stationary_seq(data_df, vis=True):
    """
        对时序数据进行差分，达到平稳

        参数：
            - data_df: 时序数据
            - vis: 是否可视化时序数据

        返回：
            - stationary_df: 平稳化的时序数据
    """
    # 差分操作
    stationary_df = data_df.diff()

    if vis:
        data_df.plot(y=config.raw_label_col, title='Raw Data')
        stationary_df.plot(y=config.raw_label_col, title='Stationary Data')
        plt.show()

    return stationary_df


def make_data_for_model(data_df):
    """
        构造训练集与测试集，并且使用t-1时刻的价格作为一个特征
        参数：
            - data_df: 时序数据
        返回：
            - train_data:   训练数据
            - test_data:    测试数据
    """
    # 使用t-1时刻的价格作为一个特征
    data_df[config.label_col] = data_df[config.raw_label_col].shift(-1)
    data_df.fillna(0, inplace=True)

    use_cols = config.stats_cols + [config.raw_label_col, config.label_col]

    # year_start_pred - 1 前（包括）的时序数据作为训练数据
    train_data = data_df.loc[:str(config.year_start_pred - 1)][use_cols]

    # # year_start_pred 后（包括）的时序数据作为测试数据
    test_data = data_df.loc[str(config.year_start_pred):][use_cols]

    print('训练数据样本个数：{}'.format(train_data.shape[0]))
    print('测试数据样本个数：{}'.format(test_data.shape[0]))

    return train_data, test_data


def scale_data(train_data, test_data):
    """
        对所有数据值做归一化到[-1, 1]

        参数：
            - train_data:   训练数据
            - test_data:    测试数据
        返回：
            - y_scaler:         标签（预测值）的scaler
            - scaled_X_train:   归一化后的训练数据特征
            - scaled_y_train:   归一化后的训练数据标签
            - scaled_X_test:    归一化后的测试数据特征
            - scaled_X_test:    归一化后的测试数据标签
    """
    # 特征上的scaler
    X_scaler = MinMaxScaler(feature_range=(-1, 1))

    # 特征列
    feat_cols = config.stats_cols + [config.raw_label_col]

    # 数据特征
    X_train = train_data[feat_cols].values
    X_test = test_data[feat_cols].values

    # 归一化后的数据特征
    scaled_X_train = X_scaler.fit_transform(X_train)
    scaled_X_test = X_scaler.transform(X_test)

    # 标签的scaler
    y_scaler = MinMaxScaler(feature_range=(-1, 1))

    # 数据标签
    y_train = train_data[config.label_col].values.reshape(-1, 1)
    y_test = test_data[config.label_col].values.reshape(-1, 1)

    # 归一化后的数据标签
    scaled_y_train = y_scaler.fit_transform(y_train)
    scaled_y_test = y_scaler.transform(y_test)

    return y_scaler, scaled_X_train, scaled_y_train, scaled_X_test, scaled_y_test


def fit_lstm(X, y):
    """
        训练LSTM模型

        参数：
            - X:    数据特征
            - y:    数据标签

        返回：
            - model:    训练好的模型
    """
    n_sample = X.shape[0]       # 样本个数
    n_feat_dim = X.shape[1]     # 特征维度

    # shape: (样本个数, time step, 特征维度)
    X = X.reshape(n_sample, config.timestep, n_feat_dim)
    # X = X.reshape(n_sample, n_feat_dim, 1)
    print(X.shape)
    # 构建模型
    model = Sequential()
    model.add(LSTM(config.nodes,
                   batch_input_shape=(config.batch_size, config.timestep, n_feat_dim),
                   stateful=True))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    print('开始训练...')

    model.fit(X, y, epochs=10, batch_size=config.batch_size, verbose=0, shuffle=False)
    '''
    for i in range(config.nb_epoch):
        print('已迭代{}次（共{}次） '.format(i + 1, config.nb_epoch))
        model.fit(X, y, epochs=1, batch_size=config.batch_size, verbose=0, shuffle=False)
        model.reset_states()
    '''
    print('训练结束...')
    # 在所有训练样本上运行一次，构建cell状态
    # why?
    model.predict(X, batch_size=config.batch_size)

    # 保存模型
    model.save(config.model_file)

    return model


def forecast_lstm(model, X):
    """
        预测

        参数：
            - model: 模型
            - X:    数据特征
        返回：
            - y_pred: 预测值
    """
    n_sample = X.shape[0]
    n_feat_dim = X.shape[1]
    X = X.reshape(n_sample, config.timestep, n_feat_dim)
    y_pred = model.predict(X, batch_size=config.batch_size)[0]
    return y_pred

