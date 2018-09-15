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
import os

# 数据集路径
data_file = './data/coinbaseUSD_1-min_data_2014-12-01_to_2018-01-08.csv'

# 结果保存路径
output_path = './output'
if not os.path.exists(output_path):
    os.makedirs(output_path)

# 数据的统计值列
stats_cols = ['Open', 'High', 'Low', 'Close', 'Volume_(BTC)', 'Volume_(Currency)']

# 原始数据的标签列
raw_label_col = 'Weighted_Price'

# 用于模型训练的标签列
label_col = 'Label_Price'

# 开始预测年份
year_start_pred = 2017

# 模型存放路径
model_file = os.path.join(output_path, 'trained_lstm_model.h5')

# LSTM模型参数
timestep = 1
nodes = 7
batch_size = 1
nb_epoch = 10
