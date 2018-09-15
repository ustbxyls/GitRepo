# -*- coding: utf-8 -*-

"""
    作者:     Robin
    版本:     1.0
    日期:     2018/02
    文件名:    config.py
    功能：     配置文件

    声明：小象学院拥有完全知识产权的权利；只限于善意学习者在本课程使用，
         不得在课程范围外向任何第三方散播。任何其他人或机构不得盗版、复制、仿造其中的创意，
         我们将保留一切通过法律手段追究违反者的权利
"""
import os

# 指定数据集路径
dataset_path = './data'

# 结果保存路径
output_path = './output'
if not os.path.exists(output_path):
    os.makedirs(output_path)

# 公共列
common_cols = ['year', 'month', 'day', 'PM_US Post']

# 每个城市对应的文件名及所需分析的列名
# 以字典形式保存，如：{城市：(文件名, 列名)}
data_config_dict = {'beijing': ('BeijingPM20100101_20151231.csv',
                                ['Dongsi', 'Dongsihuan', 'Nongzhanguan']),
                    'chengdu': ('ChengduPM20100101_20151231.csv',
                                ['Caotangsi', 'Shahepu']),
                    'guangzhou': ('GuangzhouPM20100101_20151231.csv',
                                  ['City Station', '5th Middle School']),
                    'shanghai': ('ShanghaiPM20100101_20151231.csv',
                                 ['Jingan', 'Xuhui']),
                    'shenyang': ('ShenyangPM20100101_20151231.csv',
                                 ['Taiyuanjie', 'Xiaoheyan'])
                    }
