# -*- coding: utf-8 -*-

"""
    作者:     Robin
    版本:     1.0
    日期:     2018/02
    文件名:    main.py
    功能：     主程序

    实战案例1-2：中国五大城市PM2.5数据分析 (2)
    任务：
        - 统计每个城市每天的平均PM2.5的数值
        - 基于天数对比中国环保部和美国驻华大使馆统计的污染状态

    数据集来源：https://www.kaggle.com/uciml/pm25-data-for-five-chinese-cities

    声明：小象学院拥有完全知识产权的权利；只限于善意学习者在本课程使用，
         不得在课程范围外向任何第三方散播。任何其他人或机构不得盗版、复制、仿造其中的创意，
         我们将保留一切通过法律手段追究违反者的权利
"""

import os
import pandas as pd
import numpy as np

import config


def get_china_us_pm_df(data_df, suburb_cols):
    """
        处理获取中国与美国统计的PM数据
        参数：
            - data_df:      包含城市PM值的DataFrame
            - suburb_cols:  城市对应区的列名
        返回：
            - proc_data_df:   处理后的DataFrame
    """
    pm_suburb_cols = ['PM_' + col for col in suburb_cols]

    # 取PM的均值为中国环保部在该城市的测量值
    data_df['PM_China'] = data_df[pm_suburb_cols].mean(axis=1)

    # 取出有用的列构建新的DataFrame
    proc_data_df = data_df[config.common_cols + ['city', 'PM_China']]

    # 数据预览
    print('处理后的数据预览：')
    print(proc_data_df.head())

    return proc_data_df


def preprocess_data(data_df, city_name):
    """
        预处理数据集
        参数：
            - data_df:      数据DataFrame
            - city_name:    城市名
        返回：
            - cln_data_df:  预处理后的数据集
    """
    # 数据清洗，去掉存在空值的行
    cln_data_df = data_df.dropna()

    # 重新构建索引
    cln_data_df = cln_data_df.reset_index(drop=True)

    # 添加新的一列作为城市名
    cln_data_df['city'] = city_name

    # 输出信息
    print('{}共有{}行数据，其中有效数据为{}行'.format(city_name, data_df.shape[0], cln_data_df.shape[0]))
    print('{}的前10行有效数据：'.format(city_name))
    print(cln_data_df.head())

    return cln_data_df


def add_date_col_to_df(data_df):
    """
        预处理数据集
        参数：
            - data_df:  数据DataFrame
        返回：
            - proc_data_df:  处理后的数据集
    """
    proc_data_df = data_df.copy()
    # 将'year', 'month', 'day'合并成字符串列'date'
    # 转换数据类型
    proc_data_df[['year', 'month', 'day']] = proc_data_df[['year', 'month', 'day']].astype('str')
    # 合并列
    proc_data_df['date'] = proc_data_df['year'].str.cat([proc_data_df['month'], proc_data_df['day']], sep='-')
    # 去除列
    proc_data_df = proc_data_df.drop(['year', 'month', 'day'], axis=1)
    # 调整列的顺序
    proc_data_df = proc_data_df[['date', 'city', 'PM_China', 'PM_US Post']]

    return proc_data_df


def add_polluted_state_col_to_df(day_stats):
    """
        根据每天的PM值，添加相关的污染状态
        参数：
            - day_stats:  数据DataFrame
        返回：
            - proc_day_stats: 处理后的数据集
    """
    proc_day_stats = day_stats.copy()
    bins = [-np.inf, 35, 75, 150, np.inf]
    state_lablels = ['good', 'light', 'medium', 'heavy']

    proc_day_stats['Polluted State CH'] = pd.cut(proc_day_stats['PM_China'], bins=bins, labels=state_lablels)
    proc_day_stats['Polluted State US'] = pd.cut(proc_day_stats['PM_US Post'], bins=bins, labels=state_lablels)

    return proc_day_stats


def compare_state_by_day(day_stats):
    """
        基于天数对比中国环保部和美国驻华大使馆统计的污染状态
    """
    city_names = config.data_config_dict.keys()
    city_comparison_list = []
    for city_name in city_names:
        # 找出city_name的相关数据
        city_df = day_stats[day_stats['city'] == city_name]
        # 统计类别个数
        city_polluted_days_count_ch = pd.value_counts(city_df['Polluted State CH']).to_frame(name=city_name + '_CH')
        city_polluted_days_count_us = pd.value_counts(city_df['Polluted State US']).to_frame(name=city_name + '_US')

        city_comparison_list.append(city_polluted_days_count_ch)
        city_comparison_list.append(city_polluted_days_count_us)

    # 横向组合DataFrame
    comparison_result = pd.concat(city_comparison_list, axis=1)
    return comparison_result


def main():
    """
        主函数
    """
    city_data_list = []

    for city_name, (filename, suburb_cols) in config.data_config_dict.items():
        # === Step 1. 数据获取 ===
        data_file = os.path.join(config.dataset_path, filename)
        usecols = config.common_cols + ['PM_' + col for col in suburb_cols]
        # 读入数据
        data_df = pd.read_csv(data_file, usecols=usecols)

        # === Step 2. 数据处理 ===
        # 数据预处理
        cln_data_df = preprocess_data(data_df, city_name)

        # 处理获取中国与美国统计的PM数据
        proc_data_df = get_china_us_pm_df(cln_data_df, suburb_cols)
        city_data_list.append(proc_data_df)

        print()

    # 合并5个城市的处理后的数据
    all_data_df = pd.concat(city_data_list)

    # 将'year', 'month', 'day'合并成字符串列'date'
    all_data_df = add_date_col_to_df(all_data_df)

    # === Step 3. 数据分析 ===
    # 通过分组操作获取每个城市每天的PM均值
    # 统计每个城市每天的平均PM2.5的数值
    day_stats = all_data_df.groupby(['city', 'date'])[['PM_China', 'PM_US Post']].mean()
    # 分组操作后day_stats的索引为层级索引['city', 'date']，
    # 为方便后续分析，将层级索引转换为普通列
    day_stats.reset_index(inplace=True)

    # 根据每天的PM值，添加相关的污染状态
    day_stats = add_polluted_state_col_to_df(day_stats)
    # 基于天数对比中国环保部和美国驻华大使馆统计的污染状态
    comparison_result = compare_state_by_day(day_stats)

    #  === Step 4. 结果展示 ===
    all_data_df.to_csv(os.path.join(config.output_path, 'all_cities_pm.csv'), index=False)
    day_stats.to_csv(os.path.join(config.output_path, 'day_stats.csv'))
    comparison_result.to_csv(os.path.join(config.output_path, 'comparison_result.csv'))


if __name__ == '__main__':
    main()
