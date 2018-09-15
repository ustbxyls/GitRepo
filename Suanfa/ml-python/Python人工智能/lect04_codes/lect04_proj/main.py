# -*- coding: utf-8 -*-

"""
    作者:     Robin
    版本:     1.0
    日期:     2018/03
    文件名:    main.py
    功能：     主程序

    实战案例3-1：手机价格预测(1)
    任务：使用scikit-learn建立不同的机器学习模型进行手机价格等级预测

    数据集来源： https://www.kaggle.com/vikramb/mobile-price-eda-prediction

    声明：小象学院拥有完全知识产权的权利；只限于善意学习者在本课程使用，
         不得在课程范围外向任何第三方散播。任何其他人或机构不得盗版、复制、仿造其中的创意，
         我们将保留一切通过法律手段追究违反者的权利
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import config


# 解决matplotlib显示中文问题
# 仅适用于Windows
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题


def inspect_dataset(train_data, test_data):
    """
        查看数据集
    """
    print('\n===================== 数据查看 =====================')
    print('训练集有{}条记录。'.format(len(train_data)))
    print('测试集有{}条记录。'.format(len(test_data)))

    # 可视化各类别的数量统计图
    plt.figure(figsize=(10, 5))

    # 训练集
    ax1 = plt.subplot(1, 2, 1)
    sns.countplot(x='price_range', data=train_data)

    plt.title('训练集')
    plt.xticks(rotation='vertical')
    plt.xlabel('价格等级')
    plt.ylabel('数量')

    # 测试集
    plt.subplot(1, 2, 2, sharey=ax1)
    sns.countplot(x='price_range', data=test_data)

    plt.title('测试集')
    plt.xticks(rotation='vertical')
    plt.xlabel('价格等级')
    plt.ylabel('数量')

    plt.tight_layout()
    plt.show()


def train_model(X_train, y_train, X_test, y_test, param_range, model_name='SVM'):
    """
        model_name: 默认为svm
            knn, kNN模型，对应参数为 n_neighbors
            lr, 逻辑回归模型，对应参数为 C
            svm, SVM模型，对应参数为 C
            dt, 决策树模型，对应参数为 max_dpeth

        根据给定的参数训练模型，并返回
        1. 最优模型
        2. 平均训练耗时
        3. 准确率
    """
    models = []
    scores = []
    durations = []

    for param in param_range:

        if model_name == 'kNN':
            print('训练kNN（k={}）...'.format(param), end='')
            model = KNeighborsClassifier(n_neighbors=param)
        elif model_name == 'LR':
            print('训练Logistic Regression（C={}）...'.format(param), end='')
            model = LogisticRegression(C=param)
        elif model_name == 'SVM':
            print('训练SVM（C={}）...'.format(param), end='')
            model = SVC(kernel='linear', C=param)
        elif model_name == 'DT':
            print('训练决策树（max_depth={}）...'.format(param), end='')
            model = DecisionTreeClassifier(max_depth=param)

        start = time.time()
        # 训练模型
        model.fit(X_train, y_train)

        # 计时
        end = time.time()
        duration = end - start
        print('耗时{:.4f}s'.format(duration), end=', ')

        # 验证模型
        score = model.score(X_test, y_test)
        print('准确率：{:.3f}'.format(score))

        models.append(model)
        durations.append(duration)
        scores.append(score)

    mean_duration = np.mean(durations)
    print('训练模型平均耗时{:.4f}s'.format(mean_duration))
    print()

    # 记录最优模型
    best_idx = np.argmax(scores)
    best_acc = scores[best_idx]
    best_model = models[best_idx]

    return best_model, best_acc, mean_duration


def main():
    """
        主函数
    """
    # 加载数据
    all_data = pd.read_csv(os.path.join(config.dataset_path, 'data.csv'))
    train_data, test_data = train_test_split(all_data, test_size=1/3, random_state=10)

    # 数据查看
    inspect_dataset(train_data, test_data)

    # 构建训练测试数据
    # 特征处理
    feat_names = config.feat_cols
    X_train = train_data[feat_names].values
    print('共有{}维特征。'.format(X_train.shape[1]))
    X_test = test_data[feat_names].values

    # 标签处理
    y_train = train_data[config.label_col].values
    y_test = test_data[config.label_col].values

    # 数据建模及验证
    print('\n===================== 数据建模及验证 =====================')
    model_name_param_dict = {'kNN':     [5, 10, 15],
                             'LR':      [0.01, 1, 100],
                             'SVM':     [0.01, 1, 100],
                             'DT':      [50, 100, 150]}

    # 比较结果的DataFrame
    results_df = pd.DataFrame(columns=['Accuracy (%)', 'Time (s)'],
                              index=list(model_name_param_dict.keys()))
    results_df.index.name = 'Model'
    for model_name, param_range in model_name_param_dict.items():
        _, best_acc, mean_duration = train_model(X_train, y_train, X_test, y_test,
                                                 param_range, model_name)
        results_df.loc[model_name, 'Accuracy (%)'] = best_acc * 100
        results_df.loc[model_name, 'Time (s)'] = mean_duration

    results_df.to_csv(os.path.join(config.output_path, 'model_comparison.csv'))

    # 模型及结果比较
    print('\n===================== 模型及结果比较 =====================')

    plt.figure(figsize=(10, 4))
    ax1 = plt.subplot(1, 2, 1)
    results_df.plot(y=['Accuracy (%)'], kind='bar', ylim=[60, 100], ax=ax1, title='准确率(%)', legend=False)

    ax2 = plt.subplot(1, 2, 2)
    results_df.plot(y=['Time (s)'], kind='bar', ax=ax2, title='训练耗时(s)', legend=False)
    plt.tight_layout()
    plt.savefig('./pred_results.png')
    plt.show()


if __name__ == '__main__':
    main()
