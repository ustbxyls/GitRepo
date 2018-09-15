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
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import numpy as np

import config


def preprocess_text(raw_text):
    """
        文本预处理操作
        参数：
            - raw_text  原始文本
        返回：
            - proc_text 处理后的文本
    """
    # 全部转换为小写
    raw_text = raw_text.lower()

    # 1. 使用正则表达式去除标点符号
    filter_pattern = re.compile('[%s]' % re.escape(string.punctuation))
    words_only = filter_pattern.sub('', raw_text)

    # 2. 分词
    raw_words = nltk.word_tokenize(words_only)

    # 3. 词形归一化
    wordnet_lematizer = WordNetLemmatizer()
    words = [wordnet_lematizer.lemmatize(raw_word) for raw_word in raw_words]

    # 4. 去除停用词
    filtered_words = [word for word in words if word not in stopwords.words('english')]

    proc_text = ' '.join(filtered_words)

    return proc_text


def prepare_data():
    """
        准备数据集
        返回：
            - train_data    训练数据
            - test_data     测试数据
    """
    if os.path.exists(config.proc_data_file):
        # 如果存在预处理的数据集，直接读取
        print('读取预处理结果...')
        all_data = pd.read_csv(config.proc_data_file)
    else:
        # 如果不存在预处理的数据集，需要预处理
        print('文本预处理...')
        all_data = pd.read_csv(config.data_file)

        # 添加标签
        all_data['label'] = all_data['text_type'].map(config.text_type_dict)

        # 添加预处理后的文本
        all_data['proc_text'] = all_data['text'].apply(preprocess_text)

        # 过滤掉空字符串
        all_data = all_data[all_data['proc_text'] != '']

        # 保存预处理结果
        all_data.to_csv(config.proc_data_file, index=False)

    train_data, test_data = train_test_split(all_data, test_size=1/4, random_state=0)

    return train_data, test_data


def inspect_dataset(train_data, test_data):
    """
        查看数据集
        参数：
            - train_data    训练数据
            - test_data     测试数据
    """
    print('\n===================== 数据查看 =====================')
    print('训练集有{}条记录。'.format(len(train_data)))
    print('测试集有{}条记录。'.format(len(test_data)))

    # 可视化各类别的数量统计图
    plt.figure(figsize=(10, 5))

    # 训练集
    ax1 = plt.subplot(1, 2, 1)
    sns.countplot(x='text_type', data=train_data)

    plt.title('Training Data')
    plt.xlabel('Type')
    plt.ylabel('Count')

    # 测试集
    plt.subplot(1, 2, 2, sharey=ax1)
    sns.countplot(x='text_type', data=test_data)

    plt.title('Test Data')
    plt.xlabel('Type')
    plt.ylabel('Count')

    plt.tight_layout()
    plt.show()


def do_feature_engineering(train_data, test_data):
    """
        特征工程获取文本特征
        参数：
            - train_data    训练样本集
            - test_data     测试样本集
        返回：
            - train_X       训练特征
            - test_X        测试特征
    """

    train_proc_text = train_data['proc_text'].values
    test_proc_text = test_data['proc_text'].values

    # TF-IDF特征提取
    tfidf_vectorizer = TfidfVectorizer()
    train_tfidf_feat = tfidf_vectorizer.fit_transform(train_proc_text).toarray()
    test_tfidf_feat = tfidf_vectorizer.transform(test_proc_text).toarray()

    # 词袋模型
    count_vectorizer = CountVectorizer()
    train_count_feat = count_vectorizer.fit_transform(train_proc_text).toarray()
    testcount_feat = count_vectorizer.transform(test_proc_text).toarray()

    # 合并特征
    train_X = np.hstack((train_tfidf_feat, train_count_feat))
    test_X = np.hstack((test_tfidf_feat, testcount_feat))

    return train_X, test_X





