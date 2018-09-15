# -*- coding: utf-8 -*-

"""
    作者:     Robin
    版本:     1.0
    日期:     2018/03
    文件名:    main.py
    功能：     主程序

    实战案例6：垃圾短信检测
    任务：使用文本处理，特征提取及机器学习方法检测垃圾短信

    数据集来源：https://www.kaggle.com/uciml/sms-spam-collection-dataset/

    声明：小象学院拥有完全知识产权的权利；只限于善意学习者在本课程使用，
         不得在课程范围外向任何第三方散播。任何其他人或机构不得盗版、复制、仿造其中的创意，
         我们将保留一切通过法律手段追究违反者的权利
"""

import utils
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, roc_auc_score


def main():
    """
        主函数
    """
    # 准备数据集
    train_data, test_data = utils.prepare_data()

    # 查看数据集
    utils.inspect_dataset(train_data, test_data)

    # 特征工程处理
    # 构建训练测试数据
    X_train, X_test = utils.do_feature_engineering(train_data, test_data)

    print('共有{}维特征。'.format(X_train.shape[1]))

    # 标签处理
    y_train = train_data['label'].values
    y_test = test_data['label'].values

    # 数据建模及验证
    print('\n===================== 数据建模及验证 =====================')
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)
    y_pred = nb_model.predict(X_test)

    print('准确率：', accuracy_score(y_test, y_pred))
    print('AUC值：', roc_auc_score(y_test, y_pred))


if __name__ == '__main__':
    main()

