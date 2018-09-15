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
data_file = './data/spam.csv'

# 预处理后的数据集
proc_data_file = os.path.join('./data/proc_spam.csv')

# 文本类别字典
text_type_dict = {'ham': 1,
                  'spam': 0}
