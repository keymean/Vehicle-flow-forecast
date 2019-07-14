# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 19:42:58 2018

@author: Liche
"""
import filtrate_sensor
import process
import os
import pandas as pd
import numpy as np
from scipy import hstack
from sklearn.preprocessing import LabelEncoder
import pickle
import codecs
from sklearn.metrics import accuracy_score
import datetime
def judge_sensor_data():
    if not os.path.exists('./train_data') and not os.path.exists('./test_data'):
        filtrate_sensor.filtrate_data()
def get_models():
    xgb_params = {
            'learning_rate': 0.05,  # 步长
            'n_estimators': 1000,
            'max_depth': 10,  # 树的最大深度
            'objective': 'multi:softmax',
            'num_class': 4,
            #'min_child_weight': 1,  # 决定最小叶子节点样本权重和，如果一个叶子节点的样本权重和小于min_child_weight则拆分过程结束。
            #'gamma': 0,  # 指定了节点分裂所需的最小损失函数下降值。这个参数的值越大，算法越保守
            'silent': 0,  # 输出运行信息
            'subsample': 0.8,  # 每个决策树所用的子样本占总样本的比例（作用于样本）
            'colsample_bytree': 0.8,  # 建立树时对特征随机采样的比例（作用于特征）典型值：0.5-1
            'nthread': 12,
            'seed': 27}
    rounds = 1000
    early_stopping_rounds = 30
    if not os.path.exists('./models'):
        os.mkdir('./models')
        for i in range(len(os.listdir('./train_data'))):
            model = process.training_model(i, xgb_params, rounds, early_stopping_rounds, 250)
            model.save_model(os.path.join('./models', 'model_'+str(i)+'.h5'))
def format_date(s):
        month = int(s[:2])
        day = int(s[2:4])
        h = int(s[4:6])
        m = int(s[6:])
        return datetime.datetime(2018, month, day, h , m)
def load_test(file):
    '''
    假设只有一个待预测文件
    '''
    test = pd.read_csv(file, dtype={0:str, 1:str})
    test['10'].replace('UNKNOWN_CONGESTION_LEVEL', 'NON_CONGESTION', inplace=True)
    test['begain_time'] = test['0'] + test['1']
    test['begain_time'] = test['begain_time'].map(format_date)
    led4_fit = pickle.load(open('./led4.h5', 'rb'))
    test['4'] = led4_fit.transform(test['4'].values)
    led_dict = {'NON_CONGESTION':0, 'LIGHT_CONGESTION':1, 'MEDIUM_CONGESTION':2, 'HEAVY_CONGESTION':3}
    test['10'] = test['10'].map(led_dict)
    test.drop(['0','1','2','3','5','6','7','8','9'], axis=1, inplace=True)
    return test
if __name__=="__main__":
    judge_sensor_data()
    get_models()
    if True:
        test = load_test('./test.csv')  #混合着所有传感器的信息
        info = process.predict(test, 6)
        with codecs.open('result.txt', 'w', encoding='utf8') as fin:
            fin.write(info)
        
    