# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 16:50:34 2018

@author: Liche
"""
import os
import pandas as pd
import numpy as np
from scipy import hstack
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.metrics import accuracy_score
import pickle
import datetime
def read_sensor_data(number):  #读取传感器数据
    sensor_train = pd.read_csv(os.path.join('./train_data', 'sensor_'+str(number)+'.csv'), dtype={0:str, 1:str})
    sensor_test = pd.read_csv(os.path.join('./test_data', 'sensor_'+str(number)+'.csv'), dtype={0:str, 1:str})
    sensor_train_y = sensor_train['10']
    sensor_test_y = sensor_test['10']
    sensor_train['0'] = sensor_train['0'].map(lambda x:'0'+x if len(x)==3 else x)
    sensor_train['1'] = sensor_train['1'].map(lambda x:'0'*(4-len(x))+x if len(x)<4 else x)
    sensor_train['begain_time'] = sensor_train['0'] + sensor_train['1']
    sensor_test['begain_time'] = sensor_test['0'] + sensor_test['1']
    def format_date(s):
        month = int(s[:2])
        day = int(s[2:4])
        h = int(s[4:6])
        m = int(s[6:])
        return datetime.datetime(2018, month, day, h , m)
    sensor_train['begain_time'] = sensor_train['begain_time'].map(format_date)
    sensor_test['begain_time'] = sensor_test['begain_time'].map(format_date)
    sensor_train.drop(['0','1','4'], axis=1, inplace=True)
    sensor_test.drop(['0','1','4'], axis=1, inplace=True)
    return sensor_train, sensor_test, sensor_train_y, sensor_test_y
def get_train_ago(data):
    predate = datetime.datetime(2018,1,1)  #上一个数据的日期  
    states = [np.nan for _ in range(5)]   #上一个的状态，1 2 3 4 5
    state_index = [0, 4, 3, 2, 1, 0]  #状态索引，[-1]状态索引的索引
    agos = [[] for _ in range(5)]
    for value in data.values:
        current_date = value[1].date()
        next_date = predate+pd.DateOffset(days=1)
        if predate != current_date and current_date != next_date.date():  #如果上一状态日期不同于当前状态日期，说明天数不连续
            states = [np.nan for _ in range(5)]
            predate = current_date
            state_index[-1] = 0
            states[0] = value[0]
            for i in range(5):
                agos[i].append(np.nan)
            #index += 1
        else:
            for i in range(5):
                #df.loc[index, i+2] = states[(state_index[state_index[-1]]+i)%5]
                agos[i].append(states[(state_index[state_index[-1]]+i)%5])
            states[(state_index[state_index[-1]]+4)%5] = value[0]
            state_index[-1] = (state_index[-1]+1)%5
            #index += 1
    #df.drop(['ago1','ago2','ago3','ago4','ago5'])
    for i in [4,3,2,1,0]:  #range(4, -1, -1):
        data['ago'+str(i+1)] = agos[i]
def get_totla_accumulate_train(data):   #训练集得到累计数据
    accum_count = [[] for _ in range(4)]
    start_time = 0  #连续开始日期
    predate = datetime.datetime(2018,1,1)  #上一个日期
    for i in range(data.shape[0]):
        current_date = data.iloc[i, 1].date()
        next_date = predate+pd.DateOffset(days=1)
        if predate != current_date and current_date != next_date.date():  #如果上一状态日期不同于当前状态日期，说明天数不连续
            start_time = i
            predate = current_date
            #a = df.iloc[:i, :].groupby(by=['10'], axis=0)['10'].count()  #全0
            for j in range(4):
                accum_count[j].append(0)
        else:
            a = data.iloc[start_time:i, :].groupby(by=['10'], axis=0)['10'].count()
            for j in range(4):
                accum_count[j].append(a[j] if j in a.keys() else 0)
    for i in [0,1,2,3]:  #ange(4, -1, -1):
        data['total'+str(i)] = accum_count[i]
def get_accumulate_train(data, back_time):  #累计分钟路况,针对训练集
    accum_count = [[] for _ in range(4)]
    start_time = 0  #连续开始日期
    predate = datetime.datetime(2018,1,1)  #上一个日期
    for i in range(data.shape[0]):
        current_date = data.iloc[i, 1].date()
        next_date = predate+pd.DateOffset(days=1)
        if predate != current_date and current_date != next_date.date():  #如果上一状态日期不同于当前状态日期，说明天数不连续
            start_time = i  #从当前天开始
            predate = current_date
            for j in range(4):
                accum_count[j].append(0)
        else:
            if i-start_time<back_time:  #如果累计天数小于back_time
                a = data.iloc[start_time:i, :].groupby(by=['10'], axis=0)['10'].count()
            else:
                a = data.iloc[i-back_time:i, :].groupby(by=['10'], axis=0)['10'].count()
            for j in range(4):
                accum_count[j].append(a[j] if j in a.keys() else 0)
    for i in [0,1,2,3]:  #ange(4, -1, -1):
        data['back_time'+str(back_time)+'_'+str(i)] = accum_count[i]
def get_accumulate_test(data, back_time):  #累计分钟路况,针对预测数据
    '''
    假设数据连续
    '''
    accum_count = [[0] for _ in range(4)]  #第一条数据历史累计为0
    for i in range(1, data.shape[0]-4):
        if i<back_time:  #如果累计天数小于back_time
            a = data.iloc[:i, :].groupby(by=['10'], axis=0)['10'].count()
        else:
            a = data.iloc[i-back_time:i, :].groupby(by=['10'], axis=0)['10'].count()
        for j in range(4):
            accum_count[j].append(a[j] if j in a.keys() else 0)
    for i in range(4):
        accum_count[i].extend([0,0,0,0])
    for i in [0,1,2,3]:  #ange(4, -1, -1):
        #print(data.shape, len(accum_count[i]))
        data['back_time'+str(back_time)+'_'+str(i)] = accum_count[i]
def get_totla_accumulate_test(data):   #测试集得到累计数据
    '''
    假设数据是连续的
    '''
    accum_count = [[0] for _ in range(4)]
    for i in range(1, data.shape[0]-4):  #统计到待预测数据位置(-4)
        a = data.iloc[:i, :].groupby(by=['10'], axis=0)['10'].count()
        for j in range(4):
            accum_count[j].append(a[j] if j in a.keys() else 0)
    for i in range(4):
        accum_count[i].extend([0,0,0,0])
    for i in [0,1,2,3]:  #ange(4, -1, -1):
        data['total'+str(i)] = accum_count[i]
def creat_val_data(data):   #生成待验证数据,针对需要验证的数据
    new_data = data.copy()
    get_train_ago(new_data)
    get_totla_accumulate_train(new_data)
    for back_time in [5, 10]:
        get_accumulate_train(new_data, back_time)
    new_data.drop(['10', 'begain_time'], axis=1, inplace=True)
    return new_data
def creat_test_data(data):  #生成待预测数据,针对需要预测的数据
    #final_time = data.iloc[-1, -1]
    new_data = data[['10', 'begain_time']]
    new_data = new_data.append(pd.DataFrame({'10':[0,0,0,0,0]}))  #向下追加数据
    for i in [5,4,3,2,1]:   #可以设置lagging
        ago = data['10'].values.tolist()
        for j in range(i):
            ago.insert(0, np.nan)
        for j in range(5-i):
            ago.append(np.nan)
        #df_test['ag0'+str(i)] = ago
        new_data[i] = ago
    get_totla_accumulate_test(new_data)
    for i in [5, 10]:  #, 15, 20, 25, 30]:
        get_accumulate_test(new_data, i)
    new_data.drop(['10', 'begain_time'], axis=1, inplace=True)
    return new_data
def get_last_accumulate(labels, back_time):  #得到最后一次累计的回溯天数
    accum_count = []
    if len(labels)<back_time:  #如果总共天数小于回溯累计天数
        for i in range(4):
            accum_count.append(labels.tolist().count(i))
    else:
        for i in range(4):
            accum_count.append(labels[-back_time:].count(i))
    return accum_count
def connect_predict(data, predict, lagging, pc, n):  #已有数据，已有数据标签+预测数据标签， 回溯历史数据个数，已有数据个数+已预测数据个数，已有数据+预测数据个数
    '''
    滑动窗口，连接预测数据
    假设预测数据连续
    '''
    data.reset_index(drop=True, inplace=True)
    col = data.shape[1]  #列属性
    back_times = [x for x in range(5, (col-9)//4*5+1, 5)] #得到back_times的回溯天数的列表
    if data.shape[0] < n:   #如果窗口数据小于总预测数据，向下添加数据
        #data = data.append(pd.DataFrame({'10':[np.nan, np.nan, np.nan, np.nan, np.nan]}))
        data.loc[data.shape[0], :] = [np.nan]*col  #添加一行数据
        for i in range(lagging):
            data.iloc[-1-i, i] = predict[-1]  #ago(x)
        for i in range(4):
            data.iloc[-5,i+5] = data.iloc[-6, i+5]  #copy累计路况信息pc
        data.loc[pc,'total'+str(predict[-1])] += 1  #total(x)
        for back_time in back_times:
            accum_count = get_last_accumulate(predict, back_time)
            for j in range(4):
                data.loc[pc, 'back_time'+str(back_time)+'_'+str(j)] = accum_count[j]  #back_time(x)_(x)
    else:  #data.shape[0] == n
        #print('1')
        tep = lagging-(n-pc)  #
        for i in range(n-pc):
            data.iloc[-1-i, tep+i] = predict[-1]  #ago(x)
        for i in range(4):
            #print(data.shape, pc)
            data.iloc[pc,i+5] = data.iloc[pc-1, i+5]  #copy累计路况信息-(n-pc)-1
        for back_time in back_times:
            data.loc[pc,'total'+str(predict[-1])] += 1  #total(x)
            accum_count = get_last_accumulate(predict, back_time)
            for j in range(4):
                data.loc[pc, 'back_time'+str(back_time)+'_'+str(j)] = accum_count[j]  #back_time(x)_(x)
    return data
def test(model, sensor_test, sensor_test_y, split=250):  #测试
    if split<1:
        split = int(split*sensor_test.shape[0])
    if split>sensor_test.shape[0]:
        split = sensor_test.shape[0] - 30
    pre = []  #预测数据
    data = creat_test_data(sensor_test.iloc[:split,:])
    labels = sensor_test_y[:split].tolist()  #已有数据标签
    pc = data.shape[0]-5 #指针，已预测数据个数,指向待预测数据                                                                                                    
    n = sensor_test.shape[0]  #已有数据+需要预测数据个数
    lagging = 5  #历史滑动窗口大小
    while pc<n:
        if pc<=n-lagging:  #data.shape[0]<288
            predict = model.predict(xgb.DMatrix(data.iloc[:-4,:].values))
            pre.append(int(predict[-1]))
            predict = labels + pre   #已有数据标签+预测标签
            #return predict
            pc += 1
            data = connect_predict(data, predict, lagging, pc, n)
        else:
            predict = model.predict(xgb.DMatrix(data.iloc[:-(n-pc), :].values))
            pre.append(int(predict[-1]))
            predict = labels + pre   #已有数据标签+预测标签extend出问题
            pc += 1
            if pc == n:
                break
            data = connect_predict(data, predict, lagging, pc, n)
    return sensor_test.shape[0]-split, accuracy_score(sensor_test_y[split:], pre)
    #return pre
def process_train_data(data):
    get_train_ago(data)
    get_totla_accumulate_train(data)
    for back_time in [5, 10]:
        get_accumulate_train(data, back_time)
    data.drop(['10', 'begain_time'], axis=1, inplace=True)
def predict(data, num):  #预测测试数据
    led4_fit = pickle.load(open('./led4.h5', 'rb'))
    info = ""
    for key, values in data.groupby(['4']):
        labels = values['10'].tolist()
        values.drop(['4'], axis=1, inplace=True)
        test = creat_test_data(values)
        pc = test.shape[0]-5 #指针，已预测数据个数,指向待预测数据                                                                                                    
        n = values.shape[0]+num  #已有数据+需要预测数据个数
        lagging = 5  #历史滑动窗口大小
        pre = []  #预测数据
        model = xgb.Booster(model_file=os.path.join('./models', 'model_'+str(key)+'.h5'))
        while pc<n:
            if pc<=n-lagging:  #test.shape[0]<288
                predict = model.predict(xgb.DMatrix(test.iloc[:-4,:].values))
                pre.append(int(predict[-1]))
                predict = labels + pre   #已有数据标签+预测标签
                #return predict
                pc += 1
                test = connect_predict(test, predict, lagging, pc, n)
            else:
                predict = model.predict(xgb.DMatrix(test.iloc[:-(n-pc), :].values))
                pre.append(int(predict[-1]))
                predict = labels + pre   #已有数据标签+预测标签extend出问题
                pc += 1
                if pc == n:
                    break
                test = connect_predict(test, predict, lagging, pc, n)
        pre_strs = list(map(str, pre))
        print(led4_fit.inverse_transform([key])[0],':', ' '.join(pre_strs))
        info += led4_fit.inverse_transform([key])[0]+':'+' '.join(pre_strs)+'\n'
    return info
def training_model(number, xgb_params, rounds=1000, early_stopping_rounds=30, split=250):
    sensor_train, sensor_test, sensor_train_y, sensor_test_y = read_sensor_data(number)
    if split<1:
        split = int(split*sensor_test.shape[0])
    if split>sensor_test.shape[0]:
        split = sensor_test.shape[0]-30
    process_train_data(sensor_train)
    xg_train = xgb.DMatrix(sensor_train.values, sensor_train_y)
    evaltest = creat_val_data(sensor_test.iloc[:split, :])
    dtest=xgb.DMatrix(evaltest.values, label=sensor_test_y[:split])
    watchlist = [(dtest, 'eval')]
    bst = xgb.train(xgb_params, xg_train, num_boost_round=rounds, evals=watchlist, early_stopping_rounds=early_stopping_rounds)
    data_num, acc = test(bst, sensor_test, sensor_test_y, split)
    print("第",number,"个传感器，在", data_num,"个测试数据上的得分为:",acc)
    return bst
if __name__=="__main__":
    #sensor_train, sensor_test = read_sensor_data(3)
    split = 220
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
    sensor_train, sensor_test, sensor_train_y, sensor_test_y = read_sensor_data(25)
    process_train_data(sensor_train)
    xg_train = xgb.DMatrix(sensor_train.values, sensor_train_y)
    evaltest = creat_val_data(sensor_test.iloc[:split, :])
    dtest=xgb.DMatrix(evaltest.values, label=sensor_test_y[:split])
    watchlist = [(dtest, 'eval')]
    bst = xgb.train(xgb_params, xg_train, num_boost_round=1000, evals=watchlist, early_stopping_rounds=30)
    data_num, acc = test(bst, sensor_test, sensor_test_y, split)
    