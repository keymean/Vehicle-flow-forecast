import os
import pandas as pd
import numpy as np
from scipy import hstack
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import pickle
import datetime
def filtrate_data():
    frist = 0
    train_data = [[] for _ in range(11)]
    os.mkdir('./train_data')
    os.mkdir('./test_data')
    for dir in os.listdir('./traffic/train'):
        prefix = os.path.join('./traffic/train', dir)
        for file in os.listdir(prefix):
            print(file)
            if frist == 1:
                data = pd.read_csv(os.path.join(prefix, file), sep='	', header=None, dtype={0:str, 1:str})
                for i in range(0, 11):
                    train_data[i].extend(data[i].values)
            else:
                data = pd.read_csv(os.path.join(prefix, file), header=None, dtype={0:str, 1:str})
                for i in range(0, 11):
                    train_data[i].extend(data[i].values)
        frist += 1
    train_data = pd.DataFrame(train_data).T
    test_data = [[] for _ in range(11)]
    for dir in os.listdir('./traffic/test'):
        prefix = os.path.join('./traffic/test', dir)
        for file in os.listdir(prefix):
            print(file)
            data = pd.read_csv(os.path.join(prefix, file), header=None, dtype={0:str, 1:str})
            for i in range(0, 11):
                test_data[i].extend(data[i].values)
    test_data = pd.DataFrame(test_data).T
    
    train_data[10].replace('UNKNOWN_CONGESTION_LEVEL', 'NON_CONGESTION', inplace=True)
    test_data[10].replace('UNKNOWN_CONGESTION_LEVEL', 'NON_CONGESTION', inplace=True)
    led4 = LabelEncoder()
    led4_fit = led4.fit(train_data[4].values)
    train_data[4] = led4_fit.transform(train_data[4].values)
    test_data[4] = led4_fit.transform(test_data[4].values)
    pickle.dump(led4, open('led4.h5', 'wb'))
    
    led_dict = {'NON_CONGESTION':0, 'LIGHT_CONGESTION':1, 'MEDIUM_CONGESTION':2, 'HEAVY_CONGESTION':3}
    train_data[10] = train_data[10].map(led_dict)
    test_data[10] = test_data[10].map(led_dict)
    train_data.drop([2,3,5,6,7,8,9], axis=1, inplace=True)
    test_data.drop([2,3,5,6,7,8,9], axis=1, inplace=True)
    
    for key, values in train_data.groupby(by=[4]):
        values.to_csv(os.path.join('./train_data', 'sensor_'+str(key)+'.csv'), index=False)
    for key, values in test_data.groupby(by=[4]):
        values.to_csv(os.path.join('./test_data', 'sensor_'+str(key)+'.csv'), index=False)
if __name__=="__main__":
    filtrate_data()
