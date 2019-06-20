import numpy as np
from numpy import genfromtxt

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn import tree

from sklearn.preprocessing import LabelEncoder
import pandas
import csv
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from os import walk

import pickle

for (dirpath, dirnames, filenames) in walk("./../../datasets/segments"):
    for file in filenames:
        data_path = "./../../datasets/segments/" + file
        print(data_path)
        dfX = pandas.read_csv(data_path, sep=",", header=0)
        data = dfX.values
        col_cont = ['amount', 'oldbalanceOrg', 'oldbalanceDest']  # 连续特征
        col_disc = ['trans_type', 'accountType', 'isFraud']  # 离散特征
        data_cont = dfX[col_cont]
        data_disc = dfX[col_cont]
        data_cont = (data_cont - data_cont.min(axis=0)) / (data_cont.max(axis=0)-data_cont.min(axis=0))
        p = data_cont.plot(kind='kde', linewidth=2, subplots=True, sharex=False)
        plt.legend()
        plt.savefig('./../../output/density_plot_%s.png' % (file.split(".")[0]), dpi=100, bbox_inches='tight')
        plt.show()


