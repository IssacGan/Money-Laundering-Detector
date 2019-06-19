import pickle

import numpy as np
import pandas
from sklearn import tree
import matplotlib.pyplot as plt

# read training data
dataframeX = pandas.read_csv('./../../datasets/dataset_secondary.csv', sep=",", header=0)
col_names = list(dataframeX.columns.values)

# convert to matrix for input
dataMat = dataframeX.values
X = dataMat[:, 1:-2]
Y = dataMat[:, -1]

assert (X.shape[0] == Y.shape[0]), "No of products and classes not equal"


# remove empty fields from categorical columns
def removeNulls(X, col):
    for i in range(X.shape[0]):
        if type(X[i, col]) is float and np.isnan(X[i, col]):
            X[i, col] = 'NA'


def plot_feature_importances(fea_imp, title, fea, pic_name):
    # 函数作用：绘制变量重要性柱状图 显示重要性>0的变量
    # fea_imp：方法的.feature_importances_
    # title:图的标题
    # fea：所有变量的名称
    # pic_name：要保存的图片的名称

    # 将重要性值标准化
    fea_imp = 100.0 * (fea_imp / max(fea_imp))
    # 将得分从高到低排序
    index_sorted = np.argsort(-fea_imp)  # 降序排列
    plt.figure(figsize=(16, 4))

    # 统计非0的个数
    n = (fea_imp[index_sorted] > 0).sum()
    print('重要性非0的变量共有 %d 个' % n)

    # 让X坐标轴上的标签居中显示 和n保持一致
    pos = np.arange(n) + 0.5

    # 画图只画大于0的特征重要性部分
    plt.bar(pos, fea_imp[index_sorted][:n], align='center')
    plt.xticks(pos, np.array(fea)[index_sorted][:n], rotation=90)  # 转90度就可以了！
    plt.ylabel('feature importance')
    plt.title(title)
    plt.savefig(pic_name + '.png', dpi=100, bbox_inches='tight')  # bbox_inches='tight'这个参数可以解决输出图片横轴标签显示不全的问题
    plt.show()


# train the model
model = tree.DecisionTreeClassifier()
model.fit(X, Y.astype(int))

# save the model
pickle.dump(model, open("./../../models/tree_classifier_model.dat", "wb"))
print("tree_classifier_model.dat saved in models folder")

feature_imp = model.feature_importances_
plt.bar(range(len(feature_imp)), feature_imp, color='rgb', tick_label=col_names[1:27])
plt.xticks(range(len(feature_imp)), col_names[2:27], rotation=90)
plt.savefig('feature_importance.png', dpi=100, bbox_inches='tight')
plt.show()
sorted_feature_vals = np.sort(feature_imp)
sorted_feature_indexes = np.argsort(feature_imp)
print("Significant Features in decreasing order of importance: ")
for i in reversed(sorted_feature_indexes):
    print(col_names[i + 2].replace('\t', ''), " \t->\t ", feature_imp[i])

########################################################################
# output
########################################################################
# Significant Features in decreasing order of importance:
# balance_difference_60   		  ->        0.297249822847
# balance_difference_90   		  ->        0.292937784263
# outgoing_domestic_amount_60     ->        0.112128004209
# incoming_foreign_amount_30      ->        0.0809464605337
# outgoing_domestic_amount_90     ->        0.0803045191632
# outgoing_foreign_amount_90      ->        0.0328515844364
# incoming_domestic_count_30      ->        0.0323738450738
# incoming_domestic_count_60      ->        0.0188196650244
# outgoing_foreign_amount_60      ->        0.0170659207114
# incoming_foreign_amount_60      ->        0.0163670574818
# incoming_domestic_count_90      ->        0.00795390742756
# incoming_foreign_amount_90      ->        0.00761129603537
# outgoing_foreign_amount_30      ->        0.00339013279358
# incoming_domestic_amount_90     ->        0.0
# outgoing_domestic_amount_30     ->        0.0
# balance_difference_30 		  ->        0.0
# outgoing_foreign_count_90       ->        0.0
# outgoing_foreign_count_60       ->        0.0
# incoming_foreign_count_30       ->        0.0
# outgoing_domestic_count_90      ->        0.0
# outgoing_foreign_count_30       ->        0.0
# incoming_foreign_count_90       ->        0.0
# incoming_foreign_count_60       ->        0.0
# outgoing_domestic_count_30      ->        0.0
# outgoing_domestic_count_60      ->        0.0
# incoming_domestic_amount_60     ->        0.0
