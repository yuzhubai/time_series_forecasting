#比较训练集数据与模型的拟合数据

import pandas as pd
import matplotlib.pyplot as plt

# 导入数据
flu_data = pd.read_csv(r'C:\Users\26363\Desktop\Code\Python\research\2\ARIMA\data_flu.csv',header=None)
#.diff(1)做一个时间间隔
flu_dif1 = flu_data.diff(1) #1阶差分

#对一阶差分数据在划分时间间隔
flu_dif2 = flu_dif1.diff(1) #2阶差分

train=flu_data[:418]
test=flu_data[418:]

#遍历，寻找适宜的参数
import itertools
import numpy as np
import seaborn as sns
import statsmodels.api as sm
#根据以上求得
p = 1
d = 0
q = 1

model = sm.tsa.ARIMA(train, order=(p,d,q))
results = model.fit()

predict_sunspots = results.predict(dynamic=False)
print(predict_sunspots)

#查看训练集的时间序列与预测数据
plt.figure(figsize=(12,6))
plt.plot(train)
plt.xticks(rotation=45) #旋转45度
plt.plot(predict_sunspots)
plt.show()
