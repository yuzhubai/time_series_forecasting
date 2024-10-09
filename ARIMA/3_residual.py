# 模型检验：残差的独立性
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
resid = results.resid #获取残差

#绘制
#残差的自相关系数
#当然是越小越好，残差越小，说明残差之间越独立，模型估计的越准
fig, ax = plt.subplots(figsize=(12, 5))

ax = sm.graphics.tsa.plot_acf(resid, lags=40,ax=ax)

plt.show()
