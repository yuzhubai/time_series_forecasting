#利用模型预测，将预测值与test集进行比较

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
#根据之前求得
p = 1
d = 0
q = 1

model = sm.tsa.ARIMA(train, order=(p,d,q))
results = model.fit()

# 进行预测
start_index = len(train)
end_index = len(train) + len(test) - 1
forecast = results.predict(start=start_index, end=end_index, dynamic=False)

# 将预测结果与原始数据序列合并
forecast.index = test.index

# 绘制结果
plt.figure(figsize=(12, 6))
plt.plot(train.index, train, label='Train')
plt.plot(test.index, test, label='Test')
plt.plot(forecast.index, forecast, label='Forecast', linestyle='--')
# plt.plot(predict_sunspots, label='Fit')
plt.legend()
plt.xticks(rotation=45)
plt.show()
