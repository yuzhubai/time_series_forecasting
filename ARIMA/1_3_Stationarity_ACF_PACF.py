# 图检验法：观察ACF,PACF图

import pandas as pd
import matplotlib.pyplot as plt

# 导入数据
flu_data = pd.read_csv(r'C:\Users\26363\Desktop\Code\Python\research\2\ARIMA\data_flu.csv',header=None)
#.diff(1)做一个时间间隔
flu_dif1 = flu_data.diff(1) #1阶差分

#对一阶差分数据在划分时间间隔
flu_dif2 = flu_dif1.diff(1) #2阶差分

import statsmodels.api as sm

fig = plt.figure(figsize=(12,7))

ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(flu_data, lags=20,ax=ax1)
ax1.xaxis.set_ticks_position('bottom') # 设置坐标轴上的数字显示的位置，top:显示在顶部  bottom:显示在底部

ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(flu_data, lags=20, ax=ax2)
ax2.xaxis.set_ticks_position('bottom')

plt.show()

