# ADF法

import pandas as pd
import matplotlib.pyplot as plt

# 导入数据
flu_data = pd.read_csv(r'C:\Users\26363\Desktop\Code\Python\research\2\ARIMA\data_flu.csv',header=None)
#.diff(1)做一个时间间隔
flu_dif1 = flu_data.diff(1) #1阶差分

#对一阶差分数据在划分时间间隔
flu_dif2 = flu_dif1.diff(1) #2阶差分

import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller as ADF

flu_dif1 = flu_dif1.fillna(0)
flu_dif2 = flu_dif2.fillna(0)

print(type(flu_data))
timeseries_adf = ADF(flu_data)
timeseries_diff1_adf = ADF(flu_dif1)
timeseries_diff2_adf = ADF(flu_dif2)


# 打印单位根检验结果
print('timeseries_adf : ', timeseries_adf)
print('timeseries_diff1_adf : ', timeseries_diff1_adf)
print('timeseries_diff2_adf : ', timeseries_diff2_adf)

