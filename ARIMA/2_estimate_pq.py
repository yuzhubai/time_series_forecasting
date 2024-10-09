#确定p，q
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

#AIC与BIC准则定阶

AIC = sm.tsa.stattools.arma_order_select_ic(flu_data, max_ar=4, max_ma=4, ic='aic')['aic_min_order']
# BIC
BIC = sm.tsa.stattools.arma_order_select_ic(flu_data, max_ar=4, max_ma=4, ic='bic')['bic_min_order']
print('---AIC与BIC准则定阶---')
print('the AIC is{}\nthe BIC is{}\n'.format(AIC, BIC), end='')




