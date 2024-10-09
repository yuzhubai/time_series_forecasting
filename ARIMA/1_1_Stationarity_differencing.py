#观察原始数据、一阶差分、二阶差分图，判断平稳性，确定d

import pandas as pd
import matplotlib.pyplot as plt

# 导入数据
flu_data = pd.read_csv(r'C:\Users\26363\Desktop\Code\Python\research\2\ARIMA\data_flu.csv',header=None)
#.diff(1)做一个时间间隔
flu_dif1 = flu_data.diff(1) #1阶差分

#对一阶差分数据在划分时间间隔
flu_dif2 = flu_dif1.diff(1) #2阶差分

# 创建图表
plt.figure(figsize=(15, 10))

# 原数据图表
plt.subplot(3, 1, 1)
plt.plot(flu_data, label='Original Data', color='blue')
plt.title('Original Data')
#plt.xlabel('Index')
plt.ylabel('Value')
plt.legend()

# 一阶差分图表
plt.subplot(3, 1, 2)
plt.plot(flu_dif1, label='First Difference', color='orange')
plt.title('First Difference')
#plt.xlabel('Index')
plt.ylabel('Value')
plt.legend()

# 二阶差分图表
plt.subplot(3, 1, 3)
plt.plot(flu_dif2, label='Second Difference', color='green')
plt.title('Second Difference')
plt.xlabel('Index')
plt.ylabel('Value')
plt.legend()

# 调整布局
plt.tight_layout()
plt.show()

