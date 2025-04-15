import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import json
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 加载数据
csv_file = 'sheet1.csv'
data = pd.read_csv(csv_file)

# 将时间列转换为日期时间索引
data['t_d'] = pd.to_datetime(data['t_d'])
data.set_index('t_d', inplace=True)

# 删除全为空值的列
data = data.dropna(axis=1, how='all')

# 数据清洗
data = data.replace([np.inf, -np.inf], np.nan)

# 处理缺失值
target_column = 'pH'
if target_column in data.columns:
    # 使用前向填充和后向填充
    data[target_column] = data[target_column].ffill().bfill()
    
    # 如果还有缺失值，使用中位数填充
    if data[target_column].isnull().any():
        data[target_column] = data[target_column].fillna(data[target_column].median())

timeseries = data[target_column]

# 加载模型参数
with open('model_params.json', 'r') as f:
    params = json.load(f)

# 数据分割
train_size = int(len(timeseries) * 0.8)
train_data = timeseries.iloc[:train_size]
test_data = timeseries.iloc[train_size:]

# 创建并拟合SARIMA模型
model_sarima = SARIMAX(
    train_data,
    order=params['sarima_order'],
    seasonal_order=params['sarima_seasonal_order'],
    enforce_stationarity=False,
    enforce_invertibility=False
)
results_sarima = model_sarima.fit(disp=False)

# 获取训练集和测试集的预测值
y_pred_train = results_sarima.get_prediction(start=0, end=train_size-1)
y_pred_test = results_sarima.get_prediction(start=train_size, end=len(timeseries)-1)

# 合并预测结果
y_pred_sarima = pd.concat([y_pred_train.predicted_mean, y_pred_test.predicted_mean])

# 分别计算训练集和测试集的拟合度指标
r2_train = r2_score(train_data, y_pred_train.predicted_mean)
r2_test = r2_score(test_data, y_pred_test.predicted_mean)
rmse_train = np.sqrt(np.mean((train_data - y_pred_train.predicted_mean) ** 2))
rmse_test = np.sqrt(np.mean((test_data - y_pred_test.predicted_mean) ** 2))

# 预测未来3年的数据
forecast_steps = 36  # 3年 * 12个月
forecast = results_sarima.get_forecast(steps=forecast_steps)
forecast_mean = forecast.predicted_mean

# 创建时间索引
last_date = pd.to_datetime(data.index[-1])
forecast_dates = pd.date_range(start=last_date, periods=forecast_steps+1, freq='M')[1:]

# 加载原有图像
img = plt.imread('plots/real_vs_predicted.png')
plt.figure(figsize=(12, 6))
plt.imshow(img)

# 获取图像的数据范围
ax = plt.gca()
ax.set_position([0.1, 0.1, 0.8, 0.8])

# 绘制未来预测值
plt.plot(forecast_dates, forecast_mean, label='未来3年预测', color='yellow', linestyle='--', linewidth=1.5)

# 添加置信区间
forecast_ci = forecast.conf_int()
plt.fill_between(forecast_dates, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], color='gray', alpha=0.2)

# 更新标题和标签
plt.title(f'{target_column} 时间序列预测分析（包含未来3年预测）')
plt.xlabel('时间')
plt.ylabel(f'{target_column} 值')
plt.legend()
plt.grid(True, alpha=0.3)

# 保存图表
if not os.path.exists('plots'):
    os.makedirs('plots')
if not os.path.exists('svg'):
    os.makedirs('svg')

plt.savefig('plots/prediction_3years.png', bbox_inches='tight', pad_inches=0.1)
plt.savefig('svg/prediction_3years.svg', bbox_inches='tight', pad_inches=0.1)
plt.show()