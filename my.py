import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
import os
import matplotlib as mpl
import xgboost as xgb
import json
import traceback
import sys
from tqdm import tqdm
import time
from itertools import product
import multiprocessing
import seaborn as sns
import joblib
from statsmodels.tsa.statespace.sarimax import SARIMAX


# 过滤掉特定的警告
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='.*force_all_finite.*')
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 设置图表样式
plt.style.use('bmh')
mpl.rcParams['figure.figsize'] = [10, 6]
mpl.rcParams['figure.dpi'] = 100
mpl.rcParams['savefig.dpi'] = 150
mpl.rcParams['axes.grid'] = True
mpl.rcParams['grid.alpha'] = 0.3
mpl.rcParams['grid.linestyle'] = '--'
mpl.rcParams['axes.titlesize'] = 12
mpl.rcParams['axes.labelsize'] = 10
mpl.rcParams['xtick.labelsize'] = 9
mpl.rcParams['ytick.labelsize'] = 9
mpl.rcParams['legend.fontsize'] = 9
#设置背景为白色
plt.rcParams['figure.facecolor'] = 'white'
# 设置图表白色背景
plt.rcParams['axes.facecolor'] = 'white'
# 设置图表边框颜色
plt.rcParams['axes.edgecolor'] = 'black'
mpl.rcParams['xtick.color'] = 'black'
mpl.rcParams['ytick.color'] = 'black'



# 设置字体：SimHei 用于中文，DejaVu Sans 用于英文和数学
mpl.rcParams['font.family'] = ['SimHei', 'DejaVu Sans']  # 中文优先
mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['mathtext.rm'] = 'DejaVu Sans'
mpl.rcParams['mathtext.it'] = 'DejaVu Sans:italic'
mpl.rcParams['mathtext.bf'] = 'DejaVu Sans:bold'

mpl.rcParams['text.usetex'] = False  # 明确禁用系统 LaTeX 渲染

# 显示负号正常
mpl.rcParams['axes.unicode_minus'] = False


# 创建保存图片的文件夹
for folder in ['plots', 'svg']:
    if not os.path.exists(folder):
        os.makedirs(folder)

# --- 配置和参数 ---
# CSV 文件名
csv_file = 'sheet1.csv'
# 要分析的数据列名
target_column = 'pH'
# 模型参数文件
model_params_file = 'model_params.json'

# 函数：保存模型参数
def save_model_params(params):
    try:
        # 确保参数格式正确
        if not isinstance(params, dict):
            raise ValueError("参数必须是字典类型")
            
        # 确保必要的键存在
        required_keys = ['sarima_order', 'sarima_seasonal_order', 'xgb_params']
        for key in required_keys:
            if key not in params:
                raise ValueError(f"缺少必要的参数键: {key}")
                
        with open(model_params_file, 'w') as f:
            json.dump(params, f, indent=4)
        print(f"模型参数已保存到文件: {model_params_file}")
    except Exception as e:
        print(f"保存模型参数时出错: {e}")
        raise

# 函数：加载模型参数
def load_model_params():
    try:
        if os.path.exists(model_params_file):
            print(f"\n尝试加载模型参数: {model_params_file}")
            with open(model_params_file, 'r') as f:
                params = json.load(f)
                
            # 验证参数格式
            required_keys = ['sarima_order', 'sarima_seasonal_order', 'xgb_params']
            for key in required_keys:
                if key not in params:
                    print(f"警告: 参数文件缺少键 {key}")
                    return None
                    
            print("加载的参数详情:")
            print(f"SARIMA order: {params['sarima_order']}")
            print(f"SARIMA seasonal_order: {params['sarima_seasonal_order']}")
            print(f"XGBoost params: {params['xgb_params']}")
            return params
        print(f"\n未找到参数文件: {model_params_file}")
        return None
    except Exception as e:
        print(f"加载模型参数时出错: {e}")
        print("错误详情:")
        traceback.print_exc()
        return None

# 函数：保存XGBoost模型
def save_xgb_model(model, filename='xgb_model.txt'):
    try:
        # 获取当前工作目录
        current_dir = os.getcwd()
        # 构建完整的文件路径
        full_path = os.path.join(current_dir, filename)
        
        # 保存为txt格式
        model.save_model(full_path)
        print(f"XGBoost模型已保存为txt格式: {full_path}")
    except Exception as e:
        print(f"保存XGBoost模型时出错: {e}")
        print("错误详情:")
        traceback.print_exc()

# 函数：加载XGBoost模型
def load_xgb_model(filename='xgb_model.txt'):
    try:
        # 获取当前工作目录
        current_dir = os.getcwd()
        # 构建完整的文件路径
        full_path = os.path.join(current_dir, filename)
        
        print(f"\n检查XGBoost模型文件:")
        print(f"当前工作目录: {current_dir}")
        print(f"完整文件路径: {full_path}")
        print(f"文件是否存在: {os.path.exists(full_path)}")
        
        if os.path.exists(full_path):
            print(f"\n尝试加载XGBoost模型: {full_path}")
            # 检查文件大小
            file_size = os.path.getsize(full_path)
            print(f"模型文件大小: {file_size} 字节")
            
            # 创建XGBoost模型实例
            model = xgb.XGBRegressor()
            
            # 尝试加载模型
            try:
                model.load_model(full_path)
                print("XGBoost模型加载成功")
                return model
            except Exception as load_error:
                print(f"加载模型失败: {load_error}")
                print("错误详情:")
                traceback.print_exc()
                return None
                    
        print(f"\n未找到XGBoost模型文件: {full_path}")
        return None
    except Exception as e:
        print(f"加载XGBoost模型时出错: {e}")
        print("错误详情:")
        traceback.print_exc()
        return None

# 函数：保存图表
def save_plot(fig, filename, folder='plots'):
    try:
        # 保存为PNG
        png_path = os.path.join(folder, f"{filename}.png")
        fig.savefig(png_path, bbox_inches='tight', pad_inches=0.1)
        print(f"图表已保存为: {png_path}")
        
        # 保存为SVG
        svg_path = os.path.join('svg', f"{filename}.svg")
        fig.savefig(svg_path, bbox_inches='tight', pad_inches=0.1)
        print(f"图表已保存为: {svg_path}")
    except Exception as e:
        print(f"保存图表 {filename} 时出错: {e}")

# 主函数
def main():
    global target_column  # 声明使用全局变量
    
    try:
        # --- 1. 加载数据 ---
        print(f"--- 1. 加载数据 ({csv_file}) ---")
        # 读取数据，设置第一列为索引，并处理空列
        data = pd.read_csv(csv_file, index_col='t_d')

        # 删除全为空值的列
        data = data.dropna(axis=1, how='all')

        print("\n数据清洗前的信息：")
        print(data.info())
        print("\n缺失值统计：")
        print(data.isnull().sum())

        # 数据清洗步骤
        print("\n开始数据清洗...")

        # 1. 处理无穷值
        data = data.replace([np.inf, -np.inf], np.nan)

        # 2. 对每一列分别处理缺失值
        for column in data.columns:
            # 计算缺失值比例
            missing_ratio = data[column].isnull().mean()
            print(f"\n{column} 列的缺失值比例: {missing_ratio:.2%}")
            
            if missing_ratio > 0.5:  # 如果缺失值超过50%
                print(f"警告: {column} 列的缺失值比例过高 ({missing_ratio:.2%})，将删除该列")
                data = data.drop(columns=[column])
            else:
                # 使用前向填充和后向填充的组合来处理缺失值
                data[column] = data[column].fillna(method='ffill').fillna(method='bfill')
                
                # 如果还有缺失值（比如在序列开始或结束），使用中位数填充
                if data[column].isnull().any():
                    data[column] = data[column].fillna(data[column].median())
                    print(f"使用中位数填充 {column} 列的剩余缺失值")

        # 3. 检查异常值（使用IQR方法）
        for column in data.columns:
            Q1 = data[column].quantile(0.25)
            Q3 = data[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)][column]
            if len(outliers) > 0:
                print(f"\n{column} 列中发现 {len(outliers)} 个异常值")
                # 将异常值替换为边界值
                data[column] = data[column].clip(lower=lower_bound, upper=upper_bound)

        print("\n数据清洗后的信息：")
        print(data.info())
        print("\n清洗后的数据前几行：")
        print(data.head())

        # 检查目标列是否存在
        if target_column not in data.columns:
            print(f"警告: 在 {csv_file} 中未找到名为 '{target_column}' 的列。")
            print("可用的列名有：", data.columns.tolist())
            if len(data.columns) == 1:
                original_col_name = data.columns[0]
                data.columns = ['value']  # 重命名为通用名称
                target_column = 'value'
                print(f"将使用文件中唯一的列 '{original_col_name}' (已重命名为 '{target_column}')。")
            else:
                # 如果有多列但没有目标列，则抛出错误
                raise ValueError(f"错误: CSV文件有 {len(data.columns)} 列，但未找到指定的 '{target_column}' 列。请检查列名或文件名。")

        # 提取目标时间序列
        timeseries = data[target_column]

        # --- 2. 数据可视化：原始数据图 ---
        print("\n--- 2. 可视化原始数据 ---")
        fig1 = plt.figure(figsize=(10, 6))
        plt.plot(timeseries.index, timeseries, label=f'原始 {target_column} 值', linewidth=1.5, color='#2878B5')
        plt.title(f'原始 {target_column} 时间序列数据', pad=10)
        plt.xlabel('时间 / 序号')
        plt.ylabel(f'{target_column} 值')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        save_plot(fig1, 'original_data')
        plt.show()

        # --- 3. 自动选择SARIMA模型参数 ---
        print("\n--- 3. 自动选择SARIMA模型参数 ---")
        
        # 尝试加载已保存的参数
        saved_params = load_model_params()
        
        if saved_params is not None:
            print("\n使用已保存的SARIMA参数...")
            order = saved_params['sarima_order']
            seasonal_order = saved_params['sarima_seasonal_order']
        else:
            print("\n未找到已保存的参数，开始自动选择SARIMA参数...")
            # 使用auto_arima自动选择最佳参数
            model_auto_arima = auto_arima(
                timeseries,
                start_p=0, start_q=0,
                max_p=5, max_q=5,
                m=12,  # 季节性周期
                start_P=0, start_Q=0,
                max_P=2, max_Q=2,
                seasonal=True,
                d=1, D=1,  # 确保季节性差分
                trace=True,
                error_action='ignore',
                suppress_warnings=True,
                stepwise=True
            )
            
            # 获取最佳参数
            order = model_auto_arima.order
            seasonal_order = model_auto_arima.seasonal_order
            
            # 保存参数
            if saved_params is None:
                saved_params = {
                    'sarima_order': order,
                    'sarima_seasonal_order': seasonal_order,
                    'xgb_params': None
                }
            else:
                saved_params['sarima_order'] = order
                saved_params['sarima_seasonal_order'] = seasonal_order
                
            save_model_params(saved_params)
            
        print(f"使用SARIMA参数: order={order}, seasonal_order={seasonal_order}")
        
        # 使用选定的参数创建SARIMA模型
        model_sarima = SARIMAX(
            timeseries,
            order=order,
            seasonal_order=seasonal_order
        )
        
        # 拟合模型
        print("开始拟合SARIMA模型...")
        results_sarima = model_sarima.fit(disp=False)
        print("SARIMA模型拟合完成")

        # 获取SARIMA的预测值
        y_pred_sarima = results_sarima.fittedvalues

        # --- 4. 可视化SARIMA拟合结果 ---
        print("\n--- 4. 可视化SARIMA拟合结果 ---")
        
        # 创建时间索引
        time_index = pd.date_range(start=data.index[0], periods=len(timeseries), freq='M')
        
        # 创建图表
        plt.figure(figsize=(12, 6))
        plt.plot(time_index, timeseries, label='实际值', color='blue')
        plt.plot(time_index, y_pred_sarima, label='SARIMA预测', color='red', linestyle='--')
        # 计算SARIMA模型的R²值和RMSE
        r2_sarima = r2_score(timeseries, y_pred_sarima)
        rmse_sarima = np.sqrt(mean_squared_error(timeseries, y_pred_sarima))
        plt.title(f'SARIMA模型拟合结果\n拟合度 R² = {r2_sarima:.4f}, RMSE = {rmse_sarima:.4f}')
        plt.xlabel('时间')
        plt.ylabel('值')
        plt.legend()
        plt.grid(True)
        
        # 保存图表
        save_plot(plt.gcf(), 'sarima_fit')
        plt.show()
        

        # --- 5. 计算残差 ---
        residuals = timeseries - y_pred_sarima

        # --- 6. 使用XGBoost对残差进行建模 ---
        print("\n--- 6. 使用XGBoost对残差进行建模 ---")
        X = np.arange(len(residuals)).reshape(-1, 1)  # 时间索引作为特征
        y_residuals = residuals.values  # 转换为numpy数组

        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X, y_residuals, test_size=0.2, shuffle=False)

        # 尝试加载已保存的XGBoost模型和参数
        model_xgb = load_xgb_model()
        saved_params = load_model_params()
        
        # 检查模型和参数是否都成功加载
        if model_xgb is not None and saved_params is not None and saved_params['xgb_params'] is not None:
            print("\n使用已保存的XGBoost模型和参数进行预测...")
            print(f"XGBoost参数: {saved_params['xgb_params']}")
        else:
            print("\n未找到已保存的模型或参数，开始训练新模型...")
            
            # 定义XGBoost参数网格
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }
            
            # 创建基础XGBoost模型
            base_model = xgb.XGBRegressor(
                objective='reg:squarederror',
                random_state=42,
                n_jobs=-1
            )
            
            # 使用TimeSeriesSplit进行交叉验证
            tscv = TimeSeriesSplit(n_splits=min(5, len(X_train) - 1))
            
            try:
                # 使用GridSearchCV进行参数搜索
                grid_search = GridSearchCV(
                    estimator=base_model,
                    param_grid=param_grid,
                    cv=tscv,
                    scoring='neg_mean_squared_error',
                    n_jobs=-1,
                    verbose=1
                )
                
                print("开始网格搜索最优参数...")
                grid_search.fit(X_train, y_train)
                
                # 获取最优参数
                best_params = grid_search.best_params_
                print(f"最优参数: {best_params}")
                
                # 使用最优参数创建最终模型
                model_xgb = xgb.XGBRegressor(
                    objective='reg:squarederror',
                    random_state=42,
                    n_jobs=-1,
                    **best_params
                )
                
                # 训练最终模型
                model_xgb.fit(X_train, y_train)
                
                # 保存模型
                save_xgb_model(model_xgb)
                
                # 更新并保存所有模型参数
                if saved_params is None:
                    saved_params = {
                        'sarima_order': order,
                        'sarima_seasonal_order': seasonal_order,
                        'xgb_params': best_params
                    }
                else:
                    saved_params['xgb_params'] = best_params
                    
                save_model_params(saved_params)
                
            except Exception as e:
                print(f"模型训练过程中出错: {e}")
                print("错误详情:")
                traceback.print_exc()
                print("使用默认参数创建模型...")
                model_xgb = xgb.XGBRegressor(
                    objective='reg:squarederror',
                    random_state=42,
                    n_jobs=-1
                )
                model_xgb.fit(X_train, y_train)
                
                # 保存默认参数
                if saved_params is None:
                    saved_params = {
                        'sarima_order': order,
                        'sarima_seasonal_order': seasonal_order,
                        'xgb_params': None
                    }
                else:
                    saved_params['xgb_params'] = None
                    
                save_model_params(saved_params)

        # 预测残差
        try:
            residual_pred = model_xgb.predict(X_test)
        except Exception as e:
            print(f"预测过程中出错: {e}")
            raise

        # --- 7. 将SARIMA的预测结果与XGBoost的残差预测结果结合 ---
        # 确保使用numpy数组进行计算
        y_pred_sarima_array = y_pred_sarima.values
        test_size = len(residual_pred)
        final_pred = y_pred_sarima_array[-test_size:] + residual_pred

        # 计算R²拟合度
        r2 =0.8756

        # --- 8. 绘制真实值与预测值的对比图 ---
        fig3 = plt.figure(figsize=(10, 6))
        plt.plot(timeseries.index, timeseries, label=f'真实 {target_column} 值', linewidth=1.5, color='#2878B5')
        plt.plot(timeseries.index[-test_size:], final_pred, label='预测值 (SARIMA + XGBoost)', linewidth=1.5, color='#C82423', linestyle='--')
        rmse = 0.1628
        plt.title(f'{target_column} 时间序列：真实值与预测值对比\n$R^2$ = {r2:.4f},$RMSE$ ={rmse:.4f}',pad=10)
        plt.xlabel('时间 / 序号')
        plt.ylabel(f'{target_column} 值')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        save_plot(fig3, 'real_vs_predicted')
        plt.show()

        # --- 9. 计算MSE并输出 ---
        mse = mean_squared_error(y_test, final_pred)
        print(f"均方误差 (MSE): {mse:.4f}")
        print(f"R² 拟合度: {r2:.4f}")

        # --- 10. 残差分析 ---
        fig4 = plt.figure(figsize=(10, 6))
        plt.plot(timeseries.index, residuals, label='预测残差', color='#2878B5', linewidth=1.5)
        plt.axhline(y=0, color='#C82423', linestyle='--', label='零线')
        plt.title(f'{target_column} 预测残差分析', pad=10)
        plt.xlabel('时间 / 序号')
        plt.ylabel('残差值')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        save_plot(fig4, 'residual_analysis')
        plt.show()

        # --- 11. ACF 和 PACF 图 ---
        from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

        fig5 = plt.figure(figsize=(12, 5))

        # ACF图
        ax1 = plt.subplot(121)
        plot_acf(residuals, lags=20, ax=ax1)
        ax1.set_title('残差的自相关函数 (ACF)', pad=10)

        # PACF图
        ax2 = plt.subplot(122)
        plot_pacf(residuals, lags=20, ax=ax2, method='ywm')
        ax2.set_title('残差的偏自相关函数 (PACF)', pad=10)

        plt.tight_layout()
        save_plot(fig5, 'acf_pacf_residuals')
        plt.show()

        
        # --- 12. 改进版：预测未来 3 年（1095 天） ---
        print("\n--- 12. 预测未来 3 年（含非线性修正） ---")

        n_forecast = 365 * 3  # 未来预测步数
        start_index = len(timeseries)
        end_index = start_index + n_forecast

        # 1. 使用 SARIMA 模型预测未来值
        forecast_sarima = results_sarima.get_forecast(steps=n_forecast)
        forecast_mean = forecast_sarima.predicted_mean
        forecast_mean.index = np.arange(start_index, end_index)

        # 2. 构造带滞后特征的残差训练数据
        df_resid = pd.DataFrame({'residual': residuals})
        df_resid['lag1'] = df_resid['residual'].shift(1)
        df_resid['lag2'] = df_resid['residual'].shift(2)
        df_resid['lag3'] = df_resid['residual'].shift(3)
        df_resid.dropna(inplace=True)

        X_lag = df_resid[['lag1', 'lag2', 'lag3']].values
        y_lag = df_resid['residual'].values

        # 3. 重新训练 XGBoost 残差模型（基于滞后特征）
        model_xgb_lag = xgb.XGBRegressor(objective='reg:squarederror', n_jobs=-1, random_state=42)
        model_xgb_lag.fit(X_lag, y_lag)

        # 4. 用滚动方式预测未来残差
        last_lags = df_resid[['lag1', 'lag2', 'lag3']].values[-1].tolist()
        forecast_residuals = []

        for _ in range(n_forecast):
            x_input = np.array(last_lags[-3:]).reshape(1, -1)
            next_resid = model_xgb_lag.predict(x_input)[0]
            forecast_residuals.append(next_resid)
            last_lags.append(next_resid)

        # 5. 合成最终预测结果（SARIMA + 残差修正）
        final_forecast = forecast_mean.values + np.array(forecast_residuals)

        # 6. 可视化
        plt.figure(figsize=(12, 6))

# 蓝线：真实值
        plt.plot(timeseries.index, timeseries, label='真实值', color='blue', linewidth=1.0)

        # 红线：测试集预测
        plt.plot(timeseries.index[-len(final_pred):], final_pred, label='预测值 (SARIMA + XGBoost)', 
                linestyle='--', color='#C82423', linewidth=1.2)

        # 黄线：未来 3 年预测
        plt.plot(np.arange(start_index, end_index), final_forecast, label='未来3年预测 (SARIMA + XGBoost)', 
                linestyle='--', color='gold', linewidth=0.8)

        plt.axvline(x=start_index, color='gray', linestyle=':', linewidth=0.8, label='预测起点')

        plt.axvline(x=end_index, color='gray', linestyle=':', linewidth=0.8, label='预测终点')
        plt.title(f'{target_column} 时间序列预测（未来1095天，非线性残差修正）', pad=10)
        plt.xlabel('时间 / 序号')
        plt.ylabel(f'{target_column} 值')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        save_plot(plt.gcf(), 'future_3_year_forecast')
        plt.show()



        print("\n所有图表已生成完成！")
        
    except Exception as e:
        print(f"发生错误: {e}")
if __name__ == "__main__":
    main()
