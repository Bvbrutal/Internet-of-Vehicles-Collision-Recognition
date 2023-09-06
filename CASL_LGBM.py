# 导入必要的库
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

# 加载示例数据集（可以替换为您自己的数据集）
data = pd.read_csv('your_dataset.csv')
X = data.drop('target_column', axis=1)  # 特征数据
y = data['target_column']  # 目标数据

# 将数据拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建LightGBM数据集
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# 设置LightGBM参数
params = {
    'objective': 'regression',  # 选择回归任务
    'metric': 'rmse',  # 评估指标为均方根误差
    'boosting_type': 'gbdt',  # 使用梯度提升决策树
    'num_leaves': 31,  # 决策树的叶子节点数
    'learning_rate': 0.05,  # 学习率
    'feature_fraction': 0.9,  # 每次迭代中随机选择特征的比例
    'bagging_fraction': 0.8,  # 每次迭代中随机选择数据的比例
    'bagging_freq': 5,  # bagging的频率
    'verbose': 0  # 显示训练过程
}

# 训练LightGBM模型
num_round = 100  # 迭代次数
bst = lgb.train(params, train_data, num_round, valid_sets=[test_data], early_stopping_rounds=10)

# 预测
y_pred = bst.predict(X_test, num_iteration=bst.best_iteration)

# 评估模型性能（均方根误差）
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f'均方根误差 (RMSE): {rmse}')

# 可以根据需要保存和加载模型
# bst.save_model('lgb_model.txt')
# bst = lgb.Booster(model_file='lgb_model.txt')

