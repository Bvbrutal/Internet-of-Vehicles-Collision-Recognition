import pandas as pd
import lightgbm as lgb
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# 读取数据
df = pd.read_csv(
    r'D:\huancun\WeChat Files\wxid_7fjwgckpy7m522\FileStorage\File\2023-09\dw_casl_user_sample_feature_all.csv')

# 数据预处理
# ...（你的数据预处理步骤）...

# 划分训练集和测试集
x = df.drop('sample_flag', axis=1)
y = df['sample_flag']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 定义Optuna的目标函数
def objective(trial):
    params = {
        'cat_smooth': trial.suggest_float('cat_smooth', 0.1, 10.0),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
        'num_leaves': trial.suggest_int('num_leaves', 20, 200),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 20.0),
    }

    # 设置其他参数
    params.update({
        'is_unbalance': True,
        'num_iterations': 1000,
        'objective': 'binary',
        'metric': ['binary_logloss', 'auc'],
        'early_stopping_rounds': 50,
        'random_state': 42,
    })

    # 创建LightGBM数据集
    train_data = lgb.Dataset(x_train, label=y_train)

    # 训练模型
    model = lgb.train(params, train_data, valid_sets=[(x_test, y_test)], verbose_eval=False)

    # 预测
    y_pred = model.predict(x_test, num_iteration=model.best_iteration)

    # 计算AUC
    auc = roc_auc_score(y_test, y_pred)

    return auc

# 创建Optuna研究并运行参数优化
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)

# 打印最佳参数和AUC分数
best_params = study.best_params
best_score = study.best_value
print("Best Parameters:", best_params)
print("Best AUC Score:", best_score)

# 使用最佳参数重新训练模型
best_model = lgb.train(best_params, train_data, valid_sets=[(x_test, y_test)], verbose_eval=False)

# 保存最佳模型
best_model.save_model('best_model.pkl')
