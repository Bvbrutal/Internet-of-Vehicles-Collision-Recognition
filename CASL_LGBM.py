import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv(
    r'D:\huancun\WeChat Files\wxid_7fjwgckpy7m522\FileStorage\File\2023-09\dw_casl_user_sample_feature_all.csv')

# 空值处理
df = df.drop(['id'], axis=1)  # 删除加密的特征
df.loc[df['is_comm_user'] == r'\N', 'is_comm_user'] = None  # 是否为移动用户\N置为空值
df.loc[df['is_init_comm_user'] == r'\N', 'is_init_comm_user'] = None  # 是否主动通信用户\N置为空值
df.loc[df['cust_star'] == r'\N', 'cust_star'] = None  # 客户星级\N置为空值
df.loc[df['dome_roam_flux_days'] == r'\N', 'dome_roam_flux_days'] = None  # 省际漫游流量天数\N置为空值
df.loc[df['inter_roam_flux_days'] == r'\N', 'inter_roam_flux_days'] = None  # 国际漫游流量天数\N置为空值
df.loc[df['idcard_birth_area'] == r'\N', 'idcard_birth_area'] = None  # 身份证出生地\N置为空值
df['dou'].fillna(0, inplace=True)  # 使用流量为空值的默认为0
df['is_comm_user'].fillna(0, inplace=True)  # 是否为移动用户的空值默认为0
df['is_init_comm_user'].fillna(0, inplace=True)  # 是否主动通信用户的空值默认为0
df['cust_star'].fillna(0, inplace=True)  # 客户星级的空值默认为0
df['vip_cust_id'].fillna(0, inplace=True)  # 重要客户标识的空值默认为0
df['dome_roam_flux_days'].fillna(
    int(df[df['dome_roam_flux_days'].notnull()]['dome_roam_flux_days'].astype(int).mean()),
    inplace=True)  # 省际漫游流量天数填充为平均值
df['inter_roam_flux_days'].fillna(
    int(df[df['inter_roam_flux_days'].notnull()]['inter_roam_flux_days'].astype(int).mean()),
    inplace=True)  # 国际漫游流量天数填充为平均值
df.drop(df[df['mage_status'] == r"\N"].index, inplace=True)  # 删除mage_status等于\N的35行
df.drop(df[df['educat_degree_code'] == r"\N"].index, inplace=True)  # 删除educat_degree_code等于\N的35行
df.drop(df[df['ocpn_code'] == r"\N"].index, inplace=True)  # 删除ocpn_code等于\N的35行
df.drop(df[df['idcard_birth_area'].isnull()].index, inplace=True)  # 删除身份证出生地为空值的496个数据

# 时间处理
df1 = pd.to_datetime(df['month'], format="%Y%m")
df['year'] = df1.dt.year
df['month'] = df1.dt.month
df = pd.get_dummies(df, columns=['month'], dtype=int)  # 时间独热编码
df = df.astype(float)  # 类型转换

# 训练集&测试集创建
x = df.drop('sample_flag', axis=1)  # 特征数据
y = df['sample_flag']  # 目标数据
# # 将数据拆分为训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=18)
#
# 创建LightGBM数据集
train_data = lgb.Dataset(x_train, label=y_train)
test_data = lgb.Dataset(x_test, label=y_test, reference=train_data)

# 训练模型&评分标准
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve

# 设置LightGBM参数
params = {
    'n_estimators': 3000, 'learning_rate': 0.11, 'num_leaves': 25, 'max_depth': 9, 'min_data_in_leaf': 200,
    'lambda_l1': 5, 'lambda_l2': 75, 'min_gain_to_split': 0.04711412728205311, 'bagging_fraction': 0.9,
    'bagging_freq': 1, 'feature_fraction': 0.9, 'colsample_bytree': 0.4, 'subsample': 0.30000000000000004,
    'cat_smooth': 7.674033791281722, 'min_child_samples': 42, 'reg_alpha': 0.10256752619529136,
    'reg_lambda': 1.8061729280826144, 'objective': 'binary',
    'metric': ['binary_logloss', 'auc','average_precision'],
    'num_iterations': 3000,  # 迭代次数
    'early_stopping_rounds': 100,
    'verbose_eval': False,
}

# 训练模型
model = lgb.train(params, train_data, valid_sets=[test_data])

print(model.best_iteration)
model.save_model('SL_1.pkl')
# 预测
y_pred = model.predict(x_test, num_iteration=model.best_iteration)

# 评估模型性能
auc = roc_auc_score(y_test, y_pred)

# 如果需要 Precision 和 Recall，你可以使用 sklearn 库来计算它们
from sklearn.metrics import precision_score, recall_score,f1_score,confusion_matrix,average_precision_score
threshold = 0.5  # 二分类阈值
precision = precision_score(y_test, y_pred > threshold)
recall = recall_score(y_test, y_pred > threshold)
f1 = f1_score(y_test, y_pred> threshold)
average_precision_score=average_precision_score(y_test, y_pred)

confusion_matrix=confusion_matrix(y_test, y_pred> threshold)
print(f'AUC: {auc}\nprecision: {precision}\nrecall: {recall}\nf1: {f1}\nconfusion_matrix: {confusion_matrix}\naverage_precision_score: {average_precision_score}')

model = lgb.Booster(model_file='1.pkl')
import matplotlib.pyplot as plt
import seaborn as sns
lgb.plot_importance(model, importance_type='split', max_num_features=20, figsize=(18, 6))
plt.show()

# 创建热力图
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Predicted Negative', 'Predicted Positive'],
            yticklabels=['Actual Negative', 'Actual Positive'])

plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()



# 计算精确度和召回率
precisions, recalls, _ = precision_recall_curve(y_test, y_pred)

print(y_pred[:5])
plt.figure(figsize=(6, 4))
plt.step(recall, precision, where='pre', color='b')
plt.fill_between(recalls, precisions, alpha=0.2, color='b')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve (AP = {:.2f})'.format(average_precision_score))
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.show()

fpr, tpr, thresholds = roc_curve(y_test, y_pred)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = {:.2f})'.format(auc))
plt.fill_between(fpr, tpr, alpha=0.2, color='b')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()



# optuna调参
# import optuna
# from sklearn.metrics import roc_auc_score
# def train_model_category(trial, x_train, x_test, y_train, y_test):
#     param_grid = {
#         "n_estimators": trial.suggest_int("n_estimators", 5000, 15000, step=1000),
#         "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, step=0.02),
#         "num_leaves": trial.suggest_int("num_leaves", 2 ** 2, 2 ** 6, step=4),
#         "max_depth": trial.suggest_int("max_depth", 3, 12, step=2),
#         "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 200, 10000, step=2),
#         "lambda_l1": trial.suggest_int("lambda_l1", 0, 100, step=1),
#         "lambda_l2": trial.suggest_int("lambda_l2", 0, 100, step=1),
#         "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0, 15),
#         "bagging_fraction": trial.suggest_float("bagging_fraction", 0.2, 0.95, step=0.1),
#         "bagging_freq": trial.suggest_categorical("bagging_freq", [1]),
#         "feature_fraction": trial.suggest_float("feature_fraction", 0.2, 0.95, step=0.1),
#         "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 0.9, step=0.1),
#         "subsample": trial.suggest_float("subsample", 0.2, 1, step=0.1),
#         'cat_smooth': trial.suggest_float('cat_smooth', 0.1, 10.0),
#         'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
#         'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 1.0),
#         'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 20.0),
#     }
#     # 设置固定参数
#     param_grid.update({
#         'objective': 'binary',
#         'metric': ['binary_logloss', 'auc'],
#         'num_iterations': 1000,  # 迭代次数
#         'early_stopping_rounds': 30,
#         'random_state': 42,
#         'verbose_eval' : False,
#     })
#     model = lgb.train(param_grid, lgb.Dataset(x_train, label=y_train), valid_sets=[lgb.Dataset(x_test, label=y_test)])
#     y_pred = model.predict(x_test)
#     auc = roc_auc_score(y_test, y_pred)
#     print('Full AUC score %.6f' %auc)
#     return auc
#
#
# study = optuna.create_study(direction="maximize")
# func = lambda trial: train_model_category(trial,x_train, x_test, y_train, y_test)
# study.optimize(func, n_trials=20)
# # 获取最佳超参数配置和最大化的AUC分数
# best_params = study.best_params
# best_auc = study.best_value
#
# print("Best Hyperparameters:", best_params)
# print("Best AUC:", best_auc)
