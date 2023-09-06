import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv(r'C:\Users\Polo\Desktop\dw_casl_user_sample_feature.csv')

# 空值处理
df = df.drop(['msisdn', 'imsi'], axis=1)  # 删除加密的特征
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
df['dome_roam_flux_days'].fillna(int(df[df['dome_roam_flux_days'].notnull()]['dome_roam_flux_days'].astype(int).mean()),
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
df = df.astype(int)  # 类型转换

# 训练集&测试集创建
x = df.drop('sample_flag', axis=1)  # 特征数据
y = df['sample_flag']  # 目标数据
# 将数据拆分为训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 创建LightGBM数据集
train_data = lgb.Dataset(x_train, label=y_train)
test_data = lgb.Dataset(x_test, label=y_test, reference=train_data)

# 训练模型&评分标准
from sklearn.metrics import roc_auc_score

# 设置LightGBM参数
params = {
    'cat_smooth': 8,
    'is_unbalance': True,
    'max_depth': 9,
    'metric': ['binary_logloss', 'auc'],
    'min_child_samples': 22,
    'num_iterations': 200,
    'num_leaves': 100,
    'objective': 'binary',
    'reg_alpha': 0.01,
    'reg_lambda': 8,
    'early_stopping_rounds': 50
}

# 训练模型
model = lgb.train(params, train_data, valid_sets=[test_data])

print(model.best_iteration)
model.save_model('1.pkl')
# 预测
y_pred = model.predict(x_test, num_iteration=model.best_iteration)

# 评估模型性能
auc = roc_auc_score(y_test, y_pred)

print(f'AUC: {auc}')
