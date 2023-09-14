import lightgbm as lgb
import pandas as pd

def data_deal(df):
    # 空值处理
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
    df_id = df['id']
    df = df.drop(['id'], axis=1)
    df = df.astype(float)  # 类型转换
    return df,df_id



df = pd.read_csv(r'1.csv',low_memory=False)
df,df_id=data_deal(df)
# 加载模型
model = lgb.Booster(model_file='SL_1.pkl')
# 预测
y_pred = model.predict(df, num_iteration=model.best_iteration)
df['sample_flag']=y_pred
df.insert(0,'id',df_id)
df.to_csv('dw_casl_user_sample_feature_result.csv',index=False)