import pandas as pd
import numpy as np

from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler

NAME_CONTRACT_TYPE_MAP={'Cash loans':0, 'Revolving loans':1}
CODE_GENDER_MAP={'F':0, 'M':1, 'XNA': np.nan}
FLAG_OWN_XXX_MAP={'N':0, 'Y':1}
EMERGENCYSTATE_MODE_MAP={'No':0, 'Yes':1}

col_numeric =['CNT_CHILDREN', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY',
              'AMT_GOODS_PRICE', 'REGION_POPULATION_RELATIVE',
              'DAYS_BIRTH', 'DAYS_EMPLOYED', 'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH',
              'OWN_CAR_AGE', 'CNT_FAM_MEMBERS', 'EXT_SOURCE_2', 'EXT_SOURCE_3',
              'APARTMENTS_AVG', 'BASEMENTAREA_AVG', 'YEARS_BEGINEXPLUATATION_AVG',
              'YEARS_BUILD_AVG', 'COMMONAREA_AVG', 'ELEVATORS_AVG', 'ENTRANCES_AVG',
              'FLOORSMAX_AVG', 'FLOORSMIN_AVG', 'LANDAREA_AVG', 'LIVINGAPARTMENTS_AVG',
              'LIVINGAREA_AVG', 'NONLIVINGAPARTMENTS_AVG', 'NONLIVINGAREA_AVG',
              'APARTMENTS_MODE', 'BASEMENTAREA_MODE', 'YEARS_BEGINEXPLUATATION_MODE',
              'YEARS_BUILD_MODE', 'COMMONAREA_MODE', 'ELEVATORS_MODE', 'ENTRANCES_MODE',
              'FLOORSMAX_MODE', 'FLOORSMIN_MODE', 'LANDAREA_MODE', 'LIVINGAPARTMENTS_MODE',
              'LIVINGAREA_MODE', 'NONLIVINGAPARTMENTS_MODE', 'NONLIVINGAREA_MODE',
              'APARTMENTS_MEDI', 'BASEMENTAREA_MEDI', 'YEARS_BEGINEXPLUATATION_MEDI',
              'YEARS_BUILD_MEDI', 'COMMONAREA_MEDI', 'ELEVATORS_MEDI', 'ENTRANCES_MEDI',
              'FLOORSMAX_MEDI', 'FLOORSMIN_MEDI', 'LANDAREA_MEDI', 'LIVINGAPARTMENTS_MEDI',
              'LIVINGAREA_MEDI', 'NONLIVINGAPARTMENTS_MEDI', 'NONLIVINGAREA_MEDI',
              'TOTALAREA_MODE', 'OBS_30_CNT_SOCIAL_CIRCLE', 'DEF_30_CNT_SOCIAL_CIRCLE',
              'OBS_60_CNT_SOCIAL_CIRCLE', 'DEF_60_CNT_SOCIAL_CIRCLE',
              'DAYS_LAST_PHONE_CHANGE', 'AMT_REQ_CREDIT_BUREAU_HOUR',
              'AMT_REQ_CREDIT_BUREAU_DAY', 'AMT_REQ_CREDIT_BUREAU_WEEK',
              'AMT_REQ_CREDIT_BUREAU_MON', 'AMT_REQ_CREDIT_BUREAU_QRT',
              'AMT_REQ_CREDIT_BUREAU_YEAR']

col_category=['NAME_TYPE_SUITE', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE',
              'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE',
              'REGION_RATING_CLIENT', 'REGION_RATING_CLIENT_W_CITY',
              'WEEKDAY_APPR_PROCESS_START', 'HOUR_APPR_PROCESS_START',
              'ORGANIZATION_TYPE', 'FONDKAPREMONT_MODE', 'HOUSETYPE_MODE',
              'WALLSMATERIAL_MODE']

col_flag =['FLAG_MOBIL', 'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE',
           'FLAG_PHONE', 'FLAG_EMAIL', 'REG_REGION_NOT_LIVE_REGION',
           'REG_REGION_NOT_WORK_REGION', 'LIVE_REGION_NOT_WORK_REGION',
           'REG_CITY_NOT_LIVE_CITY', 'REG_CITY_NOT_WORK_CITY',
           'LIVE_CITY_NOT_WORK_CITY', 'FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_3',
           'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_6', 'FLAG_DOCUMENT_7',
           'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_9', 'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11',
           'FLAG_DOCUMENT_12', 'FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15',
           'FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_18', 'FLAG_DOCUMENT_19',
           'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21']

def bureau_preprocessing():
    """
    bureau.csvの加工用
    """

    # load dataset
    df_bureau = pd.read_csv('bureau.csv')

    # IDごとの平均値を集計
    df_bureau = df_bureau.groupby('SK_ID_CURR').mean()

    # ID列を削除
    df_bureau = df_bureau.drop('SK_ID_BUREAU', axis=1)

    # カラム名を変更
    df_bureau.columns = ['bureau_' + c for c in df_bureau.columns]

    return df_bureau

def main():
    # 元データをロード
    df_train = pd.read_csv('application_train.csv')
    df_test = pd.read_csv('application_test.csv')

    df_train['IS_TEST']=False
    df_test['IS_TEST']=True
    df_test['TARGET'] = np.nan

    df_raw=pd.concat([df_train, df_test])
    df_raw.index = df_raw.SK_ID_CURR

    # 種類別にデータを分割
    df_numeric = df_raw[col_numeric + ['SK_ID_CURR']]
    df_category = df_raw[col_category]
    df_flag = df_raw[col_flag]

    # bureauデータと結合
    df_bureau=bureau_preprocessing()
    df_numeric = df_numeric.merge(right=df_bureau.reset_index(), how='left', on='SK_ID_CURR')
    df_numeric.index = df_numeric['SK_ID_CURR']
    df_numeric = df_numeric.drop('SK_ID_CURR', axis=1)

    # 数値形式のデータの欠損値を平均で補間
    df_numeric = df_numeric.fillna(df_numeric.mean())

    # 標準化
    stdsc = StandardScaler()
    df_numeric = pd.DataFrame(stdsc.fit_transform(df_numeric), columns=df_numeric.columns, index=df_numeric.index)

    # StandardScalerを保存
    joblib.dump(stdsc, 'standardscaler.pkl')

    # ダミー変数の生成
    df_category = pd.get_dummies(df_category, drop_first=True)

    # データを結合
    df = pd.concat([df_numeric, df_category, df_flag], axis=1)

    # 二値データの項目を数値へ変換
    df['NAME_CONTRACT_TYPE'] = df_raw.NAME_CONTRACT_TYPE.map(NAME_CONTRACT_TYPE_MAP).fillna(0)
    df['CODE_GENDER'] = df_raw.CODE_GENDER.map(CODE_GENDER_MAP).fillna(0)
    df['FLAG_OWN_CAR'] = df_raw.FLAG_OWN_CAR.map(FLAG_OWN_XXX_MAP).fillna(0)
    df['FLAG_OWN_REALTY'] = df_raw.FLAG_OWN_REALTY.map(FLAG_OWN_XXX_MAP).fillna(0)
    df['EMERGENCYSTATE_MODE'] = df_raw.EMERGENCYSTATE_MODE.map(EMERGENCYSTATE_MODE_MAP).fillna(0)

    df['label']=df_raw['TARGET']
    df['IS_TEST']=df_raw['IS_TEST']

    # split train & test data
    df_train = df[df['IS_TEST']==False]
    df_test = df[df['IS_TEST']==True]

    # save data
    df_train.to_hdf('db.h5', key='train')
    df_test.to_hdf('db.h5', key='test')
    df.to_hdf('db.h5', key='all')

    return df

if __name__ == '__main__':
    main()
