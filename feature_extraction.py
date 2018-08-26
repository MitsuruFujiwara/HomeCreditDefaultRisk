"""
各モデルで共通して使う特徴量抽出のスクリプトを分離しました。
"""

import pandas as pd
import numpy as np
import gc
import math
import time

from contextlib import contextmanager

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

# One-hot encoding for categorical columns with get_dummies
def one_hot_encoder(df, nan_as_category = True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns

def get_amt_factor(df,num_rows = None):
    tgt_cols = ['SK_ID_CURR','AMT_ANNUITY','AMT_CREDIT','AMT_GOODS_PRICE']
    tgt_cols_without = ['AMT_ANNUITY','AMT_CREDIT','AMT_GOODS_PRICE']

    data = df.loc[:,tgt_cols]
    prev = pd.read_csv('previous_application.csv', nrows = num_rows)

    temp = prev.loc[:,tgt_cols].groupby(['SK_ID_CURR'])
    grouped_prev_avg = temp.mean()
    grouped_prev_avg.columns = list(map(lambda x:x+"_AVG" ,tgt_cols_without))
    grouped_prev_avg = grouped_prev_avg.reset_index()
    grouped_prev_std = temp.std()
    grouped_prev_std.columns = list(map(lambda x:x+"_STD" ,tgt_cols_without))
    grouped_prev_std = grouped_prev_std.reset_index()

    data = pd.merge(data, grouped_prev_avg, on = 'SK_ID_CURR', how = "left")
    data = pd.merge(data, grouped_prev_std, on = 'SK_ID_CURR', how = "left")

    for col in tgt_cols_without:
        if math.isnan(np.nanmean(data.loc[:,col+"_STD"]/data.loc[:,col+"_AVG"])):
            std_avg_ratio = 0.5
        else :
            std_avg_ratio = np.nanmean(data.loc[:,col+"_STD"]/data.loc[:,col+"_AVG"])

        tgt_index = data[col+"_AVG"].isnull().index
        data.loc[tgt_index,col+"_AVG"] = data.loc[tgt_index,col]
        tgt_index = data[col+"_STD"].isnull().index
        data.loc[tgt_index,col+"_STD"] = data.loc[tgt_index,col+"_AVG"]*std_avg_ratio
        data.loc[:,col+'_FACTOR']= (data.loc[:,col]-data.loc[:,col+"_AVG"])/data.loc[:,col+"_STD"]

    output = data.loc[:,['SK_ID_CURR','AMT_ANNUITY_FACTOR','AMT_CREDIT_FACTOR','AMT_GOODS_PRICE_FACTOR']]
    output = output.set_index(['SK_ID_CURR'])
    return output

def getMomentumFactors(data, feat, key_agg, key_prim):
    """
    IDごとの平均変化幅、変化率、最終トランザクションの変化幅、変化率を返す関数

    input:
        data: テーブルのDataFrame
        feat: カラム名（string）
        key_agg: 集計用のキー(SK_ID_CURR)
        key_prim: 各テーブルの主キー(SK_ID_BUREAUなど)

    output:
        diff_mean: key_aggごとの変化幅の平均
        pct_mean: key_aggごとの変化率の平均
        diff_fin: key_aggごとの最終トランザクションの変化幅
        pct_fin: key_aggごとの最終トランザクションの変化率
    """

    # IDごとにdiffとpct changeを集計
    data['tmp_diff'] = data.groupby(key_agg)[feat].transform(pd.Series.diff)
    data['tmp_pct'] = data.groupby(key_agg)[feat].transform(pd.Series.pct_change)

    # 全トランザクションの平均
    diff_mean = data.groupby(key_agg)['tmp_diff'].mean()
    pct_mean = data.groupby(key_agg)['tmp_pct'].mean()

    # 各IDの最終トランザクションIDを取得
    id_fin = data.groupby(key_agg)[key_prim].max().tolist()

    # 最終トランザクションの変化幅と変化率
    diff_fin = data[data[key_prim].isin(id_fin)].groupby(key_agg)['tmp_diff'].mean()
    pct_fin = data[data[key_prim].isin(id_fin)].groupby(key_agg)['tmp_pct'].mean()

    del data

    return (diff_mean, pct_mean, diff_fin, pct_fin)

# correlation高い変数を削除する機能
def removeCorrelatedVariables(data, threshold):
    corr_matrix = data.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    col_drop = [column for column in upper.columns if any(upper[column] > threshold) & ('TARGET' not in column)]
    return col_drop

# 欠損値の率が高い変数を削除する機能
def removeMissingVariables(data, threshold):
    missing = (data.isnull().sum() / len(data)).sort_values(ascending = False)
    col_missing = missing.index[missing > 0.75]
    col_missing = [column for column in col_missing if 'TARGET' not in column]
    return col_missing

# Preprocess application_train.csv and application_test.csv
def application_train_test(num_rows = None, nan_as_category = False):
    # Read data and merge
    df = pd.read_csv('application_train.csv', nrows= num_rows)
    test_df = pd.read_csv('application_test.csv', nrows= num_rows)
    print("Train samples: {}, test samples: {}".format(len(df), len(test_df)))
    df = df.append(test_df).reset_index()
    # Optional: Remove 4 applications with XNA CODE_GENDER (train set)
    df = df[df['CODE_GENDER'] != 'XNA']

    docs = [_f for _f in df.columns if 'FLAG_DOC' in _f]
    live = [_f for _f in df.columns if ('FLAG_' in _f) & ('FLAG_DOC' not in _f) & ('_FLAG_' not in _f)]

    # NaN values for DAYS_EMPLOYED: 365.243 -> nan
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace= True)

    # 追加の前処理 https://github.com/neptune-ml/open-solution-home-credit/wiki/LightGBM-with-smarter-features
    df['NAME_FAMILY_STATUS'].replace('Unknown', np.nan, inplace=True)
    df['DAYS_LAST_PHONE_CHANGE'].replace(0, np.nan, inplace=True)
    df['ORGANIZATION_TYPE'].replace('XNA', np.nan, inplace=True)

    inc_by_org = df[['AMT_INCOME_TOTAL', 'ORGANIZATION_TYPE']].groupby('ORGANIZATION_TYPE').median()['AMT_INCOME_TOTAL']

    df['NEW_CREDIT_TO_ANNUITY_RATIO'] = df['AMT_CREDIT'] / df['AMT_ANNUITY']
    df['NEW_CREDIT_TO_GOODS_RATIO'] = df['AMT_CREDIT'] / df['AMT_GOODS_PRICE']
    df['NEW_DOC_IND_KURT'] = df[docs].kurtosis(axis=1)
    df['NEW_LIVE_IND_SUM'] = df[live].sum(axis=1)
    df['NEW_INC_PER_CHLD'] = df['AMT_INCOME_TOTAL'] / (1 + df['CNT_CHILDREN'])
    df['NEW_INC_BY_ORG'] = df['ORGANIZATION_TYPE'].map(inc_by_org)
    df['NEW_EMPLOY_TO_BIRTH_RATIO'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['NEW_ANNUITY_TO_INCOME_RATIO'] = df['AMT_ANNUITY'] / (1 + df['AMT_INCOME_TOTAL'])
    df['NEW_SOURCES_PROD'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']
    df['NEW_EXT_SOURCES_MEAN'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
    df['NEW_SCORES_STD'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].std(axis=1)
    df['NEW_SCORES_STD'] = df['NEW_SCORES_STD'].fillna(df['NEW_SCORES_STD'].mean())
    df['NEW_CAR_TO_BIRTH_RATIO'] = df['OWN_CAR_AGE'] / df['DAYS_BIRTH']
    df['NEW_CAR_TO_EMPLOY_RATIO'] = df['OWN_CAR_AGE'] / df['DAYS_EMPLOYED']
    df['NEW_PHONE_TO_BIRTH_RATIO'] = df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_BIRTH']
    df['NEW_PHONE_TO_BIRTH_RATIO'] = df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_EMPLOYED']
    df['NEW_CREDIT_TO_INCOME_RATIO'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']

    # ここから追加したやつ
    df['NEW_EXT_SOURCE_RATIO12'] = df['EXT_SOURCE_1'] / df['EXT_SOURCE_2']
    df['NEW_EXT_SOURCE_RATIO23'] = df['EXT_SOURCE_2'] / df['EXT_SOURCE_3']
    df['NEW_EXT_SOURCE_RATIO31'] = df['EXT_SOURCE_3'] / df['EXT_SOURCE_1']
    df['NEW_INCOME_TO_BIRTH_RATIO'] = df['AMT_INCOME_TOTAL'] / df['DAYS_BIRTH']
    df['NEW_ANNUITY_TO_BIRTH_RATIO'] = df['AMT_ANNUITY'] / df['DAYS_BIRTH']
    df['NEW_CREDIT_TO_BIRTH_RATIO'] = df['AMT_CREDIT'] / df['DAYS_BIRTH']
    df['NEW_EXT_SOURCES_MEAN12'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2']].mean(axis=1)
    df['NEW_EXT_SOURCES_MEAN23'] = df[['EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
    df['NEW_EXT_SOURCES_MEAN31'] = df[['EXT_SOURCE_3', 'EXT_SOURCE_1']].mean(axis=1)
    df['NEW_SOURCES_PROD12'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2']
    df['NEW_SOURCES_PROD23'] = df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']
    df['NEW_SOURCES_PROD31'] = df['EXT_SOURCE_3'] * df['EXT_SOURCE_1']

    # 更に追加したやつ　参考 https://github.com/neptune-ml/open-solution-home-credit/wiki/LightGBM-on-selected-features
    df['NEW_CHILDREN_RATIO'] = df['CNT_CHILDREN'] / df['CNT_FAM_MEMBERS']
    df['NEW_INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    df['NEW_PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT'] # これ被ってる気がしますが一応入れときます
    df['NEW_EXT_SOURCE_WEIGHTED'] = df['EXT_SOURCE_1']*2 + df['EXT_SOURCE_2'] * 3 + df['EXT_SOURCE_3'] * 4
    df['NEW_CNT_NON_CHILD'] = df['CNT_FAM_MEMBERS'] - df['CNT_CHILDREN']
    df['NEW_CHILD_TO_NON_CHILD_RATIO'] = df['CNT_CHILDREN'] / df['NEW_CNT_NON_CHILD']
    df['NEW_INCOME_PER_NON_CHILD'] = df['AMT_INCOME_TOTAL'] / df['NEW_CNT_NON_CHILD']
    df['NEW_CREDIT_PER_PERSON'] = df['AMT_CREDIT'] / df['CNT_FAM_MEMBERS']
    df['NEW_CREDIT_PER_CHILD'] = df['AMT_CREDIT'] / (1 + df['CNT_CHILDREN'])
    df['NEW_CREDIT_PER_NON_CHILD'] = df['AMT_CREDIT'] / df['NEW_CNT_NON_CHILD']
    df['NEW_YOUNG_AGE'] = (df['DAYS_BIRTH'] < -14000).astype(int)
    df['NEW_SHORT_EMPLOYMENT'] = (df['DAYS_EMPLOYED'] < -2000).astype(int)

    # EXT_SOURCE系大事っぽいので組み合わせ追加します by藤原
    df['NEW_SOURCES_MIN'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].min(axis=1)
    df['NEW_SOURCES_MIN12'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2']].min(axis=1)
    df['NEW_SOURCES_MIN23'] = df[['EXT_SOURCE_2', 'EXT_SOURCE_3']].min(axis=1)
    df['NEW_SOURCES_MIN31'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_3']].min(axis=1)
    df['NEW_SOURCES_MAX'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].max(axis=1)
    df['NEW_SOURCES_MAX12'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2']].max(axis=1)
    df['NEW_SOURCES_MAX23'] = df[['EXT_SOURCE_2', 'EXT_SOURCE_3']].max(axis=1)
    df['NEW_SOURCES_MAX31'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_3']].max(axis=1)
    df['NEW_SOURCES_SUM'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].sum(axis=1)
    df['NEW_SOURCES_SUM12'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2']].sum(axis=1)
    df['NEW_SOURCES_SUM23'] = df[['EXT_SOURCE_2', 'EXT_SOURCE_3']].sum(axis=1)
    df['NEW_SOURCES_SUM31'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_3']].sum(axis=1)
    df['NEW_SOURCES_MEDIAN'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].median(axis=1)
    df['NEW_SOURCES_MEDIAN12'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2']].median(axis=1)
    df['NEW_SOURCES_MEDIAN23'] = df[['EXT_SOURCE_2', 'EXT_SOURCE_3']].median(axis=1)
    df['NEW_SOURCES_MEDIAN31'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_3']].median(axis=1)

    # Polynomial Featuresを追加します。　参考： https://www.kaggle.com/willkoehrsen/start-here-a-gentle-introduction
    df['NEW_EXT_SOURCES_BIRTH_MEAN'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']].mean(axis=1)
    df['NEW_SOURCE_1^2'] = df['EXT_SOURCE_1']**2
    df['NEW_SOURCE_2^2'] = df['EXT_SOURCE_2']**2
    df['NEW_SOURCE_3^2'] = df['EXT_SOURCE_3']**2
    df['NEW_SOURCE_1^3'] = df['EXT_SOURCE_1']**3
    df['NEW_SOURCE_2^3'] = df['EXT_SOURCE_2']**3
    df['NEW_SOURCE_3^3'] = df['EXT_SOURCE_3']**3
    df['NEW_SOURCE_PROD_1_2^2'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2']**2
    df['NEW_SOURCE_PROD_1_3^2'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_3']**2
    df['NEW_SOURCE_PROD_1_B^2'] = df['EXT_SOURCE_1'] * df['DAYS_BIRTH']**2
    df['NEW_SOURCE_PROD_2_1^2'] = df['EXT_SOURCE_2'] * df['EXT_SOURCE_1']**2
    df['NEW_SOURCE_PROD_2_3^2'] = df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']**2
    df['NEW_SOURCE_PROD_2_B^2'] = df['EXT_SOURCE_2'] * df['DAYS_BIRTH']**2
    df['NEW_SOURCE_PROD_3_1^2'] = df['EXT_SOURCE_3'] * df['EXT_SOURCE_1']**2
    df['NEW_SOURCE_PROD_3_2^2'] = df['EXT_SOURCE_3'] * df['EXT_SOURCE_2']**2
    df['NEW_SOURCE_PROD_3_B^2'] = df['EXT_SOURCE_3'] * df['DAYS_BIRTH']**2
    df['NEW_SOURCE_PROD_B_1^2'] = df['DAYS_BIRTH'] * df['EXT_SOURCE_1']**2
    df['NEW_SOURCE_PROD_B_2^2'] = df['DAYS_BIRTH'] * df['EXT_SOURCE_2']**2
    df['NEW_SOURCE_PROD_B_3^2'] = df['DAYS_BIRTH'] * df['EXT_SOURCE_3']**2

    df['NEW_SOURCES_PROD1_BIRTH'] = df['EXT_SOURCE_1'] * df['DAYS_BIRTH']
    df['NEW_SOURCES_PROD2_BIRTH'] = df['EXT_SOURCE_2'] * df['DAYS_BIRTH']
    df['NEW_SOURCES_PROD3_BIRTH'] = df['EXT_SOURCE_3'] * df['DAYS_BIRTH']

    df['NEW_SOURCES_PROD12_BIRTH'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2'] * df['DAYS_BIRTH']
    df['NEW_SOURCES_PROD23_BIRTH'] = df['EXT_SOURCE_2'] * df['EXT_SOURCE_3'] * df['DAYS_BIRTH']
    df['NEW_SOURCES_PROD31_BIRTH'] = df['EXT_SOURCE_3'] * df['EXT_SOURCE_1'] * df['DAYS_BIRTH']

    df['NEW_SOURCE1_BIRTH_RATIO'] = df['EXT_SOURCE_1'] / df['DAYS_BIRTH']
    df['NEW_SOURCE2_BIRTH_RATIO'] = df['EXT_SOURCE_2'] / df['DAYS_BIRTH']
    df['NEW_SOURCE3_BIRTH_RATIO'] = df['EXT_SOURCE_3'] / df['DAYS_BIRTH']

    # さいつよっぽいやつの組み合わせ
    df['NEW_CREDIT_ANNUITY_SOURCES_RATIO'] = df['NEW_CREDIT_TO_ANNUITY_RATIO'] / df['NEW_EXT_SOURCES_MEAN']

    # Categorical features with Binary encode (0 or 1; two categories)
    for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
        df[bin_feature], uniques = pd.factorize(df[bin_feature])
    # Categorical features with One-Hot encode
    df, cat_cols = one_hot_encoder(df, nan_as_category)
    dropcolum=['FLAG_DOCUMENT_2','FLAG_DOCUMENT_4',
    'FLAG_DOCUMENT_5','FLAG_DOCUMENT_6','FLAG_DOCUMENT_7',
    'FLAG_DOCUMENT_8','FLAG_DOCUMENT_9','FLAG_DOCUMENT_10',
    'FLAG_DOCUMENT_11','FLAG_DOCUMENT_12','FLAG_DOCUMENT_13',
    'FLAG_DOCUMENT_14','FLAG_DOCUMENT_15','FLAG_DOCUMENT_16',
    'FLAG_DOCUMENT_17','FLAG_DOCUMENT_18','FLAG_DOCUMENT_19',
    'FLAG_DOCUMENT_20','FLAG_DOCUMENT_21']
    df= df.drop(dropcolum,axis=1)

    del test_df
    gc.collect()
    return df

# Preprocess bureau.csv and bureau_balance.csv
def bureau_and_balance(num_rows = None, nan_as_category = True):
    bureau = pd.read_csv('bureau.csv', nrows = num_rows)
    bb = pd.read_csv('bureau_balance.csv', nrows = num_rows)

    # 追加の特徴量　https://github.com/neptune-ml/open-solution-home-credit/blob/master/src/feature_extraction.py
    bureau['CREDIT_ACTIVE_BINARY'] = (bureau['CREDIT_ACTIVE'] != 'Closed').astype(int)
    bureau['CREDIT_ENDDATE_BINARY'] = (bureau['DAYS_CREDIT_ENDDATE'] > 0).astype(int)

    bb, bb_cat = one_hot_encoder(bb, nan_as_category)
    bureau, bureau_cat = one_hot_encoder(bureau, nan_as_category)


    # Bureau balance: Perform aggregations and merge with bureau.csv
    bb_aggregations = {'MONTHS_BALANCE': ['mean', 'min', 'max', 'size']}
    for col in bb_cat:
        bb_aggregations[col] = ['mean', 'max', 'min']

    bb_agg = bb.groupby('SK_ID_BUREAU').agg(bb_aggregations)

    # bbテーブルのモメンタムファクターを追加 # diffは意味なさそうなんでpct_changeのみ追加
    momentom_factor_bb = getMomentumFactors(bb, 'MONTHS_BALANCE', 'SK_ID_BUREAU', 'SK_ID_BUREAU')
    bb_agg['BURO_MONTHS_BALANCE_PCT_MEAN'] = momentom_factor_bb[1]

    bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])
    bureau = bureau.join(bb_agg, how='left', on='SK_ID_BUREAU')
    del bb, bb_agg
    gc.collect()

    # 追加の前処理 https://github.com/neptune-ml/open-solution-home-credit/wiki/LightGBM-with-smarter-features
    bureau['AMT_CREDIT_SUM'].fillna(0, inplace=True)
    bureau['AMT_CREDIT_SUM_DEBT'].fillna(0, inplace=True)
    bureau['AMT_CREDIT_SUM_OVERDUE'].fillna(0, inplace=True)
    bureau['CNT_CREDIT_PROLONG'].fillna(0, inplace=True)
    #bureau['DAYS_CREDIT_ENDDATE'][bureau['DAYS_CREDIT_ENDDATE'] < -40000] = np.nan
    ix = bureau[bureau['DAYS_CREDIT_ENDDATE'] < -40000].index
    bureau.loc[ix,'DAYS_CREDIT_ENDDATE'] = np.nan
    #bureau['DAYS_CREDIT_UPDATE'][bureau['DAYS_CREDIT_UPDATE'] < -40000] = np.nan
    ix = bureau[bureau['DAYS_CREDIT_UPDATE'] < -40000].index
    bureau.loc[ix,'DAYS_CREDIT_UPDATE'] = np.nan
    #bureau['DAYS_ENDDATE_FACT'][bureau['DAYS_ENDDATE_FACT'] < -40000] = np.nan
    ix = bureau[bureau['DAYS_ENDDATE_FACT'] < -40000].index
    bureau.loc[ix,'DAYS_ENDDATE_FACT'] = np.nan

    # 追加します　→表記分かりやすいように変更しましたby藤原
    bureau['CREDIT_SUM_TO_DEBT_RATIO'] = bureau['AMT_CREDIT_SUM']/bureau['AMT_CREDIT_SUM_DEBT']
    bureau['CREDIT_SUM_TO_LIMIT_RATIO'] = bureau['AMT_CREDIT_SUM']/bureau['AMT_CREDIT_SUM_LIMIT']
    bureau['CREDIT_SUM_TO_OVERDUE_RATIO'] = bureau['AMT_CREDIT_SUM']/bureau['AMT_CREDIT_SUM_OVERDUE']
    bureau['CREDIT_SUM_TO_MAX_OVERDUE_RATIO'] = bureau['AMT_CREDIT_SUM']/bureau['AMT_CREDIT_MAX_OVERDUE']
    bureau['CREDIT_SUM_TO_ANNUITY_RATIO'] = bureau['AMT_CREDIT_SUM']/bureau['AMT_ANNUITY']

    # DAYS_CREDITを起点にした特徴量
    bureau['MAX_OVERDUE_TO_DAYS_CREDIT_RATIO'] = bureau['AMT_CREDIT_MAX_OVERDUE']/bureau['DAYS_CREDIT']
    bureau['DAY_OVERDUE_TO_DAYS_CREDIT_RATIO'] = bureau['CREDIT_DAY_OVERDUE']/bureau['DAYS_CREDIT']
    bureau['ENDDATE_TO_DAYS_CREDIT_RATIO'] = bureau['DAYS_CREDIT_ENDDATE']/bureau['DAYS_CREDIT']
    bureau['ENDDATE_FACT_TO_DAYS_CREDIT_RATIO'] = bureau['DAYS_ENDDATE_FACT']/bureau['DAYS_CREDIT']
    bureau['UPDATE_TO_DAYS_CREDIT_RATIO'] = bureau['DAYS_CREDIT_UPDATE']/bureau['DAYS_CREDIT']

    # Bureau and bureau_balance numeric features
    num_aggregations = {
        'DAYS_CREDIT': ['mean', 'var', 'max', 'min', 'skew'], # 強そうな特徴量にはskewも追加する方針です。
        'DAYS_CREDIT_ENDDATE': ['mean', 'max', 'min'],
        'DAYS_CREDIT_UPDATE': ['mean', 'max', 'min'],
        'CREDIT_DAY_OVERDUE': ['mean', 'max', 'min'],
        'AMT_CREDIT_MAX_OVERDUE': ['mean', 'var', 'max', 'min', 'skew'], # 強そうな特徴量にはskewも追加する方針です。
        'AMT_CREDIT_SUM': ['mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
        'AMT_ANNUITY': ['max', 'mean'],
#        'CNT_CREDIT_PROLONG': ['sum'],
        'MONTHS_BALANCE_MIN': ['mean', 'min'],
        'MONTHS_BALANCE_MAX': ['mean', 'max'],
        'MONTHS_BALANCE_SIZE': ['mean', 'sum'],
        'CREDIT_SUM_TO_DEBT_RATIO':['mean','max','min'],
        'CREDIT_SUM_TO_LIMIT_RATIO':['mean','max','min'],
        'CREDIT_SUM_TO_OVERDUE_RATIO':['mean','max','min'],
        'CREDIT_SUM_TO_MAX_OVERDUE_RATIO':['mean', 'min'],
        'CREDIT_SUM_TO_ANNUITY_RATIO':['mean','var','max','min'],
        'MAX_OVERDUE_TO_DAYS_CREDIT_RATIO':['mean','var','max','min','skew'],
        'DAY_OVERDUE_TO_DAYS_CREDIT_RATIO':['var','skew'],
        'ENDDATE_TO_DAYS_CREDIT_RATIO':['mean','var','max','min','skew'],
        'ENDDATE_FACT_TO_DAYS_CREDIT_RATIO':['mean','var','max','min','skew'],
        'UPDATE_TO_DAYS_CREDIT_RATIO':['mean','var','max','min','skew'],
        'CREDIT_ACTIVE_BINARY':['mean'],
        'CREDIT_ENDDATE_BINARY':['mean']
    }
    # Bureau and bureau_balance categorical features
    cat_aggregations = {}
    for cat in bureau_cat: cat_aggregations[cat] = ['mean']
    for cat in bb_cat: cat_aggregations[cat + "_MEAN"] = ['mean']

    bureau_agg = bureau.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    bureau_agg.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])
    # Bureau: Active credits - using only numerical aggregations
    active = bureau[bureau['CREDIT_ACTIVE_Active'] == 1]
    active_agg = active.groupby('SK_ID_CURR').agg(num_aggregations)
    active_agg.columns = pd.Index(['ACTIVE_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(active_agg, how='left', on='SK_ID_CURR')
    del active, active_agg
    gc.collect()
    # Bureau: Closed credits - using only numerical aggregations
    closed = bureau[bureau['CREDIT_ACTIVE_Closed'] == 1]
    closed_agg = closed.groupby('SK_ID_CURR').agg(num_aggregations)
    closed_agg.columns = pd.Index(['CLOSED_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(closed_agg, how='left', on='SK_ID_CURR')

    # 新たに追加した変数です
    bureau_agg['BURO_CREDIT_ACTIVE_CLOSED_RATIO']=bureau_agg['BURO_CREDIT_ACTIVE_Active_MEAN']/bureau_agg['BURO_CREDIT_ACTIVE_Closed_MEAN']

    # モメンタムファクターの追加
    feats_mom_buro = ['DAYS_CREDIT', 'CREDIT_DAY_OVERDUE', 'DAYS_CREDIT_ENDDATE',
                      'DAYS_ENDDATE_FACT', 'AMT_CREDIT_MAX_OVERDUE', 'CNT_CREDIT_PROLONG',
                      'AMT_CREDIT_SUM', 'AMT_CREDIT_SUM_DEBT', 'AMT_CREDIT_SUM_LIMIT',
                      'AMT_CREDIT_SUM_OVERDUE', 'DAYS_CREDIT_UPDATE', 'AMT_ANNUITY']

    # bureauテーブルのモメンタムファクターを追加
    for f in feats_mom_buro:
        momentom_factor_buro = getMomentumFactors(bureau, f, 'SK_ID_CURR', 'SK_ID_BUREAU')
        bureau_agg['BURO_'+f+'_DIFF_MEAN'] = momentom_factor_buro[0]
        bureau_agg['BURO_'+f+'_PCT_MEAN'] = momentom_factor_buro[1]
        bureau_agg['BURO_'+f+'_DIFF_FIN'] = momentom_factor_buro[2]
        bureau_agg['BURO_'+f+'_PCT_FIN'] = momentom_factor_buro[3]
        del momentom_factor_buro
    bureau.drop(['SK_ID_BUREAU'], axis=1, inplace= True)
    del closed, closed_agg, bureau
    gc.collect()
    return bureau_agg

# Preprocess previous_applications.csv
def previous_applications(num_rows = None, nan_as_category = True):
    prev = pd.read_csv('previous_application.csv', nrows = num_rows)
    prev, cat_cols = one_hot_encoder(prev, nan_as_category= True)
    # Days 365.243 values -> nan
    prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace= True)
    prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace= True)
    prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace= True)
    prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace= True)
    prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace= True)

    # fill nan as zero
    prev['RATE_DOWN_PAYMENT']=prev['RATE_DOWN_PAYMENT'].fillna(0)

    # Add feature: value ask / value received percentage
    prev['APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']

    # ここから新規に追加した特徴量
    prev['CREDIT_TO_ANNUITY_RATIO'] = prev['AMT_CREDIT'] / prev['AMT_ANNUITY']
    prev['CREDIT_TO_GOODS_RATIO'] = prev['AMT_CREDIT'] / prev['AMT_GOODS_PRICE']
    prev['CREDIT_PERC_TO_ANNUITY_RATIO'] = prev['APP_CREDIT_PERC'] / prev['AMT_ANNUITY']
    prev['CREDIT_PERC_TO_GOODS_RATIO'] = prev['APP_CREDIT_PERC'] / prev['AMT_GOODS_PRICE']
    prev['ANNUITY_TO_CNT_PAYMENT_RATIO'] = prev['AMT_ANNUITY'] / prev['CNT_PAYMENT']
    prev['APPLICATION_TO_CNT_PAYMENT_RATIO'] = prev['AMT_APPLICATION'] / prev['CNT_PAYMENT']
    prev['CREDIT_TO_CNT_PAYMENT_RATIO'] = prev['AMT_CREDIT'] / prev['CNT_PAYMENT']
    prev['DOWN_PAYMENT_TO_CNT_PAYMENT_RATIO'] = prev['AMT_DOWN_PAYMENT'] / prev['CNT_PAYMENT']
    prev['GOODS_PRICE_TO_CNT_PAYMENT_RATIO'] = prev['AMT_GOODS_PRICE'] / prev['CNT_PAYMENT']
    prev['ANNUITY_TO_DAYS_DECISION_RATIO'] = prev['AMT_ANNUITY'] / prev['DAYS_DECISION']
    prev['CREDIT_TO_DAYS_DECISION_RATIO'] = prev['AMT_CREDIT'] / prev['DAYS_DECISION']
    prev['CREDIT_PERC_TO_DAYS_DECISION_RATIO'] = prev['APP_CREDIT_PERC'] / prev['DAYS_DECISION']
    prev['GOODS_TO_DAYS_DECISION_RATIO'] = prev['AMT_GOODS_PRICE'] / prev['DAYS_DECISION']

    # Previous applications numeric features
    num_aggregations = {
        'AMT_ANNUITY': [ 'max', 'mean'],
        'AMT_APPLICATION': [ 'max','mean'],
        'AMT_CREDIT': [ 'max', 'mean'],
        'APP_CREDIT_PERC': [ 'max', 'mean'],
        'AMT_DOWN_PAYMENT': [ 'max', 'mean'],
        'AMT_GOODS_PRICE': [ 'max', 'mean'],
        'HOUR_APPR_PROCESS_START': [ 'max', 'mean'],
        'RATE_DOWN_PAYMENT': [ 'max', 'mean'],
        'RATE_INTEREST_PRIMARY':['max','mean'],
        'RATE_INTEREST_PRIVILEGED':['max','mean'],
        'DAYS_DECISION': [ 'max', 'mean'],
        'CNT_PAYMENT': ['mean','sum','var','max','min','skew'],
        'CREDIT_TO_ANNUITY_RATIO':['mean','var','max','min','skew'],
        'CREDIT_TO_GOODS_RATIO':['mean','var','max','min','skew'],
        'CREDIT_PERC_TO_ANNUITY_RATIO':['mean','var','max','min','skew'],
        'CREDIT_PERC_TO_GOODS_RATIO':['mean','var','max','min','skew'],
        'ANNUITY_TO_CNT_PAYMENT_RATIO':['mean','var','max','min','skew'],
        'APPLICATION_TO_CNT_PAYMENT_RATIO':['mean','var','max','min','skew'],
        'CREDIT_TO_CNT_PAYMENT_RATIO':['mean','var','max','min','skew'],
        'DOWN_PAYMENT_TO_CNT_PAYMENT_RATIO':['mean','var','max','min','skew'],
        'GOODS_PRICE_TO_CNT_PAYMENT_RATIO':['mean','var','max','min','skew'],
        'ANNUITY_TO_DAYS_DECISION_RATIO':['mean','var','max','min','skew'],
        'CREDIT_TO_DAYS_DECISION_RATIO':['mean','var','max','min','skew'],
        'CREDIT_PERC_TO_DAYS_DECISION_RATIO':['mean','var','max','min','skew'],
        'GOODS_TO_DAYS_DECISION_RATIO':['mean','var','max','min','skew']
    }
    # Previous applications categorical features
    cat_aggregations = {}
    for cat in cat_cols:
        cat_aggregations[cat] = ['mean']

    prev_agg = prev.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])
    # Previous Applications: Approved Applications - only numerical features
    approved = prev[prev['NAME_CONTRACT_STATUS_Approved'] == 1]
    approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations)
    approved_agg.columns = pd.Index(['APPROVED_' + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])
    prev_agg = prev_agg.join(approved_agg, how='left', on='SK_ID_CURR')
    # Previous Applications: Refused Applications - only numerical features
    refused = prev[prev['NAME_CONTRACT_STATUS_Refused'] == 1]
    refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations)
    refused_agg.columns = pd.Index(['REFUSED_' + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()])
    prev_agg = prev_agg.join(refused_agg, how='left', on='SK_ID_CURR')

    # モメンタムファクターの追加
    feats_mom_prev = ['AMT_ANNUITY', 'AMT_APPLICATION', 'AMT_CREDIT', 'AMT_DOWN_PAYMENT',
                      'AMT_GOODS_PRICE', 'RATE_DOWN_PAYMENT', 'RATE_INTEREST_PRIMARY',
                      'RATE_INTEREST_PRIVILEGED', 'DAYS_DECISION', 'CNT_PAYMENT',
                      'DAYS_FIRST_DRAWING', 'DAYS_FIRST_DUE', 'DAYS_LAST_DUE_1ST_VERSION',
                      'DAYS_LAST_DUE', 'DAYS_TERMINATION']

    # prevテーブルのモメンタムファクターを追加
    for f in feats_mom_prev:
        momentom_factor_prev = getMomentumFactors(prev, f, 'SK_ID_CURR', 'SK_ID_PREV')
        prev_agg['PREV_'+f+'_DIFF_MEAN'] = momentom_factor_prev[0]
        prev_agg['PREV_'+f+'_PCT_MEAN'] = momentom_factor_prev[1]
        prev_agg['PREV_'+f+'_DIFF_FIN'] = momentom_factor_prev[2]
        prev_agg['PREV_'+f+'_PCT_FIN'] = momentom_factor_prev[3]
        del momentom_factor_prev

    del refused, refused_agg, approved, approved_agg, prev
    gc.collect()
    return prev_agg

# Preprocess POS_CASH_balance.csv
def pos_cash(num_rows = None, nan_as_category = True):
    pos = pd.read_csv('POS_CASH_balance.csv', nrows = num_rows)

    # 追加の特徴量 https://github.com/neptune-ml/open-solution-home-credit/blob/master/src/feature_extraction.py
    pos['IS_CONTRACT_STATUS_COMPLETED'] = (pos['NAME_CONTRACT_STATUS'] == 'Completed').astype(int)
    pos['POS_CASH_PAID_LATE'] = (pos['SK_DPD'] > 0).astype(int)
    pos['POS_CASH_PAID_LATE_WITH_TOLERANCE'] = (pos['SK_DPD_DEF'] > 0).astype(int)

    pos, cat_cols = one_hot_encoder(pos, nan_as_category= True)

    # added_feature
    pos['MONTHS_BALANCE_INSTALLMENT_FUTURE_RATIO'] = pos['MONTHS_BALANCE']*(-1.0)/pos['CNT_INSTALMENT_FUTURE']
    pos['MONTHS_BALANCE_INSTALLMENT_RATIO'] = pos['MONTHS_BALANCE']*(-1.0)/pos['CNT_INSTALMENT']
    pos['INSTALMENT_FUTURE_RATIO'] = pos['CNT_INSTALMENT_FUTURE']*1.0/pos['CNT_INSTALMENT']

    # Features
    aggregations = {
        'MONTHS_BALANCE': ['max', 'mean', 'size'],
        'SK_DPD': ['max', 'mean'],
        'SK_DPD_DEF': ['max', 'mean'],
        'MONTHS_BALANCE_INSTALLMENT_FUTURE_RATIO':['mean','max','min','var','skew'],
        'MONTHS_BALANCE_INSTALLMENT_RATIO':['mean','max','min','var','skew'],
        'INSTALMENT_FUTURE_RATIO':['mean','min','var','skew'],
        'IS_CONTRACT_STATUS_COMPLETED':['mean'],
        'POS_CASH_PAID_LATE':['mean'],
        'POS_CASH_PAID_LATE_WITH_TOLERANCE':['mean']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']

    pos_agg = pos.groupby('SK_ID_CURR').agg(aggregations)
    pos_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()])
    # Count pos cash accounts
    pos_agg['POS_COUNT'] = pos.groupby('SK_ID_CURR').size()

    # モメンタムファクターの追加
    feats_mom_pos = ['MONTHS_BALANCE', 'CNT_INSTALMENT', 'CNT_INSTALMENT_FUTURE']

    # posテーブルのモメンタムファクターを追加
    for f in feats_mom_pos:
        momentom_factor_pos = getMomentumFactors(pos, f, 'SK_ID_CURR', 'SK_ID_PREV')
        pos_agg['POS_'+f+'_DIFF_MEAN'] = momentom_factor_pos[0]
        pos_agg['POS_'+f+'_PCT_MEAN'] = momentom_factor_pos[1]
        pos_agg['POS_'+f+'_DIFF_FIN'] = momentom_factor_pos[2]
        pos_agg['POS_'+f+'_PCT_FIN'] = momentom_factor_pos[3]
        del momentom_factor_pos

    del pos
    gc.collect()
    return pos_agg

# Preprocess installments_payments.csv
def installments_payments(num_rows = None, nan_as_category = True):
    ins = pd.read_csv('installments_payments.csv', nrows = num_rows)
    ins, cat_cols = one_hot_encoder(ins, nan_as_category= True)
    # Percentage and difference paid in each installment (amount paid and installment value)
    ins['PAYMENT_PERC'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT']
    ins['PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']
    # Days past due and days before due (no negative values)
    ins['DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
    ins['DBD'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']
    ins['DPD'] = ins['DPD'].apply(lambda x: x if x > 0 else 0)
    ins['DBD'] = ins['DBD'].apply(lambda x: x if x > 0 else 0)
    # 追加しました
    ins['instalment_dummy'] = ins['NUM_INSTALMENT_VERSION'].map(lambda x:1 if x==0 else 0)
    ins['DAYS_PERC'] = ins['DAYS_ENTRY_PAYMENT']*1.0 / ins['DAYS_INSTALMENT']
    # Features: Perform aggregations
    aggregations = {
        'NUM_INSTALMENT_VERSION': ['nunique'],
        'DPD': ['max', 'mean', 'sum'],
        'DBD': ['max', 'mean', 'sum'],
        'PAYMENT_PERC': [ 'mean',  'var'],
        'PAYMENT_DIFF': [ 'mean', 'var'],
        'DAYS_PERC': [ 'mean', 'var'],
        'AMT_INSTALMENT': ['max', 'mean', 'sum'],
        'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
        'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']
    ins_agg = ins.groupby('SK_ID_CURR').agg(aggregations)
    ins_agg.columns = pd.Index(['INSTAL_' + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()])
    # Count installments accounts
    ins_agg['INSTAL_COUNT'] = ins.groupby('SK_ID_CURR').size()

    # モメンタムファクターの追加
    feats_mom_ins = ['NUM_INSTALMENT_VERSION', 'NUM_INSTALMENT_NUMBER', 'DAYS_INSTALMENT',
                     'DAYS_ENTRY_PAYMENT', 'AMT_INSTALMENT', 'AMT_PAYMENT']

    # insテーブルのモメンタムファクターを追加
    for f in feats_mom_ins:
        momentom_factor_ins = getMomentumFactors(ins, f, 'SK_ID_CURR', 'SK_ID_PREV')
        ins_agg['INSTAL_'+f+'_DIFF_MEAN'] = momentom_factor_ins[0]
        ins_agg['INSTAL_'+f+'_PCT_MEAN'] = momentom_factor_ins[1]
        ins_agg['INSTAL_'+f+'_DIFF_FIN'] = momentom_factor_ins[2]
        ins_agg['INSTAL_'+f+'_PCT_FIN'] = momentom_factor_ins[3]
        del momentom_factor_ins

    del ins
    gc.collect()
    return ins_agg

# Preprocess credit_card_balance.csv
def credit_card_balance(num_rows = None, nan_as_category = True):
    cc = pd.read_csv('credit_card_balance.csv', nrows = num_rows)
    cc, cat_cols = one_hot_encoder(cc, nan_as_category= True)

    # General aggregations
    # 追加の前処理 https://github.com/neptune-ml/open-solution-home-credit/wiki/LightGBM-with-smarter-features
    #cc['AMT_DRAWINGS_ATM_CURRENT'][cc['AMT_DRAWINGS_ATM_CURRENT'] < 0] = np.nan
    ix = cc[cc['AMT_DRAWINGS_ATM_CURRENT'] < 0].index
    cc.loc[ix,'AMT_DRAWINGS_ATM_CURRENT'] =np.nan
    #cc['AMT_DRAWINGS_CURRENT'][cc['AMT_DRAWINGS_CURRENT'] < 0] = np.nan
    ix = cc[cc['AMT_DRAWINGS_CURRENT'] < 0].index
    cc.loc[ix,'AMT_DRAWINGS_CURRENT'] =np.nan

    # 特徴量追加するやでー
    cc['AMT_PAYMENT_CURRENT_TOTAL_RATIO'] = cc['AMT_PAYMENT_CURRENT']*1.0/cc['AMT_PAYMENT_TOTAL_CURRENT']
    cc['CNT_DRAWINGS_ATM_CURRENT_RATIO'] = cc['CNT_DRAWINGS_ATM_CURRENT']*1.0/cc['CNT_DRAWINGS_CURRENT']
    cc['AMT_INST_MIN_REGULARITY_CURRENT_RATIO'] = cc['AMT_INST_MIN_REGULARITY']*1.0/cc['AMT_PAYMENT_CURRENT']
    cc['AMT_INST_MIN_REGULARITY_TOTAL_CURRENT_RATIO'] = cc['AMT_INST_MIN_REGULARITY']*1.0/cc['AMT_PAYMENT_TOTAL_CURRENT']
    cc['AMT_RECEIVABLE_PRINCIPAL_AMT_RECIVABLE_RATIO'] = cc['AMT_RECEIVABLE_PRINCIPAL']*1.0/cc['AMT_RECIVABLE']
    cc['AMT_RECIVABLE_AMT_TOTAL_RECEIVABLE_RATIO'] = cc['AMT_RECIVABLE']*1.0/cc['AMT_TOTAL_RECEIVABLE']

    cc['AMT_TOTAL_RECEIVABLE_CNT_INSTALMENT_MATURE_CUM'] = cc['AMT_TOTAL_RECEIVABLE']*1.0/cc['CNT_INSTALMENT_MATURE_CUM']
    cc['AMT_RECIVABLE_CNT_INSTALMENT_MATURE_CUM'] = cc['AMT_RECIVABLE']*1.0/cc['CNT_INSTALMENT_MATURE_CUM']
    cc['AMT_PAYMENT_TOTAL_CURRENT_CNT_INSTALMENT_MATURE_CUM'] = cc['AMT_PAYMENT_TOTAL_CURRENT']*1.0/cc['CNT_INSTALMENT_MATURE_CUM']
    cc['AMT_PAYMENT_CURRENT_CNT_INSTALMENT_MATURE_CUM'] = cc['AMT_PAYMENT_CURRENT']*1.0/cc['CNT_INSTALMENT_MATURE_CUM']
    cc['AMT_DRAWINGS_CURRENT_AMT_CNT_INSTALMENT_MATURE_CUM'] = cc['AMT_DRAWINGS_CURRENT']*1.0/cc['CNT_INSTALMENT_MATURE_CUM']
    cc['AMT_CREDIT_LIMIT_ACTUAL_CNT_INSTALMENT_MATURE_CUM'] = cc['AMT_CREDIT_LIMIT_ACTUAL']*1.0/cc['CNT_INSTALMENT_MATURE_CUM']
    cc['AMT_BALANCE_CNT_INSTALMENT_MATURE_CUM'] = cc['AMT_BALANCE']*1.0/cc['CNT_INSTALMENT_MATURE_CUM']

    # 追加の特徴量 https://github.com/neptune-ml/open-solution-home-credit/blob/master/src/feature_extraction.py
    cc['BALANCE_TO_LIMIT_RATIO'] = cc['AMT_BALANCE'] / cc['AMT_CREDIT_LIMIT_ACTUAL']

    # 処理も追加じゃー
    aggregations = {
        'MONTHS_BALANCE':[ 'max', 'mean', 'sum', 'var'],
        'AMT_BALANCE':[ 'max', 'mean', 'sum', 'var'],
        'AMT_CREDIT_LIMIT_ACTUAL':[ 'max', 'mean', 'sum', 'var'],
        'AMT_DRAWINGS_ATM_CURRENT':[ 'max', 'mean', 'sum', 'var'],
        'AMT_DRAWINGS_CURRENT':[ 'max', 'mean', 'sum', 'var'],
        'AMT_DRAWINGS_OTHER_CURRENT':[ 'max', 'mean', 'sum', 'var'],
        'AMT_DRAWINGS_POS_CURRENT':[ 'max', 'mean', 'sum', 'var'],
        'AMT_INST_MIN_REGULARITY':[ 'max', 'mean', 'sum', 'var'],
        'AMT_PAYMENT_CURRENT':[ 'max', 'mean', 'sum', 'var'],
        'AMT_PAYMENT_TOTAL_CURRENT':[ 'max', 'mean', 'sum', 'var'],
        'AMT_RECEIVABLE_PRINCIPAL':[ 'max', 'mean', 'sum', 'var'],
        'AMT_RECIVABLE':[ 'max', 'mean', 'sum', 'var'],
        'AMT_TOTAL_RECEIVABLE':[ 'max', 'mean', 'sum', 'var'],
        'CNT_DRAWINGS_ATM_CURRENT':[ 'max', 'mean', 'sum', 'var'],
        'CNT_DRAWINGS_CURRENT':[ 'max', 'mean', 'sum', 'var'],
        'CNT_DRAWINGS_OTHER_CURRENT':[ 'max', 'mean', 'sum', 'var'],
        'CNT_DRAWINGS_POS_CURRENT':[ 'max', 'mean', 'sum', 'var'],
        'CNT_INSTALMENT_MATURE_CUM':[ 'max', 'mean', 'sum', 'var'],
        'SK_DPD':[ 'max', 'mean', 'sum', 'var'],
        'SK_DPD_DEF':[ 'max', 'mean', 'sum', 'var'],
        'AMT_PAYMENT_CURRENT_TOTAL_RATIO': ['mean','var'],
        'CNT_DRAWINGS_ATM_CURRENT_RATIO': [ 'max', 'mean', 'sum', 'var'],
        'AMT_INST_MIN_REGULARITY_CURRENT_RATIO': [ 'max', 'mean', 'sum', 'var'],
        'AMT_INST_MIN_REGULARITY_TOTAL_CURRENT_RATIO': [ 'max', 'mean', 'sum', 'var'],
        'BALANCE_TO_LIMIT_RATIO': [ 'max', 'mean', 'sum', 'var'],
        'AMT_RECEIVABLE_PRINCIPAL_AMT_RECIVABLE_RATIO': [ 'max', 'mean', 'sum', 'var'],
        'AMT_RECIVABLE_AMT_TOTAL_RECEIVABLE_RATIO': [ 'max', 'mean', 'sum', 'var'],
        'AMT_TOTAL_RECEIVABLE_CNT_INSTALMENT_MATURE_CUM': [ 'max', 'mean', 'sum', 'var'],
        'AMT_RECIVABLE_CNT_INSTALMENT_MATURE_CUM': [ 'max', 'mean', 'sum', 'var'],
        'AMT_PAYMENT_TOTAL_CURRENT_CNT_INSTALMENT_MATURE_CUM': [ 'max', 'mean', 'sum', 'var'],
        'AMT_PAYMENT_CURRENT_CNT_INSTALMENT_MATURE_CUM': [ 'max', 'mean', 'sum', 'var'],
        'AMT_DRAWINGS_CURRENT_AMT_CNT_INSTALMENT_MATURE_CUM': [ 'max', 'mean', 'sum', 'var'],
        'AMT_CREDIT_LIMIT_ACTUAL_CNT_INSTALMENT_MATURE_CUM': [ 'max', 'mean', 'sum', 'var'],
        'AMT_BALANCE_CNT_INSTALMENT_MATURE_CUM': [ 'max', 'mean', 'sum', 'var']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']

    #cc_agg = cc.groupby('SK_ID_CURR').agg([ 'max', 'mean', 'sum', 'var'])
    cc_agg = cc.groupby('SK_ID_CURR').agg(aggregations)
    cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])
    # Count credit card lines
    cc_agg['CC_COUNT'] = cc.groupby('SK_ID_CURR').size()

    # モメンタムファクターの追加
    feats_mom_cc = ['MONTHS_BALANCE', 'AMT_BALANCE', 'AMT_CREDIT_LIMIT_ACTUAL',
                    'AMT_DRAWINGS_ATM_CURRENT', 'AMT_DRAWINGS_CURRENT', 'AMT_DRAWINGS_OTHER_CURRENT',
                    'AMT_DRAWINGS_POS_CURRENT', 'AMT_INST_MIN_REGULARITY', 'AMT_PAYMENT_CURRENT',
                    'AMT_PAYMENT_TOTAL_CURRENT', 'AMT_RECEIVABLE_PRINCIPAL', 'AMT_RECIVABLE',
                    'AMT_TOTAL_RECEIVABLE', 'CNT_DRAWINGS_ATM_CURRENT', 'CNT_DRAWINGS_CURRENT',
                    'CNT_DRAWINGS_OTHER_CURRENT', 'CNT_DRAWINGS_POS_CURRENT', 'CNT_INSTALMENT_MATURE_CUM']
    # ccテーブルのモメンタムファクターを追加
    for f in feats_mom_cc:
        momentom_factor_cc = getMomentumFactors(cc, f, 'SK_ID_CURR', 'SK_ID_PREV')
        cc_agg['CC_'+f+'_DIFF_MEAN'] = momentom_factor_cc[0]
        cc_agg['CC_'+f+'_PCT_MEAN'] = momentom_factor_cc[1]
        cc_agg['CC_'+f+'_DIFF_FIN'] = momentom_factor_cc[2]
        cc_agg['CC_'+f+'_PCT_FIN'] = momentom_factor_cc[3]
        del momentom_factor_cc

    cc.drop(['SK_ID_PREV'], axis= 1, inplace = True)
    del cc
    gc.collect()
    return cc_agg

def getAdditionalFeatures(data):
    # 作ってみた その１
    data['CREDIT_CHG_MAX']=data['AMT_CREDIT']*1.0/data['PREV_AMT_CREDIT_MAX']
    data['CREDIT_CHG_MEAN']=data['AMT_CREDIT']*1.0/data['PREV_AMT_CREDIT_MEAN']
    data['CREDIT_SUM_RATIO_BURO']=data['AMT_CREDIT']*1.0/data['BURO_AMT_CREDIT_SUM_MEAN']
    data['CREDIT_SUM_LIMIT_RATIO_BURO']=data['AMT_CREDIT']*1.0/data['BURO_AMT_CREDIT_SUM_LIMIT_MEAN']
    data['CREDIT_SUM_DEBT_RATIO_BURO']=data['AMT_CREDIT']*1.0/data['BURO_AMT_CREDIT_SUM_DEBT_MEAN']
    data['CREDIT_SUM_OVERDUE_RATIO_BURO']=data['AMT_CREDIT']*1.0/data['BURO_AMT_CREDIT_SUM_OVERDUE_MEAN']
    data['GOODS_PRICE_RATIO_CHG'] = (data['AMT_GOODS_PRICE']*1.0/data['AMT_ANNUITY'])/(data['PREV_AMT_GOODS_PRICE_MAX']*1.0/data['PREV_AMT_ANNUITY_MAX'])
    data['GOODS_PRICE_CHG']=data['AMT_GOODS_PRICE']*1.0/data['PREV_AMT_GOODS_PRICE_MAX']

    # 作ってみた その2
    data['ADD_NORMALIZED_SCORE_1'] = data['NEW_SOURCES_MEDIAN'] + data['PREV_RATE_DOWN_PAYMENT_MEAN']
    data['MINUS_NORMALIZED_SCORE_1'] = data['NEW_SOURCES_MEDIAN'] - data['PREV_RATE_DOWN_PAYMENT_MEAN']

    data['ADD_NORMALIZED_SCORE_2'] = data['NEW_SOURCES_MEDIAN'] + data['PREV_RATE_INTEREST_PRIMARY_MEAN']
    data['MINUS_NORMALIZED_SCORE_2'] = data['NEW_SOURCES_MEDIAN'] - data['PREV_RATE_INTEREST_PRIMARY_MEAN']

    data['ADD_NORMALIZED_SCORE_3'] = data['NEW_SOURCES_MEDIAN'] + data['PREV_RATE_INTEREST_PRIVILEGED_MEAN']
    data['MINUS_NORMALIZED_SCORE_3'] = data['NEW_SOURCES_MEDIAN'] - data['PREV_RATE_INTEREST_PRIVILEGED_MEAN']

    # correlation高い変数の削除
#    col_drop = removeCorrelatedVariables(data, 0.9)
#    print('There are %d columns to remove.' % (len(col_drop)))
#    data = data.drop(columns = col_drop)

    # 欠損値率の高い変数の削除
#    col_missing = removeMissingVariables(data, 0.75)
#    print('There are %d columns with more than 75%% missing values' % len(col_missing))
#    data = data.drop(columns = col_missing)

    return data

if __name__ == '__main__':
    # test
    num_rows = 5000

    # application
    df = application_train_test(num_rows)

    # bureau
    bureau = bureau_and_balance(num_rows)
    df = df.join(bureau, how='left', on='SK_ID_CURR')
    del bureau

    # prev
    prev = previous_applications(num_rows)
    df = df.join(prev, how='left', on='SK_ID_CURR')
    del prev

    # prev-additional
    prev_add = get_amt_factor(df,num_rows)
    df = df.join(prev_add, how='left', on='SK_ID_CURR')
    del prev_add
    gc.collect()

    # pos
    pos = pos_cash(num_rows)
    df = df.join(pos, how='left', on='SK_ID_CURR')
    del pos

    # ins
    ins = installments_payments(num_rows)
    df = df.join(ins, how='left', on='SK_ID_CURR')
    del ins

    # cc
    cc = credit_card_balance(num_rows)
    df = df.join(cc, how='left', on='SK_ID_CURR')
    del cc

    df = getAdditionalFeatures(df)

    print(df)
