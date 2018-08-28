import pandas as pd
import numpy as np
import lightgbm as lgb
import gc

from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns

"""
複数モデルのoutputをブレンドして最終的なsubmitファイルを生成するスクリプト。
"""

# TODO　# 0-1に置き換える最適な閾値を得るやつ
def getZeroOneThresholds(df_oof):
    # 閾値の初期値
    q_high = 1.0
    q_low = 0.0

    #
    for q in np.arange(0, 1.0001, 0.0001):
        _q = df_oof['TARGET'].quantile(q)
    q_high = df_oof['TARGET'].quantile(0.9995)
    q_low = df_oof['TARGET'].quantile(0.0005)

    test_df['TARGET'] = test_df['TARGET'].apply(lambda x: 1 if x > q_high else x)
    test_df['TARGET'] = test_df['TARGET'].apply(lambda x: 0 if x < q_low else x)

    return q_high, q_low

def main():
    # load submission files
    sub = pd.read_csv('sample_submission.csv')
    sub_lgbm = pd.read_csv('submission_add_feature_lgbm_#38.csv')
    sub_xgb = pd.read_csv('submission_add_feature_xgb_#44.csv')

    # loada out of fold data
    train_df = pd.read_csv('application_train.csv')
    oof_lgbm = pd.read_csv('oof_lgbm_#43.csv')
    oof_xgb = pd.read_csv('oof_xgb_#44.csv')

    oof_lgbm.columns=['SK_ID_CURR', 'lgbm']
    oof_xgb.columns=['SK_ID_CURR', 'xgb']
    oof_lgbm.index=oof_lgbm['SK_ID_CURR']
    oof_xgb.index=oof_xgb['SK_ID_CURR']
    train_df.index=train_df['SK_ID_CURR']

    train_df = train_df.merge(oof_lgbm, how='left', on='SK_ID_CURR')
    train_df = train_df.merge(oof_xgb, how='left', on='SK_ID_CURR')
    train_df = train_df[['SK_ID_CURR','TARGET', 'lgbm', 'xgb']].dropna()

    """
    # find best weights
    auc_bst = 0.0
    for w in np.arange(0,1.001, 0.001):
        _pred = w * train_df['lgbm'] + (1.0-w) * train_df['xgb']
        _auc = roc_auc_score(train_df['TARGET'], _pred)
        if _auc > auc_bst:
            auc_bst = _auc
            w_bst = (w, 1.0-w)
        if np.mod(w*1000, 10)==0:
            print("w: {}, auc: {}".format(w, _auc))

    print("best w: {}, best auc: {}".format(w_bst, auc_bst))
    """

    # take average of each prediction
    sub['lgbm'] = sub_lgbm['TARGET']
    sub['xgb'] = sub_xgb['TARGET']

#    sub['TARGET'] = w_bst[0]*sub_lgbm['TARGET'] + w_bst[1]*sub_xgb['TARGET']
    sub['TARGET'] = 0.5*sub_lgbm['TARGET'] + 0.5*sub_xgb['TARGET']

    # save submission file
    sub[['SK_ID_CURR', 'TARGET']].to_csv('submission_blend.csv', index= False)

if __name__ == '__main__':
    main()
