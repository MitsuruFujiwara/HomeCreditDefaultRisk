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

def main():
    # load submission files
    sub = pd.read_csv('sample_submission.csv')
    sub_lgbm = pd.read_csv('submission_add_feature_lgbm.csv')
    sub_xgb = pd.read_csv('submission_add_feature_xgb.csv')

    # loada out of fold data
    train_df = pd.read_csv('application_train.csv')
    oof_lgbm = pd.read_csv('oof_lgbm.csv')
    oof_xgb = pd.read_csv('oof_xgb.csv')

    oof_lgbm.columns=['SK_ID_CURR', 'lgbm']
    oof_xgb.columns=['SK_ID_CURR', 'xgb']
    oof_lgbm.index=oof_lgbm['SK_ID_CURR']
    oof_xgb.index=oof_xgb['SK_ID_CURR']
    train_df.index=train_df['SK_ID_CURR']

    train_df = train_df.merge(oof_lgbm, how='left', on='SK_ID_CURR')
    train_df = train_df.merge(oof_xgb, how='left', on='SK_ID_CURR')
    train_df = train_df[['SK_ID_CURR','TARGET', 'lgbm', 'xgb']].dropna()

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

    print("best w: {}, best rmse: {}".format(w_bst, auc_bst))

    # take average of each predicted values
    sub['lgbm'] = sub_lgbm['TARGET']
    sub['xgb'] = sub_xgb['TARGET']

    sub['TARGET'] = w_bst[0]*sub_lgbm['TARGET'] + w_bst[1]*sub_xgb['TARGET']

    # save submission file
    sub[['SK_ID_CURR', 'TARGET']].to_csv('submission_blend.csv', index= False)

if __name__ == '__main__':
    main()
