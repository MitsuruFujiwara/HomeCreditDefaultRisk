import pandas as pd
import numpy as np
import lightgbm as lgb
import gc

from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.optimize import minimize_scalar

"""
複数モデルのoutputをブレンドして最終的なsubmitファイルを生成するスクリプト。
"""

# 0-1に置き換える最適な閾値を得るやつ
def getZeroOneThresholds(act, pred):

    # get optimal low
    def optLow(_q):
        _threshold = pred.quantile(_q)
        _pred = pred.apply(lambda x: 0 if x < _threshold else x)
        _auc = roc_auc_score(act, _pred)
        return -1*_auc


    _result = minimize_scalar(fun=optLow, bounds=(0, 0.1), method="bounded")
    q_low = _result['x']
    print(_result)
    print('best q_low:',q_low)
    """
    auc_bst = 0.0
    for _q in np.arange(0, 0.10001, 0.0001):
        _threshold = pred.quantile(_q)
        _pred = pred.apply(lambda x: 0 if x < _threshold else x)
        _auc = roc_auc_score(act, _pred)
        if _auc > auc_bst:
            auc_bst = _auc
            q_low = _q
        print("q: {:.4f}, auc: {:.10f}".format(_q, _auc))
    print("best q_low: {:.4f}, best auc: {:.10f}".format(q_low, auc_bst))
    """

    # set low threshold
    pred = pred.apply(lambda x: 0 if x < q_low else x)


    # get optimal high
    def optHigh(_q):
        _threshold = pred.quantile(_q)
        _pred = pred.apply(lambda x: 1 if x > _threshold else x)
        _auc = roc_auc_score(act, _pred)
        return -1*_auc
    

    _result = minimize_scalar(fun=optHigh, bounds=(0.9, 1), method="bounded")
    q_high = _result['x']
    print(_result)
    print('best q_high:',q_high)
    """
    auc_bst = 0.0
    for _q in np.arange(0.9, 1.0001, 0.0001):
        _threshold = pred.quantile(_q)
        _pred = pred.apply(lambda x: 1 if x > _threshold else x)
        _auc = roc_auc_score(act, _pred)
        if _auc > auc_bst:
            auc_bst = _auc
            q_high = _q
        print("q: {:.4f}, auc: {:.10f}".format(_q, _auc))
    print("best q_high: {:.4f}, best auc: {:.10f}".format(q_high, auc_bst))
    """

    return q_high, q_low

def main():
    # load submission files
    sub = pd.read_csv('sample_submission.csv')
    sub_lgbm = pd.read_csv('submission_add_feature_lgbm_#46.csv')
    sub_xgb = pd.read_csv('submission_add_feature_xgb_#44.csv')

    # load out of fold data
    train_df = pd.read_csv('application_train.csv')
    oof_lgbm = pd.read_csv('oof_lgbm_#46.csv')
    oof_xgb = pd.read_csv('oof_xgb_#44.csv')

    # change index & columns
    oof_lgbm.columns=['SK_ID_CURR', 'lgbm']
    oof_xgb.columns=['SK_ID_CURR', 'xgb']
    oof_lgbm.index=oof_lgbm['SK_ID_CURR']
    oof_xgb.index=oof_xgb['SK_ID_CURR']
    train_df.index=train_df['SK_ID_CURR']

    # merge dfs
    train_df = train_df.merge(oof_lgbm, how='left', on='SK_ID_CURR')
    train_df = train_df.merge(oof_xgb, how='left', on='SK_ID_CURR')
    train_df = train_df[['SK_ID_CURR','TARGET', 'lgbm', 'xgb']].dropna()
    
    # get thresholds
    q_high_lgbm, q_low_lgbm = getZeroOneThresholds(train_df['TARGET'], train_df['lgbm'])
    q_high_xgb, q_low_xgb = getZeroOneThresholds(train_df['TARGET'], train_df['xgb'])

    # replace values to 0 or 1 by threshold
    train_df['lgbm'] = train_df['lgbm'].apply(lambda x: 0 if x < q_low_lgbm else x)
    train_df['lgbm'] = train_df['lgbm'].apply(lambda x: 1 if x > q_high_lgbm else x)

    sub_lgbm['TARGET'] = sub_lgbm['TARGET'].apply(lambda x: 0 if x < q_low_lgbm else x)
    sub_lgbm['TARGET'] = sub_lgbm['TARGET'].apply(lambda x: 1 if x > q_high_lgbm else x)

    train_df['xgb'] = train_df['xgb'].apply(lambda x: 0 if x < q_low_xgb else x)
    train_df['xgb'] = train_df['xgb'].apply(lambda x: 1 if x > q_high_xgb else x)

    sub_xgb['TARGET'] = sub_xgb['TARGET'].apply(lambda x: 0 if x < q_low_xgb else x)
    sub_xgb['TARGET'] = sub_xgb['TARGET'].apply(lambda x: 1 if x > q_high_xgb else x)
    
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
            print("w: {:.3f}, auc: {:.10f}".format(w, _auc))

    print("best w: {}, best auc: {:.10f}".format(w_bst, auc_bst))
    """


    # 最適化
    ## 目的関数
    def func(w):
        _pred = w * train_df['lgbm'] + (1.0-w) * train_df['xgb']
        _auc = roc_auc_score(train_df['TARGET'], _pred)
        return -1*_auc


    result = minimize_scalar(fun=func, bounds=(0, 1), method="bounded")
    w = result['x']
    print(result)
    print('best w:',w)

    # take weighted average of each prediction
    sub['lgbm'] = sub_lgbm['TARGET']
    sub['xgb'] = sub_xgb['TARGET']

#    sub['TARGET'] = w_bst[0]*sub_lgbm['TARGET'] + w_bst[1]*sub_xgb['TARGET']
    sub['TARGET'] = 0.5*sub_lgbm['TARGET'] + 0.5*sub_xgb['TARGET']

    # save submission file
    sub[['SK_ID_CURR', 'TARGET']].to_csv('submission_blend.csv', index= False)

if __name__ == '__main__':
    main()
