import pandas as pd
import numpy as np
import lightgbm as lgb
import gc

from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.optimize import minimize_scalar, minimize, differential_evolution

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

    """
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

    # 最適化
    ## 目的関数
    def func(x):
        w = x[0]
        q_high_lgbm_ = x[1]
        q_low_lgbm_ = x[2]
        q_high_xgb_ = x[3]
        q_low_xgb_ = x[4]
        q_high_pred_ = x[5]
        q_low_pred_ = x[6]

        lgbm_se = train_df['lgbm']
        xgb_se = train_df['xgb']
        lgbm_se = lgbm_se.apply(lambda x: 0 if x < q_low_lgbm_ else x)
        lgbm_se = lgbm_se.apply(lambda x: 1 if x > q_high_lgbm_ else x)
        xgb_se = xgb_se.apply(lambda x: 0 if x < q_low_xgb_ else x)
        xgb_se = xgb_se.apply(lambda x: 1 if x > q_high_xgb_ else x)

        _pred = w * lgbm_se + (1.0-w) * xgb_se
        _pred = _pred.apply(lambda x: 0 if x < q_low_pred_ else x)
        _pred = _pred.apply(lambda x: 1 if x > q_high_pred_ else x)

        _auc = roc_auc_score(train_df['TARGET'], _pred)
        return -1*_auc


    bnds = np.array([[0,1],[0.9,1],[0,0.1],[0.9,1],[0,0.1],[0.9,1],[0,0.1]])
    result = differential_evolution(func, bnds) # 微分進化法とかいう謎の強そうな最適化をやれば幸せになれるらしい

    w_bst = result['x'][0]
    q_high_lgbm = result['x'][1]
    q_low_lgbm = result['x'][2]
    q_high_xgb = result['x'][3]
    q_low_xgb = result['x'][4]
    q_high_pred = result['x'][5]
    q_low_pred = result['x'][6]
    print(result)
    print('best w:',w_bst)

    # replace values to 0 or 1 by threshold
    train_df['lgbm'] = train_df['lgbm'].apply(lambda x: 0 if x < q_low_lgbm else x)
    train_df['lgbm'] = train_df['lgbm'].apply(lambda x: 1 if x > q_high_lgbm else x)

    sub_lgbm['TARGET'] = sub_lgbm['TARGET'].apply(lambda x: 0 if x < q_low_lgbm else x)
    sub_lgbm['TARGET'] = sub_lgbm['TARGET'].apply(lambda x: 1 if x > q_high_lgbm else x)

    train_df['xgb'] = train_df['xgb'].apply(lambda x: 0 if x < q_low_xgb else x)
    train_df['xgb'] = train_df['xgb'].apply(lambda x: 1 if x > q_high_xgb else x)

    sub_xgb['TARGET'] = sub_xgb['TARGET'].apply(lambda x: 0 if x < q_low_xgb else x)
    sub_xgb['TARGET'] = sub_xgb['TARGET'].apply(lambda x: 1 if x > q_high_xgb else x)

    # take weighted average of each prediction
    sub['lgbm'] = sub_lgbm['TARGET']
    sub['xgb'] = sub_xgb['TARGET']

    sub['TARGET'] = w_bst*sub_lgbm['TARGET'] + (1 - w_bst)*sub_xgb['TARGET']
#    sub['TARGET'] = 0.5*sub_lgbm['TARGET'] + 0.5*sub_xgb['TARGET']

    # replace values to 0 or 1 by threshold
    sub['TARGET'] = sub['TARGET'].apply(lambda x: 0 if x < q_low_pred else x)
    sub['TARGET'] = sub['TARGET'].apply(lambda x: 1 if x > q_high_pred else x)

    # save submission file
    sub[['SK_ID_CURR', 'TARGET']].to_csv('submission_blend.csv', index= False)

if __name__ == '__main__':
    main()
