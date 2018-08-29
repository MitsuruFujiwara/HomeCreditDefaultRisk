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

    # set low threshold
    pred = pred.apply(lambda x: 0 if x < q_low else x)

    # get optimal high
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

    return q_high, q_low

def main():
    # load submission files
    sub = pd.read_csv('sample_submission.csv')
    sub_lgbm = pd.read_csv('submission_add_feature_lgbm_#46.csv')
    sub_xgb = pd.read_csv('submission_add_feature_xgb_best.csv')
#    sub_dnn = pd.read_csv('submission_add_feature_dnn.csv')

    # load out of fold data
    train_df = pd.read_csv('application_train.csv')
    oof_lgbm = pd.read_csv('oof_lgbm_#46.csv')
    oof_xgb = pd.read_csv('oof_xgb_best.csv')
#    oof_dnn = pd.read_csv('oof_dnn.csv')

    # change index & columns
    oof_lgbm.columns=['SK_ID_CURR', 'lgbm']
    oof_xgb.columns=['SK_ID_CURR', 'xgb']
#    oof_dnn.columns=['SK_ID_CURR', 'dnn']
    oof_lgbm.index=oof_lgbm['SK_ID_CURR']
    oof_xgb.index=oof_xgb['SK_ID_CURR']
#    oof_dnn.index=oof_xgb['SK_ID_CURR']
    train_df.index=train_df['SK_ID_CURR']

    # merge dfs
    train_df = train_df.merge(oof_lgbm, how='left', on='SK_ID_CURR')
    train_df = train_df.merge(oof_xgb, how='left', on='SK_ID_CURR')
#    train_df = train_df.merge(oof_dnn, how='left', on='SK_ID_CURR')
#    train_df = train_df[['SK_ID_CURR','TARGET', 'lgbm', 'xgb', 'dnn']].dropna()
    train_df = train_df[['SK_ID_CURR','TARGET', 'lgbm', 'xgb']].dropna()
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
#        q_high_dnn_ = x[7]
#        q_low_dnn_ = x[8]

        lgbm_se = train_df['lgbm']
        xgb_se = train_df['xgb']
#        dnn_se = train_df['dnn']
        lgbm_se = lgbm_se.apply(lambda x: 0 if x < q_low_lgbm_ else x)
        lgbm_se = lgbm_se.apply(lambda x: 1 if x > q_high_lgbm_ else x)
        xgb_se = xgb_se.apply(lambda x: 0 if x < q_low_xgb_ else x)
        xgb_se = xgb_se.apply(lambda x: 1 if x > q_high_xgb_ else x)
#        dnn_se = dnn_se.apply(lambda x: 0 if x < q_low_dnn_ else x)
#        dnn_se = dnn_se.apply(lambda x: 1 if x > q_high_dnn_ else x)

        _pred = w * lgbm_se + (1.0-w) * xgb_se
        _pred = _pred.apply(lambda x: 0 if x < q_low_pred_ else x)
        _pred = _pred.apply(lambda x: 1 if x > q_high_pred_ else x)

        _auc = roc_auc_score(train_df['TARGET'], _pred)
        return -1*_auc

    bnds = np.array([[0,1],[0.9,1],[0,0.1],[0.9,1],[0,0.1],[0.9,1],[0,0.1]])
#    bnds = np.array([[0,1],[0.9,1],[0,0.1],[0.9,1],[0,0.1],[0.9,1],[0,0.1],[0.9,1],[0,0.1]])
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
    """

    # take weighted average of each prediction
    sub['lgbm'] = sub_lgbm['TARGET']
    sub['xgb'] = sub_xgb['TARGET']

    # out of foldsの予測値
    pred_oof = 0.5*train_df['lgbm']+0.5*train_df['xgb']

    # 最適な閾値を取得
    q_high, q_low = getZeroOneThresholds(train_df['TARGET'], pred_oof)

#    sub['TARGET'] = w_bst*sub_lgbm['TARGET'] + (1 - w_bst)*sub_xgb['TARGET']
    sub['TARGET'] = 0.5*sub_lgbm['TARGET'] + 0.5*sub_xgb['TARGET']

    sub['TARGET'] = sub['TARGET'].apply(lambda x: 0 if x < q_low else x)
    sub['TARGET'] = sub['TARGET'].apply(lambda x: 1 if x > q_high else x)

    # replace values to 0 or 1 by threshold
#    sub['TARGET'] = sub['TARGET'].apply(lambda x: 0 if x < q_low_pred else x)
#    sub['TARGET'] = sub['TARGET'].apply(lambda x: 1 if x > q_high_pred else x)

    # save submission file
    sub[['SK_ID_CURR', 'TARGET']].to_csv('submission_blend.csv', index= False)

    # local validation scoreの記録用
    print(roc_auc_score(train_df['TARGET'], 0.5*train_df['lgbm']+0.5*train_df['xgb']))

if __name__ == '__main__':
    main()
