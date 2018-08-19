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

#    train_df = train_df[train_df['CODE_GENDER'] != 'XNA']

    train_df['lgbm'] = oof_lgbm['OOF_PRED']
    train_df['xgb'] = oof_xgb['OOF_PRED']

    # find best weights
    auc_bst = 0.0
    for w in np.arange(0,1, 0.001):
        _pred = w * train_df['lgbm'] + (1.0-w) * train_df['xgb']
        _act = train_df['TARGET'][_pred.notnull()]
        _pred = _pred[_pred.notnull()]
        _auc = roc_auc_score(_act, _pred)
        if _auc > auc_bst:
            auc_bst = _auc
            w_bst = (w, 1.0-w)
        print("w: {}, auc: {}".format(w, _auc))

    print(w_bst)

    # take average of each predicted values
    sub['lgbm'] = sub_lgbm['TARGET']
    sub['xgb'] = sub_xgb['TARGET']

    sub['TARGET'] = w_bst[0]*sub_lgbm['TARGET'] + w_bst[1]*sub_xgb['TARGET']

    # save submission file
    sub[['SK_ID_CURR', 'TARGET']].to_csv('submission_blend.csv', index= False)

if __name__ == '__main__':
    main()
