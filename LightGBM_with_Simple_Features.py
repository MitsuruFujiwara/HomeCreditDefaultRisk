# Forked from excellent kernel : https://www.kaggle.com/jsaguiar/updated-0-792-lb-lightgbm-with-simple-features
# From Kaggler : https://www.kaggle.com/jsaguiar
# Just removed a few min, max features. U can see the CV is not good. Dont believe in LB.

import lightgbm as lgb
import numpy as np
import pandas as pd
import gc
import time

from contextlib import contextmanager
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

from feature_extraction import bureau_and_balance, previous_applications, pos_cash, installments_payments, credit_card_balance, application_train_test, getAdditionalFeatures, get_amt_factor

warnings.simplefilter(action='ignore', category=FutureWarning)

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

# LightGBM GBDT with KFold or Stratified KFold
# Parameters from Tilii kernel: https://www.kaggle.com/tilii7/olivier-lightgbm-parameters-by-bayesian-opt/code
def kfold_lightgbm(df, num_folds, stratified = False, debug= False):

    # Divide in training/validation and test data
    train_df = df[df['TARGET'].notnull()]
    test_df = df[df['TARGET'].isnull()]

    print("Starting LightGBM. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))
    del df
    gc.collect()

    # Cross validation model
    if stratified:
        folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=47)
    else:
        folds = KFold(n_splits= num_folds, shuffle=True, random_state=47)

    # Create arrays and dataframes to store results
    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = [f for f in train_df.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]

    # 最初にsplitしないバージョンでモデルを推定します
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['TARGET'])):
        train_x, train_y = train_df[feats].iloc[train_idx], train_df['TARGET'].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['TARGET'].iloc[valid_idx]

        # set data structure
        lgb_train = lgb.Dataset(train_x,
                                label=train_y,
                                free_raw_data=False)
        lgb_test = lgb.Dataset(valid_x,
                               label=valid_y,
                               free_raw_data=False)

        # new params https://github.com/neptune-ml/open-solution-home-credit/wiki/LightGBM-clean-dynamic-features
        params = {
                'device' : 'gpu',
                'gpu_use_dp':True, #これで倍精度演算できるっぽいです
                'task': 'train',
#                'boosting_type': 'dart',
                'objective': 'binary',
                'metric': {'auc'},
                'num_threads': -1,
                'learning_rate': 0.02,
                'num_iteration': 10000,
                'num_leaves': 39,
                'colsample_bytree': 0.0587705926,
                'subsample': 0.5336340435,
                'max_depth': 7,
                'reg_alpha': 8.9675927624,
                'reg_lambda': 9.8953903428,
                'min_split_gain': 0.911786867,
                'min_child_weight': 37,
                'min_data_in_leaf': 629,
                'verbose': -1,
                'seed':int(2**n_fold),
                'bagging_seed':int(2**n_fold),
                'drop_seed':int(2**n_fold)
                }

        clf = lgb.train(
                        params,
                        lgb_train,
                        valid_sets=[lgb_train, lgb_test],
                        valid_names=['train', 'test'],
                        early_stopping_rounds= 200,
                        verbose_eval=100
                        )

        oof_preds[valid_idx] = clf.predict(valid_x, num_iteration=clf.best_iteration)
        sub_preds += clf.predict(test_df[feats], num_iteration=clf.best_iteration) / folds.n_splits

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = clf.feature_importance(importance_type='gain', iteration=clf.best_iteration)
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx])))
        del clf, train_x, train_y, valid_x, valid_y
        gc.collect()

    print('Full AUC score %.6f' % roc_auc_score(train_df['TARGET'], oof_preds))

    if not debug:
        # 提出データの予測値を保存
        test_df['TARGET'] = sub_preds
        test_df[['SK_ID_CURR', 'TARGET']].to_csv(submission_file_name, index= False)

        # out of foldの予測値を保存
        train_df['OOF_PRED'] = oof_preds
        train_df[['SK_ID_CURR', 'OOF_PRED']].to_csv(oof_file_name, index= False)

    return feature_importance_df

# Display/plot feature importance
def display_importances(feature_importance_df_, outputpath, csv_outputpath):
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:40].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]

    # importance下位の確認用に追加しました
    _feature_importance_df_=feature_importance_df_.groupby('feature').sum()
    _feature_importance_df_.to_csv(csv_outputpath)

    plt.figure(figsize=(8, 10))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig(outputpath)

def main(debug = False, use_csv=False):
    num_rows = 10000 if debug else None
    if use_csv:
        # TODO # appデータだけうまく読み込めませんでした
#        df = pd.read_csv('APP.csv', index_col=0)
        df = application_train_test(num_rows)
    else:
        df = application_train_test(num_rows)
        df.to_csv('APP.csv', index=False)
    with timer("Process bureau and bureau_balance"):
        if use_csv:
            if os.environ['USER'] == 'daiyamita':
                pass
            else:
                bureau = pd.read_csv('BUREAU.csv', index_col='SK_ID_CURR')
        else:
            bureau = bureau_and_balance(num_rows)
            if os.environ['USER'] == 'daiyamita':
                pass
            else:
                bureau.to_csv('BUREAU.csv')
        print("Bureau df shape:", bureau.shape)
        df = df.join(bureau, how='left', on='SK_ID_CURR')
        del bureau
        gc.collect()
    with timer("Process previous_applications"):
        if use_csv:
            prev = pd.read_csv('PREV.csv', index_col='SK_ID_CURR')
        else:
            prev = previous_applications(num_rows)
            prev.to_csv('PREV.csv')
        print("Previous applications df shape:", prev.shape)
        df = df.join(prev, how='left', on='SK_ID_CURR')
        # get_amt_factorここで使います
        prev_add = get_amt_factor(df,num_rows)
        df = df.join(prev_add, how='left', on='SK_ID_CURR')
        del prev, prev_add
        gc.collect()
    with timer("Process POS-CASH balance"):
        if use_csv:
            pos = pd.read_csv('POS.csv', index_col='SK_ID_CURR')
        else:
            pos = pos_cash(num_rows)
            pos.to_csv('POS.csv')
        print("Pos-cash balance df shape:", pos.shape)
        df = df.join(pos, how='left', on='SK_ID_CURR')
        del pos
        gc.collect()
    with timer("Process installments payments"):
        if use_csv:
            ins = pd.read_csv('INS.csv', index_col='SK_ID_CURR')
        else:
            ins = installments_payments(num_rows)
            ins.to_csv('INS.csv')
        print("Installments payments df shape:", ins.shape)
        df = df.join(ins, how='left', on='SK_ID_CURR')
        del ins
        gc.collect()
    with timer("Process credit card balance"):
        if use_csv:
            cc = pd.read_csv('CC.csv', index_col='SK_ID_CURR')
        else:
            cc = credit_card_balance(num_rows)
            cc.to_csv('CC.csv')
        print("Credit card balance df shape:", cc.shape)
        df = df.join(cc, how='left', on='SK_ID_CURR')
        del cc
        gc.collect()
    with timer("Process Additional Features"):
        df = getAdditionalFeatures(df)
    with timer("Run LightGBM with kfold"):
        feat_importance = kfold_lightgbm(df, num_folds= 10, stratified=True, debug= debug)
        display_importances(feat_importance ,'lgbm_importances.png', 'feature_importance_lgbm.csv')

if __name__ == "__main__":
    submission_file_name = "submission_add_feature_lgbm.csv"
    oof_file_name = "oof_lgbm.csv"
    with timer("Full model run"):
        if os.environ['USER'] == 'daiyamita':
            main(debug = True ,use_csv=False)
        else:
            main(debug = False, use_csv=True)
