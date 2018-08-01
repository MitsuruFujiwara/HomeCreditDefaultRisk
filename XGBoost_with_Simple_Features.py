# Forked from excellent kernel : https://www.kaggle.com/jsaguiar/updated-0-792-lb-lightgbm-with-simple-features
# From Kaggler : https://www.kaggle.com/jsaguiar
# Just removed a few min, max features. U can see the CV is not good. Dont believe in LB.

import lightgbm as lgb
import numpy as np
import pandas as pd
import gc
import time
import xgboost as xgb

from contextlib import contextmanager
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

from LightGBM_with_Simple_Features import bureau_and_balance, previous_applications, pos_cash, installments_payments, credit_card_balance, application_train_test

warnings.simplefilter(action='ignore', category=FutureWarning)

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

# XGBoost GBDT with KFold or Stratified KFold
def kfold_xgboost(df, num_folds, stratified = False, debug= False):

    # Divide in training/validation and test data
    train_df = df[df['TARGET'].notnull()]
    test_df = df[df['TARGET'].isnull()]

    print("Starting XGBoost. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))
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

    # final predict用にdmatrix形式のtest dfを作っておきます
    test_df_dmtrx = xgb.DMatrix(test_df[feats], label=train_df['TARGET'])

    # 最初にsplitしないバージョンでモデルを推定します
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['TARGET'])):
        train_x, train_y = train_df[feats].iloc[train_idx], train_df['TARGET'].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['TARGET'].iloc[valid_idx]

        # set data structure
        xgb_train = xgb.DMatrix(train_x,
                                label=train_y)
        xgb_test = xgb.DMatrix(valid_x,
                               label=valid_y)

        # LightGBM parameters found by Bayesian optimization
        params = {
                'device' : 'gpu',
                'task': 'train',
#                'boosting_type': 'dart',
                'objective': 'binary',
                'metric': {'auc'},
                'num_threads': -1,
                'learning_rate': 0.02,
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
                'seed':326,
                'bagging_seed':326,
                'drop_seed':326
                }

        clf = xgb.train(
                        params,
                        lgb_train,
                        num_boost_round=10000
                        evals=[(lgb_train,'train'),(lgb_test,'test')],
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
        # AUDスコアを上げるため提出ファイルの調整を追加→これは最終段階で使いましょう
        # 0or1に調整する水準を決定（とりあえず上位下位0.05%以下のものを調整）
        q_high = test_df['TARGET'].quantile(0.9995)
        q_low = test_df['TARGET'].quantile(0.0005)

        test_df['TARGET'] = test_df['TARGET'].apply(lambda x: 1 if x > q_high else x)
        test_df['TARGET'] = test_df['TARGET'].apply(lambda x: 0 if x < q_low else x)

        # 分離前モデルの予測値を保存
        test_df['TARGET'] = sub_preds
        test_df[['SK_ID_CURR', 'TARGET']].to_csv(submission_file_name, index= False)

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
        del prev
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
    with timer("Run LightGBM with kfold"):
        # 不要なカラムを落とさない方がスコア高かったのでとりあえずここはコメントアウトしてます
        # 不要なカラムを落とす処理を追加
        dropcolumns=pd.read_csv('feature_importance_not_to_use.csv')
        dropcolumns = dropcolumns['feature'].tolist()
        dropcolumns = [d for d in dropcolumns if d in df.columns.tolist()]

        df = df.drop(dropcolumns, axis=1)

        # 通常モデルのみ推定
        feat_importance = kfold_lightgbm(df, num_folds= 5, stratified=True, debug= debug)

        display_importances(feat_importance ,'lgbm_importances.png', 'feature_importance.csv')

if __name__ == "__main__":
    submission_file_name = "submission_add_feature.csv"
    submission_file_name_split = "submission_add_feature_split.csv"
    with timer("Full model run"):
        if os.environ['USER'] == 'daiyamita':
            main(debug = True ,use_csv=False)
        else:
            main(use_csv=True)
