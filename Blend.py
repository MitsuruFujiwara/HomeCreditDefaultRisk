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
とりあえず最後ロジスティック回帰で推定します。
参考:https://www.kaggle.com/eliotbarr/stacking-test-sklearn-xgboost-catboost-lightgbm
"""

"""
# 最終スコア提出用にこれコピペして使います。
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

        # TODO: ここのパラメータチューニング
        # とりあえず参考: https://www.kaggle.com/mingpeiyu/stacking-test-sklearn-xgboost-catboost-lightgbm
        params = {
                'device' : 'gpu',
                'gpu_use_dp':True, #これで倍精度演算できるっぽいです
                'task': 'train',
#                'boosting_type': 'dart',
                'objective': 'binary',
                'metric': {'auc'},
                'num_threads': -1,
#                'num_iteration': 10000,
                'learning_rate': 0.01,
                'max_depth': 1,
#                'num_leaves': 30,
#                'min_child_samples':70,
                'subsample': 0.6,
#                'subsample_freq': 1,
                'colsample_bytree': 0.8,
                'num_parallel_tree': 1,
                'min_child_weight': 1,
#                'min_gain_to_split': 0.5,
#                'reg_lambda': 100,
#                'reg_alpha': 0.0,
#                'scale_pos_weight': 1,
#                'is_unbalance': False,
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

    return feature_importance_df

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

def getData(num_rows=None):
    # load datasets
    print('Loading Datasets...')
    df = pd.read_csv('application_train.csv', nrows= num_rows)
    sub = pd.read_csv('sample_submission.csv', nrows= num_rows)
    sub['TARGET'] = np.nan

    # list of file names
    trainfiles = ['oof_lgbm.csv', 'oof_xgb.csv', 'oof_dnn.csv']
    testfiles = ['submission_add_feature_lgbm.csv',
                 'submission_add_feature_xgb.csv',
                 'submission_add_feature_dnn.csv']
    cols = ['lgbm', 'xgb', 'dnn']

    # get predicted value of each models
    for path_train, path_test, c in zip(trainfiles, testfiles, cols):
        df[c] = pd.read_csv(path_train)['OOF_PRED']
        sub[c] = pd.read_csv(path_test)['TARGET']

    df = df[cols+['TARGET', 'SK_ID_CURR']].dropna()
    sub = sub[cols+['TARGET', 'SK_ID_CURR']]

    df = pd.concat([df, sub])

    return df

def main(debug = False):
    # get dataset
    df = getData()

    # fit lightgbm
    feat_importance = kfold_lightgbm(df, num_folds= 5, stratified=True, debug= debug)

    # save feature importance
    display_importances(feat_importance ,'final_importances.png', 'feature_importance_final.csv')

if __name__ == '__main__':
    submission_file_name = 'submission_stacked.csv'
    main()
"""

def main():
    data = {}
    filepaths=['submission_add_feature.csv','submission_add_feature_xgb.csv']
    weights = [0.5, 0.5]

    for path in filepaths:
        data[path[:-4]] = pd.read_csv(path)

    ranks = pd.DataFrame(columns=data.keys())

    for key in data.keys():
        ranks[key] = data[key].TARGET.rank(method='min')
    ranks['Average'] = ranks.mean(axis=1)
    ranks['Scaled Rank'] = (ranks['Average'] - ranks['Average'].min()) / (ranks['Average'].max() - ranks['Average'].min())
    print(ranks.corr()[:1])

    ranks['Score'] = ranks[['submission_add_feature','submission_add_feature_xgb']].mul(weights).sum(1) / ranks.shape[0]
    submission_lb = pd.read_csv('submission.csv')
    submission_lb['TARGET'] = ranks['Score']
    submission_lb.to_csv("WEIGHT_AVERAGE_RANK.csv", index=None)

if __name__ == '__main__':
    main()
