
# cv: 0.747635

import numpy as np
import pandas as pd
import gc
import time

from contextlib import contextmanager
from sklearn.ensemble import RandomForestClassifier
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

def fill_NA(df):
    df=df.copy()
    Num_Features=df.select_dtypes(['float64','int64']).columns.tolist()
    df[Num_Features]= df[Num_Features].fillna(-999).replace([-np.inf, np.inf], -999)
    return df

# Random Forest with KFold or Stratified KFold
# Parameters from Tilii kernel: https://www.kaggle.com/prashantkikani/home-rf-et-xgb-cb-stack-oof-lb-0-778
def kfold_rf(df, num_folds, stratified = False, debug= False):

    # Divide in training/validation and test data
    train_df = df[df['TARGET'].notnull()]
    test_df = df[df['TARGET'].isnull()]

    print("Starting Random Forest. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))
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

    # fill nan
    train_df[feats]=fill_NA(train_df[feats])
    test_df[feats]=fill_NA(test_df[feats])

    # 最初にsplitしないバージョンでモデルを推定します
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['TARGET'])):
        train_x, train_y = train_df[feats].iloc[train_idx], train_df['TARGET'].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['TARGET'].iloc[valid_idx]

        # params from https://github.com/neptune-ml/open-solution-home-credit/blob/solution-5/configs/neptune.yaml#L104
        clf = RandomForestClassifier(
                                    n_jobs=4,
                                    n_estimators=500,
                                    criterion='gini',
                                    max_features=0.2,
                                    min_samples_split=10,
                                    min_samples_leaf=5,
#                                    class_weight=1,
                                    verbose=3,
                                    random_state=int(2**n_fold)
                                    )
        clf.fit(train_x, train_y)
        oof_preds[valid_idx] = clf.predict_proba(valid_x)[:,1]
        sub_preds += clf.predict_proba(test_df[feats])[:,1] / folds.n_splits

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = clf.feature_importances_
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
    plt.title('Random Forest Features (avg over folds)')
    plt.tight_layout()
    plt.savefig(outputpath)

def main(debug = False, use_csv=False):
    num_rows = 10000 if debug else None
    with timer("Process Read CSV"):
        df = pd.read_csv('df_selected.csv', nrows=num_rows)
    with timer("Run Random Forest with kfold"):
        feat_importance = kfold_rf(df, num_folds= 5, stratified=True, debug= debug)
        display_importances(feat_importance ,'rf_importances.png', 'feature_importance_rf.csv')

if __name__ == "__main__":
    submission_file_name = "submission_add_feature_rf.csv"
    oof_file_name = "oof_rf.csv"
    with timer("Full model run"):
        if os.environ['USER'] == 'daiyamita':
            main(debug = True ,use_csv=False)
        else:
            main(debug = False, use_csv=True)
