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
参考:https://www.kaggle.com/eliotbarr/stacking-test-sklearn-xgboost-catboost-lightgbm
"""

def main():
    data = {}
    filepaths=['submission_add_feature_lgbm.csv','submission_add_feature_xgb.csv']
    weights = [0.5, 0.5]

    for path in filepaths:
        data[path[:-4]] = pd.read_csv(path)

    ranks = pd.DataFrame(columns=data.keys())

    for key in data.keys():
        ranks[key] = data[key].TARGET.rank(method='min')
    ranks['Average'] = ranks.mean(axis=1)
    ranks['Scaled Rank'] = (ranks['Average'] - ranks['Average'].min()) / (ranks['Average'].max() - ranks['Average'].min())
    print(ranks.corr()[:1])

    ranks['Score'] = ranks[['submission_add_feature_lgbm','submission_add_feature_xgb']].mul(weights).sum(1) / ranks.shape[0]
    submission_lb = pd.read_csv('submission.csv')
    submission_lb['TARGET'] = ranks['Score']

    # AUDスコアを上げるため提出ファイルの調整を追加→これは最終段階で使いましょう
    # 0or1に調整する水準を決定（とりあえず上位下位0.05%以下のものを調整）
#    q_high = test_df['TARGET'].quantile(0.9995)
    q_low = submission_lb['TARGET'].quantile(0.1)

#    test_df['TARGET'] = test_df['TARGET'].apply(lambda x: 1 if x > q_high else x)
    submission_lb['TARGET'] = submission_lb['TARGET'].apply(lambda x: 0 if x < q_low else x)

    submission_lb.to_csv("WEIGHT_AVERAGE_RANK.csv", index=None)

if __name__ == '__main__':
    main()
