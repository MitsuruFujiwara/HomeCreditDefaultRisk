import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression

"""
複数モデルのoutputをブレンドして最終的なsubmitファイルを生成するスクリプト。
とりあえず最後ロジスティック回帰で推定します。
参考:https://www.kaggle.com/eliotbarr/stacking-test-sklearn-xgboost-catboost-lightgbm
"""

def main():
    # load datasets
    print('Loading Datasets...')
    df = pd.read_csv('application_train.csv')
    sub = pd.read_csv('submission.csv')

    # list of file names
    trainfiles = ['oof_lgbm.csv', 'oof_xgb.csv']
    testfiles = ['submission_add_feature_lgbm.csv', 'submission_add_feature_xgb.csv']
    cols = ['lgbm', 'xgb']

    # get predicted value of each models
    for path_train, path_test, c in zip(trainfiles, testfiles, cols):
        df[c] = np.log(pd.read_csv(path_train)['OOF_PRED'])
        sub[c] = np.log(pd.read_csv(path_test)['TARGET'])

    # drop Nan
    df = df[df[cols[0]].notnull()]
    # set train & test data
    trX, trY = df[cols], df['TARGET']
    valX = sub[cols]

    # とりあえずノーマルなロジスティック回帰で推定します # TODO: この部分XGBoostでやる場合が多いみたいです。
    print("Starting LogisticRegression. Train shape: {}, test shape: {}".format(trX.shape, valX.shape))
    logistic_regression = LogisticRegression(random_state=326)
    logistic_regression.fit(trX,trY)
    sub['TARGET'] = logistic_regression.predict_proba(valX)[:,1]

    # AUDスコアを上げるため提出ファイルの調整を追加
    # 0or1に調整する水準を決定（とりあえず上位下位0.05%以下のものを調整）
    """
    q_high = submission_lb['TARGET'].quantile(0.9995)
    q_low = submission_lb['TARGET'].quantile(0.0005)

    submission_lb['TARGET'] = submission_lb['TARGET'].apply(lambda x: 1 if x > q_high else x)
    submission_lb['TARGET'] = submission_lb['TARGET'].apply(lambda x: 0 if x < q_low else x)
    """

    # 最終結果を保存
    sub[['SK_ID_CURR', 'TARGET']].to_csv('submission_stacked.csv', index= False)

if __name__ == '__main__':
    main()
