import gc
import pandas as pd
import xgboost

from bayes_opt import BayesianOptimization
from LightGBM_with_Simple_Features import bureau_and_balance, previous_applications, pos_cash, installments_payments, credit_card_balance, application_train_test

# 以下参考
# https://github.com/fmfn/BayesianOptimization
# https://www.kaggle.com/tilii7/olivier-lightgbm-parameters-by-bayesian-opt/code

NUM_ROWS=None
USE_CSV=True

if USE_CSV:
    DF = application_train_test(NUM_ROWS)
    BUREAU = pd.read_csv('BUREAU.csv', index_col=0)
    PREV = pd.read_csv('PREV.csv', index_col=0)
    POS = pd.read_csv('POS.csv', index_col=0)
    INS = pd.read_csv('INS.csv', index_col=0)
    CC = pd.read_csv('CC.csv', index_col=0)
else:
    # ベストスコアのカーネルと同じ特徴量を使います
    DF = application_train_test(NUM_ROWS)
    BUREAU = bureau_and_balance(NUM_ROWS)
    PREV = previous_applications(NUM_ROWS)
    POS = pos_cash(NUM_ROWS)
    INS = installments_payments(NUM_ROWS)
    CC = credit_card_balance(NUM_ROWS)

    # save data
    DF.to_csv('APP.csv')
    BUREAU.to_csv('BUREAU.csv')
    PREV.to_csv('PREV.csv')
    POS.to_csv('POS.csv')
    INS.to_csv('INS.csv')
    CC.to_csv('CC.csv')

# concat datasets
DF = DF.join(BUREAU, how='left', on='SK_ID_CURR')
DF = DF.join(PREV, how='left', on='SK_ID_CURR')
DF = DF.join(POS, how='left', on='SK_ID_CURR')
DF = DF.join(INS, how='left', on='SK_ID_CURR')
DF = DF.join(CC, how='left', on='SK_ID_CURR')

"""
# 不要なカラムを落とす処理を追加
DROPCOLUMNS=pd.read_csv('feature_importance_not_to_use.csv')
DROPCOLUMNS = DROPCOLUMNS['feature'].tolist()
DROPCOLUMNS = [d for d in DROPCOLUMNS if d in df.columns.tolist()]

DF = df.drop(DROPCOLUMNS, axis=1)
"""

del BUREAU, PREV, POS, INS, CC

# split test & train
TRAIN_DF = DF[DF['TARGET'].notnull()]
TEST_DF = DF[DF['TARGET'].isnull()]

xgb_train = xgboost.DMatrix(TRAIN_DF.drop('TARGET', axis=1),
                        label=TRAIN_DF['TARGET'])

del TRAIN_DF, TEST_DF

def xgb_eval(gamma,
             max_depth,
             min_child_weight,
             subsample,
             colsample_bytree,
             colsample_bylevel,
             alpha,
             _lambda):

    params = {
            'objective':'gpu:binary:logistic', # GPU parameter
            'booster': 'gbtree',
            'eval_metric':'auc',
            'silent':1,
            'eta': 0.02,
            'tree_method': 'gpu_hist', # GPU parameter
            'predictor': 'gpu_predictor', # GPU parameter
            'seed':326
            }

    params['gamma'] = gamma
    params['max_depth'] = int(max_depth)
    params['min_child_weight'] = min_child_weight
    params['subsample'] = max(min(subsample, 1), 0)
    params['colsample_bytree'] = max(min(colsample_bytree, 1), 0)
    params['colsample_bylevel'] = max(min(colsample_bylevel, 1), 0)
    params['alpha'] = max(alpha, 0)
    params['lambda'] = max(_lambda, 0)

    clf = xgboost.cv(params=params,
                     dtrain=xgb_train,
                     num_boost_round=10000, # early stopありなのでここは大きめの数字にしてます
                     nfold=3,
                     metrics=["auc"],
                     folds=None,
                     early_stopping_rounds=200,
                     verbose_eval=100,
                     seed=47,
                     )
    gc.collect()
    return clf['test-auc-mean'].iloc[-1]

def main():
    # clf for bayesian optimization
    clf_bo = BayesianOptimization(xgb_eval, {'gamma':(0, 1),
                                             'max_depth': (5, 8),
                                             'min_child_weight': (0, 45),
                                             'subsample': (0.001, 1),
                                             'colsample_bytree': (0.001, 1),
                                             'colsample_bylevel': (0.001, 1),
                                             'alpha': (0, 10),
                                             '_lambda': (0, 10)
                                             })

    clf_bo.maximize(init_points=15, n_iter=25)

    res = pd.DataFrame(clf_bo.res['max']['max_params'], index=['max_params'])

    res.to_csv('max_params_xgb.csv')

if __name__ == '__main__':
    main()
