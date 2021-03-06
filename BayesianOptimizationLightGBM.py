import gc
import pandas as pd
import lightgbm

from bayes_opt import BayesianOptimization
from feature_extraction import bureau_and_balance, previous_applications, pos_cash, installments_payments, credit_card_balance, application_train_test, getAdditionalFeatures, get_amt_factor

# 以下参考
# https://github.com/fmfn/BayesianOptimization
# https://www.kaggle.com/tilii7/olivier-lightgbm-parameters-by-bayesian-opt/code

NUM_ROWS=None
USE_CSV=True

if USE_CSV:
    DF = application_train_test(NUM_ROWS)
    BUREAU = pd.read_csv('BUREAU.csv', index_col=0)
    PREV = pd.read_csv('PREV.csv', index_col=0)
    PREV_ADD = get_amt_factor(DF,NUM_ROWS)
    POS = pd.read_csv('POS.csv', index_col=0)
    INS = pd.read_csv('INS.csv', index_col=0)
    CC = pd.read_csv('CC.csv', index_col=0)
else:
    # ベストスコアのカーネルと同じ特徴量を使います
    DF = application_train_test(NUM_ROWS)
    BUREAU = bureau_and_balance(NUM_ROWS)
    PREV = previous_applications(NUM_ROWS)
    PREV_ADD = get_amt_factor(DF,NUM_ROWS)
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
DF = DF.join(PREV_ADD, how='left', on='SK_ID_CURR')
DF = DF.join(POS, how='left', on='SK_ID_CURR')
DF = DF.join(INS, how='left', on='SK_ID_CURR')
DF = DF.join(CC, how='left', on='SK_ID_CURR')
DF = getAdditionalFeatures(DF)

del BUREAU, PREV, PREV_ADD, POS, INS, CC

# split test & train
TRAIN_DF = DF[DF['TARGET'].notnull()]
FEATS = [f for f in TRAIN_DF.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]

lgbm_train = lightgbm.Dataset(TRAIN_DF[FEATS],
                              TRAIN_DF['TARGET'],
                              free_raw_data=False
                              )

del TRAIN_DF

def lgbm_eval(num_leaves,
              colsample_bytree,
              subsample,
              max_depth,
              reg_alpha,
              reg_lambda,
              min_split_gain,
              min_child_weight,
              min_data_in_leaf
              ):

    params = dict()
    params["learning_rate"] = 0.02
#    params["silent"] = True
    params['device'] = 'gpu'
#    params["nthread"] = 16
    params["application"] = "binary"
    params['seed']=326,
    params['bagging_seed']=326

    params["num_leaves"] = int(num_leaves)
    params['colsample_bytree'] = max(min(colsample_bytree, 1), 0)
    params['subsample'] = max(min(subsample, 1), 0)
    params['max_depth'] = int(max_depth)
    params['reg_alpha'] = max(reg_alpha, 0)
    params['reg_lambda'] = max(reg_lambda, 0)
    params['min_split_gain'] = min_split_gain
    params['min_child_weight'] = min_child_weight
    params['min_data_in_leaf'] = int(min_data_in_leaf)
    params['verbose']=-1

    clf = lightgbm.cv(params=params,
                      train_set=lgbm_train,
                      metrics=["auc"],
                      nfold=5,
                      folds=None,
                      num_boost_round=10000, # early stopありなのでここは大きめの数字にしてます
                      early_stopping_rounds=200,
                      verbose_eval=100,
                      seed=47,
                     )
    gc.collect()
    return clf['auc-mean'][-1]

def main():

    # clf for bayesian optimization
    clf_bo = BayesianOptimization(lgbm_eval, {'num_leaves': (32, 128),
                                              'colsample_bytree': (0.001, 1),
                                              'subsample': (0.001, 1),
                                              'max_depth': (3, 8),
                                              'reg_alpha': (0, 10),
                                              'reg_lambda': (0, 10),
                                              'min_split_gain': (0, 1),
                                              'min_child_weight': (0, 45),
                                              'min_data_in_leaf': (0, 1000),
                                              })

    clf_bo.maximize(init_points=15, n_iter=25)

    res = pd.DataFrame(clf_bo.res['max']['max_params'], index=['max_params'])

    res.to_csv('max_params_lgbm.csv')

if __name__ == '__main__':
    main()
