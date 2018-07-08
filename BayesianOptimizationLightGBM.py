import pandas as pd

# 以下参考
# https://github.com/fmfn/BayesianOptimization
# https://www.kaggle.com/tilii7/olivier-lightgbm-parameters-by-bayesian-opt/code

train_target = application_train['TARGET'].values
train_data = application_train.drop(['TARGET'], axis = 1).values

lgbm_train = lightgbm.Dataset(train_data, train_target)
fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=666)
cv_fold = fold.split(train_data, train_target)
params = dict()
params["learning_rate"] = 0.02
params["silent"] = True
params["nthread"] = 16
params["application"] = "binary"

def lgbm_eval(num_leaves,
              colsample_bytree,
              subsample,
              max_depth,
              reg_alpha,
              reg_lambda,
              min_split_gain,
              min_child_weight,
             ):
    params["num_leaves"] = int(num_leaves)
    params['colsample_bytree'] = max(min(colsample_bytree, 1), 0)
    params['subsample'] = max(min(subsample, 1), 0)
    params['max_depth'] = int(max_depth)
    params['reg_alpha'] = max(reg_alpha, 0)
    params['reg_lambda'] = max(reg_lambda, 0)
    params['min_split_gain'] = min_split_gain
    params['min_child_weight'] = min_child_weight



    clf = lightgbm.cv(params=params,
                      train_set=lgbm_train,
                      metrics=["auc"],
                      folds=cv_fold,
                      num_boost_round=5,
                      # early_stopping_rounds=100,
                      verbose_eval=1,
                     )

    return clf['auc-mean'][-1]


clf_bo = BayesianOptimization(lgbm_eval, {'num_leaves': (30, 45),
                                          'colsample_bytree': (0.1, 1),
                                          'subsample': (0.1, 1),
                                          'max_depth': (5, 15),
                                          'reg_alpha': (0, 10),
                                          'reg_lambda': (0, 10),
                                          'min_split_gain': (0, 1),
                                          'min_child_weight': (30, 45),
                                        })

clf_bo.maximize(init_points=4, n_iter=20)
