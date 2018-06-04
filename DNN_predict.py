import pandas as pd
import numpy as np

from keras.models import load_model
from sklearn.externals import joblib

from DNN_preprocessing import col_numeric, col_category, col_flag
from DNN_preprocessing import NAME_CONTRACT_TYPE_MAP, CODE_GENDER_MAP, FLAG_OWN_XXX_MAP, EMERGENCYSTATE_MODE_MAP

def main():

    # load test data
    df = pd.read_hdf('db.h5', key='test', mode='r')
    X = [np.array(df.drop(['label', 'IS_TEST'], axis=1))]

    # load model
    model = load_model('DNN_v2.h5')

    # compile
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    predict=model.predict_proba(X)

    sub_dnn = pd.DataFrame()
    sub_dnn['SK_ID_CURR'] = df.index.tolist()
    sub_dnn['TARGET'] = predict
    sub_dnn.to_csv("sub_dnn_v5.csv", index=False)

    return sub_dnn

if __name__ == '__main__':
    res = main()
    print(res)
