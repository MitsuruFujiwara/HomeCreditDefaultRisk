import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Input
from keras.layers.core import Dropout
from keras.regularizers import l2

from roc_callback import roc_callback
from LightGBM_with_Simple_Features import bureau_and_balance, previous_applications, pos_cash, installments_payments, credit_card_balance, application_train_test

"""
LightGBMと同じ特徴量でDNNの予測値を出すためのスクリプト。
最終段階でアンサンブルモデルの一部として使うかもしれないので作っておきます。
"""

def loadData(num_rows=None, use_csv=False):
    # 元データの用意。
    if use_csv:
        # 事前に用意したcsvファイルを使う場合
        df = application_train_test(num_rows) # csvがうまく読み込めないのでappだけこのままです。
        bureau = pd.read_csv('BUREAU.csv', index_col='SK_ID_CURR')
        prev = pd.read_csv('PREV.csv', index_col='SK_ID_CURR')
        pos = pd.read_csv('POS.csv', index_col='SK_ID_CURR')
        ins = pd.read_csv('INS.csv', index_col='SK_ID_CURR')
        cc = pd.read_csv('CC.csv', index_col='SK_ID_CURR')
    else:
        # 加工済みのcsvを使わない場合
        df = application_train_test(num_rows) # csvがうまく読み込めないのでappだけこのままです。
        bureau = bureau_and_balance(num_rows)
        prev = previous_applications(num_rows)
        pos = pos_cash(num_rows)
        ins = installments_payments(num_rows)
        cc = credit_card_balance(num_rows)

        df = df.join(bureau, how='left', on='SK_ID_CURR')
        df = df.join(prev, how='left', on='SK_ID_CURR')
        df = df.join(pos, how='left', on='SK_ID_CURR')
        df = df.join(ins, how='left', on='SK_ID_CURR')
        df = df.join(cc, how='left', on='SK_ID_CURR')

    del bureau, prev, pos, ins, cc

    return df

def autoencoder(encoding_dim, decoding_dim, activation, X, nb_epoch):
    # set parameters
    input_data = Input(shape=(encoding_dim,))

    # set layer
    encoded = Dense(decoding_dim, activation=activation, W_regularizer=l2(0.0001))(input_data)
    decoded = Dense(encoding_dim, activation=activation, W_regularizer=l2(0.0001))(encoded)

    # set autoencoder
    _autoencoder = Model(input=input_data, output=decoded)
    _encoder = Model(input=input_data, output=encoded)

    # compile
    _autoencoder.compile(loss='mse', optimizer='adam')

    # fit autoencoder
    _autoencoder.fit(X,X, nb_epoch=nb_epoch, verbose=1)

    return _encoder

def _model(input_dim):
    """
    モデルの定義
    モデルのパラメータなど変える場合は基本的にこの中をいじればおｋ
    """
    model = Sequential()
    model.add(Dense(output_dim=1000, input_dim=input_dim, W_regularizer=l2(0.001)))
    model.add(Activation('relu'))
    model.add(Dropout(0.8))
    model.add(Dense(output_dim=1000, input_dim=1000, W_regularizer=l2(0.001)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_dim=1000, input_dim=1000, W_regularizer=l2(0.001)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_dim=500, input_dim=1000, W_regularizer=l2(0.001)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_dim=1, input_dim=500, W_regularizer=l2(0.001)))
    model.add(Activation('sigmoid'))
    return model

def getInitialWeights(trX, num_epoc, use_saved_params=False):

    print("Starting AutoEncoder. Train shape: {}".format(trX[0].shape))


    # 各層のinitial weightsを取得するため空のモデルを生成しておきます
    base_model = _model(trX[0].shape[1])

    # モデル各層の入力・出力値を取得し、事前学習用のdimsを定義
    w = base_model.get_weights()
    dims = [w[0].shape[0]] + [_w.shape[0] for _w in w[1::2]]

    # Auto Encoderにより各層のweightを求める
    encoders = []
    for i, t in enumerate(dims[:-1]):
        _X = trX[i]
        # fit autoencoder
        _encoder = autoencoder(t, dims[i+1], 'relu', _X, num_epoc)

        # save fitted encoder
        encoders.append(_encoder)

        # generate predicted value (for next encoder)
        trX.append(_encoder.predict(_X))

        del _encoder, _X

    # set initial weights
    for i, e in enumerate(encoders):
        w[i*2] = e.get_weights()[0]
        w[i*2+1] = e.get_weights()[1]

    del base_model, encoders

    return w

def kfold_DNN(df, num_folds, stratified = False, debug= False):
    """
    DNN用の前処理など
    """
    # set feature columns
    feats = [f for f in df.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]


    # 欠損値を平均で埋めておきます
    df[feats] = df[feats].astype('float64')
    df[feats] = df[feats].replace([np.inf, -np.inf], np.nan)
    df[feats] = df[feats].fillna(df[feats].mean())

    # DNN用のスケーリング
    ms = MinMaxScaler()
    df_ms = pd.DataFrame(ms.fit_transform(df[feats]), columns=feats, index=df.index)
    df_ms['TARGET']=df['TARGET']

    # 事前学習でモデルの初期値を求める #ここではTESTデータも含めて全てのデータを使います。
    trX = [np.array(df_ms[feats])]
    weights = getInitialWeights(trX, 5)

    """
    k-foldによるDNNモデルの推定
    """
    # Divide in training/validation and test data
    train_df = df_ms[df_ms['TARGET'].notnull()]
    test_df = df_ms[df_ms['TARGET'].isnull()]

    del df, df_ms
    gc.collect()

    # Cross validation model
    if stratified:
        folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=47)
    else:
        folds = KFold(n_splits= num_folds, shuffle=True, random_state=47)

    print("Starting DNN. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))

    # Create arrays and dataframes to store results
    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
    feature_importance_df = pd.DataFrame()

    # K-folds
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['TARGET'])):
        train_x, train_y = train_df[feats].iloc[train_idx], train_df['TARGET'].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['TARGET'].iloc[valid_idx]

        # set model
        model = _model(train_x.shape[1])

        # set early stopping
        es_cb = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')

        # set model weights
        model.set_weights(weights)

        # compile model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        # training
        history = model.fit(train_x, train_y, nb_epoch=1000, verbose=1,
                            validation_data=(valid_x, valid_y),
                            callbacks=[roc_callback(training_data=(train_x, train_y),
                            validation_data=(valid_x, valid_y)),es_cb])

        oof_preds[valid_idx] = model.predict_proba(valid_x)
        sub_preds += clf.predict(test_df[feats]) / folds.n_splits

        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx])))

        del model, train_x, train_y, valid_x, valid_y, history
        gc.collect()

    if not debug:
        # AUDスコアを上げるため提出ファイルの調整を追加→これは最終段階で使いましょう
        # 0or1に調整する水準を決定（とりあえず上位下位0.05%以下のものを調整）
#        q_high = test_df['TARGET'].quantile(0.9995)
#        q_low = test_df['TARGET'].quantile(0.0005)

#        test_df['TARGET'] = test_df['TARGET'].apply(lambda x: 1 if x > q_high else x)
#        test_df['TARGET'] = test_df['TARGET'].apply(lambda x: 0 if x < q_low else x)

        # 分離前モデルの予測値を保存
        test_df['TARGET'] = sub_preds
        test_df[['SK_ID_CURR', 'TARGET']].to_csv(submission_file_name, index= False)

    """
    # save model
    model.save('DNN_v3.h5')

    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    """

def main(debug = False, use_csv=False):
    num_rows = 10000 if debug else None

    # set data
    df = loadData(num_rows, use_csv)

    # model training
    kfold_DNN(df, num_folds=5)

if __name__ == '__main__':
    submission_file_name="submission_add_feature_dnn.csv"
    main(use_csv=True)
