import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation
from keras.layers.core import Dropout
from keras.regularizers import l2

def _model():
    model = Sequential()
    model.add(Dense(output_dim=1000, input_dim=238, W_regularizer=l2(0.0001)))
    model.add(Activation('relu'))
    model.add(Dropout(0.8))
    model.add(Dense(output_dim=1000, input_dim=1000, W_regularizer=l2(0.0001)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_dim=1000, input_dim=1000, W_regularizer=l2(0.0001)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_dim=1, input_dim=1000, W_regularizer=l2(0.0001)))
    model.add(Activation('sigmoid'))
    return model

def main():
    # load Data
    df = pd.read_hdf('db.h5', key='all', mode='r')
    df = df.dropna()

    X, Y = np.array(df.drop(['label', 'IS_TEST'], axis=1)), np.array(df['label'])

    # split Data
    x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.1, random_state=18)

    # set model
    model = _model()

    # load autoencoder
    encoder0 = load_model('encoder0.h5')
    encoder1 = load_model('encoder1.h5')
    encoder2 = load_model('encoder2.h5')
    encoder3 = load_model('encoder3.h5')
#    encoder4 = load_model('encoder4.h5')


    # set initial weights
    w = model.get_weights()
    w[0] = encoder0.get_weights()[0]
    w[1] = encoder0.get_weights()[1]
    w[2] = encoder1.get_weights()[0]
    w[3] = encoder1.get_weights()[1]
    w[4] = encoder2.get_weights()[0]
    w[5] = encoder2.get_weights()[1]
    w[6] = encoder3.get_weights()[0]
    w[7] = encoder3.get_weights()[1]
#    w[8] = encoder4.get_weights()[0]
#    w[9] = encoder4.get_weights()[1]
    model.set_weights(w)

    # compile
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # training
    history = model.fit(x_train, y_train, nb_epoch=5, verbose=1, validation_data=(x_val, y_val))

    # save model
    model.save('DNN_v2.h5')

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

if __name__ == '__main__':
    main()
