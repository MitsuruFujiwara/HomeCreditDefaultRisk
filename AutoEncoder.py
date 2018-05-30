import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import Model
from keras.layers import Input, Dense
from keras.regularizers import l2

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
    _autoencoder.fit(X,X, nb_epoch=nb_epoch, verbose=2)

    return _encoder

def main():
    # load Data
    df = pd.read_hdf('db.h5', key='data', mode='r')

    # set data
    trX = [np.array(df.drop('label', axis=1))]

    # set dimension of model
    dim = [226, 300, 300, 1]

    for i, t in enumerate(dim[:-1]):
        _X = trX[i]
        # fit autoencoder
        e = autoencoder(t, dim[i+1], 'relu', _X, 100)

        # save fitted encoder
        e.save('encoder' + str(i) + '.h5')

        # generate predicted value (for next encoder)
        trX.append(e.predict(_X))

if __name__ == '__main__':
    main()
