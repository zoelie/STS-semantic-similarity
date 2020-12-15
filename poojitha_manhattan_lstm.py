import os
from sys import exit

from time import time
import datetime
import argparse
from math import exp

import tensorflow as tf

import keras
import keras.backend as K
from keras.models import Model
from keras.layers import Activation
from keras.layers import Embedding, Input
from keras.layers import LSTM, Lambda, concatenate
from keras import regularizers
from configs import * 

import numpy as np

import matplotlib
from matplotlib import pyplot as plt

from data import Data

import numpy as np
from gensim.models import KeyedVectors
import pandas as pd


class embeddings(object):
    def __init__(self, file_path, word_index):
        self.embedding_size = 300 
        self.matrix = self.get_embedding_matrix(file_path, word_index)

    def get_embedding_matrix(self, file_path, word_index):
        word2vec = KeyedVectors.load_word2vec_format(file_path, binary=True)

        # Prepare Embedding Matrix.
        matrix = np.zeros((len(word_index)+1, self.embedding_size))

        for word, i in word_index.items():
            if word not in word2vec.vocab:
                continue
            matrix[i] = word2vec.word_vec(word)

        del word2vec
        return matrix
    
    
def exponent_neg_manhattan_distance(x, hidden_size=50):
    ''' Helper function for the similarity estimate of the LSTMs outputs '''
    return K.exp(-K.sum(K.abs(x[:,:hidden_size] - x[:,hidden_size:]), axis=1, keepdims=True))

def exponent_neg_cosine_distance(x, hidden_size=50):
    ''' Helper function for the similarity estimate of the LSTMs outputs '''
    leftNorm = K.l2_normalize(x[:,:hidden_size], axis=-1)
    rightNorm = K.l2_normalize(x[:,hidden_size:], axis=-1)
    return K.exp(K.sum(K.prod([leftNorm, rightNorm], axis=0), axis=1, keepdims=True))

if __name__ == "__main__":
    print("loading_data")
    data_gen = Data(data_name, file_name, training_ratio,maximum_len,"similarity_score","training")

    train_X = data_gen.train_X
    train_Y = data_gen.train_Y
    val_X = data_gen.val_X
    val_Y = data_gen.val_Y
    vocabulary_len = data_gen.index
    maximum_len = data_gen.maximum_len

    print('Training Samples in the dataframe:', len(train_X[0]))
    print('Validation Samples in the dataframe :', len(val_X[0]))
    print('Maximum sequence length           :', maximum_len)
    print('Building Embedding Matrix')
    embedding = embeddings(embedding_file, data_gen.word_to_id)
    embedding_size = embedding.matrix.shape[1]

    print('Defining the LSRTM Model')

    seq_1 = Input(shape=(maximum_len,), dtype='int32', name='sequence1')
    seq_2 = Input(shape=(maximum_len,), dtype='int32', name='sequence2')

    embed_layer = Embedding(output_dim=embedding_size, input_dim=vocabulary_len+1, input_length=maximum_len, trainable=False)
    embed_layer.build((None,))
    embed_layer.set_weights([embedding.matrix])

    input_1 = embed_layer(seq_1)
    input_2 = embed_layer(seq_2)

    l1 = LSTM(units=hidden_size)

    output_l1 = l1(input_1)
    output_l2 = l1(input_2)

    concatination = concatenate([output_l1, output_l2], axis=-1)

    actual_output = Lambda(exponent_neg_cosine_distance, output_shape=(1,))(concatination)

    LSTM_model = Model(inputs=[seq_1, seq_2], outputs=[actual_output])

    optimizer= keras.optimizers.Adadelta(lr=learning_rate, clipnorm=1.25)

    LSTM_model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=[keras.metrics.RootMeanSquaredError()])
    LSTM_model.summary()

    training_history = LSTM_model.fit(train_X, train_Y, validation_data=(val_X, val_Y),
                        epochs=epochs, batch_size=batch_size, verbose=1)

    print("Training Finished")

    # "Accuracy"
    plt.figure(1)
    plt.plot(training_history.history['root_mean_squared_error'])
    plt.plot(training_history.history['val_root_mean_squared_error'])
    plt.title('Model RMSE')
    plt.ylabel('RMSE')
    plt.xlabel('epoch')
    plt.legend(['Training', 'Validation'], loc='upper left')
    plt.show()

    # "Loss"
    plt.figure(2)
    plt.plot(training_history.history['loss'])
    plt.plot(training_history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['Training', 'Validation'], loc='upper left')
    plt.show()
    
    #Making predictions on the validation data
    predict_Y = LSTM_model.predict(val_X)
    predict_Y = predict_Y.reshape(-1)
    corr_value = pd.Series(val_Y).corr(pd.Series(predict_Y))
    print("The correlation between the actual values and predicted vaues is: "+str(corr_value))

