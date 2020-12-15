from re import sub

import csv
import itertools
import random
from random import shuffle

import pandas as pd
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split 

import keras
from keras.preprocessing.sequence import pad_sequences
import numpy as np

class Data(object):
    def __init__(self, data_name, data_file, train_ratio=0.8, max_len=None,score_col=None,training_mode=None):
        self.file_name = data_file
        self.train_ratio = train_ratio
        self.maximum_len = max_len
        self.index = 1
        self.vocab_limit = None
        self.similarity_score = score_col
        self.sentence_cols = ["Senetence_A","Senetence_b"]
        self.training_mode = training_mode
        self.train_X = list()
        self.test_X = list()
        self.test_Y = list()
        self.train_Y = list()
        self.val_X = list()
        self.val_Y = list()
        self.vocab = set('PAD')
        self.word_to_id = {'PAD':0}
        self.id_to_word = {0:'PAD'}
        self.word_to_count = dict()
        self.run()

    def get_words_from_text(self, text):
        text = str(text)
        text = text.lower()
        text = text.split()

        return text

    def loading_data_from_file(self):
        dataframe = pd.read_csv(self.file_name, sep=',')
        dataframe = dataframe.iloc[:,1:]
        for index, row in dataframe.iterrows():
            for sequence in self.sentence_cols:
                sentence_to_index = [] 
                for word in self.get_words_from_text(row[sequence]):
                    if word not in self.vocab:
                        self.vocab.add(word)
                        self.word_to_id[word] = self.index
                        self.word_to_count[word] = 1
                        sentence_to_index.append(self.index)
                        self.id_to_word[self.index] = word
                        self.index += 1
                    else:
                        self.word_to_count[word] += 1
                        sentence_to_index.append(self.word_to_id[word])

                # Replace |sequence as word| with |sequence as number| representation
                dataframe.at[index, sequence] = sentence_to_index

        return dataframe

    def pad_sequences(self):
        if self.training_mode == "training":
            self.maximum_len = max(max(len(seq) for seq in self.train_X[0]),
                               max(len(seq) for seq in self.train_X[1]),
                               max(len(seq) for seq in self.val_X[0]),
                               max(len(seq) for seq in self.val_X[1]))

        # Zero padding
            for dataframe, col in itertools.product([self.train_X, self.val_X], [0, 1]):
                if self.maximum_len: dataframe[col] = pad_sequences(dataframe[col], maxlen=self.maximum_len)
                else : dataframe[col] = pad_sequences(dataframe[col])
        else:
            self.maximum_len = max(max(len(seq) for seq in self.test_X[0]),
                               max(len(seq) for seq in self.test_X[1]))

        # Zero padding
            for dataframe, col in itertools.product([self.test_X], [0, 1]):
                if self.maximum_len: dataframe[col] = pad_sequences(dataframe[col], maxlen=self.maximum_len)
                else : dataframe[col] = pad_sequences(dataframe[col])

    def run(self):
        # Loading data and building vocabulary.
        sentence_df = self.loading_data_from_file()
        size = len(sentence_df)

        X = sentence_df[self.sentence_cols]
        Y = sentence_df[self.similarity_score]
        
        if self.training_mode == "training":
            self.train_X, self.val_X, self.train_Y, self.val_Y = train_test_split(X, Y, train_size=self.train_ratio)
            self.train_X = [self.train_X[column] for column in self.sentence_cols]
            self.val_X = [self.val_X[column] for column in self.sentence_cols]

        # Convert labels to their numpy representations
            self.train_Y = self.train_Y.values
            self.val_Y = self.val_Y.values
        else:
            self.test_X = [X[column] for column in self.sentence_cols]
            self.test_Y = Y.values

        # Padding Sequences.
        self.pad_sequences()
