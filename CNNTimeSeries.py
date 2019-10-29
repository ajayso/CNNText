# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 10:51:44 2019

@author: Ajay Solanki
"""

#Imports

import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense,SimpleRNN,LSTM
from keras import layers
from keras.optimizers import RMSprop
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from keras.datasets import imdb
from keras.preprocessing import sequence



class CNN_1DConvnets:
    # Load the data, text and labels
    def load_data(self):
        max_features = 10000
        max_len = 500 
        (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
        x_train = sequence.pad_sequences(x_train, maxlen=max_len)
        x_test = sequence.pad_sequences(x_test, maxlen=max_len)
        print('x_train shape:', x_train.shape)
        print('x_test shape:', x_test.shape)
        print('y_train shape:', y_train.shape)
        print('y_test shape:', y_test.shape)
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        
    
    def plot(self):
        acc = self.history.history['acc']
        val_acc = self.history.history['val_acc']
        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']
        
        epochs = range(1, len(acc) + 1)
        plt.plot(epochs, acc, 'bo', label='Training acc')
        plt.plot(epochs, val_acc, 'b', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.legend()
        plt.figure()
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
        plt.show()
        
    
        

        
        
        
cnn_1DConvnets = CNN_1DConvnets()
cnn_1DConvnets.load_data()

#rnnstep2.plot()
