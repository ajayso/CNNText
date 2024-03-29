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
from keras.layers import Embedding, Flatten, Dense
from keras.utils.np_utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import json
import pandas as pd



class CNN_Text:
    # Load the data, text and labels
    def load_data(self):
        data_dir= os.getcwd()
        file_name = os.path.join(data_dir,"news.json")
        data = [json.loads(line) for line in open(file_name, 'r')]
        df = pd.DataFrame(data)
        df["category"] = df["category"].astype('category')
        df["category_cat"] = df["category"].cat.codes
        
       
        labels = []
        texts = []
        labels = df["category_cat"]
        texts = df["headline"]
        
 
        self.labels = np.asarray(labels)
        self.texts = texts

        
    
    def Tokenize(self):
        maxlen = 1000
        max_words = 20000
        
        tokenizer = Tokenizer(num_words=max_words)
        tokenizer.fit_on_texts(self.texts)
        sequences = tokenizer.texts_to_sequences(self.texts)
        print(sequences)
        
        word_indexer = tokenizer.word_index
        
        
        #print("No of words found " % len(word_indexer))
        #print('Found %s unique tokens.' % len(word_indexer))
        
        data = pad_sequences(sequences, maxlen = maxlen)
        
        print("Shape of data tensor", data.shape)
        print("Shape of label tensor", len(self.labels))
        
        self.data = data
        self.word_indexer = word_indexer
    
    def Train_Build_Model(self):
        maxlen = 1000
        max_words = 20000
        training_samples = 180000
        validation_samples = 20000
        
        indices = np.arange(self.data.shape[0])
        print(indices)
        np.random.shuffle(indices)
        data = self.data[indices]
        labels = self.labels[indices]
        
        labels = to_categorical(labels)
        
        
        x_train = data[:training_samples]
        y_train = labels[:training_samples]
        
        x_val = data[training_samples: training_samples + validation_samples]
        y_val = labels[training_samples: training_samples + validation_samples]
        
        current_path = os.getcwd()
        BASE_DIR = os.path.dirname( current_path )
        glove_dir = os.path.join(BASE_DIR, "glove.6B")
        
        embedding_index = {}
        f = open(os.path.join(glove_dir, 'glove.6B.50d.txt'),encoding="utf8")
        for line in f:
            values = line.split()
            word = values[0]
            coeff = np.asarray(values[1:], dtype='float32')
            embedding_index[word] = coeff
        f.close()
        self.embedding_index = embedding_index
        # Build an embedding matrix that can load into an Embedding Layer
        # Matrix of shape (max_words, embedding_dim)
        embedding_dim = 50
        embedding_matrix = np.zeros((max_words, embedding_dim))
        for word, i in self.word_indexer.items():
            if i < max_words:
                embedding_vector = embedding_index.get(word)
                if embedding_vector is not None:
                    embedding_matrix[i] = embedding_vector
        
            
     
        model = Sequential()
        model.add(Embedding(max_words,embedding_dim, input_length= maxlen))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(41, activation='softmax'))
        # Loading pretrained word embedding into the Embedding layer
        model.layers[0].set_weights([embedding_matrix])
        model.layers[0].trainable = False
        model.summary()
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_val= x_val
        self.y_val = y_val
        print(y_train.shape)
        print(y_val.shape)
    
    def execute(self):
        self.model.compile(
                optimizer = 'rmsprop',
                loss = 'binary_crossentropy',
                metrics=['acc'])
                
        self.history = self.model.fit(self.x_train, self.y_train,
                       epochs = 10,
                       batch_size = 32,
                       validation_data = (self.x_val, self.y_val)
                       )
                       
        self.model.save_weights('pre_trained_glove_model.h5')
    
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
        
    def tsne_plot(self):
        labels = []
        tokens = []
        max_words = 20000
        max_length = 0
        for word, i in self.word_indexer.items():
            if i < 1000 :
                
                embedding_vector = self.embedding_index.get(word)
                if embedding_vector is not None:
                    #embedding_vector = embedding_vector.reshape(50)
                    tokens.append(embedding_vector)
                    labels.append(word)
            else:
                break
        tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
        new_values = tsne_model.fit_transform(tokens)
        print(len(tokens))
        print(len(labels))
        x = []
        y = []
        for value in new_values:
            x.append(value[0])
            y.append(value[1])
        plt.figure(figsize=(16, 16))
        for i in range(len(x)):
            plt.scatter(x[i],y[i])
            plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
        plt.show()
        

        
        
        
cnn_text = CNN_Text()
cnn_text.load_data()
cnn_text.Tokenize()
cnn_text.Train_Build_Model()
cnn_text.execute()
cnn_text.plot()
cnn_text.tsne_plot()