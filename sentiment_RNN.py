#! /usr/bin/env python

#Import Necessary Packages
from __future__ import print_function
from keras.preprocessing import sequence
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Flatten
from keras import optimizers
from keras.datasets import imdb
import numpy as np
from keras.datasets import mnist
from keras.preprocessing.text import one_hot
from matplotlib import pyplot
import matplotlib

#Read In Text Datasets
#########################IMDB Dataset Preprocessing#############################
#Read In The IMDB Dataset and Print Training and Testing Lengths
#(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=5000)
#print(len(x_train), 'train sequences')
#print(len(x_test), 'test sequences')

#Pad The Training and Testing Sequences
#x_train = sequence.pad_sequences(x_train, maxlen=75)
#x_test = sequence.pad_sequences(x_test, maxlen=75)
#print('x_train shape:', x_train.shape)
#print('x_test shape:', x_test.shape)

#######################Amazon/Emotion in Text Datasets Preprocessing############

#Read In Either The Amazon Dataset or the Emotion in Text Dataset
#corpora = open("AmazonTextDataset.txt", "r").readlines()
corpora = open("EmotionInTextDataset.txt", "r").readlines()

#Create Dataset Matrix For Binary/Multiple Classification

#Binary Sentiment Matrix Creation
#corpLab = [1]*(len(corpora)/2)
#corpLab.extend([0]*(len(corpora)/2))

#Emotion Classification Matrix 
corpLab = [0]*(len(corpora)/5)
corpLab.extend([1]*(len(corpora)/5))
corpLab.extend([2]*(len(corpora)/5))
corpLab.extend([3]*(len(corpora)/5))
corpLab.extend([4]*(len(corpora)/5))
 
#Set Up Tokenizer
tokenizer = Tokenizer(nb_words=10000)
tokenizer.fit_on_texts(corpora)
textSeq = tokenizer.texts_to_sequences(corpora)
wordIndex = tokenizer.word_index

#Pad The Sentences To The Required Length
data = pad_sequences(textSeq, maxlen=75)

#Bi-Sentiment, Reshape Corpora Matrix
#(Reshape For Binary Sentiment Analysis With Single Dense Node)
#corpLab = np_utils.to_categorical(np.asarray(corpLab)).reshape(-1)
#(Reshape For Binary Sentiment Analysis With Two Dense Nodes)
#corpLab = np_utils.to_categorical(np.asarray(corpLab)).reshape(10814, 2)

#5-Sentiment, Reshape Corpora Matrix
corpLab = np_utils.to_categorical(np.asarray(corpLab)).reshape(19395, 5)

#Randomly Shuffle Data To Split Into Training And Testing
index = np.arange(data.shape[0])
np.random.shuffle(index)
data = data[index]
corpLab = corpLab[index-5]
corpSplit = int(.1 * data.shape[0])

#Split The Dataset Into Training And Testing
x_train = data[:-corpSplit]
x_test = data[-corpSplit:]
y_train = corpLab[:-corpSplit]
y_test = corpLab[-corpSplit:]

#Check Shapes Of Training and Testing
print("SHAPE XTRAIN: ", x_train.shape)
print("SHAPE YTRAIN: ", y_train.shape)
print("SHAPE XTEST: ", x_test.shape)
print("SHAPE YTEST: ", y_test.shape)

#Load The Embeddings From Word2Vec File
embedFile = open("GoogleNews-vectors-negative300.bin.gz.txt", "r")
embedIndex = {}
i = 0
for line in embedFile:
    vectors = line.split()
    word = vectors[0]
    wordNumList = np.asarray(vectors[1:], dtype='float32')
    embedIndex[word] = wordNumList
    embedDim = len(wordNumList)
    i = i + 1
    if i > 50000:
        break
embedFile.close()

#Create Embedding Matrix From Embedding Index
embedMatrix = np.zeros((len(wordIndex) + 1, embedDim))
for word, i in wordIndex.items():
    embedVector = embedIndex.get(word)
    if embedVector is not None:
       embedMatrix[i] = embedVector

#Check The Embedding Dimensions
print("EMBEDING DIM: ", embedDim)
print("\n")

#Create The Model, A Single Embedding Layer, 1/2 LSTM Layers, A Flatten Layer (For Multiple Classification)
#And A Dense Layer With 1,2, or 5 Nodes Depending On Number Of Classification Categories
print("Creating The Model...\n")
model = Sequential()
model.add(Embedding(len(wordIndex)+1, embedDim, input_length=75, weights=[embedMatrix], trainable=True)) 
#model.add(LSTM(75, dropout=0.2, recurrent_dropout=0.3, return_sequences=True))
model.add(LSTM(100, dropout=0.3, recurrent_dropout=0.2, return_sequences=True))

#5-Sentiment
model.add(Flatten())
model.add(Dense(5, activation='softmax'))
#model.add(Dense(2, activation='softmax'))
#model.add(Dense(1, activation='sigmoid'))

#Print Model Summary
model.summary()


#Compiling The Model
#Binary-Sentiment
#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#5-Sentiment
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#Train And Test The Model
print("Train And Test The Model...\n")
history = model.fit(x_train, y_train, batch_size=75, epochs=10, validation_data=(x_test, y_test)) 

#Use Pyplot to Plot The Results
pyplot.plot(history.history['loss'])
pyplot.plot(history.history['val_loss'])
pyplot.plot(history.history['acc'])
pyplot.plot(history.history['val_acc'])
pyplot.title('model train vs validation loss')
pyplot.ylabel('Accuracy/Loss as %')
pyplot.xlabel('Number Epochs')
pyplot.legend(['train_loss', 'val_loss', 'train_acc', 'val_acc'], loc='upper right')
pyplot.show()
