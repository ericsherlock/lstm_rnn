#! /usr/bin/env python

#Import Necessary Packages
from keras.preprocessing import sequence
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils
from keras.layers import LSTM
from keras.layers import Dropout
from keras.datasets import imdb
import numpy as np
from keras.datasets import mnist
from keras.preprocessing.text import one_hot
from matplotlib import pyplot

#Read In Text File
#IMDB Dataset Preprocessing
"""print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=5000)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')
print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=100)
x_test = sequence.pad_sequences(x_test, maxlen=100)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
"""
#Amazon Dataset Preprocessing
corpora = open("smallText.txt", "r").readlines()
corpLab = [1]*(len(corpora)/2)
corpLab.extend([0]*(len(corpora)/2))

#Set Up Tokenizer
tokenizer = Tokenizer(nb_words=5000)
tokenizer.fit_on_texts(corpora)
textSeq = tokenizer.texts_to_sequences(corpora)
wordIndex = tokenizer.word_index

#Pad The Sentences To The Required Length
data = pad_sequences(textSeq, maxlen=100)
corpLab = np_utils.to_categorical(np.asarray(corpLab)).reshape(-1)
print('Shape of Sentence Matrix', data.shape)
print('Shape of Label Matrix', corpLab.shape)

#ch
#Randomly Shuffle Data To Split Into Training And Testing
index = np.arange(data.shape[0])
np.random.shuffle(index)
data = data[index]
corpLab = corpLab[index]
corpSplit = int(.1 * data.shape[0])

#Split The Dataset Into Training And Testing
x_train = data[:-corpSplit]
x_test = data[-corpSplit:]
y_train = corpLab[:-corpSplit]
y_test = corpLab[-corpSplit:]
#ch

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
    if i > 5000:
        break
embedFile.close()

#Create Embedding Matrix From Embedding Index
embedMatrix = np.zeros((len(wordIndex) + 1, embedDim))
for word, i in wordIndex.items():
    embedVector = embedIndex.get(word)
    if embedVector is not None:
       embedMatrix[i] = embedVector

#Create The Model
print("Creating The Model...\n")
model = Sequential()
model.add(Embedding(len(wordIndex)+1, embedDim, input_length=100, weights=[embedMatrix], trainable=True)) 
model.add(LSTM(125, dropout=0.4, recurrent_dropout=0.4, return_sequences=True))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))
model.summary()


#Compiling The Model With Loss, Optimizer Functions
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#Training And Testing Model
print("Train And Test The Model...\n")
history = model.fit(x_train, y_train, batch_size=50, epochs=2, validation_data=(x_test, y_test)) 

#Plot The Results
pyplot.plot(history.history['loss'])
pyplot.plot(history.history['val_loss'])
pyplot.plot(history.history['acc'])
pyplot.plot(history.history['val_acc'])
pyplot.title('model train vs validation loss')
pyplot.ylabel('Accuracy/Loss as %')
pyplot.xlabel('Number Epochs')
pyplot.legend(['train_loss', 'val_loss', 'train_acc', 'val_acc'], loc='upper right')
pyplot.show()
