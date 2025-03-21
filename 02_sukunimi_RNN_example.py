# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 20:38:52 2019

@author: Lari
"""

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.utils import to_categorical

# %% Load data

# Load document into memory
def load_doc(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text

raw_text = load_doc('rhyme.txt')
print(raw_text)

# %% Remove newlines

tokens = raw_text.split()
raw_text = ' '.join(tokens)

# %% Create invididual input sequences
length = 10
sequences = list()
for i in range(length, len(raw_text)):
    seq = raw_text[i-length:i+1]
    sequences.append(seq)

# %% Encode input sequences to integers

chars = sorted(list(set(raw_text)))
mapping = {c:i for i,c in enumerate(chars)}

enc_sequences = list()
for seq in sequences:
    encoded_seq = [mapping[char] for char in seq]
    enc_sequences.append(encoded_seq)

vocab_size = len(mapping)

# %% Split into input and output + do one-hot encoding

split_sequences = np.array(enc_sequences)
X, y = split_sequences[:,:-1], split_sequences[:,-1]

onehot_X = to_categorical(X, num_classes=vocab_size)
onehot_y = to_categorical(y, num_classes=vocab_size)

# %%

model = Sequential()
model.add(LSTM(75, input_shape=(onehot_X.shape[1], onehot_X.shape[2])))
model.add(Dense(vocab_size, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(onehot_X, onehot_y, epochs=100, verbose=1)

# %%

in_text = 'Sing a son'
#in_text = 'Hello worl'

for _ in range(100):
    encoded = [mapping[char] for char in in_text]
    encoded = encoded[-10:]
    encoded = to_categorical(encoded, num_classes=vocab_size)
    encoded = encoded.reshape(1,10,37)
    yhat = model.predict_classes(encoded)
  
    out_char = ''
    for char, index in mapping.items():
        if index == yhat:
            out_char = char
            break
    in_text += char