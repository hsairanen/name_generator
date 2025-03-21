# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 20:38:52 2019

@author: Lari
"""

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.utils import to_categorical
import modellingdata_RNN as md

# %% Load data

WORD_LENGTH_LOWER_LIMIT = 0
WORD_LENGTH_UPPER_LIMIT = 50
SAMPLE_SIZE = 3000

#raw_text = md.get_modellingdata(word_length_upper_limit=WORD_LENGTH_UPPER_LIMIT, 
#                                       word_length_lower_limit=WORD_LENGTH_LOWER_LIMIT,
#                                       sample_size=SAMPLE_SIZE,
#                                       select_places = False)

#%% Modelling data variations

#raw_text = pd.read_pickle('./md_tuplavokaalit.pkl')
raw_text = pd.read_pickle('./md_konsonanttiloppuiset.pkl')
#raw_text = pd.read_excel('stadin_slangi_nimet.xlsx', header=None, dtype=str)
#raw_text = pd.read_excel('nimet_ruotsiksi.xlsx', header=None, dtype=str)
#cat_text = raw_text

# raw_text = np.load('german_names.npy', allow_pickle=True)
# raw_text = pd.DataFrame(raw_text)

cat_text = raw_text.str.cat(sep=' ')
#cat_text = raw_text


# %% Create invididual input sequences
length = 10
sequences = list()
for i in range(length, len(cat_text)):
    seq = cat_text[i-length:i+1]
    sequences.append(seq)

# %% Encode input sequences to integers

chars = sorted(list(set(cat_text)))
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

# %% Train the model

EPOCHS = 7

model = Sequential()
model.add(LSTM(50, input_shape=(onehot_X.shape[1], onehot_X.shape[2]), return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(50, input_shape=(onehot_X.shape[1], onehot_X.shape[2])))
model.add(Dense(vocab_size, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(onehot_X, onehot_y, epochs=EPOCHS, verbose=1, shuffle=True)

# %% Generate names

IN_TEXT = 'sairanen paunonen kohonen'
N_CHARS_GENERATED = 5000
PROB_ORDINAL_NB = 0             # 0 = the largest, 1 = the second largest etc.
RANDOMIZE = True                 
RANDOMIZER_LIMITS = [0.01, 1]

generated_names = IN_TEXT
counter = 0

for _ in range(N_CHARS_GENERATED):
    # Select the last letters of the generated text
    generated_names_end = generated_names[-length:]
    encoded = [mapping[char] for char in generated_names_end]
    encoded = to_categorical(encoded, num_classes=vocab_size)
    encoded = encoded.reshape(1, length, vocab_size)

    # Predict the probability distribution of the next letter
    #yhat = model.predict_classes(encoded)
    yhat = model.predict_proba(encoded)

    # Select the next letter based on the order of the probabilities
    tmp=pd.DataFrame(data=yhat.T, columns=['probs']).reset_index()
    if RANDOMIZE == True:
        # Randomizer
        tmp['randomizer'] = np.random.uniform(RANDOMIZER_LIMITS[0], RANDOMIZER_LIMITS[1], len(tmp))
        tmp['randomized_probs'] = tmp['probs'] * tmp['randomizer']
        tmp.loc[0, ('randomized_probs')] = tmp['probs'][0]
        tmp.sort_values(by='randomized_probs', ascending=False, inplace=True)
        yhat = np.array([tmp.iloc[0]['index']], dtype=np.int64)
    else:
        tmp.sort_values(by='probs', ascending=False, inplace=True)
        yhat = np.array([tmp.iloc[PROB_ORDINAL_NB]['index']], dtype=np.int64)
      
    # Convert the index of the letter into text
    out_char = ''
    for char, index in mapping.items():
        if index == yhat:
            out_char = char
            break
    generated_names += char
    
    # Counter for following the progression
    counter += 1
    if counter % 500 == 0:
        print("Generating char %i of %i" % (counter, N_CHARS_GENERATED))
    

# Remove seed name from output
generated_names = generated_names[len(IN_TEXT):]

# Split into separate names
generated_names = np.unique(np.array(generated_names.split(" ")))

# Remove names that exist in input dataset
nimet = md.get_all_names()
generated_names_fakes = np.unique(generated_names[~np.isin(generated_names, nimet)])

# Remove names from input dataset
generated_names_fakes = np.unique(generated_names[~np.isin(generated_names, raw_text)])

print(generated_names_fakes)
#print(str(generated_names_fakes).replace(" ", "\n"))

#%%
# Store the generated names and the model parameters

# Get the previous results from the excel
#old_res = pd.read_excel('results.xlsx')

# Make the new id for the results to be stored 
#ID = old_res['ID'].max() + 1

# The new results
#new_res = pd.DataFrame(generated_names_fakes, columns=['NAMES'])
#new_res['ID'] = ID
#new_res = new_res.loc[:,['ID','NAMES']]
#new_res['WORD_LENGTH_LOWER_LIMIT'] = WORD_LENGTH_LOWER_LIMIT
#new_res['WORD_LENGTH_UPPER_LIMIT'] = WORD_LENGTH_UPPER_LIMIT
#new_res['SAMPLE_SIZE'] = SAMPLE_SIZE
#new_res['LENGTH'] = length
#new_res['EPOCHS'] = EPOCHS
#new_res['RANDOMIZE'] = RANDOMIZE
#new_res['RANDOMIZER_LIMITS_LOWER'] = RANDOMIZER_LIMITS[0]
#new_res['RANDOMIZER_LIMITS_UPPER'] = RANDOMIZER_LIMITS[1]
#new_res['PROB_ORDINAL_NB'] = PROB_ORDINAL_NB

# Combine the old and the new results
#res = old_res.append(new_res, ignore_index=True)

# Write to excel
#res.to_excel('results.xlsx', index=False)
