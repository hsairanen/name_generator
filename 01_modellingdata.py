# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 18:48:08 2019

@author: saihei
"""


import numpy as np
import pandas as pd
import Functions as func
import random
from keras.utils import to_categorical

def get_modellingdata(word_length_upper_limit, 
                      word_length_lower_limit,
                      select_places=True,
                      drop_foreign_letters=True):
    
    sukunimet_text = np.load('.\input_data\sukunimet_text.npy')
    paikannimet_text = np.load('.\input_data\paikannimet_text.npy')

    data_all = []
    data_all.extend(sukunimet_text)
    if select_places == True:
        data_all.extend(paikannimet_text)
    
    data_all = np.unique(data_all)
    data_all = pd.Series(data_all)

    # Remove foreign letters
    if drop_foreign_letters == True:
        data_all = func.remove_names_with_foreign_alphabets(data_all, 
                                                            [' ','b','c','d','f','g','q','x','z','å'])
    # Limit word length
    data = data_all[(data_all.str.len() <= word_length_upper_limit) & 
                    (data_all.str.len() >= word_length_lower_limit)]

    # Reset index
    data = data.reset_index(drop=True)

    # Shuffle data
    ind = np.array(range(len(data)))
    random.shuffle(ind)
    data = data[ind]

    # Encode characters

    nimet = [list(func.pad(i, word_length_upper_limit, ' ')) for i in data]

    abcDict = func.get_abcDict()

    encoded_real = []
    for i in nimet:
        encoded_real_tmp1  = [abcDict[char] for char in i]
        encoded_real_tmp2  = to_categorical(encoded_real_tmp1, num_classes=func.accepted_characters_cnt)
        encoded_real.append(encoded_real_tmp2) 

    encoded_real = np.array(encoded_real)

    return encoded_real