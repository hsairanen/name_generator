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
                      sample_size=100,
                      select_with_letter='',
                      select_places=True):
    
    #sukunimet_text = np.load('.\input_data\sukunimet_text.npy')
    #paikannimet_text = np.load('.\input_data\paikannimet_text.npy')

    # Upgraded numpy
    sukunimet_text = np.load('.\input_data\sukunimet_text.npy', allow_pickle=True)
    paikannimet_text = np.load('.\input_data\paikannimet_text.npy', allow_pickle=True)

    data_all = []
    data_all.extend(sukunimet_text)
    if select_places == True:
         data_all.extend(paikannimet_text)

    data_all = np.unique(data_all)
    data_all = pd.Series(data_all)
  
    # Limit word length
    data = data_all[(data_all.str.len() <= word_length_upper_limit) & 
                    (data_all.str.len() >= word_length_lower_limit)]

    # Filter data
    del_contains = ['w','c','å','z','q','x','mohamed','khalidi','mahmoud','skog', 'sch']
    
    for i in del_contains:
        data = data[-data.str.contains(i)]

    
    del_end = ['ford','ing', 'blom', 'gg', 'th', 'fer', 'dig', 'lig', 'wall',
               'feldt','flyckt','löf','strand','bom','felt','dahl','vik','dig',
               'roth','man','gren','lund','backa','berg','borg','son','holm',
               'ström','roos','bohm','stedt','kvist','elm','bäck','fors','kins',
               'sef','ff','ov','ck','v','w','c','d', 'nen']

    for i in del_end:
        data = data[-data.str.endswith(i)]

    # Choose names with a specific letter
    if select_with_letter != '':
        data = data[data.str.contains(select_with_letter)]
           
    # Reset index
    data = data.reset_index(drop=True)
    
     # Shuffle data
    ind = np.array(range(len(data)))
    random.shuffle(ind)
    data = data[ind]
   
    # Sample size
    if sample_size > data.shape[0]:
            sample_size = data.shape[0]
    
    data = data[0:sample_size-1]
     
    # Combine words as one string
    data = data.str.cat(sep=' ')
     
    return data

# =============================================================================
    
def get_all_names(select_places=False):
    
    #sukunimet_text = np.load('.\input_data\sukunimet_text.npy')
    #paikannimet_text = np.load('.\input_data\paikannimet_text.npy')

    # Upgraded numpy
    sukunimet_text = np.load('.\input_data\sukunimet_text.npy', allow_pickle=True)
    paikannimet_text = np.load('.\input_data\paikannimet_text.npy', allow_pickle=True)

    data_all = []
    data_all.extend(sukunimet_text)
    if select_places == True:
         data_all.extend(paikannimet_text)

    data_all = np.unique(data_all)
     
    return data_all
