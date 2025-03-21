# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 23:22:30 2019

@author: saihei
"""

import numpy as np
from itertools import chain, repeat, islice
import pandas as pd
import random

def get_abcDict():
    asciiList = [' ']
    asciiList.extend([chr(i) for i in range(97,123)])
    asciiList.extend(['å', 'ä', 'ö'])    
    abcDict = dict((c,i) for i, c in enumerate(asciiList))
    return(abcDict)

accepted_characters_cnt = len(get_abcDict())

def get_numDict():
    abcDict = get_abcDict()
    dict_tmp = pd.DataFrame.from_dict(abcDict, orient='index', columns=['number'])  
    dict_tmp = dict_tmp.reset_index()
    dict_tmp.columns = ['letter', 'number']
    numDict = dict(zip(dict_tmp.number, dict_tmp.letter))
    return numDict

def check_letters(s):
    names_split = [list(i) for i in s]
    
    chars_merged = []
    for x in names_split:
      chars_merged.extend(x)
    
    chars_merged = np.array(chars_merged)
    return(np.unique(chars_merged))


def replace_letters(s):
    s = [w.replace(' ', '') for w in s]
    s = [w.replace("'", '') for w in s]
    s = [w.replace('-', '') for w in s]
    s = [w.replace('á', 'a') for w in s]
    s = [w.replace('ą', 'a') for w in s]
    s = [w.replace('ć', 'c') for w in s]
    s = [w.replace('ç', 'c') for w in s]
    s = [w.replace('é', 'e') for w in s]
    s = [w.replace('è', 'e') for w in s]
    s = [w.replace('ê', 'e') for w in s]
    s = [w.replace('ë', 'e') for w in s]
    s = [w.replace('ę', 'e') for w in s]
    s = [w.replace('ł', 'l') for w in s]
    s = [w.replace('ń', 'n') for w in s]
    s = [w.replace('î', 'i') for w in s]
    s = [w.replace('ï', 'i') for w in s]
    s = [w.replace('ó', 'o') for w in s]
    s = [w.replace('ø', 'o') for w in s]
    s = [w.replace('ô', 'o') for w in s]
    s = [w.replace('û', 'u') for w in s]
    s = [w.replace('ü', 'y') for w in s]
    s = [w.replace('ś', 's') for w in s]
    s = [w.replace('ß', 'ss') for w in s]
    s = [w.replace('ź', 'z') for w in s]
    s = [w.replace('ż', 'z') for w in s]
    return(s)

def remove_names_with_foreign_alphabets(data,alphabets):
    for char in alphabets:
        data = data[data.str.contains(char)==False]
    return(data)

def pad_infinite(iterable, padding=None):
   return chain(iterable, repeat(padding))

def pad(iterable, size, padding=None):
   return islice(pad_infinite(iterable, padding), size)

   
# Apufunktiot

def get_names_db(data, word_cnt):
    pointer = 0
    while pointer+word_cnt <= len(data):
        yield data[pointer:pointer+word_cnt,]
        pointer += word_cnt 

def preds_to_letters(words):
    pred_letter = []
    numDict = get_numDict()
    for i in words:
        pred_letter_tmp = [numDict[char] for char in i]
        pred_letter.append(pred_letter_tmp)
    pred_letter = [''.join(word) for word in pred_letter]
    return pred_letter

def get_names_random_sample(data, sample_size):
    ind = np.array(range(len(data)))
    sample_ind = [random.choice(ind) for i in range(sample_size)]
    encoded_real_sample = data[sample_ind,]
    return encoded_real_sample
        





