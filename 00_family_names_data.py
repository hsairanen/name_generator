# -*- coding: utf-8 -*-

# %reset

"""
Created on Sun Sep  1 12:30:08 2019

@author: saihei
"""

import Functions as func
import pandas as pd	
import numpy as np

#%%

# Load finnish family names

df = pd.read_excel('.\data\sukunimitilasto-2019-08-07-vrk.xlsx')
df = df.rename(columns = {"Sukunimi":"sukunimi", "Nimenhaltijoita yhteensä":"frekvenssi"})


#%%

df['sukunimi'] = df['sukunimi'].str.lower()

df['sukunimi'] = func.replace_letters(df['sukunimi'])

func.check_letters(df['sukunimi'])

#%%

# Save data

np.save('.\input_data\sukunimet_text.npy', df['sukunimi'])

    