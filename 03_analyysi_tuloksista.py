# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 22:49:09 2020

@author: Lari
"""

import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET

#%% Results from the model
df = pd.read_excel('.\results\results.xlsx')

#%% Filter out Finnish words

# Finnish words
tree = ET.parse('.\kotus-sanalista-v1\kotus-sanalista_v1.xml')
root = tree.getroot()

sanat = []
for w in root.findall('st'):
    sana = w.find('s').text
    sanat.append(sana)

# Filter out if the results include meaningful Finnish words    
df = df[-df['NAMES'].isin(sanat)]

#%% Other filters

# More than 3 letters
df = df[df['NAMES'].str.len() > 3]

# No typical endings
del_end = ['vaara','nen','mäki','lahti','kangas','korpi','salo','kari','maa','niemi','järvi','koski','kallio','der','la',
           'talo','saari','ranta', 'lampi', 'oja', 'aho', 'laakso', 'karja', 'karju']

for i in del_end:
    df = df[-df['NAMES'].str.endswith(i)]

# No typical beginning
del_start = ['kangas']

for i in del_start:
    df = df[-df['NAMES'].str.startswith(i)]

# No letters ä, ö and å
del_char = ['ä','ö','å']

for i in del_char:
    df = df[-df['NAMES'].str.contains(i)]

# Filter out if the name exists
existing_names = ['valama', 'karvio', 'kortte', 'salio', 'salkio', 'sarani', 'harvisto', 'sinell', 'riikka',
                  'vilma', 'martell', 'kuro']

df = df[-df['NAMES'].isin(existing_names)]

# Drop duplicates
names_to_assess = df['NAMES'].drop_duplicates()

# Sort
names_to_assess = names_to_assess.sort_values()


#%%

names_to_assess.to_excel('.\results\names_to_assess.xlsx', index=False)

#%%

# karvisto
# korkio
# harinto
# sandes

