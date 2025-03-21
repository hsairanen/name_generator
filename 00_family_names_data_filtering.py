# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 17:25:11 2020

@author: Lari
"""

import pandas as pd	
import numpy as np
import Functions as func
import xml.etree.ElementTree as ET

#%%

df = pd.read_excel('.\data\sukunimitilasto-2019-08-07-vrk.xlsx')
df = df.rename(columns = {"Sukunimi":"sukunimi", "Nimenhaltijoita yhteensä":"frekvenssi"})

#%%

df['sukunimi'] = df['sukunimi'].str.lower()
df['sukunimi'] = func.replace_letters(df['sukunimi'])

#%% Konsonanttifiltteri

df['vokaalipaate'] = 0
paate_filter = df['sukunimi'].str.endswith(('a', 'e', 'i', 'o',
                                           'u', 'y', 'ä', 'ö', 'å'))
df.loc[paate_filter,'vokaalipaate'] = 1

#%%

df['paate_suomi'] = 0

paate_filter = df['sukunimi'].str.endswith(('nen', 'sto', 'vaara', 'mäki',
                                           'lahti', 'kangas', 'korpi', 'salo',
                                           'kari', 'maa', 'niemi', 'järvi',
                                           'koski', 'kallio', 'talo', 'saari',
                                           'ranta', 'lampi', 'oja', 'aho',
                                           'laakso', 'karja', 'karju',
                                           'la', 'lä', 'pää', 'stö',
                                           'virta', 'kivi', 'juuri'))
df.loc[paate_filter,'paate_suomi'] = 1

df['paate_ruotsi'] = 0

paate_filter = df['sukunimi'].str.endswith(('man', 'holm', 'ström', 
                                            'gren', 'lund', 'fors', 
                                           'son', 'roos', 'qvist', 'berg',
                                           'blom', 'gård', 'backa', 'der',
                                           'vik', 'bäck', 'strand', 'skog',
                                           'ff', 'back', 'felt', 'll',
                                           'us', 'hjelm', 'näs', 'dal',
                                           'löf'))
df.loc[paate_filter,'paate_ruotsi'] = 1

#%% Filter out Finnish words

# Finnish words
tree = ET.parse('.\kotus-sanalista-v1\kotus-sanalista_v1.xml')
root = tree.getroot()

sanat = []
for w in root.findall('st'):
    sana = w.find('s').text
    sanat.append(sana)

# Filter out if the results include meaningful Finnish words    
df['sanakirja_suomi'] = 0
sanakirja_suomi_filter = df['sukunimi'].str.contains('|'.join(sanat))
df.loc[sanakirja_suomi_filter,'sanakirja_suomi'] = 1

#%% Export

df.to_excel('.\input_data\filtered_family_names.xlsx', index=False)