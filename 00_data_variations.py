# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 10:45:49 2020

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

#%% Filters

# Loppuvokaalit
loppuvokaali_filter = df['sukunimi'].str.endswith(('a', 'e', 'i', 'o',
                                           'u', 'y', 'ä', 'ö', 'å'))

# Loppuu i, g
i_filter = df['sukunimi'].str.endswith('i')
g_filter = df['sukunimi'].str.endswith('g')

# Ääkköset
aakkoset_filter = df['sukunimi'].str.contains('ä|ö|å')

# Tuplavokaalit
tuplavokaali_filter = df['sukunimi'].str.contains('aa|ee|ii|uu|yy') 

# Suomalaiset paatteet 
paate_filter_suomi = df['sukunimi'].str.endswith(('nen', 'sto', 'vaara', 'mäki',
                                           'lahti', 'kangas', 'korpi', 'salo',
                                           'kari', 'maa', 'niemi', 'järvi',
                                           'koski', 'kallio', 'talo', 'saari',
                                           'ranta', 'lampi', 'oja', 'aho',
                                           'laakso', 'karja', 'karju',
                                           'la', 'lä', 'pää', 'stö',
                                           'virta', 'kivi', 'juuri','vuo', 'kka', 'kko', 'kartano',
                                           'kkä', 'kkö', 'suu', 'linna', 'lehto', 'niitty', 'tupa',
                                           'luoma', 'haara', 'viita', 'korva', 'keto', 'siira', 'kettu',
                                           'kaarto', 'harju', 'heimo', 'puro', 'niitty', 'kaarre', 'murto',
                                           'vuopio', 'luoto', 'naattu', 'luusua', 'nka', 'nki', 'hasko', 'amo', 'imo','umo',
                                           'silta', 'tie', 'torppa', 'pohja', 'palo', 'puu', 'koivu', 'riutta',
                                           'piirto', 'aukee', 'kaapo', 'niva', 'huhta', 'mies', 'kyyny', 'vainio', 'kanta',
                                           'hako', 'matta', 't', 'sivu', 'ja', 'na', 'ma', 'halme', 'luhta', 'rinta', 'rinne',
                                           'laita', 'aalto', 'alho', 'suo', 'kannas', 'kouvo', 'kanto', 'siimes', 'jalka', 'ruusu'))

# Ruotsalaiset paatteet
paate_filter_ruotsi = df['sukunimi'].str.endswith(('man', 'holm', 'ström', 
                                            'gren', 'lund', 'fors', 
                                           'son', 'roos', 'qvist', 'berg',
                                           'blom', 'gård', 'backa', 'der',
                                           'vik', 'bäck', 'strand', 'skog',
                                           'ff', 'back', 'felt', 'll',
                                           'us', 'hjelm', 'näs', 'dal',
                                           'löf'))

# Venäläiset paatteet
paate_filter_venaja = df['sukunimi'].str.endswith(('ov', 'ova'))


# Ala tai ylä edessä
etuliite_filter = df['sukunimi'].str.startswith(('ala', 'ylä', 'ali', 'yli', 'uusi'))

# Suomen sanakirjan sanat
tree = ET.parse('.\kotus-sanalista-v1\kotus-sanalista_v1.xml')
root = tree.getroot()

sanat = []
for w in root.findall('st'):
    sana = w.find('s').text
    sanat.append(sana)

sanakirja_suomi_filter = df['sukunimi'].isin(sanat)

#%% Apply filters

df = df.loc[-paate_filter_suomi]

df = df.loc[-paate_filter_ruotsi]

df = df.loc[-paate_filter_venaja]

df = df.loc[-etuliite_filter]

df = df.loc[-aakkoset_filter]


#%% Create modelling data (tuplavokaalit)

md1 = df.loc[tuplavokaali_filter, 'sukunimi']

md1 = md1.loc[-i_filter]

md1.to_pickle('./md_tuplavokaalit.pkl')


#%% Create modelling data (konsonanttiloppuiset)

md2 = df.loc[-loppuvokaali_filter, 'sukunimi']

md2 = md2.loc[-aakkoset_filter]

md2.to_pickle('./md_konsonanttiloppuiset.pkl')
