# -*- coding: utf-8 -*-

# %reset

"""
Created on Mon Sep  2 19:22:14 2019

@author: saihei
"""

import Functions as func
import numpy as np
import pandas as pd

#%%

# Area data from Statistics Finland
df = pd.read_excel('./data/paavo_1_he.xlsx')
df.columns = ['alue','population']

postal_area = df.alue.str.split(' ').str[1]

cities_tmp = df.alue.str.split('(', expand=True)
cities = cities_tmp[1].str.split(' ').str[0]
cities = cities.str.replace('(', '', regex=False)
cities = cities.str.replace(')', '', regex=False)

data_all = []
data_all.extend(cities)
data_all.extend(postal_area)

#%%

# Area data from Maanmittauslaitos
df_xml = np.load('paikannimet_xml.npy')

df_xml = pd.DataFrame(df_xml)
df_xml.columns = ['alue_xml']

df_xml['alue_xml'] = df_xml['alue_xml'].str.replace('<pnr:kirjoitusasu>', '', regex=False)
df_xml['alue_xml'] = df_xml['alue_xml'].str.replace('</pnr:kirjoitusasu>', '', regex=False)

data_all.extend(df_xml['alue_xml'])

#%%          

# Clean the whole data

data_all = pd.DataFrame(data_all)
data_all.columns = ['alue']

data_all.drop_duplicates(inplace=True)

data_all['alue'] = data_all['alue'].str.lower()

data_all['alue'] = func.replace_letters(data_all['alue'])

data_all = data_all[data_all['alue'].str.contains('ǧ') == False]
data_all = data_all[data_all['alue'].str.contains('ǩ') == False]
data_all = data_all[data_all['alue'].str.contains('ǯ') == False]
data_all = data_all[data_all['alue'].str.contains('ʒ') == False]
data_all = data_all[data_all['alue'].str.contains('õ') == False]
data_all = data_all[data_all['alue'].str.contains('â') == False]
data_all = data_all[data_all['alue'].str.contains('ŋ') == False]
data_all = data_all[data_all['alue'].str.contains('ŧ') == False]
data_all = data_all[data_all['alue'].str.contains('č') == False]
data_all = data_all[data_all['alue'].str.contains('đ') == False]
data_all = data_all[data_all['alue'].str.contains('š') == False]
data_all = data_all[data_all['alue'].str.contains('ž') == False]
data_all = data_all[data_all['alue'].str.contains('_') == False]
data_all = data_all[data_all['alue'].str.contains('1') == False]
data_all = data_all[data_all['alue'].str.contains('2') == False]
data_all = data_all[data_all['alue'].str.contains('3') == False]
data_all = data_all[data_all['alue'].str.contains('4') == False]
data_all = data_all[data_all['alue'].str.contains('5') == False]
data_all = data_all[data_all['alue'].str.contains('6') == False]
data_all = data_all[data_all['alue'].str.contains('7') == False]
data_all = data_all[data_all['alue'].str.contains('8') == False]
data_all = data_all[data_all['alue'].str.contains('9') == False]
data_all = data_all[data_all['alue'].str.contains('\(') == False]
data_all = data_all[data_all['alue'].str.contains('\)') == False]
data_all = data_all[data_all['alue'].str.contains('\.') == False]
data_all = data_all[data_all['alue'].str.contains('\,') == False]
data_all = data_all[data_all['alue'].str.contains('\:') == False]

func.check_letters(data_all['alue'])


#%%

# Save the data
    
np.save('paikannimet_text.npy', data_all['alue'])

    