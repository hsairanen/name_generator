# -*- coding: utf-8 -*-
#%reset -f

"""
Created on Sun Sep  8 14:54:45 2019
@author: saihei
"""
#%reset

import numpy as np
import tensorflow as tf
import random
import pandas as pd
import Functions as func
import modellingdata as mdata


#%%

# Gan model functions

def define_discriminator(word_length):    
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(15, input_dim=word_length, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(15, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(2, activation=tf.nn.softmax))
    #tf.keras.utils.plot_model(model, to_file='testi.png', 
    #                                   show_shapes=True, show_layer_names=True)
    model.compile(optimizer='adam', 
                   loss='sparse_categorical_crossentropy', 
                   metrics=['accuracy'])
    return model

# Latent space generates noise
def generate_latent_noise(n, size):
    latent_noise = np.random.rand(n, size)*30
    return latent_noise

def define_generator(noise_length, word_length):    
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(20, input_dim=noise_length, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(20, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(word_length, activation=tf.nn.relu))
    model.compile(optimizer='adam', 
                   loss='sparse_categorical_crossentropy', 
                   metrics=['accuracy'])
    return model

# Generates noise and runs the generator using the noise as input
def generate_fake_words(generator, noise_length, noise_n):
    noise = generate_latent_noise(noise_n, noise_length)
    dataX_fake = generator.predict(noise)
    dataY_fake = np.zeros(noise_n)
    return dataX_fake, dataY_fake

# Generates noise labeled real for gan
def generate_noise_for_gan(noise_length, noise_n):
    dataX_real = generate_latent_noise(noise_n, noise_length)
    dataY_real = np.ones(noise_n)
    return dataX_real, dataY_real

def define_gan(discriminator, generator):
    discriminator.trainable = False
    model = tf.keras.models.Sequential()
    model.add(generator)
    model.add(discriminator)
    model.compile(optimizer='adam', 
                   loss='sparse_categorical_crossentropy')
    return model

def choose_generated_names(generator, discriminator, n_words_generated,
                           n_words_returned, noise_length):
    # Create empty dataframe for storing words and their evaluations
    res_words = pd.DataFrame()

    # Generate names
    latent_noise = generate_latent_noise(n_words_generated, noise_length)
    generated_names = generator.predict(latent_noise)
    generated_names_eval = discriminator.predict(generated_names)

    # Names changed to strings
    generated_names = np.rint(generated_names)
    generated_names[generated_names > 29] = 0
    res_words['generated_names'] = func.preds_to_letters(generated_names)
        
    # Store evaluations and sort by evaluation
    res_words['evaluation'] = generated_names_eval[:,1]
    res_words.sort_values(by='evaluation', ascending=False, inplace=True)

    # Rule 1: no foreign characters
    preserved_words = func.remove_names_with_foreign_alphabets(res_words['generated_names'], 
                                         ['b','c','d','f','g','q','x','z','å','ä','ö'])
    res_words = res_words[res_words['generated_names'].isin(preserved_words)]    

    # Rule 2:
    res_words['generated_names'] = res_words['generated_names'].str.replace(' ', '')

    # Return n_words_returned top words
    return res_words.head(n_words_returned)


#%%

random.seed(9001)

WORD_LENGTH_UPPER_LIMIT = 7
WORD_LENGTH_LOWER_LIMIT = 7

NOISE_LENGTH = 8
NOISE_N = 100
DISC_NOISE_N = 100
WORD_LENGTH = WORD_LENGTH_UPPER_LIMIT

n_loops = 10000
WORD_CNT = 100
#data = get_names_db(WORD_CNT)


#%%

encoded_real = mdata.get_modellingdata(WORD_LENGTH_UPPER_LIMIT,
                                       WORD_LENGTH_LOWER_LIMIT)


#%%

# Luodaan mallit
di_model = define_discriminator(WORD_LENGTH)

ge_model = define_generator(NOISE_LENGTH, WORD_LENGTH)

gan_model = define_gan(di_model, ge_model)

for i in range(n_loops):    
    
    train_dataX_real = func.get_names_random_sample(WORD_CNT)    
    #train_dataX_real = next(data)
    train_dataY_real = np.ones(len(train_dataX_real))
  
    # Treenataan diskriminaattoria oikeilla sukunimillä
    # Diskriminaattori tietää, että oikeita sukunimiä
    di_model.train_on_batch(train_dataX_real, train_dataY_real) 
    
    # Treenataan diskriminaattoria generaattorin luomilla väärillä sanoilla
    # Diskriminaattori tietää, että vääriä sanoja
    train_dataX_fake, train_dataY_fake = generate_fake_words(ge_model,
                                                             NOISE_LENGTH,
                                                             DISC_NOISE_N)
    di_model.train_on_batch(train_dataX_fake, train_dataY_fake)
    
    
    # Treenataan gan-mallia (generaattoria) kohinalla, jonka label on real 
    train_dataX_real_gan, train_dataY_real_gan = generate_noise_for_gan(NOISE_LENGTH,
                                                             NOISE_N)
    
    gan_model.train_on_batch(train_dataX_real_gan, train_dataY_real_gan)
    
    if(i % 100 == 0):
        print(i)

#%%

# Generaattori

N_WORDS_GENERATED = 100
N_WORDS_RETURNED = 20

choose_generated_names(ge_model, di_model, N_WORDS_GENERATED,
                       N_WORDS_RETURNED, NOISE_LENGTH)



#%%

# Oikeat sukunimet

tdn2 = di_model.predict(encoded_real[0:10,])

print('Oikeat sukunimet', tdn2)

print(func.preds_to_letters(encoded_real[0:10,]))

#for i in range(len(tdn2)):
#    print(np.argmax(tdn2[i]))
    

