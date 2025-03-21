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
import matplotlib.pyplot as plt

#%%

# Clip model weights to a given hypercube
class ClipConstraint(tf.keras.constraints.Constraint):
    # Set clip value when initialized
    def __init__(self, clip_value):
        self.clip_value = clip_value
    
    def __call__(self, weights):
        return tf.keras.backend.clip(weights, -self.clip_value, self.clip_value)
    
    def get_config(self):
        return {'clip_value': self.clip_value}
    

#%%
        
NODES=200    

# Loss function
def wasser_loss_function(y_true, y_pred):
    return tf.keras.backend.mean(y_true * y_pred)

# Gan model functions

def define_discriminator(word_length):
    #weight initialization
    init = tf.keras.initializers.RandomNormal(stddev=0.02)
    # weight constraint
    const = ClipConstraint(0.01)
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(word_length,30)))
    model.add(tf.keras.layers.Dense(NODES, 
                                    activation=tf.nn.relu, 
                                    kernel_initializer=init, 
                                    kernel_constraint=const))
    model.add(tf.keras.layers.Dense(NODES, 
                                    activation=tf.nn.relu,
                                    kernel_initializer=init, 
                                    kernel_constraint=const))
    model.add(tf.keras.layers.Dense(1, 
                                    activation=tf.keras.activations.linear)) # Hyvyysluku
    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.00005), 
                   loss=wasser_loss_function)
    return model

# Latent space generates noise
def generate_latent_noise(n, size):
    latent_noise = np.random.rand(n, size)*30
    return latent_noise

def define_generator(noise_length, word_length):    
    #weight initialization
    init = tf.keras.initializers.RandomNormal(stddev=0.02)
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(NODES, 
                                    input_dim=noise_length, 
                                    activation=tf.nn.relu, 
                                    kernel_initializer=init))
    model.add(tf.keras.layers.Dense(NODES, 
                                    activation=tf.nn.relu,
                                    kernel_initializer=init))
    model.add(tf.keras.layers.Dense(word_length*30, 
                                    activation=tf.nn.relu, 
                                    kernel_initializer=init))
    model.add(tf.keras.layers.Reshape((word_length,30)))
    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.00005), 
                   loss=wasser_loss_function)
    return model

# Generates noise and runs the generator using the noise as input
def generate_fake_words(generator, noise_length, noise_n):
    noise = generate_latent_noise(noise_n, noise_length)
    dataX_fake = generator.predict(noise)   
    dataY_fake = np.ones(noise_n) # Nyt 1
    return dataX_fake, dataY_fake

# Generates noise labeled real for gan
def generate_noise_for_gan(noise_length, noise_n):
    dataX_real = generate_latent_noise(noise_n, noise_length)
    dataY_real = np.ones(noise_n)*(-1) # Nyt -1
    return dataX_real, dataY_real

def define_gan(discriminator, generator):
    discriminator.trainable = False
    model = tf.keras.models.Sequential()
    model.add(generator)
    model.add(discriminator)
    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.00005), 
                   loss=wasser_loss_function)
    return model


#%%
    
def choose_best_names(generator, discriminator, n_words_generated,
                      n_words_returned, noise_length):
        
    # Create empty dataframe for storing words and their evaluations
    res_words = pd.DataFrame()

    # Generate names
    latent_noise = generate_latent_noise(1000, NOISE_LENGTH)
    generated_names = ge_model.predict(latent_noise)
    generated_names_eval = di_model.predict(generated_names)
   
    # Generate names
#    latent_noise = generate_latent_noise(n_words_generated, noise_length)
#    generated_names = generator.predict(latent_noise)
#    generated_names_eval = discriminator.predict(generated_names)

    gen_names=[]
    for i in generated_names: 
        xx=np.argmax(i,axis=1)
        gen_names.append(xx)
    
    # Names changed to strings
    res_words['generated_names'] = func.preds_to_letters(gen_names)
    
    # Store evaluations and sort by evaluation
    res_words['evaluation'] = generated_names_eval
    res_words.sort_values(by='evaluation', ascending=True, inplace=True)

    # Rule 1: no foreign characters
    preserved_words = func.remove_names_with_foreign_alphabets(res_words['generated_names'], 
                                         ['b','c','d','f','g','q','x','z','å','ä','ö'])
    res_words = res_words[res_words['generated_names'].isin(preserved_words)]    

    # Rule 2: no empty spaces
    res_words['generated_names'] = res_words['generated_names'].str.replace(' ', '')

    # Rule 3: no duplicates
    res_words = res_words.groupby(['generated_names']).min().sort_values(by=['evaluation'])
    
    # Return n_words_returned top words
    return res_words.head(n_words_returned)



#%%

WORD_LENGTH_UPPER_LIMIT = 7
WORD_LENGTH_LOWER_LIMIT = 7

encoded_real = mdata.get_modellingdata(WORD_LENGTH_UPPER_LIMIT,
                                       WORD_LENGTH_LOWER_LIMIT, 
                                       select_places=True)


#%%

random.seed(9001)

NOISE_LENGTH = WORD_LENGTH_LOWER_LIMIT*30+200
NOISE_N = 100
DISC_NOISE_N = 100
WORD_LENGTH = WORD_LENGTH_UPPER_LIMIT

n_loops = 200
WORD_CNT = 500

# Luodaan mallit
di_model = define_discriminator(WORD_LENGTH)

ge_model = define_generator(NOISE_LENGTH, WORD_LENGTH)

gan_model = define_gan(di_model, ge_model)

# Record loss variables
di_real_loss = []
di_fake_loss = []
ge_loss = []
res_words = pd.DataFrame()

for i in range(n_loops):    
 
    for _ in range(50):
   
        train_dataX_real = func.get_names_random_sample(encoded_real, WORD_CNT)    
        train_dataY_real = np.ones(len(train_dataX_real))*(-1) # Nyt -1
      
        # Treenataan diskriminaattoria oikeilla sukunimillä
        # Diskriminaattori tietää, että oikeita sukunimiä
        tmp = di_model.train_on_batch(train_dataX_real, train_dataY_real) 
        di_real_loss.append(tmp)
        
        # Treenataan diskriminaattoria generaattorin luomilla väärillä sanoilla
        # Diskriminaattori tietää, että vääriä sanoja
        train_dataX_fake, train_dataY_fake = generate_fake_words(ge_model,
                                                                 NOISE_LENGTH,
                                                                 DISC_NOISE_N)
    
        tmp = di_model.train_on_batch(train_dataX_fake, train_dataY_fake)
        di_fake_loss.append(tmp)
    
    # Treenataan gan-mallia (generaattoria) kohinalla, jonka label on real 
    train_dataX_real_gan, train_dataY_real_gan = generate_noise_for_gan(NOISE_LENGTH,
                                                                        NOISE_N)
    
    tmp = gan_model.train_on_batch(train_dataX_real_gan, train_dataY_real_gan)
    ge_loss.append(tmp)
    
    if(i % 10 == 0):
        
        N_WORDS_GENERATED = 100
        N_WORDS_RETURNED = 20
        
        tmp_words = choose_best_names(ge_model, di_model, N_WORDS_GENERATED,
                               N_WORDS_RETURNED, NOISE_LENGTH)

        tmp_words['n_loops'] = i

        res_words = res_words.append(tmp_words)
    
    if(i % 10 == 0):
        print(i)

#%%

N_WORDS_GENERATED = 100
N_WORDS_RETURNED = 30

choose_best_names(ge_model, di_model, N_WORDS_GENERATED,
                       N_WORDS_RETURNED, NOISE_LENGTH)

 
#%%
        
plt.plot(di_real_loss, label='Discriminator loss: real names')
plt.plot(di_fake_loss, label='Discriminator loss: generated names')
plt.legend()
plt.show()

plt.plot(ge_loss, label='Generator loss')
plt.legend()
plt.show()