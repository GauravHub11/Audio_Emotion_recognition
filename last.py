#!/usr/bin/env python
# coding: utf-8

# In[1]:


from numba import cuda
cuda.select_device(0)
cuda.close()


# In[10]:


import keras
from keras.models import Model
import record_audio
from util.feature_extraction import get_audio_features
from util.feature_extraction import get_features_dataframe
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import *


# In[6]:


from keras.models import model_from_json
json_file = open('./Trained_Models/ggg.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("./Trained_Models/ggg.h5")
print("Loaded model from disk")


# In[ ]:


for i in range(0,20):
    
    record_audio.Record()

    demo_audio_path = './demo_audio.wav'
    sampling_rate = 20000 
#     ipd.Audio(demo_audio_path)
    demo_mfcc, demo_pitch, demo_mag, demo_chrom = get_audio_features(demo_audio_path,sampling_rate)

    mfcc = pd.Series(demo_mfcc)
    pit = pd.Series(demo_pitch)

    mag = pd.Series(demo_mag)
    C = pd.Series(demo_chrom)
    demo_audio_features = pd.concat([mfcc,pit,mag,C],ignore_index=True)
    demo_audio_features= np.expand_dims(demo_audio_features, axis=0)
    demo_audio_features= np.expand_dims(demo_audio_features, axis=2)

    livepreds = loaded_model.predict(demo_audio_features, 
                             batch_size=32, 
                             verbose=0)

    emotions=["anger","disgust","fear","happy","neutral", "sad", "surprise"]
    index = livepreds.argmax(axis=1).item()
    print(emotions[index])
    
