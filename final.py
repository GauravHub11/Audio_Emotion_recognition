#!/usr/bin/env python
# coding: utf-8

# In[1]:


from numba import cuda
cuda.select_device(0)
cuda.close()


# In[2]:


import Audio_Record_UI
from util.feature_extraction import get_audio_features
from util.feature_extraction import get_features_dataframe
import pandas as pd
import numpy as np


# In[3]:


# loading json and creating model
from keras.models import model_from_json
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("./Trained_Models/Speech_Emotion_Recognition_Model.h5")
print("Loaded model from disk")


# In[4]:


demo_audio_path = './Output.wav'
sampling_rate = 20000 
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
                         verbose=1)
emotions=["anger","disgust","fear","happy","neutral", "sad", "surprise"]
index = livepreds.argmax(axis=1).item()




# In[5]:


import tkinter as tk
from tkinter import *


# In[6]:


window=tk.Tk()
window.title("Audio Emotion Recognition")
window.configure(background='black')
Label(window,text=emotions[index],bg='black',fg='white',font='none 40 bold').grid(row=0,column=0,sticky=W)
window.mainloop()    


# In[7]:


from numba import cuda
cuda.select_device(0)
cuda.close()


# In[ ]:




