#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import librosa     #for Audio Analysis
import librosa.display    #need to import from librosa

#import audio playback widget
import IPython.display as ipd
from IPython.display import Image

import os
import pickle


# In[2]:


data,sampling_rate=librosa.load('/home/gaurav/Downloads/Datasets/anger/anger001.wav')
ipd.Audio(data=data,rate=sampling_rate)


# In[3]:


plt.figure(figsize=(15,5))
librosa.display.waveplot(data,sr=sampling_rate)


# In[4]:


D=librosa.stft(data)
print(D)


# In[5]:


log_power=librosa.amplitude_to_db(D,ref=np.max)
librosa.display.specshow(log_power,x_axis='time',y_axis='linear')
plt.colorbar()


# In[6]:


#log-frequency axis

librosa.display.specshow(log_power,x_axis='time',y_axis='log')
plt.colorbar()


# In[7]:


# path = '/home/gaurav/Downloads/Datasets'
# lst = []

# for subdir, dirs, files in os.walk(path):
#     for file in files:
#         try:
            
#         #Load librosa array, obtain mfcss, store the file and the mcss information in a new array
#             X, sample_rate = librosa.load(os.path.join(subdir,file), res_type='kaiser_fast')
#             mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0) 
#         # The instruction below converts the labels (from 1 to 8) to a series from 0 to 7
#         # This is because our predictor needs to start from 0 otherwise it will try to predict also 0.
#             file = int(file[7:8]) - 1 
#             arr = mfccs, file
#             lst.append(arr)
#       # If the file is not valid, skip it
      
#         except ValueError:
#             continue


# In[8]:


dataset_path = os.path.abspath('/home/gaurav/Downloads/Datasets')
destination_path = os.path.abspath('/home/gaurav/Downloads')
# To shuffle the dataset instances/records
randomize = True
# for spliting dataset into training and testing dataset
split = 0.8
# Number of sample per second for example (16KHz)
sampling_rate = 20000 
emotions=["anger","disgust","fear","happy","neutral", "sad", "surprise"]


# In[9]:


# loading dataframes using dataset module 
from util import dataset
df, train_df, test_df = dataset.create_and_load_meta_csv_df(dataset_path, destination_path, randomize, split)


# In[10]:


train_df


# In[11]:


print("Actual Audio :",df['path'][0])
print("Labels:",df['label'][0])


# 
# ### Labels Assigned for emotions : 
# - 0 : anger
# - 1 : disgust
# - 2 : fear
# - 3 : happy
# - 4 : neutral 
# - 5 : sad
# - 6 : surprise

# In[12]:


unique_label=df.label.unique()
unique_label.sort()
print("Unique Labels in Emotion Dataset are:")
print(*unique_label,sep=',')
unique_label_count=df.label.value_counts(sort=False)
print('\n\nCount of Unique label in emotion dataset are:')
print(*unique_label_count,sep=',')


# In[13]:


# Histogram of classes 

plt.bar(unique_label,unique_label_count,align='center',width=0.5,color='r')
plt.xlabel('Index of labels')
plt.ylabel('Count of labels')
plt.title('Histogram for labels in Emotion Datsets')
plt.show()


# ## Data Pre-Processing
# 
# Calculating MFCC, Pitch, magnitude, Chroma features.

# In[14]:


from util.feature_extraction import get_audio_features
from util.feature_extraction import get_features_dataframe

# train_features,train_label=get_features_dataframe(train_df,sampling_rate)
# test_features,test_label=get_features_dataframe(test_df,sampling_rate)

# train_features.to_pickle('./features_dataframe/train_features')
# train_label.to_pickle('./features_dataframe/train_label')
# test_features.to_pickle('./features_dataframe/test_features')
# test_label.to_pickle('./features_dataframe/test_label')
                         
                         
train_features = pd.read_pickle('./features_dataframe/train_features')
train_label = pd.read_pickle('./features_dataframe/train_label')
test_features = pd.read_pickle('./features_dataframe/test_features')
test_label = pd.read_pickle('./features_dataframe/test_label')


# In[15]:


train_features=train_features.fillna(0)
test_features=test_features.fillna(0)


# In[16]:


# By using .ravel() : Converting 2D to 1D e.g. (2044,1) -> (2044,). To prevent DataConversionWarning

X_train=np.array(train_features)
y_train=np.array(train_label).ravel()
X_test=np.array(test_features)
y_test=np.array(test_label).ravel()


# In[17]:


y_train[:5]
print(X_train.shape)


# In[18]:


# build Neural network and Create Desired Model
import keras
from keras.models import Model
from keras.models import Sequential
from keras.layers import Conv1D,MaxPool1D  #AveragePooling1D
from keras.layers import Flatten,Dropout,Activation,BatchNormalization      #Input
from keras.layers import Dense    #Embedding
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from kerastuner import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters
from kerastuner.engine.hypermodel import HyperModel



# In[19]:


#one-hot encoding
lb=LabelEncoder()

y_train=np_utils.to_categorical(lb.fit_transform(y_train))
y_test=np_utils.to_categorical(lb.fit_transform(y_test))


# In[20]:


y_train[:5]
print(y_train.shape[1])


# 
# 
# ### Changing dimension for CNN model

# In[21]:



x_traincnn=np.expand_dims(X_train,axis=2)             #its shows the depth of 1 for CNN model
x_testcnn=np.expand_dims(X_test,axis=2)


# In[22]:




x_traincnn.shape


# ### Model creation

# In[23]:


model = Sequential()
model.add(Conv1D(256, 8, padding='same',input_shape=(x_testcnn.shape[1],x_traincnn.shape[2]))) #1
model.add(Activation('relu'))

model.add(Conv1D(256, 8, padding='same')) #2
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(MaxPool1D(pool_size=(8)))

# model.add(Conv1D(128, 8, padding='same')) #3
# model.add(Activation('relu')) 

model.add(Conv1D(128, 8, padding='same')) #4
model.add(Activation('relu'))
model.add(Dropout(0.5))

# model.add(Conv1D(128, 5, padding='same')) #5
# model.add(Activation('relu'))

model.add(Conv1D(128, 8, padding='same')) #6
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(MaxPool1D(pool_size=(8)))

# model.add(Conv1D(64, 5, padding='same')) #7
# model.add(Activation('relu'))

model.add(Conv1D(64, 8, padding='same')) #8
model.add(Activation('relu'))
model.add(Flatten())

model.add(Dense(y_train.shape[1])) #9
model.add(Activation('softmax'))
opt = keras.optimizers.rmsprop(lr=0.00001,decay=1e-6)


model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])


# In[24]:


# from keras.wrappers.scikit_learn import KerasClassifier
# from sklearn.model_selection import GridSearchCV


# In[25]:


# model=KerasClassifier(build_fn=Sequential())
# learning=[0.001,0.0001]
# epochs=np.array([100,200,300,400,500])
# batches=np.array([5,10,15,20])


# In[26]:


# param=dict(nb_epoch=epochs,batch_size=batches,lr=learning)
# grid=GridSearchCV(estimator=model,param_grid=param).fit(x_traincnn,y_train)


# In[27]:


cnn=model.fit(x_traincnn,y_train,batch_size=20,epochs=500,validation_data=(x_testcnn,y_test))


# ### Loss visualization

# In[28]:


plt.plot(cnn.history['loss'])
plt.plot(cnn.history['val_loss'])
plt.title('Model loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(['train','test'],loc='upper left')
plt.show() 


# In[29]:


model_name = 'Speech_Emotion_Recognition_Model(78-57).h5'
save_dir = os.path.join(os.getcwd(), 'Trained_Models')
# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)


# In[ ]:


import json
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)


# In[ ]:


# loading json and creating model
from keras.models import model_from_json
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("./Trained_Models/Speech_Emotion_Recognition_Model.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
loaded_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
score = loaded_model.evaluate(x_testcnn, y_test, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))


# ### Prediction on Test Data

# In[ ]:


preds=loaded_model.predict(x_testcnn,batch_size=32,verbose=1)


# In[ ]:


print(preds)
pred=preds.argmax(axis=1)
pred=np_utils.to_categorical(lb.fit_transform(pred))
pred


# In[ ]:


from sklearn.metrics import classification_report
cr=classification_report(y_test,pred)
print(cr)


# In[ ]:


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test.argmax(axis=1),pred.argmax(axis=1))
print(cm)


# In[ ]:


import seaborn as sns
# visualize confusion matrix with seaborn heatmap

cm_matrix = pd.DataFrame(data=cm, columns=['anger:0', 'disgust:1','fear:2','happy:3','neutral:4','sad:5','suprise:6'], 
                                 index=['Actual anger:0', 'Actual disgust:1','Actual fear:2','Actual happy:3','Actual neutral:4','Actual sad:5','Actual suprise:6'])
sns.heatmap(cm_matrix, annot=True,fmt='d', cmap='Blues',linewidths=1)


# In[ ]:


import record_audio
record_audio.Record()


# In[ ]:


demo_audio_path = './demo_audio.wav'
ipd.Audio(demo_audio_path)


# In[ ]:


demo_mfcc, demo_pitch, demo_mag, demo_chrom = get_audio_features(demo_audio_path,sampling_rate)

mfcc = pd.Series(demo_mfcc)
pit = pd.Series(demo_pitch)
mag = pd.Series(demo_mag)
C = pd.Series(demo_chrom)
demo_audio_features = pd.concat([mfcc,pit,mag,C],ignore_index=True)


# In[ ]:


demo_audio_features= np.expand_dims(demo_audio_features, axis=0)
demo_audio_features= np.expand_dims(demo_audio_features, axis=2)


# In[ ]:


demo_audio_features.shape


# In[ ]:


livepreds = loaded_model.predict(demo_audio_features, 
                         batch_size=32, 
                         verbose=1)


# In[ ]:


livepreds


# In[ ]:


emotions=["anger","disgust","fear","happy","neutral", "sad", "surprise"]
index = livepreds.argmax(axis=1).item()
index


# In[ ]:


print(emotions[index])


# In[ ]:




