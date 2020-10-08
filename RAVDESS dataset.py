#!/usr/bin/env python
# coding: utf-8

# # You'll need to install the following libraries with !pip top run this code.!!

# In[1]:


get_ipython().system('pip install librosa')


# In[2]:


get_ipython().system('pip install soundfile')


# In[3]:


get_ipython().system('pip install numpy')


# In[4]:


get_ipython().system('pip install sklearn')


# In[4]:


# Install this Microsoft Visual C++ 14.0 before installing
get_ipython().system('pip3 install pyaudio #(Optional) ')


# In[3]:


# Make the necessary imports

import librosa
import soundfile
import os, glob, pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


# In[7]:


#DataFlair - Extract features (mfcc, chroma, mel) from a sound file

def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate=sound_file.samplerate
        if chroma:
            stft=np.abs(librosa.stft(X))
        result=np.array([])
        if mfcc:
            mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result=np.hstack((result, mfccs))
        if chroma:
            chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result=np.hstack((result, chroma))
        if mel:
            mel=np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
            result=np.hstack((result, mel))
    return result


# In[8]:


#DataFlair - Emotions in the RAVDESS dataset
emotions={
  '01':'neutral',
  '02':'calm',
  '03':'happy',
  '04':'sad',
  '05':'angry',
  '06':'fearful',
  '07':'disgust',
  '08':'surprised'
}

#DataFlair - Emotions to observe
observed_emotions=['calm', 'happy', 'fearful', 'disgust']


# In[9]:


#DataFlair - Load the data and extract features for each sound file

def load_data(test_size=0.2):
    x,y=[],[]
    for file in glob.glob("E:\Resume 2020 Projects Extreme\speech-emotion-recognition-ravdess-data\\Actor_*\\*.wav"):
        file_name=os.path.basename(file)
        emotion=emotions[file_name.split("-")[2]]
        if emotion not in observed_emotions:
            continue
        feature=extract_feature(file, mfcc=True, chroma=True, mel=True)
        x.append(feature)
        y.append(emotion)
    return train_test_split(np.array(x), y, test_size=test_size, random_state=9)


# In[10]:


#speech-emotion-recognition-ravdess-data- Split the dataset
x_train,x_test,y_train,y_test=load_data(test_size=0.25)


# In[11]:


#speech-emotion-recognition-ravdess-data - Get the shape of the training and testing datasets
print((x_train.shape[0], x_test.shape[0]))


# In[12]:


#speech-emotion-recognition-ravdess-data- Get the number of features extracted
print(f'Features extracted: {x_train.shape[1]}')


# In[14]:


#speech-emotion-recognition-ravdess-data- Initialize the Multi Layer Perceptron Classifier
model=MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(300,), learning_rate='adaptive', max_iter=500)


# In[15]:


#speech-emotion-recognition-ravdess-data - Train the model
model.fit(x_train,y_train)


# In[16]:


#speech-emotion-recognition-ravdess-data - Predict for the test set
y_pred=model.predict(x_test)


# In[17]:


#speech-emotion-recognition-ravdess-data - Calculate the accuracy of our model
accuracy=accuracy_score(y_true=y_test, y_pred=y_pred)

#speech-emotion-recognition-ravdess-data - Print the accuracy
print("Accuracy: {:.2f}%".format(accuracy*100))


# In[ ]:




