#!/usr/bin/env python
# coding: utf-8

# In[15]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dropout, Dense
from keras.utils import np_utils
import tensorflow


# In[17]:


x_data = pd.read_csv('fashion-mnist_train.csv')
test = pd.read_csv('fashion-mnist_test.csv')


# In[42]:


x_ = np.array(x_data)
x = x_[:,1:]
x = x/255.0
y = x_[:,0]

test_ = np.array(test)
x_t =  test_[:,1:]
x_t = x_t/255.0
y_t = test_[:,0]


# In[43]:


#reshaping for network

x_train = x.reshape((-1,28,28,1))
y_train = np_utils.to_categorical(y)  # to create one hot notation
print(x_train.shape, y_train.shape)

x_test = x_t.reshape((-1,28,28,1))
y_test = np_utils.to_categorical(y_t)  # to create one hot notation
print(x_test.shape, y_test.shape)


# In[ ]:





# In[ ]:





# In[ ]:





# In[28]:


#creating CNN model
model = Sequential()
model.add(Convolution2D(32,(3,3), activation = 'relu', input_shape = (28,28,1)))
model.add(Convolution2D(64,(3,3), activation = 'relu'))
model.add(Dropout(.25))
model.add(MaxPooling2D(2,2))

model.add(Convolution2D(128,(3,3), activation = 'relu'))
model.add(Convolution2D(64,(3,3), activation = 'relu'))
model.add(Dropout(.25))
model.add(MaxPooling2D(2,2))

model.add(Convolution2D(32,(2,2), activation = 'relu'))
model.add(Convolution2D(16,(2,2), activation = 'relu', data_format = 'channels_first'))
model.add(Flatten())

model.add(Dense(10,activation = 'softmax'))

model.summary()


# In[29]:


model.compile(loss ='categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])


# In[30]:


hist = model.fit(x_train, y_train, epochs = 10, shuffle = True, batch_size = 256, validation_split = 0.2)


# In[44]:


plt.figure(0)
plt.plot(hist.history['loss'], 'g')
plt.plot(hist.history['val_loss'], 'b')
plt.plot(hist.history['accuracy'], 'r')
plt.plot(hist.history['val_accuracy'], 'black')
plt.show()


# In[45]:


hist.history.keys()


# In[46]:


print(np.mean(hist.history['accuracy']), np.mean(hist.history['val_accuracy']))


# In[47]:


len(hist.history['accuracy'])


# In[48]:


model.evaluate(x_test,y_test)


# In[ ]:




