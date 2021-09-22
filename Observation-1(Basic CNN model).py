#!/usr/bin/env python
# coding: utf-8

# # Observation-1

# ## Basic CNN Model

# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
tf.__version__
import os
from keras.preprocessing.image import ImageDataGenerator
os. getcwd()


# In[ ]:


os. getcwd()


# In[ ]:


train_datagen=ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)
training_set = train_datagen.flow_from_directory(
    '/content/drive/MyDrive/Deep Learning Dataset - Cat vs Dog Classifier/dog-cat-full-dataset-master/dog-cat-full-dataset-master/data/train',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary')


# In[ ]:


test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory(
        '/content/drive/MyDrive/Deep Learning Dataset - Cat vs Dog Classifier/dog-cat-full-dataset-master/dog-cat-full-dataset-master/data/test',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')


# In[ ]:


#building the CNN
#initialising the CNN
cnn=tf.keras.models.Sequential()
#convolutional layer prepration(First layer prepration)
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64,64,3]))
#pooling process
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
#adding a second convolutional layer
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
#falttening
cnn.add(tf.keras.layers.Flatten())
#full conncetion
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
#Output layer
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))


# In[ ]:


#compiling the CNN
cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#training the CNN on training set and evalating on the test set
cnn.fit(x=training_set, validation_data=test_set, epochs=25)


# In[ ]:


from keras.preprocessing import image
test_image=image.load_img('/content/drive/MyDrive/single_prediction/cat_or_dog_1.jpg', target_size=(64, 64))
test_image=image.img_to_array(test_image)
test_image=np.expand_dims(test_image, axis=0)
result=cnn.predict(test_image)
training_set.class_indices
if result[0][0]==1:
    prediction='dog'
else:
    prediction='cat'


# In[ ]:


print(prediction)


# In[ ]:


test_image=image.load_img('/content/drive/MyDrive/single_prediction/cat.220.jpg', target_size=(64, 64))
test_image=image.img_to_array(test_image)
test_image=np.expand_dims(test_image, axis=0)
result=cnn.predict(test_image)
training_set.class_indices
if result[0][0]==1:
    prediction='dog'
else:
    prediction='cat'


# In[ ]:


print(prediction)


# In[ ]:


test_image=image.load_img('/content/drive/MyDrive/single_prediction/dog.169.jpg', target_size=(64, 64))
test_image=image.img_to_array(test_image)
test_image=np.expand_dims(test_image, axis=0)
result=cnn.predict(test_image)
training_set.class_indices
if result[0][0]==1:
    prediction='dog'
else:
    prediction='cat'


# In[ ]:


print(prediction)

