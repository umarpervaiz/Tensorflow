#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
import os
import zipfile
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from os import path, getcwd, chdir
from PIL import *
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator

path = f"{getcwd()}/../tmp2/happy-or-sad.zip"

zip_ref = zipfile.ZipFile(path, 'r')
zip_ref.extractall("/tmp/h-or-s")
zip_ref.close()


# In[34]:


a = mpimg.imread(path)
plt.imshow(a)


# In[38]:


#Train_happy_sad_model
def train_happy_sad_model():

    DESIRED_ACCURACY = 0.999

    class myCallback(tf.keras.callbacks.Callback):
         def on_epoch_end(self, epoch, logs = {}):
                if(logs.get('acc') > DESIRED_ACCURACY):
                    print("\nReached the desired accuracy level so cancelling training")
                    self.model.stop_training = True

    callbacks = myCallback()

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, (3,3), activation = 'relu', input_shape = (150,150,3)),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(32, (3,3), activation = 'relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64, (3,3), activation = 'relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation = 'relu'),
        tf.keras.layers.Dense(1, activation = 'sigmoid')
    ])

    from tensorflow.keras.optimizers import RMSprop

    model.compile(loss = 'binary_crossentropy',
                 optimizer = RMSprop(lr = 0.001),
                 metrics = ['acc'])

    train_datagen = ImageDataGenerator(rescale = 1/255)

    train_generator = train_datagen.flow_from_directory(
    "/tmp/h-or-s",
    target_size = (150,150),
    batch_size = 180,
    class_mode = 'binary')
    
    history = model.fit_generator(
    train_generator,
    steps_per_epoch = 8,
    epochs = 15,
    callbacks = [callbacks])
    # model fitting
    return history.history['acc'][-1]


# In[39]:


train_happy_sad_model()

