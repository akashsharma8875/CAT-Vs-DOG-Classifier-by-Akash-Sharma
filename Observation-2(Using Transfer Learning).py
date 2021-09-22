#!/usr/bin/env python
# coding: utf-8

# # CAT VS DOG CLASSIFIER

# ## Observation-2(Using Transfer Learning)

# In[ ]:


#required Model Importation
import tensorflow as tf
import os
from tensorflow.keras.preprocessing import image_dataset_from_directory
import numpy as np
import matplotlib.pyplot as plt


# # **`Dataset Path Allocation and Extraction`**
# 
# 

# In[ ]:


URL = 'https://drive.google.com/drive/folders/1fBDldyiNV4WpRC6rEKGIBdw_bMLNDnsZ?usp=sharing'


# In[ ]:


train_dir = os.path.join('/content/drive/MyDrive/Deep Learning Dataset - Cat vs Dog Classifier/dog-cat-full-dataset-master/dog-cat-full-dataset-master/data/train')
validation_dir = os.path.join('/content/drive/MyDrive/Deep Learning Dataset - Cat vs Dog Classifier/dog-cat-full-dataset-master/dog-cat-full-dataset-master/data/test')


# **Creating Training and validation(test) dataset**

# In[ ]:


BATCH_SIZE = 32
IMG_SIZE = (160, 160)
train_dataset = image_dataset_from_directory(train_dir,
                                             shuffle=True,
                                             batch_size=BATCH_SIZE,
                                             image_size=IMG_SIZE)


# In[ ]:


validation_dataset = image_dataset_from_directory(validation_dir,
                                             shuffle=True,
                                             batch_size=BATCH_SIZE,
                                             image_size=IMG_SIZE)


# **Validating the Seperation Size of validation dataset**

# In[ ]:


valdation_batches = tf.data.experimental.cardinality(validation_dataset)
print('Number of validation batches: %d' % tf.data.experimental.cardinality(validation_dataset))


# In[ ]:


test_batches = valdation_batches // 5


# In[ ]:


test_dataset = validation_dataset.take(test_batches)
print('Number of test batches: %d' % tf.data.experimental.cardinality(test_dataset))


# In[ ]:


validation_dataset = validation_dataset.skip(test_batches)
print('Number of validation batches: %d' % tf.data.experimental.cardinality(validation_dataset))


# In[ ]:


class_names = train_dataset.class_names
class_names


# # **Visualising the Training Dataset**

# **Plot one Image**

# In[ ]:


for image, label in train_dataset.take(1):
    image = image.numpy().astype("uint8")
    plt.imshow(image[0])
    plt.title(class_names[label[0]])
    plt.axis('off')
    plt.show()


# **Plotting Multiple training Dataset**

# In[ ]:


plt.figure(figsize=(20, 20))
for images, labels in train_dataset.take(1):
    for i in range(25):
        ax = plt.subplot(5, 5, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")


# # **Configuring the Dataset for Performance**

# In[ ]:


data_augmentation = tf.keras.Sequential([
tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
  tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
])


# **Plotting One Augmented Image**

# In[ ]:


plt.figure(figsize=(10, 10))
for image, label in train_dataset.take(1):
    first_image = image[0]
    print(f'first image shape: {first_image.shape}')
    expaned_dims_first_image =  tf.expand_dims(first_image, 0)
    print(f'Expaned dims of first image:   {expaned_dims_first_image.shape}')
    
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        
        augmented_image=data_augmentation(expaned_dims_first_image)
        
        #rescale augmented_image
        augmented_image = augmented_image[0] / 255
        plt.imshow(augmented_image)
        plt.axis('off')


# # **RESCALLING THE PIXEL VALUES**
# 

# **Creating the Base model from the pre-trained MobileNet V2**

# In[ ]:


preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
preprocess_input


# In[ ]:


IMG_SIZE


# In[ ]:


IMG_SHAPE = IMG_SIZE + (3,)
IMG_SHAPE


# In[ ]:


base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')
base_model


# In[ ]:


image_batch, label_batch = next(iter(train_dataset))
feature_batch = base_model(image_batch)
print(feature_batch.shape)


# **Freezing The Convolutional Base**

# **Model Summary**

# In[ ]:


base_model.trainable = False
base_model.summary()


# **Adding a Classification Head**

# In[ ]:


global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)
print(feature_batch_average.shape)


# **Applying a Dense Layer on Feature batch average**

# In[ ]:


prediction_layer = tf.keras.layers.Dense(1)
prediction_batch = prediction_layer(feature_batch_average)
print(prediction_batch.shape)


# # **Building a Model**

# In[ ]:


inputs = tf.keras.Input(shape=(160, 160, 3))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)
model


# **Compiling the Model**

# In[ ]:


base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate), loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.summary()


# **View the trainable variables of model**

# In[ ]:


len(model.trainable_variables)


# **Checking the intial Loss and accuracy on validation dataset**
# 
# 

# In[ ]:


loss0, accuracy0 = model.evaluate(validation_dataset)


# In[ ]:


print("initial loss: {:.2f}".format(loss0))
print("initial accuracy: {:.2f}".format(accuracy0))


# # **Training the model**

# In[ ]:


EPOCHS = 10
history = model.fit(train_dataset,
                    epochs=EPOCHS,
                    validation_data=validation_dataset)


# **Plotting the loss and accuracy**

# In[ ]:


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']


# In[ ]:


plt.figure(figsize=(8, 8))
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')
plt.show()


# In[ ]:


plt.figure(figsize=(8, 8))
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Loss')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()


# # **Fine tuning of pre-trained model**

# In[ ]:


print("Number of layers in the base model: ", len(base_model.layers))


# **Unfreezing the top layers of the model**

# In[ ]:


base_model.trainable = True
# Fine-tune from this layer onwards
fine_tune_at = 100
# Freeze all the layers before the `fine_tune_at` layer
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable =  False


# # **Re-Compile the Model**

# In[ ]:


model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), optimizer = tf.keras.optimizers.RMSprop(lr=base_learning_rate/10),
              metrics=['accuracy'])
model.summary()


# # **Continuous training of Model**

# In[ ]:


len(model.trainable_variables)
FINE_TUNE_EPOCHS = 10
TOTAL_EPOCHS =  EPOCHS + FINE_TUNE_EPOCHS
history_fine = model.fit(train_dataset,
                         epochs=TOTAL_EPOCHS,
                         initial_epoch=history.epoch[-1],
                         validation_data=validation_dataset)


# **Replotting the model of loss and accuracy with Validation Dataset**

# In[ ]:


acc += history_fine.history['accuracy']
val_acc += history_fine.history['val_accuracy']
loss += history_fine.history['loss']
val_loss += history_fine.history['val_loss']


# In[ ]:


plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.ylim([0.8, 1])
plt.plot([EPOCHS-1,EPOCHS-1], plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.ylim([0, 1.0])
plt.plot([EPOCHS-1,EPOCHS-1], plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()


# **Evaluating the Model after tuning on test dataset**

# In[ ]:


loss, accuracy = model.evaluate(test_dataset)


# In[ ]:


print('Test loss :', loss)
print('Test accuracy :', accuracy)


# # **Predicting the Test Dataset**

# In[ ]:


image_batch, label_batch = test_dataset.as_numpy_iterator().next()
predictions = model.predict_on_batch(image_batch).flatten()
print(predictions)


# In[ ]:


# Apply a sigmoid since our model returns logits
predictions = tf.nn.sigmoid(predictions)
predictions = tf.where(predictions < 0.5, 0, 1)
print('Predictions:\n', predictions.numpy())
print('Labels:\n', label_batch)


# # **Plotting the prediction**

# In[ ]:


plt.figure(figsize=(20, 20))
for i in range(25):
    ax = plt.subplot(5, 5, i + 1)
    plt.imshow(image_batch[i].astype("uint8"))
    plt.title(class_names[predictions[i]])
    plt.axis("off")
    plt.show()


# In[ ]:




