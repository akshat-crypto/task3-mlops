#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.applications import VGG16


# In[2]:


model = VGG16(weights = 'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', 
                 include_top = False, 
                 input_shape = (224, 224, 3))


# In[3]:


model.summary()


# In[4]:


for layer in model.layers:
    layer.trainable = False


# In[5]:


layer.__class__


# In[6]:


layer.__class__.__name__


# In[7]:


layer.__class__.__name__, layer.trainable


# In[8]:


enumerate(model.layers)


# In[9]:


for (i,layer) in enumerate(model.layers):
    print(str(i) + " "+ layer.__class__.__name__, layer.trainable)


# In[10]:


def LayerAddflatten(bottom_model, num_classes):
    """creates the top or head of the model that will be 
    placed ontop of the bottom layers"""
    top_model = bottom_model.output
    top_model = Flatten(name = "flatten")(top_model)
    top_model = Dense(526, activation = "relu")(top_model)
    top_model = Dense(263, activation = "relu")(top_model)
    top_model = Dense(num_classes, activation = "sigmoid")(top_model)
    return top_model


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[11]:


from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model

num_classes = 2

FC_Head = LayerAddflatten(model, num_classes)

modelnew = Model(inputs=model.input, outputs=FC_Head)

print(modelnew.summary())


# In[12]:


from keras.preprocessing.image import ImageDataGenerator

train_data_dir = 'cnn_dataset/training_set/'
validation_data_dir = 'cnn_dataset/test_set/'

# Let's use some data augmentaiton 
train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=45,
      width_shift_range=0.3,
      height_shift_range=0.3,
      horizontal_flip=True,
      fill_mode='nearest')
 
validation_datagen = ImageDataGenerator(rescale=1./255)

train_batchsize = 16
val_batchsize = 10
 
train_generator = train_datagen.flow_from_directory(
        'cnn_dataset/training_set/',
        target_size=(224, 224),
        batch_size=train_batchsize,
        class_mode='categorical')
 
validation_generator = validation_datagen.flow_from_directory(
        'cnn_dataset/test_set/',
        target_size=(224, 224 ),
        batch_size=val_batchsize,
        class_mode='categorical')


# In[13]:


from keras.optimizers import RMSprop


# In[14]:


modelnew.compile(loss = 'categorical_crossentropy',
              optimizer = RMSprop( lr = 0.001 ),
              metrics = ['accuracy'])


# In[15]:


nb_train_samples = 1190
nb_validation_samples = 170
epochs = 3
batch_size = 20
history = modelnew.fit_generator(
    train_generator,
    steps_per_epoch = nb_train_samples // batch_size,
    validation_data = validation_generator,
    validation_steps = nb_validation_samples // batch_size,
    epochs = 1)


# In[ ]:


result_accuracy = history.history['accuracy']


# In[ ]:


#import subprocess , sys
#subprocess.Popen("C:\\Users\\Akshat\\Desktop\\mlops\\advance\\task-mlops\\dog_cat-transfer_learning_tweak" , shell=True)
#result_accuracy_f = result_accuracy


# In[ ]:


print ("model is trained and accuracy is:" , result_accuracy)


# In[ ]:


modelnew.save('dog_cat_transfer-learning.h5')


# In[ ]:


#import subprocess , sys
#subprocess.Popen("C:\\Users\\Akshat\\Desktop\\mlops\\advance\\task-mlops\\test2.ipynb" , shell=True)
#x = 74


# In[ ]:


#code for adding diff files.


# In[ ]:


modelnew.save('dog_cat_transfer-learning.pk1')

