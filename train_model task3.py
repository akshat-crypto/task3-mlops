#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf 


# In[2]:


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Restrict TensorFlow to only use the fourth GPU
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')

        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


# In[3]:


from keras.applications import VGG16


# In[4]:


model = VGG16(weights = 'imagenet', 
                 include_top = False, 
                 input_shape = (224, 224, 3))


# In[5]:


model.summary()


# In[6]:


for layer in model.layers:
    layer.trainable = False


# In[7]:


layer.__class__


# In[8]:


layer.__class__.__name__


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


# In[11]:


from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model


# In[12]:


num_classes = 2

FC_Head = LayerAddflatten(model, num_classes)

modelnew = Model(inputs=model.input, outputs=FC_Head)


# In[13]:


print(modelnew.summary())


# In[14]:


from keras.preprocessing.image import ImageDataGenerator


# In[15]:


train_data_dir = 'C:/Users/MAC/Desktop/MLOPS/DATASETS/datasets/160647_367971_bundle_archive_3/data/train_set/'
validation_data_dir = 'C:/Users/MAC/Desktop/MLOPS/DATASETS/datasets/160647_367971_bundle_archive_3/data/test_set/'


# In[16]:


train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=45,
      width_shift_range=0.3,
      height_shift_range=0.3,
      horizontal_flip=True,
      fill_mode='nearest')


# In[17]:


validation_datagen = ImageDataGenerator(rescale=1./255)

train_batchsize = 16
val_batchsize = 10


# In[18]:


train_generator = train_datagen.flow_from_directory(
        'C:/Users/MAC/Desktop/MLOPS/DATASETS/datasets/160647_367971_bundle_archive_3/data/train_set/',
        target_size=(224, 224),
        batch_size=train_batchsize,
        class_mode='categorical')


# In[19]:


validation_generator = validation_datagen.flow_from_directory(
'C:/Users/MAC/Desktop/MLOPS/DATASETS/datasets/160647_367971_bundle_archive_3/data/test_set/',
        target_size=(224, 224 ),
        batch_size=val_batchsize,
        class_mode='categorical')


# In[20]:


from keras.optimizers import RMSprop


# In[21]:


modelnew.compile(loss = 'categorical_crossentropy',
              optimizer = RMSprop( lr = 0.001 ),
              metrics = ['accuracy'])


# In[22]:


nb_train_samples = 1190
nb_validation_samples = 170
#epochs = 3
batch_size = 20
history = modelnew.fit_generator(
    train_generator,
    steps_per_epoch = nb_train_samples // batch_size,
    validation_data = validation_generator,
    validation_steps = nb_validation_samples // batch_size,
    epochs = 5)


# In[23]:


modelnew.save('test1.h5')


# In[90]:


result_accuracy = history.history['accuracy']


# In[91]:


print (result_accuracy)


# In[92]:


print ("the best accuracy this model will get is: " , result_accuracy)


# In[93]:


result_accuracy.reverse()
result_accuracy = result_accuracy[0]
result_accuracy = int(result_accuracy*100)


# In[94]:


print (result_accuracy)


# In[105]:


f=open("accuracy.txt","w+")
f.write(str(result_accuracy))
f.close()


# In[102]:


file1 = open("accuracy.txt","r+")  
result_accuracy=file1.read()


# In[103]:


print(result_accuracy)


# In[ ]:





# In[ ]:




