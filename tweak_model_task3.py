#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf 
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


# In[2]:


from keras.models import load_model


# In[3]:


def recreate_model():
  from keras.applications import VGG16
  model = VGG16(weights = 'imagenet', 
                 include_top = False, 
                 input_shape = (224, 224, 3))
  for layer in model.layers:
    layer.trainable = False
  return (model)


# In[4]:



def LayerAddF(bottom_model , num_classes , neurons):
  top_model = bottom_model.output
  top_model = Flatten(name = "flatten")(top_model)
  top_model = Dense(neurons, activation='relu')(top_model)
  top_model = Dense(int(neurons/2),activation='relu')(top_model)
  top_model = Dense(int(neurons/4),activation='relu')(top_model)
  top_model = Dense(int(neurons/8),activation='relu')(top_model)
  top_model = Dense(int(neurons/16), activation ='relu')(top_model)
  top_model = Dense(int(neurons/32) , activation ='relu')(top_model)
  top_model = Dense(num_classes , activation='softmax')(top_model)
  return top_model


# In[5]:


def LayerAdds(bottom_model , num_classes , neurons):
  top_model = bottom_model.output
  top_model = Flatten(name = "flatten")(top_model)
  top_model = Dense(int(neurons), activation='relu')(top_model)
  top_model = Dense(int(neurons/2),activation='relu')(top_model)
  top_model = Dense(int(neurons/4),activation='relu')(top_model)
  top_model = Dense(int(neurons/8),activation='relu')(top_model)
  top_model = Dense(int(neurons/16) , activation ='relu')(top_model)
  top_model = Dense(num_classes , activation='softmax')(top_model)
  return top_model


# In[6]:


from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model


# In[7]:


val1 = 92

val2 = 80

val3 = 75


# In[8]:


file1 = open("C:/Users/MAC/Desktop/task3/task3-mlops/accuracy.txt","r+")  
result_accuracy=file1.read()


# In[9]:


print(result_accuracy)


# In[10]:


result_accuracy=int(result_accuracy)


# In[11]:


model= recreate_model()


# In[12]:


from keras.preprocessing.image import ImageDataGenerator


# In[13]:


train_data_dir = 'C:/Users/MAC/Desktop/MLOPS/DATASETS/datasets/160647_367971_bundle_archive_3/data/train_set/'
validation_data_dir = 'C:/Users/MAC/Desktop/MLOPS/DATASETS/datasets/160647_367971_bundle_archive_3/data/test_set/'


# In[14]:


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
        'C:/Users/MAC/Desktop/MLOPS/DATASETS/datasets/160647_367971_bundle_archive_3/data/train_set/',
        target_size=(224, 224),
        batch_size=train_batchsize,
        class_mode='categorical')
 
validation_generator = validation_datagen.flow_from_directory(
        'C:/Users/MAC/Desktop/MLOPS/DATASETS/datasets/160647_367971_bundle_archive_3/data/test_set/',
        target_size=(224, 224 ),
        batch_size=val_batchsize,
        class_mode='categorical')


# In[15]:



        
if (result_accuracy > val1):
  print ("best model has been trained with accuracy: " , result_accuracy)
  
elif (result_accuracy>val2 & result_accuracy<val1): 
  num_classes = 2
  model = recreate_model()
  neurons = 1024
  FC_Head = LayerAdds(model, num_classes , neurons)
  modelnew = Model(inputs=model.inputs, outputs=FC_Head)
  print(modelnew.summary())
  epochs=5
  for i in range (1,6):
    epochs = epochs+5
    from keras.optimizers import RMSprop
    modelnew.compile(loss = 'categorical_crossentropy',
              optimizer = RMSprop( lr = 0.001 ),
              metrics = ['accuracy'])
    nb_train_samples = 1190
    nb_validation_samples = 170
    batch_size = 20
    history = modelnew.fit_generator(
       train_generator,
       steps_per_epoch = nb_train_samples // batch_size,
       validation_data = validation_generator,
       validation_steps = nb_validation_samples // batch_size,
       epochs = epochs)
  result_accuracy_final = history.history['accuracy']
  ##result_accuracy_final = int(result_accuracy_final)
  result_accuracy_final.reverse()
  result_accuracy_final = result_accuracy_final[0]
  result_accuracy_final=int(result_accuracy_final*100)
  if (result_accuracy_final<result_accuracy):
    print("The accuracy of model is not increasing after changing epochs")
  else:
    print("The accuracy of model is increasing after changing epochs")

else:
  model = recreate_model()
  num_classes = 2
  neurons = 2048
  FC_Head = LayerAddF(model, num_classes , neurons)
  modelnew = Model(inputs=model.inputs, outputs=FC_Head)
  print(modelnew.summary())
  epochs=5
  for i in range (1,3):
    epochs = epochs+5
    from keras.optimizers import RMSprop
    modelnew.compile(loss = 'categorical_crossentropy',
              optimizer = RMSprop( lr = 0.001 ),
              metrics = ['accuracy'])
    nb_train_samples = 1190
    nb_validation_samples = 170
    batch_size = 20
    history = modelnew.fit_generator(
       train_generator,
       steps_per_epoch = nb_train_samples // batch_size,
       validation_data = validation_generator,
       validation_steps = nb_validation_samples // batch_size,
       epochs = epochs)
  result_accuracy_final = history.history['accuracy']
    ##result_accuracy_final = int(result_accuracy_final)
  result_accuracy_final.reverse()
  result_accuracy_final = result_accuracy_final[0]
  result_accuracy_final=int(result_accuracy_final*100)
  if (result_accuracy_final<result_accuracy):
    print("The accuracy of model is not increasing after changing epochs")
  else:
    print("The accuracy of model is increasing after changing epochs")

result_accuracy_final = history.history['accuracy']
    ##result_accuracy_final = int(result_accuracy_final)
result_accuracy_final.reverse()
result_accuracy_final = result_accuracy_final[0]
result_accuracy_final=int(result_accuracy_final*100)
if (result_accuracy_final<result_accuracy):
  print("The accuracy of model is not increasing after changing epochs")
else:
  print("The accuracy of model is increasing after changing epochs")


# In[16]:


print(result_accuracy_final)


# In[ ]:




