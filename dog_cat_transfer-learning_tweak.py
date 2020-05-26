#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.models import load_model


# In[2]:


model = load_model('dog_cat_transfer-learning.h5')


# In[3]:


#from dog_cat-transfer_learning 


# In[ ]:


#import dog_cat-transfer_learning.result_accuracy_f


# In[4]:


model.summary()


# In[5]:


for layer in model.layers[:-18]:
    layer.trainable = False


# In[6]:


for (i,layer) in enumerate(model.layers):
    print(str(i) + " "+ layer.__class__.__name__, layer.trainable)


# In[7]:


#def LayerAddflatten(bottom_model, num_classes):
    """creates the top or head of the model that will be 
    placed ontop of the bottom layers"""
    #top_model = bottom_model.output
    #top_model = Flatten(name = "flatten")(top_model)
    #top_model = Dense(526, activation = "relu")(top_model)
    #top_model = Dense(263, activation = "relu")(top_model)
    #top_model = Dense(num_classes, activation = "sigmoid")(top_model)
    #return top_model


# In[22]:


def LayerAdd(bottom_model, num_classes,):
    """creates the top or head of the model that will be 
    placed ontop of the bottom layers"""
    top_model = bottom_model.output
    top_model = Dense(526, activation = "relu")(top_model)
    top_model = Dense(263, activation = "relu")(top_model)
    top_model = Dense(num_classes, activation = "sigmoid")(top_model)
    return top_model


# In[23]:


model.summary()


# In[24]:


model.input


# In[25]:


from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model

num_classes = 2

FC_Head = LayerAdd(model, num_classes)

modelnew = Model(inputs=model.input, outputs=FC_Head)

print(modelnew.summary())


# In[ ]:


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


# In[ ]:


from keras.optimizers import RMSprop
modelnew.compile(loss = 'categorical_crossentropy',
                          optimizer = RMSprop(lr = 0.001),
                          metrics = ['accuracy'])


# In[11]:


nb_train_samples = 1190
nb_validation_samples = 170
history = modelnew.fit_generator(
                        train_generator,
                        steps_per_epoch = nb_train_samples // batch_size,
                        validation_data = validation_generator,
                        validation_steps = nb_validation_samples // batch_size,
                        epochs = 1)


# In[ ]:


def LayerRemove():
    modelnew.summary()
    modelnew.layers.pop()
    modelnew.summary()

#x = MaxPooling2D()(model.layers[-1].output)
#o = Activation('sigmoid', name='loss')(x)

#model2 = Model(input=in_img, output=[o])
#model2.summary()


# In[ ]:


val1 = .85


# In[ ]:


val2 = .80


# In[ ]:


val3 = .75


# In[ ]:


for i in range (1 , 5):
    
    for x in result_accuracy_f:
        if ( x >= val1):
             print("Model trained successfully in the given range.")
            break
            
       #elif (x>=val2 & x<=val1):
            #modelnew = LayerRemove()
            #num_classes = 2
            #from keras.models import Sequential
            #from keras.layers import Dense, Dropout, Activation, Flatten
            #from keras.layers import Conv2D, MaxPooling2D
            #from keras.models import Model

            #num_classes = 2

            #FC_Head = LayerAdd(modelnew, num_classes)

            #modelnew = Model(inputs=model.input, outputs=FC_Head)

            #print(modelnew.summary())
        
            #from keras.optimizers import Adam
            #modelnew.compile(loss = 'categorical_crossentropy',
              #            optimizer = 'Adam',
               #           metrics = ['accuracy'])
        
            #histry = modelnew.fit_generator(
                        #train_generator,
                        #validation_data = validation_generator,
                        #steps_per_epoch=300,
                        #epochs = 3)
        else:
            modelnew = LayerRemove()
            num_classes = 2
            from keras.models import Sequential
            from keras.layers import Dense, Dropout, Activation, Flatten
            from keras.layers import Conv2D, MaxPooling2D
            from keras.models import Model

            num_classes = 2

            FC_Head = LayerAdd(modelnew, num_classes)

            modelnew = Model(inputs=model.input, outputs=FC_Head)

            print(modelnew.summary())
            
            from keras.optimizers import RMSprop
            modelnew.compile(loss = 'categorical_crossentropy',
                          optimizer = RMSprop(lr = 0.001),
                          metrics = ['accuracy'])
        
            history = modelnew.fit_generator(
                        train_generator,
                        validation_data = validation_generator,
                        
                        epochs = 1)


# In[ ]:


modelnew.save('Final_Trained_model.h5')


# In[ ]:


result_accuracy = history.history['accuracy']


# In[ ]:


print ("the best accuracy this model will get is: " , result_accuracy)


# In[1]:


import sys


# In[ ]:


sys.exit(result_accuracy)

