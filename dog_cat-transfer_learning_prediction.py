#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.models import load_model


# In[2]:


model = load_model('Final_Trained_model.h5')


# In[3]:


from keras.preprocessing import image


# In[4]:


test_image = image.load_img('cnn_dataset/single_prediction/cat_or_dog_2.jpg', 
               target_size=(224,224))


# In[5]:


type (test_image)


# In[6]:


test_image.shape


# In[7]:


import numpy as np


# In[8]:


test_image = np.expand_dims(test_image, axis=0)


# In[9]:


test_image.shape


# In[10]:


result = model.predict (test_image)


# In[11]:


if result[0][0] == 1.0:
    print('dog')
else:
    print('cat')


# In[ ]:


#to predict random  imgs in the loaded model
#import os
#import cv2
#import numpy as np
#from os import listdir
#from os.path import isfile, join

#dog_cat_dict = {"[0]": "dog", 
                #"[1]": "cat",
                #}

#dog_cat_dict_n = {"n0": "dog", 
                  #"n1": "cat",
                 #}

#def draw_test(name, pred, im):
    #animal = dog_cat_dict[str(pred)]
    #BLACK = [0,0,0]
    #expanded_image = cv2.copyMakeBorder(im, 80, 0, 0, 100 ,cv2.BORDER_CONSTANT,value=BLACK)
    #cv2.putText(expanded_image, animal, (20, 60) , cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,255), 2)
    #cv2.imshow(name, expanded_image)

#def getRandomImage(path):
    """function loads a random images from a random folder in our test path """
    #folders = list(filter(lambda x: os.path.isdir(os.path.join(path, x)), os.listdir(path)))
    #random_directory = np.random.randint(0,len(folders))
    #path_class = folders[random_directory]
    #print("Class - " + dog_cat_dict_n[str(path_class)])
    #file_path = path + path_class
    #file_names = [f for f in listdir(file_path) if isfile(join(file_path, f))]
    #random_file_index = np.random.randint(0,len(file_names))
    #image_name = file_names[random_file_index]
    #return cv2.imread(file_path+"/"+image_name)    

#for i in range(0,10):
    #input_im = getRandomImage("cnn_dataset/single_prediction/")
    #input_original = input_im.copy()
    #input_original = cv2.resize(input_original, None, fx=0.5, fy=0.5, interpolation = cv2.INTER_LINEAR)
    
    #input_im = cv2.resize(input_im, (224, 224), interpolation = cv2.INTER_LINEAR)
    #input_im = input_im / 255.
    #input_im = input_im.reshape(1,224,224,3) 
    
    # Get Prediction
    #res = np.argmax(classifier.predict(input_im, 1, verbose = 0), axis=1)
    
    # Show image with predicted class
    #draw_test("Prediction", res, input_original) 
    #cv2.waitKey(0)

#cv2.destroyAllWindows()

