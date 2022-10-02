---
layout: post
title: Facial Expressions Detector using ResNets
featured: true 
---
## Problem Statement 

Facial expression recognition or computer-based facial expression recognition system is important because of its ability to mimic human coding skills. Facial expressions and other gestures convey nonverbal communication cues that play an important role in interpersonal relations. These cues complement speech by helping the listener to interpret the intended meaning of spoken words. Therefore, facial expression recognition, because it extracts and analyzes information from an image or video feed, it is able to deliver unfiltered, unbiased emotional responses as data.
Its applications include, detecting the customer experience and can also be used during driver monitoring system.

![input](/assets/images/f1.png)

## Importing Libraries 

* Essentially we will be using Tensorflow and Kersas API for image classification.
* cv or opencv used for computer vision
* pandas is used for data manipulation
* numpy is used for numerical analysis
* seaborn and matplotlib is used for data visualisation
* monokai theme is used so that we can see the x and y labels clearly on the graph

```python
import cv2
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import tensorflow as tf
import pickle
from tensorflow.python.keras import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, optimizers
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, LearningRateScheduler
from IPython.display import display
from tensorflow.keras import backend as K
from jupyterthemes import jtplot
jtplot.style(theme='monokai', context='notebook', ticks=True, grid=False) 
# setting the style of the notebook to be monokai theme  
# this line of code is important to ensure that we are able to see the x and y axes clearly
# If you don't run this code line, you will notice that the xlabel and ylabel on any plot is black on black and it will be hard to see them. 
```
## Reading the csv file

```python
# read the csv files
emotion_df = pd.read_csv('emotion.csv')
emotion_df
```

Output
We have an input in from of pixels and a target class that is called emotion. Each individual row represents an image.

```python

emotion	pixels
0	0	70 80 82 72 58 58 60 63 54 58 60 48 89 115 121...
1	0	151 150 147 155 148 133 111 140 170 174 182 15...
2	2	24 32 36 30 32 23 19 20 30 41 21 22 32 34 21 1...
3	2	20 17 19 21 25 38 42 42 46 54 56 62 63 66 82 1...
4	3	77 78 79 79 78 75 60 55 47 48 58 73 77 79 57 5...
...	...	...
24563	3	0 39 81 80 104 97 51 64 68 46 41 67 53 68 70 5...
24564	0	181 177 176 156 178 144 136 132 122 107 131 16...
24565	3	178 174 172 173 181 188 191 194 196 199 200 20...
24566	0	17 17 16 23 28 22 19 17 25 26 20 24 31 19 27 9...
24567	3	30 28 28 29 31 30 42 68 79 81 77 67 67 71 63 6...
24568 rows × 2 columns
```
Since the pixels are in the string format we will convert them into an array format so that it is easier to use them for model building

```python
# function to convert pixel values in string format to array format
def string2array(x):
  return np.array(x.split(' ')).reshape(48, 48, 1).astype('float32')
emotion_df['pixels'] = emotion_df['pixels'].apply(lambda x: string2array(x))
emotion_df['pixels'][2]
```
Output

```python
array([[[ 24.],
        [ 32.],
        [ 36.],
        ...,
        [173.],
        [172.],
        [173.]],

       [[ 25.],
        [ 34.],
        [ 29.],
        ...,
        [173.],
        [172.],
        [173.]],

       [[ 26.],
        [ 29.],
        [ 25.],
        ...,
        [172.],
        [172.],
        [174.]],

       ...,

       [[159.],
        [185.],
        [157.],
        ...,
        [157.],
        [156.],
        [153.]],

       [[136.],
        [157.],
        [187.],
        ...,
        [152.],
        [152.],
        [150.]],

       [[145.],
        [130.],
        [161.],
        ...,
        [142.],
        [143.],
        [142.]]], dtype=float32)
```

## Checking for null values

We will check if any null value is present in the dataset or not, if present we will either remove it or fill it with some values.

```python
# checking for the presence of null values in the data frame
emotion_df.isnull().sum()
```

Output

```python
emotion    0
 pixels    0
pixels     0
dtype: int64
```

## Label encoding the emotions

```python
label_to_text = {0:'anger', 1:'disgust', 2:'sad', 3:'happiness', 4: 'surprise'}
```

## Data Visualisation

In this block of code, we will visualise the images corresponding to the 5 emotion categories mentioned.

```python
emotions = [0,1,2,3,4]

for i in emotions:
  data = emotion_df[emotion_df['emotion'] == i][:1]
  img = data['pixels'].item()
  img = img.reshape(48,48)
  plt.figure()
  plt.title(label_to_text[i])
  plt.imshow(img, cmap= 'gray')

```
![input](/assets/images/f2.png)

## Perform Data Augumentation

Data Augumentation is done so that if our image is rotated slight to the left or right or flipped, we can still classify the image correctly.
In this block, X or the input feature is the pixels and y or the target feature is the emotion. We encode the emotion using get dummy method.

```python
# split the dataframe to features and labels
# from keras.utils import to_categorical

X = emotion_df['pixels']
# y = to_categorical(emotion_df['emotion'])
y = pd.get_dummies(emotion_df['emotion'])

X = np.stack(X, axis = 0)
X = X.reshape(24568, 48, 48, 1)

print(X.shape, y.shape)
```
Output

```python
(24568, 48, 48, 1) (24568, 5)
```

## Splitting the data into Train , Test and Validation sets 

We split the 10 % data into test and the remaining 90% is kept for training. Out of the test, we split 50% into test and 50%. We need to define a validation set to check if the model is overfitting or not. If the training error decreases and the validation error also decreases that means our model is well trained, if they do not resonate that indicates that the model is overfitted. And we need to apply early stopping to prevent the model from memorising the outputs.

```python
# spliting the dataframe in to train,test and validation data frames

from sklearn.model_selection import train_test_split

X_train, X_Test, y_train, y_Test = train_test_split(X,y,test_size = 0.1, shuffle = True)
X_val, X_Test, y_val, y_Test = train_test_split(X_Test,y_Test, test_size = 0.5, shuffle = True)
```


## Data Normalisation

To scale the values between 0 and 1, we divide the values by 255. As 0 indicates black and 255 indicates white.

```python
# image pre-processing

X_train = X_train/255
X_val   = X_val /255
X_Test  = X_Test/255
```

## Image Generator 
 It generates batches of images 

```python
train_datagen = ImageDataGenerator(
rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    fill_mode = "nearest"
)
```

## Understanding intuition behind CNN 

* CNN is used for image classification.
* The image passes through the kernel that extracts the important features of the image.
* Then in pooling layer, the image is compressed or downsampled.
* Pixels are then flattened and passed into fully connected ANN for classification

![input](/assets/images/f3.png)

## Why are RESNETS used ?

* ResNets are used because CNN suffers from vanishing gradient problem.
* ResNets use skip connection thattrains 152 layers without vanishing gradient problem 
* It is a superior algorithm.

![input](/assets/images/f4.png)

## Building and Training a deep learning model- Res-Block

 We have a res-block that has various blocks such as Convolution block and two Identity blocks.
 
 ![input](/assets/images/f5.png)

 ```python
 def res_block(X, filter, stage):

  # Convolutional_block
  X_copy = X

  f1 , f2, f3 = filter

  # Main Path
  X = Conv2D(f1, (1,1),strides = (1,1), name ='res_'+str(stage)+'_conv_a', kernel_initializer= glorot_uniform(seed = 0))(X)
  X = MaxPool2D((2,2))(X)
  X = BatchNormalization(axis =3, name = 'bn_'+str(stage)+'_conv_a')(X)
  X = Activation('relu')(X) 

  X = Conv2D(f2, kernel_size = (3,3), strides =(1,1), padding = 'same', name ='res_'+str(stage)+'_conv_b', kernel_initializer= glorot_uniform(seed = 0))(X)
  X = BatchNormalization(axis =3, name = 'bn_'+str(stage)+'_conv_b')(X)
  X = Activation('relu')(X) 

  X = Conv2D(f3, kernel_size = (1,1), strides =(1,1),name ='res_'+str(stage)+'_conv_c', kernel_initializer= glorot_uniform(seed = 0))(X)
  X = BatchNormalization(axis =3, name = 'bn_'+str(stage)+'_conv_c')(X)


  # Short path
  X_copy = Conv2D(f3, kernel_size = (1,1), strides =(1,1),name ='res_'+str(stage)+'_conv_copy', kernel_initializer= glorot_uniform(seed = 0))(X_copy)
  X_copy = MaxPool2D((2,2))(X_copy)
  X_copy = BatchNormalization(axis =3, name = 'bn_'+str(stage)+'_conv_copy')(X_copy)

  # ADD
  X = Add()([X,X_copy])
  X = Activation('relu')(X)

  # Identity Block 1
  X_copy = X


  # Main Path
  X = Conv2D(f1, (1,1),strides = (1,1), name ='res_'+str(stage)+'_identity_1_a', kernel_initializer= glorot_uniform(seed = 0))(X)
  X = BatchNormalization(axis =3, name = 'bn_'+str(stage)+'_identity_1_a')(X)
  X = Activation('relu')(X) 

  X = Conv2D(f2, kernel_size = (3,3), strides =(1,1), padding = 'same', name ='res_'+str(stage)+'_identity_1_b', kernel_initializer= glorot_uniform(seed = 0))(X)
  X = BatchNormalization(axis =3, name = 'bn_'+str(stage)+'_identity_1_b')(X)
  X = Activation('relu')(X) 

  X = Conv2D(f3, kernel_size = (1,1), strides =(1,1),name ='res_'+str(stage)+'_identity_1_c', kernel_initializer= glorot_uniform(seed = 0))(X)
  X = BatchNormalization(axis =3, name = 'bn_'+str(stage)+'_identity_1_c')(X)

  # ADD
  X = Add()([X,X_copy])
  X = Activation('relu')(X)

  # Identity Block 2
  X_copy = X


  # Main Path
  X = Conv2D(f1, (1,1),strides = (1,1), name ='res_'+str(stage)+'_identity_2_a', kernel_initializer= glorot_uniform(seed = 0))(X)
  X = BatchNormalization(axis =3, name = 'bn_'+str(stage)+'_identity_2_a')(X)
  X = Activation('relu')(X) 

  X = Conv2D(f2, kernel_size = (3,3), strides =(1,1), padding = 'same', name ='res_'+str(stage)+'_identity_2_b', kernel_initializer= glorot_uniform(seed = 0))(X)
  X = BatchNormalization(axis =3, name = 'bn_'+str(stage)+'_identity_2_b')(X)
  X = Activation('relu')(X) 

  X = Conv2D(f3, kernel_size = (1,1), strides =(1,1),name ='res_'+str(stage)+'_identity_2_c', kernel_initializer= glorot_uniform(seed = 0))(X)
  X = BatchNormalization(axis =3, name = 'bn_'+str(stage)+'_identity_2_c')(X)

  # ADD
  X = Add()([X,X_copy])
  X = Activation('relu')(X)

  return X
```

Convolution Block and Identity Blocks have the following components that are mentioned in the code.

![input](/assets/images/f6.png)

## Building the Final Model

![input](/assets/images/f8.png)


```python
input_shape = (48, 48, 1)

# Input tensor shape
X_input = Input(input_shape)

# Zero-padding
X = ZeroPadding2D((3, 3))(X_input)

# 1 - stage
X = Conv2D(64, (7, 7), strides= (2, 2), name = 'conv1', kernel_initializer= glorot_uniform(seed = 0))(X)
X = BatchNormalization(axis =3, name = 'bn_conv1')(X)
X = Activation('relu')(X)
X = MaxPooling2D((3, 3), strides= (2, 2))(X)

# 2 - stage
X = res_block(X, filter= [64, 64, 256], stage= 2)

# 3 - stage
X = res_block(X, filter= [128, 128, 512], stage= 3)

# 4 - stage
# X = res_block(X, filter= [256, 256, 1024], stage= 4)

# Average Pooling
X = AveragePooling2D((2, 2), name = 'Averagea_Pooling')(X)

# Final layer
X = Flatten()(X)
X = Dense(5, activation = 'softmax', name = 'Dense_final', kernel_initializer= glorot_uniform(seed=0))(X)

model_emotion = Model( inputs= X_input, outputs = X, name = 'Resnet18')

model_emotion.summary()
```
Output

```python
Model: "Resnet18"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_1 (InputLayer)            [(None, 48, 48, 1)]  0                                            
__________________________________________________________________________________________________
zero_padding2d (ZeroPadding2D)  (None, 54, 54, 1)    0           input_1[0][0]                    
__________________________________________________________________________________________________
conv1 (Conv2D)                  (None, 24, 24, 64)   3200        zero_padding2d[0][0]             
__________________________________________________________________________________________________
bn_conv1 (BatchNormalization)   (None, 24, 24, 64)   256         conv1[0][0]                      
__________________________________________________________________________________________________
activation (Activation)         (None, 24, 24, 64)   0           bn_conv1[0][0]                   
__________________________________________________________________________________________________
max_pooling2d (MaxPooling2D)    (None, 11, 11, 64)   0           activation[0][0]                 
__________________________________________________________________________________________________
res_2_conv_a (Conv2D)           (None, 11, 11, 64)   4160        max_pooling2d[0][0]              
__________________________________________________________________________________________________
max_pooling2d_1 (MaxPooling2D)  (None, 5, 5, 64)     0           res_2_conv_a[0][0]               
__________________________________________________________________________________________________
bn_2_conv_a (BatchNormalization (None, 5, 5, 64)     256         max_pooling2d_1[0][0]            
__________________________________________________________________________________________________
activation_1 (Activation)       (None, 5, 5, 64)     0           bn_2_conv_a[0][0]                
__________________________________________________________________________________________________
res_2_conv_b (Conv2D)           (None, 5, 5, 64)     36928       activation_1[0][0]               
__________________________________________________________________________________________________
bn_2_conv_b (BatchNormalization (None, 5, 5, 64)     256         res_2_conv_b[0][0]               
__________________________________________________________________________________________________
activation_2 (Activation)       (None, 5, 5, 64)     0           bn_2_conv_b[0][0]                
__________________________________________________________________________________________________
res_2_conv_copy (Conv2D)        (None, 11, 11, 256)  16640       max_pooling2d[0][0]              
__________________________________________________________________________________________________
res_2_conv_c (Conv2D)           (None, 5, 5, 256)    16640       activation_2[0][0]               
__________________________________________________________________________________________________
max_pooling2d_2 (MaxPooling2D)  (None, 5, 5, 256)    0           res_2_conv_copy[0][0]            
__________________________________________________________________________________________________
bn_2_conv_c (BatchNormalization (None, 5, 5, 256)    1024        res_2_conv_c[0][0]               
__________________________________________________________________________________________________
bn_2_conv_copy (BatchNormalizat (None, 5, 5, 256)    1024        max_pooling2d_2[0][0]            
__________________________________________________________________________________________________
add (Add)                       (None, 5, 5, 256)    0           bn_2_conv_c[0][0]                
                                                                 bn_2_conv_copy[0][0]             
__________________________________________________________________________________________________
activation_3 (Activation)       (None, 5, 5, 256)    0           add[0][0]                        
__________________________________________________________________________________________________
res_2_identity_1_a (Conv2D)     (None, 5, 5, 64)     16448       activation_3[0][0]               
__________________________________________________________________________________________________
bn_2_identity_1_a (BatchNormali (None, 5, 5, 64)     256         res_2_identity_1_a[0][0]         
__________________________________________________________________________________________________
activation_4 (Activation)       (None, 5, 5, 64)     0           bn_2_identity_1_a[0][0]          
__________________________________________________________________________________________________
res_2_identity_1_b (Conv2D)     (None, 5, 5, 64)     36928       activation_4[0][0]               
__________________________________________________________________________________________________
bn_2_identity_1_b (BatchNormali (None, 5, 5, 64)     256         res_2_identity_1_b[0][0]         
__________________________________________________________________________________________________
activation_5 (Activation)       (None, 5, 5, 64)     0           bn_2_identity_1_b[0][0]          
__________________________________________________________________________________________________
res_2_identity_1_c (Conv2D)     (None, 5, 5, 256)    16640       activation_5[0][0]               
__________________________________________________________________________________________________
bn_2_identity_1_c (BatchNormali (None, 5, 5, 256)    1024        res_2_identity_1_c[0][0]         
__________________________________________________________________________________________________
add_1 (Add)                     (None, 5, 5, 256)    0           bn_2_identity_1_c[0][0]          
                                                                 activation_3[0][0]               
__________________________________________________________________________________________________
activation_6 (Activation)       (None, 5, 5, 256)    0           add_1[0][0]                      
__________________________________________________________________________________________________
res_2_identity_2_a (Conv2D)     (None, 5, 5, 64)     16448       activation_6[0][0]               
__________________________________________________________________________________________________
bn_2_identity_2_a (BatchNormali (None, 5, 5, 64)     256         res_2_identity_2_a[0][0]         
__________________________________________________________________________________________________
activation_7 (Activation)       (None, 5, 5, 64)     0           bn_2_identity_2_a[0][0]          
__________________________________________________________________________________________________
res_2_identity_2_b (Conv2D)     (None, 5, 5, 64)     36928       activation_7[0][0]               
__________________________________________________________________________________________________
bn_2_identity_2_b (BatchNormali (None, 5, 5, 64)     256         res_2_identity_2_b[0][0]         
__________________________________________________________________________________________________
activation_8 (Activation)       (None, 5, 5, 64)     0           bn_2_identity_2_b[0][0]          
__________________________________________________________________________________________________
res_2_identity_2_c (Conv2D)     (None, 5, 5, 256)    16640       activation_8[0][0]               
__________________________________________________________________________________________________
bn_2_identity_2_c (BatchNormali (None, 5, 5, 256)    1024        res_2_identity_2_c[0][0]         
__________________________________________________________________________________________________
add_2 (Add)                     (None, 5, 5, 256)    0           bn_2_identity_2_c[0][0]          
                                                                 activation_6[0][0]               
__________________________________________________________________________________________________
activation_9 (Activation)       (None, 5, 5, 256)    0           add_2[0][0]                      
__________________________________________________________________________________________________
res_3_conv_a (Conv2D)           (None, 5, 5, 128)    32896       activation_9[0][0]               
__________________________________________________________________________________________________
max_pooling2d_3 (MaxPooling2D)  (None, 2, 2, 128)    0           res_3_conv_a[0][0]               
__________________________________________________________________________________________________
bn_3_conv_a (BatchNormalization (None, 2, 2, 128)    512         max_pooling2d_3[0][0]            
__________________________________________________________________________________________________
activation_10 (Activation)      (None, 2, 2, 128)    0           bn_3_conv_a[0][0]                
__________________________________________________________________________________________________
res_3_conv_b (Conv2D)           (None, 2, 2, 128)    147584      activation_10[0][0]              
__________________________________________________________________________________________________
bn_3_conv_b (BatchNormalization (None, 2, 2, 128)    512         res_3_conv_b[0][0]               
__________________________________________________________________________________________________
activation_11 (Activation)      (None, 2, 2, 128)    0           bn_3_conv_b[0][0]                
__________________________________________________________________________________________________
res_3_conv_copy (Conv2D)        (None, 5, 5, 512)    131584      activation_9[0][0]               
__________________________________________________________________________________________________
res_3_conv_c (Conv2D)           (None, 2, 2, 512)    66048       activation_11[0][0]              
__________________________________________________________________________________________________
max_pooling2d_4 (MaxPooling2D)  (None, 2, 2, 512)    0           res_3_conv_copy[0][0]            
__________________________________________________________________________________________________
bn_3_conv_c (BatchNormalization (None, 2, 2, 512)    2048        res_3_conv_c[0][0]               
__________________________________________________________________________________________________
bn_3_conv_copy (BatchNormalizat (None, 2, 2, 512)    2048        max_pooling2d_4[0][0]            
__________________________________________________________________________________________________
add_3 (Add)                     (None, 2, 2, 512)    0           bn_3_conv_c[0][0]                
                                                                 bn_3_conv_copy[0][0]             
__________________________________________________________________________________________________
activation_12 (Activation)      (None, 2, 2, 512)    0           add_3[0][0]                      
__________________________________________________________________________________________________
res_3_identity_1_a (Conv2D)     (None, 2, 2, 128)    65664       activation_12[0][0]              
__________________________________________________________________________________________________
bn_3_identity_1_a (BatchNormali (None, 2, 2, 128)    512         res_3_identity_1_a[0][0]         
__________________________________________________________________________________________________
activation_13 (Activation)      (None, 2, 2, 128)    0           bn_3_identity_1_a[0][0]          
__________________________________________________________________________________________________
res_3_identity_1_b (Conv2D)     (None, 2, 2, 128)    147584      activation_13[0][0]              
__________________________________________________________________________________________________
bn_3_identity_1_b (BatchNormali (None, 2, 2, 128)    512         res_3_identity_1_b[0][0]         
__________________________________________________________________________________________________
activation_14 (Activation)      (None, 2, 2, 128)    0           bn_3_identity_1_b[0][0]          
__________________________________________________________________________________________________
res_3_identity_1_c (Conv2D)     (None, 2, 2, 512)    66048       activation_14[0][0]              
__________________________________________________________________________________________________
bn_3_identity_1_c (BatchNormali (None, 2, 2, 512)    2048        res_3_identity_1_c[0][0]         
__________________________________________________________________________________________________
add_4 (Add)                     (None, 2, 2, 512)    0           bn_3_identity_1_c[0][0]          
                                                                 activation_12[0][0]              
__________________________________________________________________________________________________
activation_15 (Activation)      (None, 2, 2, 512)    0           add_4[0][0]                      
__________________________________________________________________________________________________
res_3_identity_2_a (Conv2D)     (None, 2, 2, 128)    65664       activation_15[0][0]              
__________________________________________________________________________________________________
bn_3_identity_2_a (BatchNormali (None, 2, 2, 128)    512         res_3_identity_2_a[0][0]         
__________________________________________________________________________________________________
activation_16 (Activation)      (None, 2, 2, 128)    0           bn_3_identity_2_a[0][0]          
__________________________________________________________________________________________________
res_3_identity_2_b (Conv2D)     (None, 2, 2, 128)    147584      activation_16[0][0]              
__________________________________________________________________________________________________
bn_3_identity_2_b (BatchNormali (None, 2, 2, 128)    512         res_3_identity_2_b[0][0]         
__________________________________________________________________________________________________
activation_17 (Activation)      (None, 2, 2, 128)    0           bn_3_identity_2_b[0][0]          
__________________________________________________________________________________________________
res_3_identity_2_c (Conv2D)     (None, 2, 2, 512)    66048       activation_17[0][0]              
__________________________________________________________________________________________________
bn_3_identity_2_c (BatchNormali (None, 2, 2, 512)    2048        res_3_identity_2_c[0][0]         
__________________________________________________________________________________________________
add_5 (Add)                     (None, 2, 2, 512)    0           bn_3_identity_2_c[0][0]          
                                                                 activation_15[0][0]              
__________________________________________________________________________________________________
activation_18 (Activation)      (None, 2, 2, 512)    0           add_5[0][0]                      
__________________________________________________________________________________________________
Averagea_Pooling (AveragePoolin (None, 1, 1, 512)    0           activation_18[0][0]              
__________________________________________________________________________________________________
flatten (Flatten)               (None, 512)          0           Averagea_Pooling[0][0]           
__________________________________________________________________________________________________
Dense_final (Dense)             (None, 5)            2565        flatten[0][0]                    
==================================================================================================
Total params: 1,174,021
Trainable params: 1,165,445
Non-trainable params: 8,576
```

## Training the model and defining the early stopping function

```python
# train the network
model_emotion.compile(optimizer = "Adam", loss = "categorical_crossentropy", metrics = ["accuracy"])
# Recall that the first facial key points model was saved as follows: FacialKeyPoints_weights.hdf5 and FacialKeyPoints-model.json

# using early stopping to exit training if validation loss is not decreasing even after certain epochs (patience)
earlystopping = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = 20)

# save the best model with lower validation loss
checkpointer = ModelCheckpoint(filepath = "FacialExpression_weights.hdf5", verbose = 1, save_best_only=True)

history = model_emotion.fit(train_datagen.flow(X_train, y_train, batch_size=64),
	validation_data=(X_val, y_val), steps_per_epoch=len(X_train) // 64,
	epochs= 20, callbacks=[checkpointer, earlystopping])
```

Output

```python
Train for 345 steps, validate on 1228 samples
Epoch 1/20
344/345 [============================>.] - ETA: 0s - loss: 1.4257 - accuracy: 0.4219
Epoch 00001: val_loss improved from inf to 1.49407, saving model to FacialExpression_weights.hdf5
345/345 [==============================] - 338s 980ms/step - loss: 1.4256 - accuracy: 0.4219 - val_loss: 1.4941 - val_accuracy: 0.4153
Epoch 2/20
275/345 [======================>.......] - ETA: 1:06 - loss: 1.1768 - accuracy: 0.5217
```

## Accessing the performance of the model

```python
with open('Emotion-model.json', 'r') as json_file:
    json_savedModel= json_file.read()
    
# load the model architecture 
model_emotion = tf.keras.models.model_from_json(json_savedModel)
model_emotion.load_weights('FacialExpression_weights.hdf5')
model_emotion.compile(optimizer = "Adam", loss = "categorical_crossentropy", metrics = ["accuracy"])
score = model_emotion.evaluate(X_Test, y_Test)
print('Test Accuracy: {}'.format(score[1]))
```

Output

The model accuracy is 72% 

```python
2s 2ms/sample - loss: 0.7766 - accuracy: 0.7177
Test Accuracy: 0.7176566123962402
```
## Visualising the Training Accuracy vs Validation Accuracy
 
 If both of these show an upward trend that means our model is well trained.

 ```python
 accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(accuracy))

plt.plot(epochs, accuracy, 'bo', label='Training Accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

```

Output

![input](/assets/images/f9.png)

## Visualising the Training Loss vs Validation Loss

```python
plt.plot(epochs, loss, 'ro', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and Validation loss')
plt.legend()
```
Output

![input](/assets/images/f10.png)

## Creating Confusion Matrix

```python
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true, predicted_classes)
plt.figure(figsize = (10, 10))
sns.heatmap(cm, annot = True, cbar = False)
```

![input](/assets/images/f11.png)

## Lets visualise the results!

We will print a grid of 25 images along with their predicted/true label

```python
L = 5
W = 5

fig, axes = plt.subplots(L, W, figsize = (24, 24))
axes = axes.ravel()

for i in np.arange(0, L*W):
    axes[i].imshow(X_Test[i].reshape(48,48), cmap = 'gray')
    axes[i].set_title('Prediction = {}\n True = {}'.format(label_to_text[predicted_classes[i]], label_to_text[y_true[i]]))
    axes[i].axis('off')

plt.subplots_adjust(wspace = 1)
```

Output

![input](/assets/images/f12.png)

