# Kyle MacLaughlin's Resume Projects

### Summary

Here I have compiled into one Jupyter notebook my implementations of a few statistical and machine learning tasks applying common methods to datasets from a variety of fields. Along the way, I include some implementation details and my motivations for choosing them, but I have also included some data visualizations to make the process and results more intuitive.

### Table of Contents

1. [Detecting Tumors With A Deep Convolutional Network](#Detecting-Tumors-With-A-Deep-Convolutional-Network-brainmedical_symbol)
    1. [Results](#Testing-the-model)
2. [A Recommender System For Netflix Movies](#A-Recommender-System-For-Netflix-Movies-popcorndvd)
3. [Sentiment Analysis In Product Reviews](#Sentiment-Analysis-In-Product-Reviews)

## Detecting Tumors With A Deep Convolutional Network :brain::medical_symbol:

### Background

One challenge in modern medicine is identifying tumors and other anomalies on brain MR scans. The goal of developing increasingly sophisticated systems for identifying tumors (and other visible abnormalities) is to aid radiologists in their meticulous searches. In this section, I will create a deep convolutional neural network in order to classify scans as positive (tumorous) or negative (typical).

### Implementation Details

In order to create this model, I downloaded the _Br35H :: Brain Tumor Detection 2020_ image dataset published on Kaggle (https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection) by Ahmed Hamada. This dataset contains two folders labelled "yes" (positive) and "no" (negative), each containing 1500 images of axial slices taken from MR scans. I will segment two-thirds of the data (2000 images) into a training set, and the rest will be placed in the test set. Only one model will be used (for simplicity), so no validation set is required.

### Preparing the data


```python
"""
    Written by Kyle MacLaughlin in 2022
    - A deep convolutional classifier model to detect brain tumors -
"""

# Begin with imports

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Sequential
import matplotlib.pyplot as plt
import matplotlib.image as plt_img
from skimage.transform import resize
```


```python
# Load the data
no = [plt_img.imread(f'data/Br35H/no/no{i}.jpg') for i in range(1500)]
yes = [plt_img.imread(f'data/Br35H/yes/y{i}.jpg') for i in range(1500)]

# Convert each image to normalized grayscale (we can assume no alpha from .jpg)
for i in range(1500):
    if len(no[i].shape) == 3:
        no[i] = np.mean(no[i], axis=(-1,)) / 255
    if len(yes[i].shape) == 3:
        yes[i] = np.mean(yes[i], axis=(-1,)) / 255

# Peak at first five scans from each category
fig, axs = plt.subplots(2, 5, figsize=(16,6), dpi=200)
fig.suptitle('First five negative/positive scans', fontsize=24)

for i in range(5):
    axs[0,i].imshow(no[i], cmap='bone')
    axs[1,i].imshow(yes[i], cmap='bone')
```


    
![png](README_files/README_3_0.png)
    



```python
# These images could clearly use some cleanup;
# We will make each image square, then scale it to match the size of the smallest image
data = []

def padded(img):
    r, c = img.shape
    if r == c:
        return img
    elif r > c:
        left = (r - c) // 2
        right = r - c - left
        return np.pad(img, [(0,0), (left, right)])
    else:
        top = (c - r) // 2
        bot = c - r - top
        return np.pad(img, [(top, bot), (0,0)])

for img in no:
    sq_img = padded(img)
    data += [[sq_img, False]]

for img in yes:
    sq_img = padded(img)
    data += [[sq_img, True]]

min([x[0].shape[0] for x in data])
```




    175




```python
# Since the smallest size is 175 x 175, we will resize all images to match this
X = []
y = []

# We will introduce a categorical encoding for outputs:
# negative -> [0,1], positive -> [1,0]
for img,cat in data:
    X += [img if img.shape[0] == 175 else resize(img, (175, 175))]
    y += [[1 * cat, 1 - 1 * cat]]
    
# Make into NumPy arrays with input channel added to shape
X = np.array(X)[:,:,:,None]
y = np.array(y)

# Finally, we will shuffle the data and create training and test sets
indices = list(range(3000))
np.random.shuffle(indices)
X = X[indices]
y = y[indices]

train_X, train_y = X[:2000], y[:2000]
test_X, test_y = X[2000:], y[2000:]
```

### Building & training the model

When we finally build our model, we identify increasingly high-level features with a series of convolutional layers, followed by a sudden dimensionality reduction, and we feed this into our output layer. The output layer consists of two units with a softmax activation for our categorical data, which is accompanied by an appropriate binary cross-entropy loss function. For our dimensionality-reduction layers, we include regularization penalties to ensure that parameters do not grow too large.


```python
model1 = tf.keras.Sequential([
    layers.Input((175, 175, 1)),
    layers.Conv2D(64, (3,3)),
    layers.Conv2D(32, (5,5)),
    layers.Conv2D(16, (5,5)),
    layers.Conv2D(8, (7,7), strides=(2,2)),
    layers.Conv2D(4, (7,7), strides=(2,2)),
    layers.Flatten(),
    layers.Dense(50, activation='relu',
                 kernel_regularizer='l2', bias_regularizer='l2'),
    layers.Dense(2, activation='softmax')
], 'Tumor-Detector')

model1.compile(optimizer='adam', loss='binary_crossentropy')
model1.summary()
```

    Model: "Tumor-Detector"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     conv2d_71 (Conv2D)          (None, 173, 173, 64)      640       
                                                                     
     conv2d_72 (Conv2D)          (None, 169, 169, 32)      51232     
                                                                     
     conv2d_73 (Conv2D)          (None, 165, 165, 16)      12816     
                                                                     
     conv2d_74 (Conv2D)          (None, 80, 80, 8)         6280      
                                                                     
     conv2d_75 (Conv2D)          (None, 37, 37, 4)         1572      
                                                                     
     flatten_14 (Flatten)        (None, 5476)              0         
                                                                     
     dense_29 (Dense)            (None, 50)                273850    
                                                                     
     dense_30 (Dense)            (None, 2)                 102       
                                                                     
    =================================================================
    Total params: 346,492
    Trainable params: 346,492
    Non-trainable params: 0
    _________________________________________________________________



```python
model1.fit(train_X, train_y, batch_size=64, epochs=3)
```

    Epoch 1/3
    32/32 [==============================] - 208s 6s/step - loss: 1.5178
    Epoch 2/3
    32/32 [==============================] - 210s 7s/step - loss: 1.0078
    Epoch 3/3
    32/32 [==============================] - 215s 7s/step - loss: 0.6601





    <keras.callbacks.History at 0x138644970>



### Testing the model

Now that we have constructed our model and fitted it to the training set, we will put it to the test! We will compute the balanced error rate (BER) of the model on the testing set. This is the mean of the false positive and negative rates, and provides a metric for how well the model generalizes to the rest of the data.


```python
# Make predictions from test set inputs
pred = model1.predict(test_X)

# Ignore confidence levels, apply firm labels
pred = 1 * (pred > 0.5)

# Identify true/false positives & negatives
true = pred * test_y
false = pred * (1 - test_y)

# The column-wise sums yield +/-
tp, tn = np.sum(true, axis=0)
fp, fn = np.sum(false, axis=0)

fpr = fp / (fp + tn) # false positive rate
fnr = fn / (fn + tp) # false negative rate
ber = 0.5 * (fpr + fnr)
print(f'Balanced error rate of model over test set: {ber}')
```

    32/32 [==============================] - 17s 538ms/step
    Balanced error rate of model over test set: 0.09001100990891803


We observe a BER of 0.09, meaning our model is fairly accurate. While this result is not state of the art, it could certainly be useful in the context of flagging potential tumors for radiologists to examine&mdash;especially with a tweak to the classification threshold.


```python
# Peak at first 15 scans from test set and predictions
fig, axs = plt.subplots(3, 5, figsize=(16,10), dpi=200)
fig.suptitle('First fifteen test scans & predictions', fontsize=24)

for i in range(15):
    tumor = 'Tumor' if pred[i,0] > 0 else 'No Tumor'
    corr = 'Correct' if np.sum(pred[i] * test_y[i]) > 0 else 'Incorrect'
    axs[i//5, i%5].imshow(test_X[i,:,:,0], cmap='bone')
    axs[i//5, i%5].set_title(f'Prediction {i+1}:\n{tumor} ({corr})', fontsize=12)
    axs[i//5, i%5].set_xticks([])
    axs[i//5, i%5].set_yticks([])
```


    
![png](README_files/README_12_0.png)
    


## A Recommender System For Netflix Movies :popcorn::dvd:

### Background
In 2006, Netflix issued a challenge that was hard (for us plebeians) to ignore: any team that developed a recommender system for movies using data collected by the streaming service that could beat their in-house _Cinematch_ algorithm's error score by at least 10% would receive a $1 million prize. In spite of the competition having ended more than a decade ago, I have decided to put my skills to the test by developing a recommender system of my own using the same dataset.

### Implementation Details
I downloaded the official _Netflix Prize_ dataset published by Netflix on Kaggle (https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data). In this example, I aim to beat the company's original RMSE (root-mean-square error) of 0.9525 as per the [contest rules](https://web.archive.org/web/20100106185508/http://www.netflixprize.com//rules). For my model, I will use a function that was demonstrated in my recommender systems class. Be forewarned, there will be a lot of math ahead (feel free to skip to the code/results).

The model in question is a function of the average rating across the database (the mean of the per-movie mean user ratings), each user's average deviation from the mean, each movie's average deviation from the mean, and an alignment score between the user and the movie. Here is the function in all of its glory:
$$f(u,m) = \mu + \delta_u + \delta_m + \alpha_u \cdot \beta_m$$

where $\mu$ is the average rating, $\delta_u$ is user $u$'s average deviation from the mean, $\delta_m$ is movie $m$'s average deviation from the mean, and $\alpha_u$ and $\beta_m$ are the respective $k$-dimensional representations of the user and the movie respectively. The loss function we are trying to minimize is the RMSE with a small regularization penalty applied, given by

$$L(\alpha, \beta, \delta) = \sqrt{\sum_{u,m} f(u,m) - R_{u,m}^2}$$


```python

```

## Sentiment Analysis In Product Reviews

### Background
It is often useful to determine more information about users' preferences than merely their binary or numerical ratings of a product or service. For example, a small or mid-sized business might collect data from its customers, but those data might be too sparse to explain the consumers' reasons for making purchases or leaving reviews. In such cases, it is useful to identify latent factors in their thinking, referred to as _sentiment_.


```python

```
