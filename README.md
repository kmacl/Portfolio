# Kyle MacLaughlin's Resume Projects

## Summary

Here I have compiled into one Jupyter notebook my implementations of a few statistical and machine learning tasks applying common methods to datasets from a variety of fields.

#### Table of Contents

1. [Predicting User Gender From Review Text](#Predicting-User-Gender-From-Review-Text)
2. [Detecting Tumors With A Deep Convolutional Network](#Detecting-Tumors-With-A-Deep-Convolutional-Network)
3. [Sentiment Analysis In Product Reviews](#Sentiment-Analysis-In-Product-Reviews)

## Predicting User Gender From Review Text

### Background

Often it is useful to be able to predict the probability that the user of some service belongs to a particular demographic category when it is unspecified. Having such additional information could potentially bolster the efficacy of a recommender system. Moreover, there are analogous scenarios across many domains in which we might take interest in determining the probability that a generated data point belongs to some latent category. Therefore, we will use logistic regression to predict the probability that a user is 

### Implementation Details




```python

```

## Detecting Tumors With A Deep Convolutional Network

### Background

One challenge in modern medicine is identifying tumors and other anomalies on brain MR scans. The goal of developing increasingly sophisticated systems for identifying tumors (and other visible abnormalities) is to aid radiologists in their meticulous searches. In this section, I will create a deep convolutional neural network in order to classify scans as positive (tumorous) or negative (typical).

### Implementation Details

In order to create this model, I have downloaded the _Br35H :: Brain Tumor Detection 2020_ image dataset published on Kaggle (https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection) by Ahmed Hamada. This dataset contains two folders labelled "yes" (positive) and "no" (negative), each containing 1500 images of axial slices taken from MR scans. I will segment 2/3 of the data (1000 images from each category) into a training set, and the rest will be placed in the testing set. Only one model will be used (for simplicity), so no validation set is required.


```python
import numpy as np
import pandas as pd
import tensorflow as tf
```

## Sentiment Analysis In Product Reviews


```python

```
