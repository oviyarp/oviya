#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 15:21:45 2021

@author: oviyapavanan
"""

import pandas as pd
import matplotlib.pyplot as plt
from pandas import set_option
from pandas import read_csv
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from numpy import set_printoptions
from pandas.plotting import scatter_matrix
# load boston data set. 

#read boston file
da1 = pd.read_csv('/Users/oviyapavanan/Downloads/boston.csv')

df = pd.DataFrame(da1)
  #removing Extra column:
da1 = df.drop(['Unnamed: 0'], axis= 1)
print(da1)

names = ['crim','zn','indus','chas','nox','rm','age','dis','rad','tax','ptratio','black','lstat','medv']




# save features as pandas dataframe put columns zero to thirteen as X and column 14as Y
############

X = da1.drop(da1.columns[13], axis= 1)
Y = da1.drop(da1.columns[0:13], axis= 1)

# separate features and response into two different arrays (Numpy array). 

array = da1.values
m = array[:, 0:13]
n = array[:, 13]

# First perform exploratory data analysis 
# look at the first 20 rows of data
############

daa1 = da1.head(20)
############

# descriptive statistics: mean, max, min, count, 25 percentile, 50 percentile, 75 percentile
############

#descriptive statistics:
set_option('display.width', 100)
set_option('precision', 1)
description = da1.describe()
print(description)
############

# we look at the distribution of data 
#histogram of da1:
da1.hist()
plt.show()
############
    
# perform data scaling by normalizing only the X (we don't normally perform transformation on the Y/output)
############
# to do
   #Normalization:
scaler = Normalizer().fit(X)
normalizedX = scaler.transform(X)
  #summarize transformation:
set_printoptions(precision = 3)
print(normalizedX[0:5,:])

# you can make a new data frame with the normalized data
dataNormdf = pd.DataFrame(normalizedX, columns = names[0:13])
dataNormdf['medv'] = Y

# calculate the descriptive statistics after normalization

set_option('display.width', 100)
set_option('precision', 1)
Norm_description = dataNormdf.describe()
print(Norm_description)
############

# plot histogram of X and compare with histograms before normalization.
# Does normalization improves the data for predictive modeling?
'''Yes Normalization improves data better when we see the shapes of distribution
       but when we look the values of normalized descriptive statistics, 
       the mean and standard deviation values are not in correct range 
       and it's not done properly'''

dataNormdf.hist() #histogram after normalization
plt.show()
############

# perform data scaling by standardizing only the X (we don't normally perform transformation on the Y/output)
############
# to do
scaler = StandardScaler().fit(X)
rescaledX = scaler.transform(X)
#summarize transformation:
set_printoptions(precision = 3)
print(rescaledX[0:5,:])
############

#new dataframe with standardized data
dataStanddf = pd.DataFrame(rescaledX, columns = names[0:13])
dataStanddf['medv'] = Y

# calculate the descriptive statistics after standardization

#descriptive statistics after standardisation:
set_option('display.width', 100)
set_option('precision', 1)
Stand_description = dataStanddf.describe()
print(Stand_description)
############

# plot histogram of X and compare with histograms before standardizing.
# Does standardizing improves the data for predictive modeling?
   
'''standardized histogram looks same before and after standardization
   and there is no change in shape but change in limit values only.
   The mean and standard deviation values are in correct range and standardization is done properly.'''
############

dataStanddf.hist() #histogram after normalization
plt.show()

#############

# scatter plot of all data below
############
plt.figure()
scatter_matrix(da1, c = 'red')
plt.show()

