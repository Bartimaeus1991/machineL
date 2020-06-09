# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 23:15:29 2020

@author: SRIV0
"""

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# import the libraries that are required for viewing the image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Lets start with k nearest neighbors from a library
# For that we need to import it from scikit learn. We will deal with this next time
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier 


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# import the libraries that are required for viewing the image
import matplotlib.pyplot as plt

print("packages imported")
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Data input. This is a dataframe of width L and length 785. 1st column has the number and the next 784 columns have the digit
digit_data=pd.read_csv("C:/Users/SRIV0/Downloads/digit-recognizer/train.csv")

print("Data opened")
# Choose a random number from the frame and plot it to see how it looks

number_in_one_line=digit_data.iloc[8].to_numpy()
number=np.reshape(number_in_one_line[1:],(28,28))



print("Image plotted")


#Convert dataframe to numpy array X and y

train_data=digit_data.values
y=train_data[:,0]
X=train_data[:,1:784]

# Split this into training and validation data. Random state decides the internal random number generator, stratify is for stratified sampling.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state=42, stratify=y)

print("Data_split")




#Now put the previous lines in a loop and get the error for different k's
k_values=np.array([1,2,3,4,5,6,7])
train_accuracy = np.empty(len(k_values))
test_accuracy = np.empty(len(k_values))

#loop for testing



# Loop over different values of k
for i, k in enumerate(k_values):
    # Setup a k-NN Classifier with k neighbors: knn
    knn = KNeighborsClassifier(n_neighbors=k)
    print(i)
    print(k)
    # Fit the classifier to the training data
    knn.fit(X_train, y_train)
    
    #Compute accuracy on the training set
    train_accuracy[i] = knn.score(X_train, y_train)

    #Compute accuracy on the testing set
    test_accuracy[i] = knn.score(X_test, y_test)




plt.title('k-NN: Varying Number of Neighbors')
plt.plot(k_values, test_accuracy, label = 'Testing Accuracy')
plt.plot(k_values, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()


