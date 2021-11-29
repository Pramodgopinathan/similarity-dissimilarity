# Abstract
Similarly, it is a measurement in which we would like to understand each object within the database are alike, whereas dissimilarity measures the difference in objects in the database. This study tells to what degree objects are similar or dissimilar. The measurement is based on the distance between each features in the database. So if distances are lesser, that means both the measured objects are identical, whereas they are not similar when distance are more. Understanding similarly and dissimilarity would be important when using clustering or some classification, and anomaly detection.

# Methods
Below are the following method of measuring the distance between the objects  <br />
**Euclidean Distance** is based on the Pythagoras theorem. The Euclidean distance between any two points, whether the points are 2D or 3D space, is used to measure the length of a segment connecting the two points. 
![](https://github.com/Pramodgopinathan/similarity-dissimilarity/blob/856ae39167d7dcd62a7ac4f68e77b4501e93cb1d/Euclidean_Distance.png) <br/>
**Example** - Car manufacturer company want to give the ads to the users who are interested in buying their product, so for this we have a dataset that contains multiple user's information through the social network. <br/>
Lets Estimated Salary and age consider for the independent variable and the purchased variable is dependent variable. <br/><br/>
Ref: Python code in the repository

Importing Library, independent and dependent variable
```python:
import numpy as nm  
import matplotlib.pyplot as mtp  
import pandas as pd

data_set= pd.read_csv('user_data.csv')  

x= data_set.iloc[:, [2,3]].values  
y= data_set.iloc[:, 4].values  
```
Lets split the into test and train
```python:
from sklearn.model_selection import train_test_split  
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.25, random_state=0)  
```
Lets now standardise the scaler so that machine is able to interpreate between 0s and 1s.
```python:
from sklearn.preprocessing import StandardScaler    
st_x= StandardScaler()    
x_train= st_x.fit_transform(x_train)    
x_test= st_x.transform(x_test)  
```

# Reference
https://www.javatpoint.com/k-nearest-neighbor-algorithm-for-machine-learning
