# Abstract
Similarly, it is a measurement in which we would like to understand each object within the database are alike, whereas dissimilarity measures the difference in objects in the database. This study tells to what degree objects are similar or dissimilar. The measurement is based on the distance between each features in the database. So if distances are lesser, that means both the measured objects are identical, whereas they are not similar when distance are more. Understanding similarly and dissimilarity would be important when using clustering or some classification, and anomaly detection.

# Methods
Below are the following method of measuring the distance between the objects  <br /> <br /> 
## **Euclidean Distance** 
is based on the Pythagoras theorem. The Euclidean distance between any two points, whether the points are 2D or 3D space, is used to measure the length of a segment connecting the two points. 
![](https://github.com/Pramodgopinathan/similarity-dissimilarity/blob/856ae39167d7dcd62a7ac4f68e77b4501e93cb1d/Euclidean_Distance.png) <br/>
**Example** - Car manufacturer company want to give the ads to the users who are interested in buying their product, so for this we have a dataset that contains multiple user's information through the social network. <br/>
Lets Estimated Salary and age consider for the independent variable and the purchased variable is dependent variable. <br/><br/>
Ref: Python code in the repository

### Importing Library, independent and dependent variable
```python:
import numpy as nm  
import matplotlib.pyplot as mtp  
import pandas as pd

data_set= pd.read_csv('user_data.csv')  

x= data_set.iloc[:, [2,3]].values  
y= data_set.iloc[:, 4].values  
```
### Lets split the into test and train
```python:
from sklearn.model_selection import train_test_split  
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.25, random_state=0)  
```
### Lets now standardise the scaler so that machine is able to interpreate between 0s and 1s.
```python:
from sklearn.preprocessing import StandardScaler    
st_x= StandardScaler()    
x_train= st_x.fit_transform(x_train)    
x_test= st_x.transform(x_test)  
```
### Fitting K-NN classifier to the Training data:
Now we will fit the K-NN classifier to the training data. To do this we will import the KNeighborsClassifier class of Sklearn Neighbors library. After importing the class, we will create the Classifier object of the class. The Parameter of this class will be <br /> <br /> 
n_neighbors: To define the required neighbors of the algorithm. Usually, it takes 5. <br />
metric='minkowski': This is the default parameter and it decides the distance between the points. <br />
**p=2: It is equivalent to the standard Euclidean metric.** <br />
And then we will fit the classifier to the training data. Below is the code for it: <br />
```python:
from sklearn.neighbors import KNeighborsClassifier  
classifier= KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2 )  
classifier.fit(x_train, y_train)  
y_pred= classifier.predict(x_test)  
```
### Now we will create the Confusion Matrix for our K-NN model to see the accuracy of the classifier
```python:
from sklearn.metrics import confusion_matrix  
cm= confusion_matrix(y_test, y_pred)  
```
we can see there are 64+29= 93 correct predictions and 3+4= 7 incorrect predictions <br />
### Conclusion 
The predicted output is well good as most of the red points are in the red region and most of the green points are in the green region. However, there are few green points in the red region and a few red points in the green region. So these are the incorrect observations that we have observed in the confusion matrix(7 Incorrect output).

## **Manhattan Distance**
The Manhattan distance (aka taxicab distance) is a measure of the distance between two points on a 2D plan when the path between these two points has to follow the grid layout. 

![](https://github.com/Pramodgopinathan/similarity-dissimilarity/blob/fdcc765727710238fa2a5215ab26683ad70dd41f/Manhattan%20Distance.png)

Manhattan Blocks <br />
It is based on the idea that a taxi will have to stay on the road and will not be able to drive through buildings!

![](https://github.com/Pramodgopinathan/similarity-dissimilarity/blob/fd4704e7353921d70bfa190fe9f8fc3b9913a9e8/Manhattan%20Distance%20-%20Block.png)

## **Cosine Similarity & Cosine Distance**
Cosine Similarity is very widely used in recomedation system. 
Suppose there are two points P1 and P2 and if similarity between two point increases then distance would decrease, sameway if distance between two point increases then similarity decrease.

![](https://github.com/Pramodgopinathan/similarity-dissimilarity/blob/62f7e84d862078852ea813679e71b8dad84f82f4/Cosine%20Similarity%20Part%201.png)

In this example, If we want to find simarity between point P1 and P2 then we have to find angle assiming that angle is around 45 degree. <br />

Cosine similarity = Cosine theta (Angle between P1 and P2)

Also angle would be around -1 to 1 <br />
Example: Cosine 45 = 0.53


![](https://github.com/Pramodgopinathan/similarity-dissimilarity/blob/62f7e84d862078852ea813679e71b8dad84f82f4/Cosine%20Similarity%20Part%202.png)



# Reference
https://www.javatpoint.com/k-nearest-neighbor-algorithm-for-machine-learning

