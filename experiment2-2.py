# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 18:14:36 2018

@author: shail
"""

import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  

dataset = pd.read_csv('C:\Users\shail\Downloads\data1.csv')  
X = dataset[['Telecommunication Infrastructure Index']]
y = dataset['E-Government Rank']  
from sklearn.cross_validation import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)  
from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(X_train, y_train)  
coeff_df = pd.DataFrame(regressor.coef_, X.columns, columns=['Coefficient'])  
 
y_pred = regressor.predict(X_test)  
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
from sklearn import metrics  
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred))) 

plt.scatter(X_test, y_test,  color='blue')
plt.title('Test Data')
plt.xlabel('Telecommunication Infrastructure Index')
plt.ylabel('Rank')
plt.xticks(())
plt.yticks(())
 
plt.show()

plt.plot(X_test, regressor.predict(X_test), color='red',linewidth=3)
print('accuracy',regressor.score(X_test,y_test))