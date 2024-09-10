# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 11:06:43 2024

@author: udayk
"""
#importing required libraries

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

df=pd.read_csv(r"C:\Users\udayk\Desktop\PRACTICE\Dataset\Salary Data.[Linear reg].csv")

# to display the first 5 data's
df.head()

df.shape #dimension of the dataset(rows,columns)

df.info() # provides information about data types and non null values in each column


df.isnull().sum()  #checks for missing values in each column

df.describe()  #summary stats(mean,SD,min,max...)

df.columns  # list all column names

df1 = df.dropna()  # removing rows with missing values
df1.isnull().sum()  # confirms there are no missing values after dropping them.

df1['Age'].unique()   # displays unique values in the Age column

df1['Gender'].unique()

df1['Education Level'].unique()

df1['Job Title'].unique()

df1.columns


# Choosing only the features that is relevant for the prediction
# You can also keep 'Job Title' in the dataframe but i haven't kept it
df2= df1.drop( ['Age', 'Gender','Job Title'], axis=1)
df2.head(10)


# Converting the Categorial data into numbers
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
df2['Education Level'] = label_encoder.fit_transform(df2['Education Level'])
df2.head()


# Splitting the data into x and y for training and testing purpose
x=df2[['Education Level', 'Years of Experience']]
x.shape

x.head(5)

y=df2[['Salary']]
y.head(5)

# Splitting the above data into training and testing data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size= 0.2,random_state = 9598)

# to know about the shape of traing and testing data

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# As it is the Regrssion Problem lets try all the Regression Algorith to get the best accuracy
from sklearn.linear_model import LinearRegression
model = LinearRegression()
# To train and fit the model
model.fit(x_train,y_train)

# to predict the data
y_pred = model.predict(x_test)
y_pred

# to know what the orignal prediction was for the testing data
y_test

#to Calculate the Accuracy of the Model
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error,mean_squared_error

# to print the mean absolute error(MAE)
mean_absolute_error(y_test,y_pred)

# to print the mean absolute percentage
per_e = mean_absolute_percentage_error(y_test,y_pred)
per_e

# to print the accuracy in percentage
acc1 = (1-per_e)*100
acc1
