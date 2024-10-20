#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 11:05:51 2024

@author: tusker
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
#%%
data=pd.read_csv("/mnt/f/Practice/Python/PythonPractice/cardio_train.csv", sep=";")
#data=pd.read_csv("cardio_train.csv", sep=";")
#%%                 
# dimensions of the dataframe
print("Data shape: " , data.shape )

#print("Data head: ", data.head() )
#%%
# Print column names of the dataframe

#print("Column names : ", data.columns.tolist())
for column in data.columns:
    print(column)
#%%
#Select values of the colums age into an array
ages = data['age'].values
#print(ages)
#%%
# Compute the year of the age values

years_age = ages//365
print(years_age)

#%%
# Compute the mean of the age values
# mean_age = np.mean(years_age)
mean_age = sum(years_age)/len(years_age)
print(mean_age)

#%%
# Compute maximum and minimum of the age values

max_age = years_age.max() #64
min_age = years_age.min() #29

#print('statistics CVD data ', data.describe().transpose() )
#%%
''' Data Preprocessing'''
# blood pressure cannnot be negative
data = data[(data['ap_lo'] <= 370) &\
(data['ap_lo'] > 0)].reset_index(drop=True)
print("errors ap_lo dropped data.shape: ",data.shape)

# The highest blood pressure is 370
data = data[(data['ap_hi'] <= 370) &\
(data['ap_hi'] > 0)].reset_index(drop=True)
print("errors ap_hi dropped data.shape: ",data.shape)

# Systolic blood pressure (ap_hi) is always higher than Diastolic blood pressure (ap_lo)
data = data[ ( data['ap_hi'] ) >= ( data['ap_lo'] ) ].\
        reset_index(drop=True)
print("errors ap_lo > ap_hi dropped data.shape: ",data.shape)

# Split data into training and testing sets

"""X_training_data,X_testing_data = train_test_split(data, 
    test_size=0.15,train_size=0.85,random_state=42)"""
y = data['cardio']
X = data.drop(['cardio'],axis=1).copy()
 
X_train, X_test, y_train, y_test = train_test_split(
    X,y,test_size=0.15, random_state=42)

print(f'training data: { X_train.shape }, \
      Testing data: {X_test.shape}')
      
#%%
# Scale data using standard scaler
ss = StandardScaler()

# [ 0 - 1 ]
X_train_scaled = ss.fit_transform( X_train )

X_test_scaled = ss.fit_transform( X_test )

#%%
# Create a Logistic Regression Model

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

# Train the model
model.fit( X_train_scaled, y_train )

# Predict the output
y_pred = model.predict( X_test_scaled )

# Compare the predicted output with the real value
from sklearn.metrics import confusion_matrix

conf_matrix = confusion_matrix( y_test, y_pred )

print( conf_matrix )

# Print the importance of the features
importances = pd.DataFrame( data={'Attribute' : X_train.columns, \
                                  'Importance': model.coef_[0] } )
importances = importances.sort_values( by = 'Importance', 
                                      ascending = False )
print("\nimportances coefficients Logistic Regression-based model\n",
      importances )

# Print all features and their values of one proband
print( X_test.iloc[0] )

#Visualize feature important in a bar chart
plt.bar( x = importances['Attribute'],\
        height = importances['Importance'], color='#087E8B' )
plt.title('Feature importances Logistic Regression-based model', 
          size = 20)
plt.xticks( rotation = 'vertical' )
plt.show()
#%%
log_model = sm.Logit( y_train,sm.add_constant( X_train ) )
log_result = log_model.fit()
print( log_result.summary2() )

# Feature importance : calculation of these coefficients 
print("Exponents of coefficients\n", 
      np.exp( log_result.params ).sort_values( ascending=False ) )

# sd
print("np.std(X_train)\n",np.std( X_train ) )
