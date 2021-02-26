# importing required libraries
import warnings
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
warnings.filterwarnings('ignore')

#Loading dataset
data = pd.read_csv("hiring.csv")
#View of data
print(data.head(10))
#Shape of data
print(data.shape)
print(data.info())
##Checking missing values in dataset
print(data.isnull().sum())
#We can see that experience & test_score has some missing values.
#Now we fill missing values in dataset
data['experience'].fillna(0, inplace = True)
data['test_score(out of 10)'].fillna(data['test_score(out of 10)'].mean(), inplace = True)

#again check missing values in dataset
print("Number of null values in dataset:\n",data.isnull().sum())
#We can see that there is no missing values in datset

#Now converting categorical variable to numerical
def word_to_number(word):
    word_dict = {"one":1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8, 'nine': 9, 'ten': 10,
                 'eleven': 11, 'twelve': 12, 'zero':0, 0:0}

    return word_dict[word]

data['experience'] = data['experience'].apply(lambda x: word_to_number(x))

print(data.head(10))
print()
print(data.dtypes)
print()
#Now splitting dataset into independent variable and dependent variable
X = data.iloc[:,:-1]
y = data.iloc[:,-1]
print(X.head())
print(y.head())

#Now create regression model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
#train the model
regressor.fit(X,y)

#Now save model to disk
import pickle
#pickle.dump(regressor, open("salary_model.pkl", 'wb'))

#Load the model for prediction
load_model = pickle.load(open('salary_model.pkl', 'rb'))

print(load_model.predict([[0,8,8]]))
