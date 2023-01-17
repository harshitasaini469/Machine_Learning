# Steps in data preprocessing

# 1. Get the Dataset

# 2. Importing Libraries
from tkinter import Label
import numpy as np
import matplotlib.pyplot as mpt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import StandardScaler  

# 3. Importing the dataset
data_set = pd.read_csv('Data.csv')

# Extracting independent variable
x = data_set.iloc[:, :-1].values
# print(x)

# Extracting dependent variable
y = data_set.iloc[:, 3].values

# print(y)

# 4. Handling missing data (Replacing missing data with mean value)
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
# fitting imputer object to the independent variable x.
imputer = imputer.fit(x[:, 1:3])
# Replacing missing data with the calculated mean value
x[:, 1:3] = imputer.transform(x[:, 1:3])
# print(x)

# 5. Encoding categorical Data :
# Since machine learning model completely works on mathematics and numbers, 
# but if our dataset would have a categorical variable, then it may create 
# trouble while building the model. So it is necessary to encode these categorical 
# variables into numbers.

# For country variable

label_encoder_x=LabelEncoder()
x[:,0]=label_encoder_x.fit_transform(x[:,0])
# print(x)

# Dummy encoding using oneHotEncoder
ct=ColumnTransformer([('Country',OneHotEncoder(),[0])],remainder='passthrough')
x=np.array(ct.fit_transform(x),dtype=np.float)
print(x)

# for purchased variable
labelencoder_y=LabelEncoder()
y=labelencoder_y.fit_transform(y)
print(y)

# 6) Splitting the Dataset into the Training set and Test set
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# 7) Feature Scaling
st_x=StandardScaler()
x_train=st_x.fit_transform(x_train)

x_test=st_x.fit_transform(x_test)

print(x_train)
print(x_test)