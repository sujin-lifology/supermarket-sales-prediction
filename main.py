import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics

data = pd.read_csv('/content/Train.csv')

new_column_names = {
    'Item_Identifier':'Identifier',
    'Item_Fat_Content': 'Fat_Content',
    'Item_Weight':'Weight',
    'Item_Visibility': 'Visibility',
    'Item_Type':'Type',
    'Item_MRP':'MRP',
    'Outlet_Establishment_Year':'Establishment_Year',
    'Outlet_Location_Type':'Location_Type',
    'Item_Outlet_Sales':'Outlet_Sales'
}

data = data.rename(columns=new_column_names)
data.head()
data.shape

data.info()
data.isnull().sum()

data['Weight'].mean()

data['Weight'].fillna(data['Weight'].mean(), inplace=True)
data['Outlet_Size'].mode()
mode_size = data.pivot_table(values='Outlet_Size', columns='Outlet_Type', aggfunc=(lambda x: x.mode()[0]))
missing = data['Outlet_Size'].isnull()
data.loc[missing, 'Outlet_Size'] = data.loc[missing,'Outlet_Type'].apply(lambda x: mode_size[x])

data.isnull().sum()
data.describe()

sns.set()

plt.figure(figsize=(7,7))
sns.distplot(data['Weight'])
plt.show()

plt.figure(figsize=(7,7))
sns.distplot(data['Visibility'])
plt.show()

plt.figure(figsize=(7,7))
sns.distplot(data['MRP'])
plt.show()

plt.figure(figsize=(7,7))
sns.distplot(data['Outlet_Sales'])
plt.show()

plt.figure(figsize=(7,7))
sns.countplot(x='Establishment_Year', data=data)
plt.show()

plt.figure(figsize=(7,7))
sns.countplot(x='Fat_Content', data=data)
plt.show()

plt.figure(figsize=(30,6))
sns.countplot(x='Type', data=data)
plt.show()

plt.figure(figsize=(7,7))
sns.countplot(x='Outlet_Size', data=data)
plt.show()

data.head()
data['Fat_Content'].value_counts()
data.replace({'Fat_Content': {'low fat':'Low Fat','LF':'Low Fat', 'reg':'Regular'}}, inplace=True)
data['Fat_Content'].value_counts()

# Label encoding
encoder = LabelEncoder()
data['Identifier'] = encoder.fit_transform(data['Identifier'])

data['Fat_Content'] = encoder.fit_transform(data['Fat_Content'])

data['Type'] = encoder.fit_transform(data['Type'])

data['Outlet_Identifier'] = encoder.fit_transform(data['Outlet_Identifier'])

data['Outlet_Size'] = encoder.fit_transform(data['Outlet_Size'])

data['Location_Type'] = encoder.fit_transform(data['Location_Type'])

data['Outlet_Type'] = encoder.fit_transform(data['Outlet_Type'])

data.describe()

X_data = data.drop(columns='Outlet_Sales', axis=1)
Y_data = data['Outlet_Sales']

X_training_data, X_testing_data, Y_training_data, Y_testing_data = train_test_split(X_data, Y_data, test_size=0.2, random_state=2)
print(X_data.shape, X_training_data.shape, X_testing_data.shape)

# XGBoost Regressor
model = XGBRegressor()
model.fit(X_training_data, Y_training_data)
training_data = model.predict(X_training_data)
r2_training_data = metrics.r2_score(Y_training_data, training_data)
print('R Squared value for training data = ', r2_training_data)

testing_data = model.predict(X_testing_data)
r2_testing_data = metrics.r2_score(Y_testing_data, testing_data)
print('R Squared value for testing data = ', r2_testing_data)