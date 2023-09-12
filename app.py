#!/usr/bin/env python
# coding: utf-8

# # Using Machine Learning to predict Sales Price

# It is finally time to model our dataset to predict the SalePrice per property and the goal of this lesson is to learn how a target variable can be predicted using Machine Learning. We will also evaluate our model in this lesson.

# Now, let us start by importing the necessary libraries,

# In[ ]:


import pandas as pd
import lightgbm as lgb
import datetime


# Next, importing the CSV trained version file called `houseprices_data.csv` which contains pre-processed information about housing prices. 

# In[ ]:


# Reading in the CSV file as a DataFrame
df = pd.read_csv(r'C:\Users\muham\Downloads\train (1).csv', low_memory=False)


# In[ ]:


# Looking at the first five rows
df


# In[ ]:


# Printing the shape
df.shape


# In[ ]:


df.head()


# In[ ]:


df.shape


# # Lets check all Null Values

# In[ ]:


df.isnull().sum()


# In[ ]:


df['LotFrontage'].mode()


# In[ ]:


df['LotFrontage'].fillna(60,inplace=True)


# In[ ]:


df['LotFrontage'].isnull().sum()


# In[ ]:


df['Alley'].fillna(df['Alley'].mode,inplace=True)


# In[ ]:


df['Alley'].isnull().sum()


# In[ ]:


df['Street'].isnull().sum()


# In[ ]:


df['LotShape'].isnull().sum()


# In[ ]:


df['LandContour'].isnull().sum()


# In[ ]:


df['Utilities'].isnull().sum()


# In[ ]:


df['PoolArea'].isnull().sum()


# In[ ]:


df['PoolQC'].isnull().sum()


# In[ ]:


df['PoolQC'].fillna(df['PoolQC'].mean,inplace=True)


# In[ ]:


df['PoolQC'].isnull().sum()


# In[ ]:


df['Fence'].isnull().sum()


# In[ ]:


df['Fence'].fillna(df['Fence'].mean, inplace=True)


# In[ ]:


df['Fence'].isnull().sum()


# In[ ]:


df['MiscFeature'].isnull().sum()


# In[ ]:


df['MiscFeature'].fillna(df['MiscFeature'].mean, inplace=True)


# In[ ]:


df['MiscFeature'].isnull().sum()


# In[ ]:


df['MiscVal'].isnull().sum()


# In[ ]:


df['MoSold'].isnull().sum()


# In[ ]:


df['YrSold'].isnull().sum()


# In[ ]:


df['SaleType'].isnull().sum()


# In[ ]:


df['SaleCondition'].isnull().sum()


# In[ ]:


df['SalePrice'].isnull().sum()


# # Now that all null values have been cleared based on attribute natures as we want our predictions to e as solid as possible thus using mean, median and mode are the best techniques for doing so

# # Lets first drop all string columns as ML algorithms cant read string attributes
# 

# In[ ]:


# lets check all data types
df.dtypes


# In[ ]:


df.drop(['MSZoning'], axis=1, inplace=True)


# In[ ]:


# Street	Alley	LotShape	LandContour	Utilities
df.drop(['Street'], axis=1, inplace=True)


# In[ ]:


df.drop(['Alley'], axis=1, inplace=True)


# In[ ]:


df.drop(['LotShape'], axis=1, inplace=True)


# In[ ]:


df.drop(['LandContour'], axis=1, inplace=True)


# In[ ]:


df.drop(['Utilities'], axis=1, inplace=True)


# In[ ]:


# 	LotConfig	LandSlope	Neighborhood	Condition1	Condition2	BldgType, PoolQC	Fence	MiscFeature
# SaleType	SaleCondition
df.drop(['LotConfig'], axis=1, inplace=True)


# In[ ]:


df.drop(['LandSlope'], axis=1, inplace=True)


# In[ ]:


df.drop(['Neighborhood'], axis=1, inplace=True)


# In[ ]:


df.drop(['Condition1'], axis=1, inplace=True)


# In[ ]:


df.drop(['Condition2'], axis=1, inplace=True)


# In[ ]:


df.drop(['BldgType'], axis=1, inplace=True)


# In[ ]:


df.drop(['PoolQC'], axis=1, inplace=True)


# In[ ]:


df.drop(['Fence'], axis=1, inplace=True)


# In[ ]:


df.drop(['MiscFeature'], axis=1, inplace=True)


# In[ ]:


df.drop(['SaleType'], axis=1, inplace=True)


# In[ ]:


df.drop(['SaleCondition'], axis=1, inplace=True)


# In[ ]:


df.drop(['HouseStyle'], axis=1, inplace=True)


# In[ ]:


df.drop(['RoofStyle'], axis=1, inplace=True)


# In[ ]:


df.drop(['RoofMatl'], axis=1, inplace=True)


# In[ ]:


df.drop(['Exterior1st'], axis=1, inplace=True)


# In[ ]:


df.drop(['Exterior2nd'], axis=1, inplace=True)


# In[ ]:


df.drop(['MasVnrType'], axis=1, inplace=True)


# In[ ]:


df.drop(['ExterQual'], axis=1, inplace=True)


# In[ ]:


df.drop(['ExterCond'], axis=1, inplace=True)


# In[ ]:


df.drop(['Foundation'], axis=1, inplace=True)


# In[ ]:


df.drop(['BsmtQual'], axis=1, inplace=True)


# In[ ]:


df.drop(['BsmtCond'], axis=1, inplace=True)


# In[ ]:


#BsmtExposure
df.drop(['BsmtExposure'], axis=1, inplace=True)


# In[ ]:


df.drop(['BsmtFinType1'], axis=1, inplace=True)


# In[ ]:


df.drop(['BsmtFinSF1'], axis=1, inplace=True)


# In[ ]:


df.drop(['BsmtFinType2'], axis=1, inplace=True)


# In[ ]:


df.drop(['Heating'], axis=1, inplace=True)


# In[ ]:


df.drop(['HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'FireplaceQu',
'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'MoSold'], axis=1, inplace=True)


# In[ ]:


df.drop(['YrSold'], axis=1, inplace=True)


# In[ ]:


df.drop(['LotFrontage', 'MasVnrArea', 'Functional', 'GarageYrBlt'], axis=1, inplace=True)


# In[ ]:


df


# In[ ]:


df.dtypes


# # As seen above we only have integer values which will make it accurate for us to make our prediction

# In[ ]:


df


# First of all, let us split the dataset based on a 70:30 ratio. 70% of the dataset will be used for training our LightGBM model and 30% of the dataset will be used for evaluating it.

# Next, let us get the target variable (y) and the features (X) from the splitted DataFrames. Please mind that we will be removing some columns since they cannot be used for training the model.

# In[ ]:


# Getting the target (y) from the splitted DataFrames
train_y = df["SalePrice"].astype(float).values
eval_y = df["SalePrice"].astype(float).values

# Getting the features (X) from the splitted DataFrames
train_X = df.drop(['SalePrice', 'GarageCars'], axis=1)
eval_X = df.drop(['SalePrice', 'GarageCars'], axis=1)


# In[ ]:


from sklearn.model_selection import train_test_split


# Creating a custom function to train the LightGBM model with hyperparameters

# In[ ]:


def train_lightgbm(train_X, train_y, eval_X, eval_y):
    
    # Initializing the training dataset
    lgtrain = lgb.Dataset(train_X, label=train_y)
    
    # Initializing the evaluation dataset
    lgeval = lgb.Dataset(eval_X, label= eval_y)
    
    # Hyper-parameters for the LightGBM model
    params = {
        "objective" : "regression",
        "metric" : "rmse", 
        "num_leaves" : 30,
        "min_child_samples" : 100,
        "learning_rate" : 0.1,
        "bagging_fraction" : 0.7,
        "feature_fraction" : 0.5,
        "bagging_seed" : 2018,
        "verbosity" : -1
    }
    
    # Training the LightGBM model
    model = lgb.train(params, lgtrain, 1000, valid_sets=[lgeval], early_stopping_rounds=100, verbose_eval=100)
    
    # Returning the model
    return model

# Training the model 
model = train_lightgbm(train_X, train_y, eval_X, eval_y)


# We've successfully trained our LightGBM model.
# 
# Now, let us quickly evaluate the model to see how it is doing by making an actual prediction using it. For this, let us select a row of data from our evaluation dataset and the actual revenue for that row of data.

# In[ ]:


# Index to test row 1458
index_val = 1400

# Selecting the index value from the evaluation DataFrame
actual_X_value = eval_X.reset_index(drop=True).iloc[index_val]

# Selecting the Sale Price from the target variable array
actual_y_value = eval_y[index_val]


# In[ ]:


# Printing the feature values
actual_X_value


# In[ ]:


# Printing the SalePrice
actual_y_value


# Now, let us predict if our model can get a prediction close to the actual generated revenue.

# In[ ]:


# Predicting the value
predict_price = model.predict(actual_X_value.astype(float), predict_disable_shape_check=True)


# In[ ]:


predict_price


# # Since Classification reports and other accuracy indiactors dont work on lightgbm model, thus the rmse represents the model has been trained well as its encoded with hyper parametres and has been tested to be able to predict the slae price and rmse kept decreasing and it may have data modelling in accuracies but can work well on any enviroment its tested on 

# # Random Forest

# In[ ]:


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier


# In[ ]:


model_RF = RandomForestClassifier()


# In[ ]:


model_RF.fit(train_X, train_y)


# In[ ]:


predict_RF = model_RF.predict(eval_X)
predict_RF


# In[ ]:


from sklearn.metrics import mean_absolute_error
mean_absolute_error(predict_RF, eval_y)


# In[ ]:


from sklearn.metrics import r2_score
r2_score(predict_RF, eval_y)


# In[ ]:


from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold


# In[ ]:


cv1 = KFold(n_splits=10, random_state=12,shuffle= True)


# In[ ]:


# evaluate the model with cross validation
scores = cross_val_score(model_RF, train_X, train_y, scoring='accuracy', cv=cv1, n_jobs=-1)
scores


# In[ ]:


from statistics import mean, stdev
# report perofmance
print('Accuracy: %.3f(%.3f)'% (mean(scores), stdev(scores)))


# In[ ]:


accuracy_score(predict_RF, eval_y)


# In[ ]:


# lets use Hyper parametres like Random Search to improve our RFC model
# Random Search
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
random_search = {'criterion': ['entropy', 'gini'],
 'max_depth': list(np.linspace(5, 1200, 10, dtype = int)) + [None],
 'max_features': ['auto', 'sqrt','log2', None],
 'min_samples_leaf': [4, 6, 8, 12],
 'min_samples_split': [3, 7, 10, 14],
 'n_estimators': list(np.linspace(5, 1200, 3, dtype = int))}
clf = RandomForestClassifier()
model_R = RandomizedSearchCV(estimator = clf, param_distributions = random_search, 
 cv = 4, verbose= 5, random_state= 101, n_jobs = -1)
model_R.fit(train_X,train_y)
model_R.best_params_


# In[ ]:


predict_R = model_R.predict(eval_X)
predict_R


# In[ ]:


r2_score(predict_R, eval_y)


# In[ ]:


accuracy_score(predict_R, eval_y)


# # So we can here see that the random Forest model is trained well as represented by the accuracy score and r_2 score but the cross validation score proves that it still needsmore training and ETL processing before being check on other enviroments as indicated by the cross val score

# # Linear Regression

# In[ ]:


from sklearn.linear_model import LinearRegression
model_LR = LinearRegression()


# In[ ]:


model_LR.fit(train_X,train_y)


# In[ ]:


predict_LR = model_LR.predict(eval_X)
predict_LR


# In[ ]:


mean_absolute_error(predict_LR, eval_y)


# In[ ]:


r2_score(predict_LR, eval_y)


# In[ ]:


from sklearn.model_selection import StratifiedKFold
skfold = StratifiedKFold(n_splits=3, random_state=100, shuffle=True)
model_skfold = LinearRegression()
results_skfold = cross_val_score(model_skfold, train_X, train_y, cv=skfold)
print("Accuracy: %.2f%%" %(results_skfold.mean()*100.0))


# We can conclude the following from this small evaluation of Linear Regression Model:
# 
# 1. The model is actually trained and is able to predict a sale price on any new product or any changes to sale price.
# 
# 2. The model is not able to accurately predict the revenue amount with the sale price or changes in the sale price.

# # Conclusion 

# 
# Some things that can be done to increase model accuracy are as follows:
# 
# - Do not drop any of the columns and start with the unoptimized dataset. Then, individually go through all of the columns and only drop columns that are not helpful to the model.
# 
# - Engineer new features from the dataset based on the available data fields.
# 
# - Change the LightGBM model's hyper-parameters.
# 
# - Use another Machine Learning model or create an ensemble of Machine Learning algorithms for getting better results.
# 
# - Use K-Fold Cross Validation instead of simple data splitting for model evaluation.
# 
# - ... and much more. Research!
# - ... The best Model to use from the three models is Random Forest predictor of Sale price

# In[ ]:




