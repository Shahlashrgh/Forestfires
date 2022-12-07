#!/usr/bin/env python
# coding: utf-8

# # Predicting Forest Fires Burn Area
# 
# ### Shahla Eshraghi
# 
# 
# ## Introduction
# 
# Forest fires are a common occurance we hear in the news to regions like Australia and California. We thought being able to understand factors that impact the size of fires would be interesting to study considering these natural events happen so close to home. 
# 
# A team at the University of Minho, Portugal collected data on local fires in a Northeastern region of Portugul along with a variety of meterological data. We wanted to understand the following question: "How good is meteorological data at predicting how much area (1 hm2) the fire consumes?
# 
# In order to answer those questions, we created different types of regression models (Linear Regression, Decision Tree model, SVR, and Random Forest) and applied certain tools (gridsearch, Lasso, and Ridge) to improve model performance.
# 

# # Requirements
# 
# The following libraries need to be imported before running the program code. Please note that we surpressed warnings for clean readability of each output. Warning messages shown before supression indicated a change of function name/arguement, but does not interfere with the process. Lastly, the dataset is a csv file downloaded from UCI directory.

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import utils
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor 
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.svm import SVR
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso

warnings.filterwarnings('ignore')

# we renamed the downloaded csv file to "forestfires_100"
df=pd.read_csv('forestfires_100.csv')


# ## Data Visualization
# 
# Preparation work had to be done first to understand the data we are working with to understand patterns and correlations. Also to determine if we need to transform and manipulate the data.

# In[2]:


df.info()


# In[3]:


print(df.shape)


# In[4]:


df["FFMC"] = df["FFMC"].astype(int)
df["DMC"] = df["DMC"].astype(int)
df["DC"] = df["DC"].astype(int)
df["ISI"] = df["ISI"].astype(int)
df["temp"] = df["temp"].astype(int)
df["wind"] = df["wind"].astype(int)
df["rain"] = df["rain"].astype(int)
df["area"] = df["area"].astype(int)


# In order to create a model, we needed to change all the data types to an integer.

# In[5]:


print(df.dtypes)


# In[6]:


#Checking for null values.
df.isnull().sum()


# In[7]:


#Statistical description of our target variable.
print(df['area'].describe())


# In[8]:


#Violin Plot
plt.figure(figsize = (10,5))
ax= sns.violinplot(df['area'])
plt.show()


# We can see that most of the forest fire in our dataset are relatively small. 
# As a lot of area data was equal to 0, we can convert this data to a binary variable, in order know if a fire took place, area equal to 1, or not, area equal to 0.

# In[9]:


dfc=df.copy(deep=True) #We create a copy of the dataset to prevent from chaning the data used for the models
for i in dfc.index:
    if (dfc["area"][i]!=0):
        dfc["area"][i]=1      


# In[10]:


#frequency distribution for fire happening
pd.crosstab(index=dfc['area'], columns='count')


# After converting the target variable to number, we see that in our dataset, 274 (56%) instances resulted in no fire while 232 (44%) instances resulted in a fire.

# In[11]:


#Values description for temperature

print(dfc['temp'].describe())


# In[12]:


#bar diagram for temperature
plt.hist(dfc['temp'], bins=35, range=[0,34])
plt.title('Distribution of Temperature')
plt.xlabel('Temperature')
plt.ylabel('Count')
plt.show()


# In[13]:


dfc["temp"] = dfc["temp"].astype(int)
ax = sns.countplot('temp',hue='area', data=dfc)
ax.set_title('Temperature if a fire is happening or not') #if area = 1, a fire took place
ax.set_xlabel('temp')
ax.set_ylabel('count')
plt.show()


# In[14]:


#numerical summary for temperature
dfc.groupby('area')['temp'].describe()


# We can therefore see that Temperature does not have a big influence on fire happening, since there are more fire happening for low temperatures than high temperatures.

# 

# In[15]:


#frequency distribution for Relative Humidity
pd.crosstab(index=dfc['RH'], columns='count')


# In[16]:


#Values description for RH

print(dfc['RH'].describe())


# In[17]:


#bar diagram for RH
plt.hist(dfc['RH'], bins=100, range=[10,100])
plt.title('Distribution of Relative Humidity')
plt.xlabel('Relative Humidity')
plt.ylabel('Count')
plt.show()


# Relative humidity was quite low most of the time.

# In[18]:


dfc["RH"] = dfc["RH"].astype(int)
ax = sns.countplot('RH',hue='area', data=dfc)
ax.set_title('Relative Humidity if a fire is happening or not') #if area = 1, a fire took place
ax.set_xlabel('Relative Humidity')
ax.set_ylabel('count')
plt.show()


# More forestfires are happening when the relative humidity is low.

# In[19]:


#frequency distribution for Wind
pd.crosstab(index=dfc['wind'], columns='count')


# In[20]:


#Values distribution for Wind

print(dfc['wind'].describe())


# In[21]:


#bar diagram for Wind
plt.hist(dfc['wind'], bins=10, range=[0,10])
plt.title('Distribution of Wind')
plt.xlabel('Wind')
plt.ylabel('Count')
plt.show()


# In[22]:


dfc["wind"] = dfc["wind"].astype(int)
ax = sns.countplot('wind',hue='area', data=dfc)
ax.set_title('Wind if a fire is happening or not') #if area = 1, a fire took place
ax.set_xlabel('Wind')
ax.set_ylabel('count')
plt.show()


# There are more fire happening when the wind is high.

# In[23]:


#frequency distribution for rain
pd.crosstab(index=dfc['rain'], columns='count')


# In[24]:


#Values description for Rain

print(dfc['rain'].describe())


# In[25]:


#bar diagram for Rain
plt.hist(dfc['rain'], bins=10, range=[0,7])
plt.title('Distribution of Rain')
plt.xlabel('Rain')
plt.ylabel('Count')
plt.show()


# There has been no rain for most of the time.

# In[26]:


dfc["rain"] = dfc["rain"].astype(int)
ax = sns.countplot('rain',hue='area', data=dfc)
ax.set_title('Rain if a fire is happening or not') #if area = 1, a fire took place
ax.set_xlabel('Rain')
ax.set_ylabel('count')
plt.show()


# Rain equal 0 wether there has been a fire or not, but most of the time, there were no fire when rain didn't occur. Moreover, there has been fires when rain occured. We can then conclude rain does not really have an influence on fire appearance.

# ## Correlation plot

# In[27]:


#Correlogram
sns.pairplot(df)
plt.show()


# In[28]:


plt.figure(figsize=(12,10))
sns.heatmap(df.corr(),annot=True,fmt='.2f')


# Looking at the correlation plot, we see that no features correlate to our target variable "area"

# # Dataset Preparation
# 
# After understanding our data, we are ready to start building our models. 
# 
# We dropped two categorical features "X" and "Y". These spatial data points had no bearing on the target variable and showed location of each fire, which is out of scope for this project. In addition we dropped "month" and "day", as these features complicated the model and was out of scope for the project. 
# 
# The dataset contained no null values. Lastly, we changed all features that remained to integer datatype from float64 to allow our models to run properly.
# 
# The new dataframe created was assigned to df_new and used for X and Y variables. We eventually applied feature removal and had to created a new dataset called df_features. This dataframe was used for X1 and Y1 variables. 
# 
# In this workbook, SVR, Decision Tree, and Random Forest models are using df_features for the best performance possible.

# In[29]:


#Feature removal
#4,5,7 ISI, temp, wind
df_new = df.drop(["X","Y","month","day"], axis=1).astype(np.int)
df_features = df.drop(["X","Y","RH","month","day","FFMC","DMC","DC"], axis=1).astype(np.int)

x1 = df_features.drop(["area"], axis=1)
y1 = df_features[["area"]]

print("df_features: ",df_features.columns)
print("x1: ",x1.columns)
print("y1: ",y1.columns) 

x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, test_size=0.2, random_state=4)


# # Linear Regression
# 
# For our first model, we createa a linear regression using "wind" feature on our target variable. Looking at the model performance (MSE and r2), we see this model is not a good fit.

# In[30]:


# creating an instance of LinearRegression class
x = df['wind']
y = df["area"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)

reg = linear_model.LinearRegression()
print(reg)


# In[31]:


print(y_train.shape) 
y_train = y_train[:,None] # y_train = y_train.reshape(-1, 1)
print(y_train.shape)
x_train = x_train[:,None]
print(x_train.shape)
x_test = x_test[:,None]
y_test = y_test[:,None]


# In[32]:


print(type(x_train))
print(type(y_train))
print(type(x_test))
print(type(y_test))


# In[33]:


# y = mx + b here m is the coefficient (or slope) of x and b is the intercept
reg.fit(x_train, y_train)
print(reg.coef_) # 
print(reg.intercept_)


# In[34]:


# area = 0.21390022 wind + 2.14145839


# In[35]:


yhat = reg.predict(x_test)


# In[36]:


print(yhat[0])


# In[37]:


mse_test = mean_squared_error(y_test, yhat)
print(mse_test)


# In[38]:


# y_t_predict is the predicted y values for the x_train data
y_t_predict = reg.predict(x_train)

# note that y_train is the true y value
mse_train = mean_squared_error(y_train, y_t_predict)
mse_test = mean_squared_error(y_test, yhat)
print("mse_train :",mse_train)
print("mse_test :",mse_test)
print("r2_train :", r2_score(y_train, y_t_predict))
print("r2_test: ", r2_score(y_test, yhat))


# In[39]:


r1= mse_test/mse_train
diff1=np.abs(mse_train - mse_test)

print("r1: ", r1)
print("diff1: ",diff1)


# # Lasso Regression
# 
# For our second model, we ran a Lasso Regression to minimize the impact of unimportant features. After looking at the performance, this model was not a good fit.

# In[40]:


x = df_new.drop("area", axis=1)
y = df_new[["area"]]
print(x.columns)


# In[41]:


lassoReg = Lasso(alpha=0.05 ,  normalize=False)

lassoReg.fit(x1_train,y1_train)

y_test_pred = lassoReg.predict(x1_test)
y_train_pred = lassoReg.predict(x1_train)


mse_test = mean_squared_error(y1_test, y_test_pred)

mse_train = mean_squared_error(y1_train, y_train_pred)

print("mse_train: ", mse_train)
print("mse_test: ", mse_test)

r2_lasso_train = r2_score(y1_train, y_train_pred)
r2_lasso_test = r2_score(y1_test, y_test_pred)

print("r2_lasso_train: ", r2_lasso_train)
print("r2_lasso_test: ", r2_lasso_test)


# # Ridge Regression
# 
# For our third model, we ran a Ridge Regression to minimize the impact of unimportant features. After looking at the performance, this model was not a good fit.

# In[42]:


print("LassoReg.coef: ",lassoReg.coef_)
print("LassoReg.intercept: ",lassoReg.intercept_)


# In[43]:


ridgeReg = Ridge(alpha=3, normalize=False)

ridgeReg.fit(x1_train,y1_train)

y_train_pred = ridgeReg.predict(x1_train)
y_test_pred = ridgeReg.predict(x1_test)

mse_test = mean_squared_error(y1_test, y_test_pred)
mse_train = mean_squared_error(y1_train, y_train_pred)

r2_ridge_train = r2_score(y1_train, y_train_pred)
r2_ridge_test = r2_score(y1_test, y_test_pred)

print("mse_train: ", mse_train)
print("mse_test: ", mse_test)

print("r2_ridge_train: ", r2_ridge_train)
print("r2_ridge_test: ", r2_ridge_test)


# # Feature Removal
# 
# We used the coefficients from lasso and ridge regressions to determine which features should be removed from the dataset. We concluded that the new dataset should only contain 'ISI', 'temp', 'wind', 'rain' based on highest coefficient values and removing values equal to 0.
# 
# See Data Preparation section above for the new dataset variables referenced.

# In[44]:


print("ridgeReg.coef: ",ridgeReg.coef_)
print("ridgeReg.intercept: ",ridgeReg.intercept_)


# In[45]:


print("LassoReg.coef: ",lassoReg.coef_)
print("LassoReg.intercept: ",lassoReg.intercept_)


# # Hyperparameters

# In[46]:


lassoTestlow = Lasso(alpha=0.1, normalize = False)
lassoTestmed = Lasso(alpha=0.15, normalize = False)
lassoTesthigh = Lasso(alpha=0.2, normalize = False)

lassoTestlow.fit(x_train,y_train)
lassoTestmed.fit(x_train,y_train)
lassoTesthigh.fit(x_train,y_train)

x_Lasso = [0.1, 0.15, 0.2]
y_Lasso = [lassoTestlow.coef_, lassoTestmed.coef_, lassoTesthigh.coef_]

print("++++++++++ Lasso ++++++++++")

print("for alpha=0.1: ", y_Lasso[0])
print("for alpha=0.15: ",y_Lasso[1])
print("for alpha=0.2: ", y_Lasso[2])



ridgeTestlow = Ridge(alpha=0.1, normalize = False)
ridgeTestmed = Ridge(alpha=0.15, normalize = False)
ridgeTesthigh = Ridge(alpha=0.2, normalize = False)

ridgeTestlow.fit(x_train,y_train)
ridgeTestmed.fit(x_train,y_train)
ridgeTesthigh.fit(x_train,y_train)

x_Ridge = [0.1, 0.15, 0.2]
y_Ridge = [ridgeTestlow.coef_, ridgeTestmed.coef_, ridgeTesthigh.coef_]

print("\n++++++++++ Ridge ++++++++++")
print("for alpha=0.1: ", y_Ridge[0])
print("for alpha=0.15: ",y_Ridge[1])
print("for alpha=0.2: ", y_Ridge[2])


# # Decision Tree
# 
# Now using the new dataset with removed features, we ran the decision tree as our fourth model. We applied gridsearchCV to determine the optimal # of trees and random_state for the best performance. After considering the MSE and r2 values on the test/train data, we found a model with strong performance.
# 
# mse_train: 31.65
# mse_test: 305.22
# r2_train: 0.83
# r2_test: -0.91

# In[47]:


'''
model = DecisionTreeRegressor()

max_depth = [3,4,5,20]
random_state= [0,1,2,3,4]

grid = dict(max_depth=max_depth,random_state=random_state)
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=3)
grid_result = grid_search.fit(x1_train, np.ravel(y1_train))

print("\nGridSearhCV best model:\n ")
print('The best score: ', grid_result.best_score_)
print('The best parameters:', grid_result.best_params_)
'''


# Using gridsearchCV, the output states the best parameters to use is max_depth = 3 and random_state = 0. However for the sake of validation, we tried different numbers for these parameters and determined that max_depth = 20 produces the best performance for the model.

# In[48]:


tree = DecisionTreeRegressor(max_depth=20 , random_state = 0)
cross_val_score(tree, x1, y1, cv=10)
tree.fit(x1_train, y1_train)

y_train_pred = tree.predict(x1_train)
y_test_pred = tree.predict(x1_test)

mse_test = mean_squared_error(y1_test, y_test_pred)
mse_train = mean_squared_error(y1_train, y_train_pred)

print("mse_train: ", mse_train)
print("mse_test: ", mse_test)

r2_train = r2_score(y1_train, y_train_pred)
r2_test = r2_score(y1_test,y_test_pred) # returns the r-squared value

print("r2_train: ", r2_train)
print("r2_test: ", r2_test)


# In[49]:


r1 = mse_test/mse_train
diff1 = np.abs(mse_train - mse_test)
print("r1: ",r1)
print("diff1: ",diff1)


# # Random Forest
# 
# We wanted to see if Random Forest Regression would yield similar results to the decision tree. As our fifth model, we ran this model and optimized for the # of trees and random_state. This model also yielded good performance for the dataset.
# 
# mse_train: 62.76
# mse_test: 63.52
# r2_train: 0.67
# r2_test: 0.60

# In[50]:


#x, y = make_regression(n_features=8, n_informative=2,random_state=0, shuffle=False)
rfc = RandomForestRegressor(n_estimators = 100, max_depth=15, random_state=2)
rfc.fit(x1, y1)


# In[51]:


y_hat = rfc.predict(x1_train)
y_pred = rfc.predict(x1_test)

mse_test = mean_squared_error(y1_test, y_pred)
mse_train = mean_squared_error(y1_train, y_hat)
print("mse_train: ", mse_train)
print("mse_test: ", mse_test)

r2_train = r2_score(y1_train, y_hat)
r2_test = r2_score(y1_test,y_pred) 
print("r2_train: ", r2_train)
print("r2_test: ", r2_test)


# In[52]:


r1= mse_test/mse_train
diff1=np.abs(mse_train - mse_test)

print("r1: ",r1)
print("diff1: ",diff1)


# In[53]:


xp = df_new.iloc[:, 4:5].values #as shown with the print, selects the 4th/temperature column
yp = df_new.iloc[:, 8].values #as shown with the print, selects the 8th/area column

# fit the regressor with x and y data 
rfc.fit(xp, yp)

Y_pred = rfc.predict(np.array([6.5]).reshape(1, 1))

# Visualising the Random Forest Regression results

# arange for creating a range of values from min value of x to max value of x 
#with a difference of 0.01 between two consecutive values 
X_grid = np.arange(min(xp), max(xp), 0.01)

# reshape for reshaping the data into a len(X_grid)*1 array, i.e. to make a column out of the X_grid value 
X_grid = X_grid.reshape((len(X_grid), 1))

# Scatter plot for original data 
plt.scatter(xp, yp, color = 'blue')


# plot predicted data 
plt.plot(X_grid, rfc.predict(X_grid), 
 color = 'green') 
plt.title('Random Forest Regression with temperature considering area') 
plt.xlabel('temp') 
plt.ylabel('Area') 
plt.show()


# We graphed the Random Forest model to see how well it predicted the data points.

# ## SVR using GridsearchCV

# We decided to run SVR model as well since we had strong model performance for Decision Tree and Random Forest. However it did not yield good results.
# 
# Evaluate the current algorithm and variety of algorithms by creating test harness for diverse algorithms in conjunction with resampling techniques like cross validation, variable importance and implementing Gridsearch.Improve Result by playing with hyperparameters and innovative methods like ensembles.

# In[54]:


parameters = {'kernel':['linear', 'rbf', 'poly'], 'C':[1, 5, 10]}

s = SVR()

clf = GridSearchCV(s, param_grid = parameters, cv = 3, verbose=True, n_jobs=-1)

final_clf = clf.fit(x1_train, y1_train)


#print(sorted(final_clf.cv_results_.keys()))
print("Best parameters: ",final_clf.best_estimator_)
print(sorted(final_clf.cv_results_.keys()))


# In[55]:


#Create SVR
svregressor = SVR(kernel='linear', C=10) 

#Train the model
svregressor.fit(x1_train, y1_train) 

yhat = svregressor.predict(x1_test)
y_pred = svregressor.predict(x1_train)


# In[56]:


yhat = svregressor.predict(x1_test)
mse_test = mean_squared_error(y1_test, yhat)
y_t_predict = svregressor.predict(x1_train)
yhat = svregressor.predict(x1_test)

mse_train = mean_squared_error(y1_train, y_pred)
print("mse_train: ",mse_train)
print("mse_test: ",mse_test)

print("r-squared for test data: ", r2_score(y1_test, yhat))
print("r-squared for train data: ", r2_score(y1_train, y_t_predict))


# In[57]:


r1 = mse_test/mse_train

diff1 = np.abs(mse_train - mse_test)

print("r1: ",r1)

print("diff1: ",diff1)


# # Conclusion
# The best performing one was Random Forest Regressor. While Decision tree also yielded strong performance, we think Random Forest is better for the following reasons:
# 
# 1. Random forest shows a moderately strong r2 value for both test train datasets. 
# 
# 2. MSE for test and train is significantly lower than Decision Tree.
# 
# 3. Random forest has high consistency for MSE and r2 while decision tree tends to fluctuate. We think a consistent,   moderately strong r2 value is better option over highest r2 value.
# 

# # References
# 
# - dataset : https://archive.ics.uci.edu/ml/datasets/Forest+Fires
# 
# 
# - Random Forest plot: https://www.geeksforgeeks.org/random-forest-regression-in-python/
# 
# 
# - More refrences: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
#   
# https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html?highlight=decision%20tree%20regressor#sklearn.tree.DecisionTreeRegressor
# 
# 
# 
