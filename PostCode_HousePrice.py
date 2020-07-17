#!/usr/bin/env python
# coding: utf-8

# Importing all libraries

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# Importing dataset

# In[2]:


#As csv file is zipped, we are using compression='gzip'
pc_hp=pd.read_csv('PostcodeHouseprice.csv.gz',compression='gzip')


# Exploratory Data Analysis

# In[3]:


pc_hp.head()


# In[4]:


pc_hp.describe()


# In[5]:


pc_hp.count()


# In[6]:


sns.distplot(pc_hp['avg(price)'], kde=False, bins=100)

#Just to observe around where mean would come


# Data Pre-Processing for Finding Out if Burglary is Associated with Area of Affluence or Deprivation

# In[7]:


#Finding mean avg(price)
#So, it can be used to classify if postcode is area of affluence or not.
#If average price of a postcode will be above mean price, we will classify it as 'area of affluence'
#If average price of a postcode will be below mean price, we will classify it as 'area of deprivation'

mean_of_avg = pc_hp["avg(price)"].mean()
mean_of_avg


# We are using numeric value '0' for area of affluence, and numeric value '1' for area of deprivation

# In[8]:


pc_hp['area_type'] = np.where(pc_hp['avg(price)'] >= mean_of_avg, 0,1)


# In[9]:


pc_hp


# In[10]:


#Dropping PostCode column now because pd.to_numeric function of Pandas would either 'ignore' it or 
#'coerce' it to 'NaN' - both seems irrelevant

pc_hp_new = pc_hp.drop('PostCode', axis=1)


# In[11]:


pc_hp_new


# In[12]:


sns.countplot(x='area_type',data=pc_hp_new)


# We can see that count of row of burglaries are more in area of deprivation, but its sufficient to finally 
# say that number of burglaries are more in area of deprivation, so we are going to get sum of number of burlagaries
# of each cases in area of deprivation

# In[13]:


sns.barplot(x='area_type',y='number_of_burglaries',data=pc_hp_new, estimator=np.sum)


# Now, we can easily see that, after taking sum of number of burglaries, that indeed yes, number of cases of burglaries are defnitely more in area of deprivation

# Also, we are going to use Logistic Regression to be assured of our result

# In[14]:


#Training model for Logistic Regression, and importing relevant files thereon


# In[15]:


X = pc_hp_new.drop('area_type',axis=1)
y = pc_hp_new['area_type']


# In[16]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# In[17]:


#taking training size as 70% and test 30%

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)


# In[18]:


#creating instance of logistic Regression model and Fitting data

logreg = LogisticRegression()
logreg.fit(X_train,y_train)


# In[19]:


predictions = logreg.predict(X_test)


# In[20]:


#importing classification report and confusion matrix from Scikit-Learn

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# In[21]:


print(classification_report(y_test,predictions))
print('\n')
print(confusion_matrix(y_test,predictions))

#Confusion matrix with (R1,C1) as TN, (R1,C2) as FP, (R2,C1) as FN, and (R2,C2) as TP 


# Thus, it can be easily said that burglary is more closely associated with Area of Deprivation

# Now, for the next task of finding out if the number of cases of bruglaries are increasing or decreasing, we are going to use Linear Regression, and with finding coefficient (positive or negative), we could determine if number of burglaries are increasing or decreasing

# In[22]:


#As now task is to create LinearRegression and time series, we need time as timedate64(ns) and in column
#So, adding day column, and changing year, month, day to single period column
#Then dropping rest of the columns to create a dataframe with only 'Period' as single column 

pc_hp['Day']='01'
pc_hp_3 =pc_hp.drop(['PostCode', 'avg(price)','area_type','number_of_burglaries'], axis=1)
pc_hp_3['Period']=pd.to_datetime(pc_hp_3)
period=pc_hp_3.drop(['Year','Month','Day'], axis=1)
period


# In[23]:


#Concatenating dataframe with 'Period' as single column with main 'pch_price' dataframe

y=pd.concat([pc_hp,period],axis=1)
y.columns


# In[24]:


X= pc_hp[['number_of_burglaries','avg(price)']]


# In[25]:


y=y['Period']


# In[26]:


#Now, applying Linear Regression
from sklearn.linear_model import LinearRegression


# In[27]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)


# In[28]:


linreg=LinearRegression()


# In[29]:


linreg.fit(X_train,y_train)


# In[31]:


linreg.coef_


# In[32]:


X_train.columns


# In[33]:


final_df=pd.DataFrame(linreg.coef_,X.columns,columns=['Coefficient'])


# In[34]:


final_df


# We do not need to fully implement Linear Regression model; with finding coefficient value for 'number_of_burglaries' it can be adjudged if the cases are increasing or decreasing.
# And, with the negative coefficient, we can easily state that there's a decrease in the number of burglaries

# Next, we are going to some data-pre-processing again for Time Series Analysis

# In[35]:


#Again, creating new dataframe with all columns of 'pc_hp'
#and concatenating it with 'period' dataframe with 'Period' as col

new_df=pd.concat([pc_hp,period],axis=1)


# In[36]:


new_df


# In[37]:


#Dropping irrelevant rest of the column
#Grouping all rows with same period and taking sum to come up with 'number of burglaries' in each month
#FINALLY, creating a dataframe 'ts_data' that could be used for time series analysis

z=new_df.drop(['PostCode','Year','Month','Day', 'avg(price)','area_type'], axis=1)
z1=z.groupby('Period')
ts_data=z1.sum()
ts_data


# Importing libraries for Time Series Analysis

# In[38]:


from matplotlib import pyplot
from pandas import Series
from pandas import datetime
from statsmodels.tsa.stattools import adfuller
from matplotlib.pylab import rcParams


# Plotting Times Series graph

# In[39]:


plt.plot(ts_data)
plt.xlabel('Period')
plt.ylabel('No_of_burglaries')
plt.title('Time Series of Burglaries Cases')
plt.rcParams["figure.figsize"] = [12,6]


# In[40]:


#Taking rolling mean, and standard deviation

rolmean = ts_data.rolling(10).mean()
rolstd = ts_data.rolling(10).std()


# In[41]:


#Plotting Rolling Mean and Standard Deviation

plt.rcParams["figure.figsize"] = [12,6]
orig = plt.plot(ts_data, color='blue',label='Original Graph')
mean = plt.plot(rolmean, color='red', label='Rolling Mean')
std = plt.plot(rolstd, color='black', label = 'Standard Dev')
plt.legend(loc='best')
plt.xlabel('Period')
plt.ylabel('No_of_burglaries')
plt.title('Rolling Mean & Standard Deviation')
plt.show(block=False)
ts_data.plot(style='k.')
pyplot.show()


# In[42]:


def adf_test(timeseries):
    #Perform Dickey-Fuller test:
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
       dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)


# In[43]:


#apply adf test on the series
adf_test(ts_data['number_of_burglaries'])


# As Critical Value(1%), Critical Value(5%), Critical Value(10%) is less than Test Statistic. It is a non-stationary series

# In[ ]:




