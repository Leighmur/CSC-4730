
# coding: utf-8

# 

In[5]:


get_ipython().magic('pylab inline')


# 

In[6]:



# all the imports



import numpy as np
import pandas as pd


pd.set_option('display.height', 1000)

pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)

pd.set_option('display.width', 1000)



import matplotlib.pyplot as plt

import seaborn as sns; sns.set()


import csv
from sklearn.svm 
import SVC
from sklearn.pipeline 
import Pipeline
from sklearn.model_selection 
import GridSearchCV 
from sklearn.datasets 
import load_files
from sklearn.model_selection 
import train_test_split 
from sklearn.feature_extraction.text 
import TfidfVectorizer
from sklearn.metrics 
import accuracy_score, f1_score  
from sklearn.metrics 
import classification_report 


# 

In[7]:




data = pd.read_csv('acs2015_county_data.csv')  

#names the data

#data.info()                                   
#Shows the data


dataHateMap = pd.read_csv('MyHateMap.csv')    
#names the hate map data
#dataHateMap.info()                                #Shows the hate map data


data.head(10)


# 

In[8]:



#This cell will clean the data for each data set


dataHateMap = dataHateMap.dropna() 

#drops all NaN 

data = data.dropna()               

#drops all NaN



#dataHateMap = dataHateMap.drop('City', axis=1)             
#drops the city column


#drops the following columns


data = data.drop(['County','CensusId','Income','IncomeErr','IncomePerCap','IncomePerCapErr','FamilyWork'], axis=1)

data.head(10)

#data.columns


# 

In[9]:



# we're given the numbers in percentages for each county, so

# what we will do is convert those into raw numbers,

# and then group them by state



percentages = ['Hispanic', 'White',
       'Black', 'Native', 'Asian', 'Pacific', 'Citizen', 'Poverty',
       'ChildPoverty', 'Professional', 'Service', 'Office', 'Construction',
       'Production', 'Drive', 'Carpool', 'Transit', 'Walk', 'OtherTransp',
       'WorkAtHome', 'MeanCommute', 'Employed', 'PrivateWork', 'PublicWork',
       'SelfEmployed', 'Unemployment']


for i in percentages:
    
	data[i] = round(data['TotalPop'] * data[i] / 100)
    

data = data.groupby('State', as_index=False).sum()

data.head()


# 

In[10]:



# Now, we will reconvert those into percentages



for i in percentages:
    
	data[i] = round(data[i] / data['TotalPop'] * 100)


data.head()


# 

In[11]:



# Now, we will merge the data and dataHateMap 
dataframes

dataHateMap.head()

merged = data.merge(dataHateMap, on='State')
merged.to_csv("output.csv", index=False)

merged.head()


# 

In[12]:



# Now the merged datacan be sorted through at will



merged.sort_values(['Poverty','Total'], ascending=[False,False]).head(10)


# 

In[13]:



# now we will visualize the data using linear regression. Specifically,

# we will examine how the poverty rate of a state effects the total number

# of hate groups in a state


sns.lmplot(x="Poverty", y="Total", data=merged)
plt.show()


# 

In[14]:




# We can use any number of data visualization tools to examine the data



sns.lmplot(x="Black", y="Neo Confederate", data=merged)
plt.show()



# Here, we see the percent of black people a state has correlates with the

# number of Neo Confederate hate groups a state has

