#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None) 
pd.set_option('display.max_colwidth', -1)


# In[10]:


hateMapData = pd.read_csv('Cleaned_Hate_Map_READY.csv')

hateMapData = hateMapData.sort_values(by='Hate Crimes/100,000', ascending=False)

hateMapData.head()


# In[11]:


hateMapData.corr()


# In[12]:


cleanedData = pd.read_csv('Cleaned_Data_READY.csv')

cleanedData = cleanedData.sort_values(by = 'Hate Crimes/100,000', ascending = False)

cleanedData.head()


# In[13]:


cleanedData.corr()


# In[7]:


plt.matshow(hateMapData.corr())
plt.show()


# In[8]:


plt.matshow(cleanedData.corr())
plt.show()


# In[9]:


#hateMapData = hateMapData.dropna()

#hateMapData.head(50)
cleanedData.head(50)


# In[ ]:





# In[ ]:





# In[ ]:




