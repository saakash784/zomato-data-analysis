#!/usr/bin/env python
# coding: utf-8

# # pandas
# 

# In[25]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[26]:


dataframe = pd.read_csv("C:/Users/USER/Zomato data .csv")
print(dataframe.head())
data = pd.DataFrame({
    "online_order": [True, False, True, False, True, False, True, False, True, False]
})


# In[27]:


def handleRate(value):
    value=str(value).split('/')
    value=value[0];
    return float(value)
 
dataframe['rate']=dataframe['rate'].apply(handleRate)
print(dataframe.head())


# In[28]:


dataframe.info()


# In[29]:


sns.countplot(x=dataframe['listed_in(type)'])
plt.xlabel("Type of restaurant")


# In[30]:


grouped_data = dataframe.groupby('listed_in(type)')['votes'].sum()
result = pd.DataFrame({'votes': grouped_data})
plt.plot(result, c="green", marker="o")
plt.xlabel("Type of restaurant", c="black", size=20)
plt.ylabel("Votes", c="black", size=20)


# In[31]:


max_votes = dataframe['votes'].max()
restaurant_with_max_votes = dataframe.loc[dataframe['votes'] == max_votes, 'name']
 
print("Restaurant(s) with the maximum votes:")
print(restaurant_with_max_votes)


# In[32]:


sns.countplot(x=data['online_order'])


# In[33]:


plt.hist(dataframe['rate'],bins=5)
plt.title("Ratings Distribution")
plt.show()


# In[37]:


couple_data=dataframe['approx_cost(for two people)']
sns.countplot(x=couple_data)


# In[35]:


plt.figure(figsize = (6,6))
sns.boxplot(x = 'online_order', y = 'rate', data = dataframe)


# In[36]:




pivot_table = dataframe.pivot_table(index='listed_in(type)', columns='online_order', aggfunc='size', fill_value=0)
sns.heatmap(pivot_table, annot=True, cmap="YlGnBu", fmt='d')
plt.title("Heatmap")
plt.xlabel("Online Order")
plt.ylabel("Listed In (Type)")
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




