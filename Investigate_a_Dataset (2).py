#!/usr/bin/env python
# coding: utf-8

# >
# 
# # Project: Investigate a Dataset - [No_show appointments ]
# 
# ## Table of Contents
# <ul>
# <li><a href="#intro">Introduction</a></li>
# <li><a href="#wrangling">Data Wrangling</a></li>
# <li><a href="#eda">Exploratory Data Analysis</a></li>
# <li><a href="#conclusions">Conclusions</a></li>
# </ul>

# <a id='intro'></a>
# ## Introduction
# 
# ### Dataset Description 
# 
# > **** This dataset collects information from 100k medical appointments in Brazil and is focused on the question of whether or not patients show up for their appointment. A number of characteristics about the patient are included in each row \n
# 
# **columns:
# 
# *‘ScheduledDay’: tells us on what day the patient set up their appointment.
# 
# *‘Neighborhood’:indicates the location of the hospital.
# 
# *‘Scholarship’ :indicates whether or not the patient is enrolled in Brasilian welfare program Bolsa Família.
# 
# ****Be careful about the encoding of the last column: it says ‘No’ if the patient showed up to their appointment, and ‘Yes’ if they did not show up.
# 
# ### Question(s) for Analysis
# **What factors are important for us to know in order to predict if a patient will show up for their scheduled appointment?
# 
# > **Tip**: Once you start coding, use NumPy arrays, Pandas Series, and DataFrames where appropriate rather than Python lists and dictionaries. Also, **use good coding practices**, such as, define and use functions to avoid repetitive code. Use appropriate comments within the code cells, explanation in the mark-down cells, and meaningful variable names. 

# In[1]:


# Use this cell to set up import statements for all of the packages that you
#   plan to use.
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns 
# Remember to include a 'magic word' so that your visualizations are plotted
#   inline with the notebook. See this page for more:
#   http://ipython.readthedocs.io/en/stable/interactive/magics.html
# Upgrade pandas to use dataframe.explode() function. 


# In[ ]:


# Upgrade pandas to use dataframe.explode() function. 
get_ipython().system('pip install --upgrade pandas==0.25.0')


# <a id='wrangling'></a>
# ## Data Wrangling
# 
# > 

# In[2]:


# Load your data and print out a few lines. Perform operations to inspect data
#   types and look for instances of missing or possibly errant data.
df=pd.read_csv('no_show.csv')
df.head(3)


# In[3]:


#the Raws and columns of the data
df.shape


# In[4]:


#check the duplicated raws of the data
df.duplicated().sum() 


# No duplicted raws

# In[5]:


#check the data info
df.info()


# No missing values 

# In[6]:


#describe our data 
df.describe()


# *The average of ages is 37 years
# 
# *The min of age is mistake becauese it is -1 
# 
# *The max of age is 115 years

# 
# ### Data Cleaning
# 
#  

# In[7]:


# After discussing the structure of the data and any problems that need to be
#   cleaned, perform those cleaning steps in the second part of this section.
# we do not need the information like : PatientId ,AppointmentID,ScheduledDay,AppointmentDay
df.drop(columns=['PatientId','AppointmentID','ScheduledDay','AppointmentDay'],inplace=True)
df.head()


# In[8]:


#Rename a column to fcilitate the analysis 
df.rename(columns={'No-show':'No_show'},inplace=True)
df.head(3)


# In[9]:


# The minimum of age does not make sense 
#Find this patient 
df[df['Age']<0]


# In[10]:


#drop this patient 
df.drop(99832 , inplace= True)
df[df['Age']<0]


# Now our data is cleaned 

# <a id='eda'></a>
# ## Exploratory Data Analysis
# 
# 
# ### Analysing all the characteristics of our data 

# In[11]:


#Analysing our data by the hist plot
df.hist(figsize=(10,10));


# -Most of the patients are recieved the SMS about double those who did not receive
# 
# -the patient is enrolled in 'Brasilian welfare program Bolsa Família' about 10 % 
# 
# -Most of the patients does not suffer from (Alcoholism ,Diabetes,Handcap)
# 
# -About 20000 of the patient suffers from the Hipertension

# In[12]:


#Numbers of the patient who showed 
df_show=df[df['No_show']=='No']
df_show.count()


# In[13]:


#Numbers of the patient who did not show 
df_noshow=df[df['No_show']=='Yes']
df_noshow.count()


# The number of the patients who did not show is about 25 % 

# In[14]:


# Comparsion according to the gender
df.groupby(['Gender','No_show']).No_show.count().plot(kind='pie',figsize=(10,10))
df.groupby(['Gender','No_show']).No_show.count ;


# In[39]:


# Comparsion according to the gender by another way to get the exactly numbers 
df.groupby('Gender')['No_show'].value_counts()


# The percentage of Females who did not attend is close to that of Males 
# so the Gender is insignificant

# In[32]:


#Making a function to help us in the analysis
def plot (xVar): 
    """
    "This function Make graphics easier"
    the input is the name of the column 
    the function will return the hist plot
    """
    df_show[xVar].hist(label='Show',color='blue')
    df_noshow[xVar].hist(label='No show',color='red')
    plt.xlabel(xVar,color='red')
    plt.ylabel('The number of people',color='red')
    plt.title('Comparsion according to the {}'.format(xVar),color='red')
    plt.legend;


# In[33]:


# Comparsion according to the Age
plot('Age')


# -Most of patients who show are in the age  0 to 10 and about from 35 to 70

# In[16]:


# More details about the patient who show according to the age 
df_show.Age.value_counts()


# In[19]:


# More details about the patient who no_show according to the age 
df_noshow.Age.value_counts()


# In[21]:


# Comparsion according to the Neighbourhood
df_show.Neighbourhood.value_counts().plot(kind='bar',label='Show',color='blue',figsize=(9,9))
df_noshow.Neighbourhood.value_counts().plot(kind='bar',label='No show',color='red',figsize=(9,9))
plt.xlabel('Neighbourhood',color='red')
plt.ylabel('The number of people',color='red')
plt.title('Comparsion according to the Neighbourhood',color='red')
plt.legend;


# In[22]:


#find which Neighbourhood has the most show
df_show.Neighbourhood.value_counts()


# The Neighbourhood is a strong significant 
# 
# -' JARDIM CAMBURI' has the maximum number of the show
# 
# -' PARQUE INDUSTRIAL'has the minimum number of the show
# 

# In[34]:


# Comparsion according to the Hipertension
plot('Hipertension')


# Hipertension is insignificant

# In[36]:


# Comparsion according to the SMS_received
plot('SMS_received')


# The Most of patients who showed did not recieve SMS , it is strange 

# In[37]:


# Comparsion according to the Diabetes
plot('Diabetes')


# Diabetes is insignificant

# In[38]:


# Comparsion according to the Handcap
plot('Handcap')


# Handcap is insignificant

# <a id='conclusions'></a>
# ## Conclusions
# 
# > *At the end i could say that the Age is an important sign , the most of the show is between 0 to 10 & 35 to 70 
# 
# > *The Neighbourhood is a strong significant 
# 
#            - JARDIM CAMBURI: has the maximum number of the show
# 
#            - PARQUE INDUSTRIAL: has the minimum number of the show
# 
# > *I could not take the SMS as a sign because it is strange as The Most of patients who showed did not recieve SMS
# 
# > *Other characteristics is insignificant
# 
#      ##limitations :1) I think that we may need weather conditions, as they may have an effect on patients not attending if they are bad
# 
#                      2)We must know the social conditions of patients, as they certainly have an impact on their neglect of their health
#      
#   #Notes: This analysis is only my opinion
# 
# ## Submitting your Project 
# 

# In[ ]:


from subprocess import call
call(['python', '-m', 'nbconvert', 'Investigate_a_Dataset.ipynb'])

