
# coding: utf-8

# In[14]:


import matplotlib.pyplot as plt
import numpy as np
import math

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

#get_ipython().magic('matplotlib inline')


# In[3]:


def sigmoid(x):
    a = []
    for item in x:
        a.append(1/(1+math.exp(-item)))
    return a


# In[4]:


#x-axis as a 1D array with 100 elements, from -10 to 10 
x = np.arange(-10., 10., 0.2)


# In[5]:


sig = sigmoid(x)


# In[30]:


#Figure
fig = plt.gcf()
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

#Plot 
plt.plot(x, sig)
plt.grid(True)
plt.xlabel(r'x', fontsize=20)
plt.ylabel(r'y', fontsize=20)
plt.title(r"Sigmoid Function", fontsize=20, color='black')

#Show
plt.show()
#Save
fig.savefig('sigmoid_function.eps', format='eps', dpi=1000)

