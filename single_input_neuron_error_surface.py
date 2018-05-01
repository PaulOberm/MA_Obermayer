
# coding: utf-8

# In[251]:


'''
    Explanation: The error surface of a single input neuron by applying either a linear or sigmoid activation function.
    ------------
'''

import matplotlib.pyplot as plt
import numpy as np
import math

from scipy import ndimage
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits import mplot3d

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

#get_ipython().magic('matplotlib inline')


# In[252]:


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)


# In[34]:


def objective_function(neuron_output, train_output): 
    E = 0.5 * (neuron_output - train_output)**2
    return E


# In[240]:


def single_input_neuron_function(bias, weight, input_value=0.7, activation = 'linear'): 
    #Intermediate state: z
    z = input_value * weight + bias
    
    #Activation function
    if activation == 'linear':
        y = linear_activation(z)
    elif activation == 'sigmoid':
        y = sigmoid(z)
    else: 
        print('Activation function not implemented!')
        exit()
    return y


# In[234]:


def sigmoid(z):
    y = (1/(1+math.exp(-z)))
    return y


# In[233]:


def linear_activation(z, factor = 0.5):
    y = factor * z
    return y


# In[241]:


#1st dimension: bias
bias = np.arange(-40., 40., 2)
#2nd dimension: weight
weight = np.arange(-40., 40., 2)

#2D arrays
X, Y = np.meshgrid(bias, weight)
print('Bias matrix: {}'.format(X.shape))
print('Weight matrix: {}'.format(Y.shape))


# In[242]:


#3rd dimension: error
error = np.zeros((bias.shape[0], weight.shape[0]))

for b_idx, b in enumerate(bias): 
    for w_idx, w in enumerate(weight): 
        error[b_idx, w_idx] = objective_function(single_input_neuron_function(b, w), train_output = 0.3)
    
print('Error matrix: {}'.format(error.shape))


# In[294]:


#Figure
fig = plt.figure()
fig.set_size_inches(18.5, 10.5)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.grid(True)

#Plot
ax = fig.gca(projection='3d')

# Plot the surface.
surf = ax.plot_surface(X, Y, error, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

#Labels
ax.set_xlabel(r'Bias $\theta$', fontsize=20)
ax.set_ylabel(r'Weight $w$', fontsize=20)
ax.set_zlabel(r'Error $E$', fontsize=20)
#Rotation
ax.view_init(60, 20)
plt.title(r"Single input neuron error surface", fontsize=20, color='black')

#Drawing arrows
a = Arrow3D([X[0,2], X[3,0]], [Y[0,2], Y[3,0]], [error[0,2], error[3,0]], mutation_scale=20, arrowstyle='-|>', color='g')
ax.add_artist(a)
a = Arrow3D([X[3,0], X[2,6]], [Y[3,0], Y[2,6]], [error[3,0], error[2,6]], mutation_scale=20, arrowstyle='-|>', color='g')
ax.add_artist(a)
a = Arrow3D([X[2,6], X[5,0]], [Y[2,6], Y[5,0]], [error[2,6], error[5,0]], mutation_scale=20, arrowstyle='-|>', color='g')
ax.add_artist(a)
a = Arrow3D([X[5,0], X[3,8]], [Y[5,0], Y[3,8]], [error[5,0], error[3,8]], mutation_scale=20, arrowstyle='-|>', color='g')
ax.add_artist(a)
a = Arrow3D([X[3,8], X[7,2]], [Y[3,8], Y[7,2]], [error[3,8], error[7,2]], mutation_scale=20, arrowstyle='-|>', color='g')
ax.add_artist(a)
a = Arrow3D([X[7,2], X[5,12]], [Y[7,2], Y[5,12]], [error[7,2], error[5,12]], mutation_scale=20, arrowstyle='-|>', color='g')
ax.add_artist(a)
a = Arrow3D([X[5,12], X[10,6]], [Y[5,12], Y[10,6]], [error[5,12], error[10,6]], mutation_scale=20, arrowstyle='-|>', color='g')
ax.add_artist(a)

a = Arrow3D([X[35,0], X[5,39]], [Y[35,0], Y[5,39]], [error[35,0], error[5,39]], mutation_scale=20, color='r')
ax.add_artist(a)

#Plot
plt.show()

#Save
fig.savefig('single_input_neuron_error_surface.eps', format='eps', dpi=1000)

