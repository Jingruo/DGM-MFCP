# -*- coding: utf-8 -*-
"""DGM_MFC.py

We construct a deep neural network to train the algorithm called Deep Galerkin Method (DGM).

The DGM is used to solve high dimensional PDE coming from the Hamilton-Jacobi-Bellman (HJB) equation
in Mean Field Control Problem (MFCP).

"""

import tensorflow as tf

"""# Model"""

# LSTM-like layer used in DGM - modification of keras Layer class
class LSTMLayer(tf.keras.layers.Layer):
    
    # constructor/initializer function (automatically called when new instance of class is created) 
    def __init__(self, units, trans = "tanh"):
        '''
        Args: 
            units (int): number of units in each layer 
            trans (str): nonlinear activation function
                         one of: "tanh" (default), "relu", or "sigmoid"
        Returns: 
            customized keras Layer object used as intermediate layers in DGM
        
        '''
        
        # create an instance of a keras Layer object (call initialize function of superclass of LSTMLayer)
        super().__init__() 
        self.units = units
        
        if trans == "tanh":
            self.trans = tf.nn.tanh
        elif trans == "relu":
            self.trans = tf.nn.relu
        elif trans == "sigmoid":
            self.trans = tf.nn.sigmoid
        
    # define LSTMLayer parameters
    def build(self,input_shape):
        
        # U matrix (weighting for original inputs X)
        self.Uz = self.add_weight(shape = (input_shape[-1], self.units),
                                  initializer = 'random_normal',
                                  trainable = True)
        self.Ug = self.add_weight(shape = (input_shape[-1], self.units),
                                  initializer = 'random_normal',
                                  trainable = True)
        self.Ur = self.add_weight(shape = (input_shape[-1], self.units),
                                  initializer = 'random_normal',
                                  trainable = True)
        self.Uh = self.add_weight(shape = (input_shape[-1], self.units),
                                  initializer = 'random_normal',
                                  trainable = True)
        
        # super().build(input_shape)   
        
        # W matrix (weighting for outputs from previous layer)
        self.Wz = self.add_weight(shape = (self.units, self.units),
                                  initializer = 'random_normal',
                                  trainable = True)
        self.Wg = self.add_weight(shape = (self.units, self.units),
                                  initializer = 'random_normal',
                                  trainable = True)
        self.Wr = self.add_weight(shape = (self.units, self.units),
                                  initializer = 'random_normal',
                                  trainable = True)
        self.Wh = self.add_weight(shape = (self.units, self.units),
                                  initializer = 'random_normal',
                                  trainable = True)
        
        # bias vector
        self.bz = self.add_weight(shape = (self.units,),
                                  initializer = 'random_normal',
                                  trainable = True)
        self.bg = self.add_weight(shape = (self.units,),
                                  initializer = 'random_normal',
                                  trainable = True)
        self.br = self.add_weight(shape = (self.units,),
                                  initializer = 'random_normal',
                                  trainable = True)
        self.bh = self.add_weight(shape = (self.units,),
                                  initializer = 'random_normal',
                                  trainable = True)
        
    # main function to be called 
    def call(self,X,S):
        '''Compute output of a LSTMLayer for given inputs X and S.    
        Args:            
            X: data input
            S: output of previous layer
        
        Returns: 
            S_new: input to next LSTMLayer
        '''  
        
        # compute components of LSTMLayer output
        Z = self.trans(tf.add(tf.add(tf.matmul(X,self.Uz), tf.matmul(S,self.Wz)), self.bz))
        G = self.trans(tf.add(tf.add(tf.matmul(X,self.Ug), tf.matmul(S,self.Wg)), self.bg))
        R = self.trans(tf.add(tf.add(tf.matmul(X,self.Ur), tf.matmul(S,self.Wr)), self.br))
        H = self.trans(tf.add(tf.add(tf.matmul(X,self.Uh), tf.matmul(tf.multiply(S,R), self.Wh)), self.bh))
        
        # compute LSTMLayer outputs
        S_new = tf.add(tf.multiply(tf.subtract(tf.ones_like(G), G), H), tf.multiply(Z,S))
        return S_new

# Fully connected (dense) layer - modification of keras Layer class
class DenseLayer(tf.keras.layers.Layer):
    
    # constructor/initializer function (automatically called when new instance of class is created)
    def __init__(self, units, trans = None):
        '''
        Args:
            units (int): number of units in each layer 
            trans (str): nonlinear activation function
                         one of: "tanh", "relu", "sigmoid", or None (default)
                         None means identity map 
        Returns: 
            customized keras (fully connected) Layer object
        '''        
        
        # create an instance of a keras Layer object (call initialize function of superclass of DenseLayer)
        super().__init__()
        self.units = units
        
        if trans:
            if trans == "tanh":
                self.trans = tf.tanh
            elif trans == "relu":
                self.trans = tf.nn.relu
            elif trans == "sigmoid":
                self.trans = tf.nn.sigmoid
        else:
            self.trans = trans
        
    # define DenseLayer parameters
    def build(self,input_shape):
        
        # W matrix (weighting for outputs from previous layer)
        self.W = self.add_weight(shape = (input_shape[-1], self.units),
                                  initializer = 'random_normal',
                                  trainable = True)
        
        # bias vector
        self.b = self.add_weight(shape = (self.units,),
                                  initializer = 'random_normal',
                                  trainable = True)
    
    
    # main function to be called 
    def call(self,X):
        '''Compute output of a DenseLayer for a given input X.
        Args:                        
            X: input to layer
        Returns:
            S: input to next layer
        '''
        
        # compute DenseLayer output
        S = tf.add(tf.matmul(X, self.W), self.b)
        
        if self.trans: 
            S = self.trans(S)
        return S

# Neural network architecture used in DGM - modification of keras Model class
class DGMNet(tf.keras.Model):
    
    # constructor/initializer function (automatically called when new instance of class is created)
    def __init__(self, units, n_layers, final_trans = None):
        '''
        Args:
            units (int):       number of units in each layer
            n_layers (int):    number of intermediate LSTM layers
            final_trans (str): nonlinear activation function used in final layer
                               one of: "tanh" (default), "relu", or "sigmoid"
        Returns: 
            customized keras Model object representing DGM neural network
        '''  
        
        # create an instance of a keras Model object (call initialize function of superclass of DGMNet)
        super().__init__()
        
        # define initial layer as fully connected
        self.initial_layer = DenseLayer(units, trans = "tanh")
        
        # define intermediate LSTM layers
        self.n_layers = n_layers
        self.LSTMLayerList = []
        
        for _ in range(self.n_layers):
            self.LSTMLayerList.append(LSTMLayer(units,trans = "tanh"))
            
        # define final layer as fully connected with a single output (function value)
        self.final_layer = DenseLayer(1, trans = final_trans)
    
    # main function to be called  
    def call(self,x,t):
        '''Run the DGM model and obtain fitted function value at the inputs (t,x).
        Args:
            x: sampled space inputs
            t: sampled time inputs
        Returns:
            result: fitted function value
        '''  
        
        # define initial inputs as time-space pairs
        X = tf.concat([x,t],1)
        
        # call initial layer
        initial = self.initial_layer
        S = initial(X)
        
        # call intermediate LSTM layers
        for i in range(self.n_layers):
            LSTM = self.LSTMLayerList[i]
            S = LSTM(X,S)
        
        # call final layer
        final = self.final_layer
        result = final(S)
        return result

"""# Mean Field Control Problem"""

import numpy as np
import scipy.stats as spstats
import matplotlib.pyplot as plt

# Neural network parameters
n_layers = 3
nodes_per_layer = 20
learning_rate = 0.001

# MFCP parameters

d = 10

t_low = 0.0
T = 20

m_low = 0.0
m_high = 1.0
M = 20

# c = np.ones(shape = [d, d], dtype = np.float32)

c = np.array([[0.0, 2.0, 5.0, 3.0, 1.0, 2.0, 3.0, 5.0, 10.0, 1.0],
              [2.0, 0.0, 5.0, 3.0, 1.0, 2.0, 3.0, 5.0, 10.0, 1.0],
              [10.0, 2.0, 0.0, 3.0, 1.0, 2.0, 3.0, 5.0, 10.0, 1.0],
              [3.0, 2.0, 5.0, 0.0, 1.0, 2.0, 3.0, 5.0, 10.0, 1.0],
              [1.0, 2.0, 5.0, 3.0, 0.0, 2.0, 3.0, 5.0, 10.0, 1.0],
              [10.0, 2.0, 5.0, 3.0, 1.0, 0.0, 3.0, 5.0, 10.0, 1.0],
              [2.0, 2.0, 5.0, 3.0, 1.0, 2.0, 0.0, 5.0, 10.0, 1.0],
              [2.0, 2.0, 5.0, 3.0, 1.0, 2.0, 3.0, 0.0, 10.0, 1.0],
              [5.0, 2.0, 5.0, 3.0, 1.0, 2.0, 3.0, 5.0, 0.0, 1.0],
              [1.0, 2.0, 5.0, 3.0, 1.0, 2.0, 3.0, 5.0, 10.0, 0.0]])


# Terminal cost function g
def g(m, i):
    s = 0
    for i in range(d):
        s += 10 * m[i]
    return s

# Hamiltonian function H
def a_star(r):
    result = tf.clip_by_value(r, 0, M)
    return result

def Hamilton(m, z, i):
    s = 0
    for j in range(d):
        s -= a_star(z[j]*(-1)*c[i][j]) * z[j]
        s -= 1/2 * pow(abs(a_star(z[j]*(-1)*c[i][j])), 2)
    s += a_star(z[i]*(-1)*c[i][i]) * z[i]
    s += 1/2 * pow(abs(a_star(z[i]*(-1)*c[i][i])), 2)
    s = s - 2*m[i]
    return s

# Training parameters
sampling_stages  = 30   # number of times to resample new time-space domain points
steps_per_sample = 10    # number of SGD steps to take before re-sampling

# Sampling parameters
nSim_interior = 30
nSim_terminal = 10

# Sampling function - randomly sample time-space pairs
def sampler(nSim_interior, nSim_terminal):
    ''' Sample time-space points from the function's domain;
        here each space point is the probability vector of staying at each state;
        points are sampled uniformly on the interior of the domain as well as at the terminal time points.
    Args:
        nSim_interior: number of space points in the interior of the function's domain to sample 
        nSim_terminal: number of space points at terminal time to sample (terminal condition)
    ''' 
    
    # Sampler 1st: domain interior
    t_interior = np.random.uniform(low=t_low, high=T, size=[nSim_interior, 1])
    m_interior = np.random.uniform(low=m_low, high=m_high, size=[nSim_interior, d])
    m_interior_sum = np.sum(m_interior,axis=1).reshape([nSim_interior,1])
    m_interior = m_interior/m_interior_sum

    # Sampler 2nd: spatial boundary
    # no spatial boundary condition for this problem
    
    # Sampler 3rd: initial/terminal condition
    t_terminal = T * np.ones((nSim_terminal, 1))
    m_terminal = np.random.uniform(low=m_low, high=m_high, size = [nSim_terminal, d])
    m_terminal_sum = np.sum(m_terminal,axis=1).reshape([nSim_terminal,1])
    m_terminal = m_terminal/m_terminal_sum
    
    
    t_interior = tf.convert_to_tensor(t_interior,dtype="float32")
    m_interior = tf.convert_to_tensor(m_interior,dtype="float32")
    t_terminal = tf.convert_to_tensor(t_terminal,dtype="float32")
    m_terminal = tf.convert_to_tensor(m_terminal,dtype="float32")
    
    return t_interior, m_interior, t_terminal, m_terminal

# Loss function for HJB equation of the MFC problem
def loss(model, t_interior, m_interior, t_terminal, m_terminal, nSim_interior, nSim_terminal):
    ''' Compute total loss for training.
    Args:
        model:         DGM model object
        t_interior:    sampled time points in the interior of the function's domain
        m_interior:    sampled space points in the interior of the function's domain
        t_terminal:    sampled time points at terminal time (vector of terminal times)
        m_terminal:    sampled space points at terminal time
        nSim_interior: number of space points in the interior of the function's domain to sample 
        nSim_terminal: number of space points at terminal time to sample (terminal condition)
    ''' 
    
    t = t_interior
    m = m_interior
    
    # compute derivatives at current sampled points in the interior
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(m)
        V = model(m,t)
    Vm = tape.gradient(V,m)
    
    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch(t)
        V = model(m,t)
    Vt = tape2.gradient(V,t)
    
    sum_list = []
    for k in range(nSim_interior):
        s = 0
        for i in range(d):
            partial_i = Vm[k][i]
            z = Vm[k] - partial_i
            s += m[k][i] * Hamilton(m, z, i)
        sum_list.append(s)
    sum = tf.stack(sum_list)
    
    diff_V = -Vt + sum
    L1 = tf.reduce_mean(tf.square(diff_V)) 
    
    # Loss term 2nd: boundary condition
    # no boundary condition for this problem
    
    # Loss term 3rd: initial/terminal condition
    r_list = []
    for k in range(nSim_terminal):
        l = 0
        for i in range(d):
            l += m_terminal[k][i] * g(m_terminal[k], i)
        r_list.append(l)
    target_value = tf.stack(r_list)

    fitted_value = model(m_terminal, t_terminal)
    # compute average L2-error of terminal condition
    L3 = tf.reduce_mean(tf.square(fitted_value - target_value))

    return L1, L3

# Gradient function (of loss function)
def grad(model, t_interior, m_interior, t_terminal, m_terminal, nSim_interior, nSim_terminal):
    V = model(m_interior,t_interior)
    W1 = model.initial_layer.W
    b1 = model.initial_layer.b
    W_last = model.final_layer.W
    b_last = model.final_layer.b
    Uz_list = []
    Ug_list = []
    Ur_list = []
    Uh_list = []
    Wz_list = []
    Wg_list = []
    Wr_list = []
    Wh_list = []
    bz_list = []
    bg_list = []
    br_list = []
    bh_list = []
    for i in range(n_layers):
        Uz_list.append(model.LSTMLayerList[i].Uz)
        Ug_list.append(model.LSTMLayerList[i].Ug)
        Ur_list.append(model.LSTMLayerList[i].Ur)
        Uh_list.append(model.LSTMLayerList[i].Uh)
        Wz_list.append(model.LSTMLayerList[i].Wz)
        Wg_list.append(model.LSTMLayerList[i].Wg)
        Wr_list.append(model.LSTMLayerList[i].Wr)
        Wh_list.append(model.LSTMLayerList[i].Wh)
        bz_list.append(model.LSTMLayerList[i].bz)
        bg_list.append(model.LSTMLayerList[i].bg)
        br_list.append(model.LSTMLayerList[i].br)
        bh_list.append(model.LSTMLayerList[i].bh)
    
    parameter_set = ([W1,b1,W_last,b_last] + Uz_list + Ug_list + Ur_list + Uh_list + Wz_list + Wg_list
                    + Wr_list + Wh_list + bz_list + bg_list + br_list + bh_list)
    
    with tf.GradientTape(persistent=True) as tape:
        L1, L3 = loss(model, t_interior, m_interior, t_terminal, m_terminal, nSim_interior, nSim_terminal)
        Loss = L1 + L3
    return tape.gradient(Loss, parameter_set), parameter_set

"""# Train"""

# Set up network
model = DGMNet(nodes_per_layer, n_layers)
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# Train network
for i in range(sampling_stages):
    # sample uniformly from the required regions
    t_interior, m_interior, t_terminal, m_terminal = sampler(nSim_interior, nSim_terminal)
    
    # for a given sample, take the required number of SGD steps
    for j in range(steps_per_sample):
        grads, parameters = grad(model, t_interior, m_interior, t_terminal, m_terminal, nSim_interior, nSim_terminal)
        optimizer.apply_gradients(zip(grads,parameters))
        if j % 10 == 0:
            L1, L3 = loss(model, t_interior, m_interior, t_terminal, m_terminal, nSim_interior, nSim_terminal)
            L1 = L1.numpy()
            L3 = L3.numpy()
            Loss = L1 + L3
            print(Loss, L1, L3, i)

# Make 3D plot for the value function (applicable for d = 2)
fig = plt.figure(figsize = (10, 10))
ax = plt.axes(projection='3d')
ax.grid()
n_plot = 100

t_plot = np.linspace(t_low, T, n_plot, dtype = np.float32)
m_plot = np.linspace(0.0, 1.0, n_plot, dtype = np.float32)
m_new = np.array([m_plot, 1.0-m_plot]).T
V = model(m_new, t_plot.reshape(-1, 1))
V = abs(V)

t, m = np.meshgrid(t_plot, m_plot)
surf = ax.plot_surface(t, m, V, cmap = plt.cm.cividis)
ax.set_title('Value function')

# Set axes label
ax.set_xlabel('time', labelpad=20)
ax.set_ylabel('m1', labelpad=20)
ax.set_zlabel('value function', labelpad=20)

ax.view_init(10, 40)
plt.show()

# Multilayer Perceptron for distribution \mu

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

intput_dim = 1
output_dim = d
hidden_dim = 30

model2 = tf.keras.models.Sequential()
model2.add(tf.keras.Input(shape=(1,)))
model2.add(tf.keras.layers.Dense(hidden_dim, activation='relu'))
model2.add(tf.keras.layers.Dense(hidden_dim, activation='relu'))
model2.add(tf.keras.layers.Dense(d, activation='sigmoid'))

# Do the sampling

def sampler2(nSim_interior, nSim_terminal):
    t_interior2 = np.random.uniform(low=t_low, high=T, size=[nSim_interior, 1])
    t_interior2 = tf.convert_to_tensor(t_interior, dtype="float32")

    t_terminal2 = T * np.ones((nSim_terminal, 1))
    t_terminal2 = tf.convert_to_tensor(t_terminal, dtype="float32")
    
    return t_interior2, t_terminal2

# Compute the loss function for distribution \mu

def loss2(t_interior2, t_terminal2):
    t = t_interior2
    u = model2(t_interior)
    u_sum = np.sum(u.numpy(),axis=1).reshape([nSim_interior,1])
    u = u/u_sum
    V = model(u, t)
    V = abs(V)
    
    # This is for the interior condition
    
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(u)
        V = model(u,t)
    Vu = tape.gradient(V,u)
    
    # in the process of calculating gradients, we apply the jacobian and reshape it
    sum_list = []
    z = np.arange(d, dtype = np.float32)
    for i in range(d):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(t)
            u = model2(t_interior)
            u_sum = np.sum(u.numpy(),axis=1).reshape([nSim_interior, 1])
            u = u/u_sum
        u_t = tape.jacobian(u, t)
        boolean_mask = tf.cast(u_t, dtype=tf.bool)              
        u_t = tf.boolean_mask(u_t, boolean_mask, axis=0)
        u_t = u_t.reshape(nSim_interior, d)
        for k in range(nSim_interior):
            partial_i = Vu[k][i]
            z = Vu[k] - partial_i
            s = 0
            for j in range(d):
                s = s + a_star((-1) * z[i]) * u[k][j] - a_star((-1) * z[j]) * u[k][i]
            sum_list.append(u_t[k][i] - s)

    diff_V = tf.stack(sum_list)
    L1 = tf.reduce_mean(tf.square(diff_V))
    
    # This is for the terminal condition
    
    # Note that the initial states could be set randomly according to the situation,
    # here we just set it as 1 for the 1st state and 0 for others
    r_list = []
    for k in range(nSim_terminal):
        r_list.append([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    target = tf.stack(r_list)
    u_terminal2 = model2(t_terminal)
    u_terminal2_sum = np.sum(u_terminal2.numpy(),axis=1).reshape([nSim_terminal,1])
    fitted = u_terminal2/u_terminal2_sum
    L3 = tf.reduce_mean(tf.square(fitted))

    return L1, L3

# Train network
for i in range(sampling_stages):
    t_interior2, t_terminal2 = sampler2(nSim_interior, nSim_terminal)
    
    # for a given sample, take the required number of SGD steps
    for j in range(steps_per_sample):
        opt = tf.keras.optimizers.SGD(learning_rate=0.1)
        with tf.GradientTape() as tape:
            L1, L3 = loss2(t_interior2, t_terminal2)
            loss = L1 + L3
            if j % 10 == 0:
              print(loss.numpy(), L1.numpy(), L3.numpy(), i)
            grads = tape.gradient(loss, model.trainable_weights)
            opt.apply_gradients(zip(grads, model.trainable_weights))

# Make line graph for the distribution \mu
n_plot = 10

def get_u_t(st_t, end_t):
  t_arr = np.linspace(st_t, end_t, n_plot)[:,np.newaxis]
  u_arr = model2(t_arr)
  u_sum = np.sum(u_arr.numpy(),axis=1).reshape([n_plot, 1])
  u_arr = u_arr/u_sum
  return t_arr, u_arr

t, u_arr = get_u_t(t_low, T)
#print(u_arr)

def get_index(t, u_arr, index):
  return u_arr[:,index]

u1 = get_index(t, u_arr, 0)
u2 = get_index(t, u_arr, 1)
u3 = get_index(t, u_arr, 2)
u4 = get_index(t, u_arr, 3)
u5 = get_index(t, u_arr, 4)
u6 = get_index(t, u_arr, 5)
u7 = get_index(t, u_arr, 6)
u8 = get_index(t, u_arr, 7)
u9 = get_index(t, u_arr, 8)
u10 = get_index(t, u_arr, 9)

plt.text(3, 0.4, r'$\mu_1$')
plt.text(3, 0.4, r'$\mu_2$')
plt.text(3, 0.4, r'$\mu_3$')
plt.text(3, 0.4, r'$\mu_4$')
plt.text(3, 0.4, r'$\mu_5$')
plt.text(3, 0.4, r'$\mu_6$')
plt.text(3, 0.4, r'$\mu_7$')
plt.text(3, 0.4, r'$\mu_8$')
plt.text(3, 0.4, r'$\mu_9$')
plt.text(3, 0.4, r'$\mu_10$')

plt.plot(t[0:n_plot], u1, color = 'lightcoral', label = '$\mu_1$')
plt.plot(t[0:n_plot], u2, color = 'dodgerblue', label = '$\mu_2$')
plt.plot(t[0:n_plot], u3, color = 'yellowgreen', label = '$\mu_3$')
plt.plot(t[0:n_plot], u4, color = 'grey', label = '$\mu_4$')
plt.plot(t[0:n_plot], u5, color = 'violet', label = '$\mu_5$')
plt.plot(t[0:n_plot], u6, color = 'cornsilk', label = '$\mu_6$')
plt.plot(t[0:n_plot], u7, color = 'limegreen', label = '$\mu_7$')
plt.plot(t[0:n_plot], u8, color = 'lightskyblue', label = '$\mu_8$')
plt.plot(t[0:n_plot], u9, color = 'bisque', label = '$\mu_9$')
plt.plot(t[0:n_plot], u10, color = 'lightpink', label = '$\mu_{10}$')
plt.title('Distribution', fontsize = 14)
plt.xlabel('Time', fontsize = 14)
plt.ylabel('Proportion', fontsize = 14)
plt.legend(loc='upper right')
plt.show()
