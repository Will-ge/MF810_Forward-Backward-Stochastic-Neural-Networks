""" 
Project MF810: Tensorflow conversion
Original code from: Maziar Raissi
https://maziarraissi.github.io/FBSNNs/
"""


import numpy as np
import tensorflow as tf
import time
from abc import ABC, abstractmethod
from tensorflow.keras import backend as K

def custom_activation(x):
    # activation function used in NN
    return K.sin(x)



class FBSNN(ABC): # Forward-Backward Stochastic Neural Network
    def __init__(self, Xi, T,
                       M, N, D,
                       layers):
        
        self.Xi = Xi # initial point
        self.T = T # terminal time
        
        self.M = M # number of trajectories
        self.N = N # number of time snapshots
        self.D = D # number of dimensions
        
        # layers
        self.layers = layers # (D+1) --> 1
        
        # initialize NN
        self.initialize_NN(layers)
    
    def initialize_NN(self, layers):
        '''initializes the NN with random weights and 0 biases
        '''
        self.nn = tf.keras.Sequential()
        num_layers = len(layers) 
        for l in range(0,num_layers-2):
            self.nn.add(tf.keras.layers.Dense(layers[l+1], 
                                              input_shape = (layers[l],), 
                                          use_bias = True, 
                                          activation = custom_activation))
        self.nn.add(tf.keras.layers.Dense(layers[-1], 
                                          input_shape = (layers[-2],), 
                                          use_bias = True, 
                                          activation = 'linear'))

        return
        
   
    def neural_net(self, X): #, weights, biases):
        '''returns the result of the network for the given weights and biases
        '''
        weights = self.nn.weights[::2]
        biases = self.nn.weights[1::2]
        num_layers = len(weights) + 1
        H = X   # H is the evolution of X through the network
        for l in range(0,num_layers-2):
            W = tf.cast(weights[l], tf.float64)
            b = tf.cast(biases[l], tf.float64)
            H = tf.sin(tf.add(tf.linalg.matmul(H, W), b))
        W = tf.cast(weights[-1], tf.float64)
        b = tf.cast(biases[-1], tf.float64)
        Y = tf.add(tf.linalg.matmul(H, W), b)

        return Y
    
    def net_u(self, t, X): # M x 1, M x D
        with tf.GradientTape() as tape:
            tape.watch(X)
            u = tf.cast(self.neural_net(tf.concat([t,X], 1)), tf.float64) # M x 1
        Du = tape.gradient(u, X)  # M x D

        return u, Du

    def Dg_tf(self, X): # M x D
        with tf.GradientTape() as tape:
            tape.watch(X)
            g = self.g_tf(X)

        return tape.gradient(g, X)# M x D
        
    def loss_function(self, t, W, Xi): # M x (N+1) x 1, M x (N+1) x D, 1 x D
        loss = 0
        X_list = []
        Y_list = []
        t0 = t[:,0,:]
        W0 = W[:,0,:]
        X0 = tf.tile(Xi,[self.M,1]) # M x D  # Repeats Xi M times
        Y0, Z0 = self.net_u(t0,X0) # M x 1, M x D
        
        X_list.append(X0)
        Y_list.append(Y0)
        
        for n in range(0,self.N):
            t1 = t[:,n+1,:]
            W1 = W[:,n+1,:]
            # Euler-Maruyama scheme for X, Y
            X1 = X0 + self.mu_tf(t0,X0,Y0,Z0)*(t1-t0) + tf.squeeze(tf.matmul(self.sigma_tf(t0,X0,Y0),tf.expand_dims(W1-W0,-1)))
            Y1_tilde = Y0 + (self.phi_tf(t0,X0,Y0,Z0)*(t1-t0) + 
                             tf.reduce_sum(input_tensor=Z0*tf.squeeze(tf.matmul(self.sigma_tf(t0,X0,Y0),tf.expand_dims(W1-W0,-1))), axis=1, keepdims = True))
            Y1, Z1 = self.net_u(t1,X1)
            
            loss += tf.reduce_sum(input_tensor=tf.square(Y1 - Y1_tilde))
            
            t0 = t1
            W0 = W1
            X0 = X1
            Y0 = Y1
            Z0 = Z1
            
            X_list.append(X0)
            Y_list.append(Y0)

        loss += tf.reduce_sum(input_tensor=tf.square(Y1 - self.g_tf(X1)))
        loss += tf.reduce_sum(input_tensor=tf.square(Z1 - self.Dg_tf(X1)))

        X = tf.stack(X_list,axis=1)
        Y = tf.stack(Y_list,axis=1)

        return loss, X, Y, Y[0,0,0]

    def fetch_minibatch(self):
        '''returns the time and Brownian Motion for every t
        '''
        T = self.T
        
        M = self.M
        N = self.N
        D = self.D
        
        Dt = np.zeros((M,N+1,1)) # M x (N+1) x 1
        DW = np.zeros((M,N+1,D)) # M x (N+1) x D
        
        dt = T/N
        
        Dt[:,1:,:] = dt
        DW[:,1:,:] = np.sqrt(dt)*np.random.normal(size=(M,N,D))
        
        t = np.cumsum(Dt,axis=1) # M x (N+1) x 1
        W = np.cumsum(DW,axis=1) # M x (N+1) x D
        
        return t, W
    
    def train(self, N_Iter, learning_rate):
        
        opt = tf.keras.optimizers.Adam(learning_rate = learning_rate)
        start_time = time.time()
        
        for it in range(N_Iter):
            
            t_batch, W_batch = self.fetch_minibatch() # M x (N+1) x 1, M x (N+1) x D
        
            with tf.GradientTape() as tape:
                tape.watch(self.nn.trainable_weights)
                loss_value = self.loss_function(t_batch, W_batch, self.Xi)[0]
            gradients = tape.gradient(loss_value, self.nn.trainable_weights)
            opt.apply_gradients(zip(gradients, self.nn.trainable_weights))
            
            if it % 100 == 0:
                elapsed = time.time() - start_time
                loss_value, _, _1, Y0_value = self.loss_function(t_batch, W_batch, self.Xi)
                print('It: %d, Loss: %.3e, Y0: %.3f, Time: %.2f, learning rate: %.3f' % 
                      (it, loss_value, Y0_value, elapsed, learning_rate))

                start_time = time.time()
                
            
    def predict(self, Xi_star, t_star, W_star):
        _, X_star, Y_star, _1 = self.loss_function(t_star, W_star, Xi_star)
        
        return X_star, Y_star
    
    ###########################################################################
    @abstractmethod
    def phi_tf(self, t, X, Y, Z): # M x 1, M x D, M x 1, M x D
        pass # M x1
    
    @abstractmethod
    def g_tf(self, X): # M x D
        pass # M x 1
    
    @abstractmethod
    def mu_tf(self, t, X, Y, Z): # M x 1, M x D, M x 1, M x D
        M = self.M
        D = self.D
        return np.zeros([M,D]) # M x D
    
    @abstractmethod
    def sigma_tf(self, t, X, Y): # M x 1, M x D, M x 1
        M = self.M
        D = self.D
        return tf.linalg.diag(tf.ones([M,D])) # M x D x D
    ###########################################################################