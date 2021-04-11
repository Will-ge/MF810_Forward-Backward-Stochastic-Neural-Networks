""" 
Project MF810: Tensorflow conversion
Original code from: Maziar Raissi
https://maziarraissi.github.io/FBSNNs/

Team 2: 
"Shang, Dapeng" <dpshang@bu.edu>,
"Leng, Congshan" <amier98@bu.edu>,
"Xia, Shiyue" <syxia@bu.edu>,
"Chang, Jung Cheng" <junchang@bu.edu>,
"Deng, Hanrui" <hanrui@bu.edu>,
"Ge, Jinjing" <willge@bu.edu>
"Vivero Cabeza, Diego" <dvivero@bu.edu>
"""

'''We left part of the original code commented out
for comparison purposes
'''

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
        
        # We don't need to initialize session
        # tf session 
        # self.sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=True,
        #                                               log_device_placement=True))
        
        # tf placeholders and graph (training)
        # self.learning_rate = tf.compat.v1.placeholder(tf.float32, shape=[])
        # self.t_tf = tf.compat.v1.placeholder(tf.float32, shape=[M, self.N+1, 1]) # M x (N+1) x 1
        # self.W_tf = tf.compat.v1.placeholder(tf.float32, shape=[M, self.N+1, self.D]) # M x (N+1) x D
        # self.Xi_tf = tf.compat.v1.placeholder(tf.float32, shape=[1, D]) # 1 x D

        # self.loss, self.X_pred, self.Y_pred, self.Y0_pred = self.loss_function(self.t_tf, self.W_tf, self.Xi_tf)
                
        # optimizers
        # self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate = self.learning_rate)
        # self.train_op = self.optimizer.minimize(self.loss)
        # self.train_op = tf.keras.optimizers.Adam(learning_rate=1e-3)
        
        # initialize session and variables
        # init = tf.compat.v1.global_variables_initializer()
        # self.sess.run(init)
    
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
        
    # def xavier_init(self, size):
    #     '''generates random normal numbers with less than 2 std from the mean
    #     '''
    #     in_dim = size[0]
    #     out_dim = size[1]        
    #     xavier_stddev = np.sqrt(2/(in_dim + out_dim))
    #     return tf.Variable(tf.random.truncated_normal([in_dim, out_dim],
    #                                             stddev=xavier_stddev, dtype =tf.float64), dtype=tf.float64)
    
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
        # u = self.neural_net(tf.concat([t,X], 1), self.weights, self.biases) # M x 1
        # Du = tf.gradients(ys=u, xs=X)[0] # M x D

        with tf.GradientTape() as tape:
            tape.watch(X)
            u = tf.cast(self.neural_net(tf.concat([t,X], 1)), tf.float64) # M x 1
        Du = tape.gradient(u, X)  # M x D

        return u, Du

    def Dg_tf(self, X): # M x D
        #tf.gradients(ys=self.g_tf(X), xs=X)[0] 

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

            # tf_dict = {self.Xi_tf: self.Xi, self.t_tf: t_batch, self.W_tf: W_batch, self.learning_rate: learning_rate}
            # self.sess.run(self.train_op, tf_dict)
            
            if it % 100 == 0:
                elapsed = time.time() - start_time
                # loss_value, Y0_value, learning_rate_value = self.sess.run([self.loss, self.Y0_pred, self.learning_rate], tf_dict)
                loss_value, _, _1, Y0_value = self.loss_function(t_batch, W_batch, self.Xi)
                print('It: %d, Loss: %.3e, Y0: %.3f, Time: %.2f, learning rate: %.3f' % 
                      (it, loss_value, Y0_value, elapsed, learning_rate))

                start_time = time.time()
                
            
    def predict(self, Xi_star, t_star, W_star):
        
        # tf_dict = {self.Xi_tf: Xi_star, self.t_tf: t_star, self.W_tf: W_star}
        
        # X_star = self.sess.run(self.X_pred, tf_dict)
        # Y_star = self.sess.run(self.Y_pred, tf_dict)
        
        _, X_star, Y_star, _1 = self.loss_function(t_star, W_star, Xi_star)
        
        return X_star, Y_star
    
    ###########################################################################
    ############################# Change Here! ################################
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