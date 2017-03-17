#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is the code for an 8x3x8 autoencoder. In this code, it successfully learns
the 8-bit identity matrix. The number of nodes per layer is easily changeable
by changing the values of each layer size
"""
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return(1/(1+np.exp(-x)))

np.set_printoptions(precision=2,threshold=np.nan)
color_map = {0:'r',1:'b',2:'c',3:'m',4:'g',5:'k'} 


# Set adjustable input and output layer sizes
input_layer_size = 8
hidden_layer_size = 3
output_layer_size = 8

# Set alpha, or learning rate
alphas = [50]
# Set lambda, or weight decay parameter. In this case, I actually found that this
# parameter did not help, so I set it to 0
lam = 0# When the weight decay coefficient is big the penalty for big weights is also big, when it is small weights can freely grow

# Build 8-bit identity matrix for training
M = np.zeros([8,8])
for i in range(8):
    M[i,i] = 1
number_samples = 8
          
y = M # for autoencoder, want output same as input
index = 0

# Train autoencoder - allow for variation in alphas
# Since training matrix is relatively small, use batch gradient descent
for alpha in alphas:
    mean_error=[]
    x_values = []
    label = "alpha:" + str(alpha)
    np.random.seed(1)
    # Initialize delta weight and delta bias matrices
    del_W_L1 = np.zeros([input_layer_size,hidden_layer_size])
    del_B_L1 = np.zeros([1,hidden_layer_size])
    
    del_W_L2 = np.zeros([hidden_layer_size,output_layer_size])
    del_B_L2 = np.zeros([1,output_layer_size])
    
    # Initialize starting weight and bias value 
    W1 = np.zeros_like(del_W_L1)
    
    for i in range(input_layer_size):
        W1[i,:] = np.random.normal(0,0.01,hidden_layer_size)
    B1 = np.random.normal(0,0.01,hidden_layer_size)
    
    # Initialize weight matrix (2) and bias value (2)
    W2 = np.zeros_like(del_W_L2)
    for i in range(hidden_layer_size):
        W2[i] = np.random.normal(0,0.01,output_layer_size)
    B2 = np.random.normal(0,0.01,output_layer_size)
        
    for iterations in range(2000000):
        del_W_L1 = np.zeros([input_layer_size,hidden_layer_size])
        del_B_L1 = np.zeros([1,hidden_layer_size])
        
        del_W_L2 = np.zeros([hidden_layer_size,output_layer_size])
        del_B_L2 = np.zeros([1,output_layer_size])

        # Feed forward:
            
        # Calculate inputs to second layer:
        z2 = np.dot(M,W1)+B1
        # Calculate activation on second layer
        a2 = sigmoid(z2)
        # Calculate inputs to final layer
        z3 = np.dot(a2,W2)+B2
        # Calculate activation on final layer
        a3 = sigmoid(z3)
        
        # Calculate errors
        error_L3 = -(y-a3)
        if iterations % 100000 == 0:
            print_string = "Error after " + str(iterations) + ":"
            error = np.mean(np.abs(error_L3))**2
            mean_error.append(error)
            x_values.append(iterations)
            print(print_string,str(np.mean(np.abs(error_L3))))
        error_L3 *= (a3*(1-a3)) #output layer
        error_L2 = np.dot(error_L3,W2.T) * (a2*(1-a2)) # hidden layer
    
        for i in range(number_samples):
            # Calculate desired partial derivatives
            grad_cost_L2_W = np.multiply(np.matrix(a2[i]).T, np.matrix(error_L3[i]))
            grad_cost_L2_B = error_L3[i]               
                         
            grad_cost_L1_W = np.multiply(np.matrix(M[i]).T, np.matrix(error_L2[i]))
            grad_cost_L1_B = error_L2[i]               
                         
            # Set weight and bias changes
            del_W_L1 += grad_cost_L1_W 
            del_B_L1 += grad_cost_L1_B
        
            del_W_L2 += grad_cost_L2_W 
            del_B_L2 += grad_cost_L2_B
        
        # Upate weight and bias parameters
        W1 = W1 - alpha*((1/number_samples*del_W_L1) + lam*W1)
        B1 = B1 - alpha*(1/number_samples * del_B_L1)
        
        W2 = W2 - alpha*((1/number_samples*del_W_L2) + lam*W2)
        B2 = B2 - alpha*(1/number_samples * del_B_L2)
        
        # If MSE error sufficiently low, stop gradient descent
        if error < 1E-10:
            break
    # Plot results for varying alphas. This was commented out once optimal
    # alpha is determined
    """
    plt.plot(x_values,mean_error,color=color_map[index],label=label)
    plt.legend()
    plt.xlabel("Iterations")
    plt.ylabel("Mean squared error")
    #plt.xlim([7000,9000])
    #plt.ylim([-0.001,0.005])
    index += 1
    """

# test against identity matrix
M = np.zeros([8,8])
for i in range(8):
    M[i,i] = 1
# Calculate inputs to second layer:
z2 = np.dot(M,W1)+B1
# Calculate activation on second layer
a2 = sigmoid(z2)
# Calculate inputs to final layer
z3 = np.dot(a2,W2)+B2
# Calculate activation on final layer
a3 = sigmoid(z3)
print(M)
print(a3)
