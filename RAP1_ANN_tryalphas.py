#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is the code for my 3 layer ANN (1 hidden layer) that uses a sigmoidal 
activation function. It predicts the probability of a DNA sequence being a
Rap1 binding site. This code specifically tests different learning rates, or
alphas

@author: alisonsu
"""

from final_project import *
import numpy as np
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn import metrics

def sigmoid(x):
    """
    This is the activation function
    """
    return(1/(1+np.exp(-x)))

def training(M_train_b,y_train_b, W1,W2,B1,B2, mean_error, x_values):
    """
    This function trains the neural network by performing forward and back propagation
    """
    number_samples = M_train_b.shape[0] #number of samples in minibatch
    del_W_L1 = np.zeros([input_layer_size,hidden_layer_size])
    del_B_L1 = np.zeros([1,hidden_layer_size])
        
    del_W_L2 = np.zeros([hidden_layer_size,output_layer_size])
    del_B_L2 = np.zeros([1,output_layer_size])

    # Feed forward:                
    # Calculate inputs to second layer:
    z2 = np.dot(M_train_b,W1)+B1
    # Calculate activation on second layer
    a2 = sigmoid(z2)
    # Calculate inputs to final layer
    z3 = np.dot(a2,W2)+B2
    # Calculate activation on final layer
    a3 = sigmoid(z3)
    
    # Calculate errors
    error_L3 = -(y_train_b-a3)
    if iterations % 100 == 0:
        print_string = "Error after " + str(iterations) + ":"
        error = np.mean(np.abs(error_L3))
        mean_error.append(error)
        if index == 0:
            print(print_string,str(np.mean(np.abs(error_L3))))
            x_values.append(iterations)

    error_L3 = np.multiply(error_L3,(a3*(1-a3))) #output layer
    error_L2 = np.multiply(np.dot(error_L3,W2.T),(a2*(1-a2))) # hidden layer

    for i in range(number_samples):
        # Calculate desired partial derivatives
        grad_cost_L2_W = np.multiply(np.matrix(a2[i]).T, np.matrix(error_L3[i]))
        grad_cost_L2_B = error_L3[i]               
                     
        grad_cost_L1_W = np.multiply(np.matrix(M_train_b[i]).T, np.matrix(error_L2[i]))
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
    
    return(W1, W2, B1, B2, mean_error, x_values)

#Read in positive sequences
pos_seqs = utils.read_pos_seqs("rap1-lieb-positives.txt")

# Give all positive sequences an output value of 1 
y_pos = np.ones([len(pos_seqs)])

# Read in negative sequences and make sure they are all sequences (len>0)
neg_seqs = utils.read_neg_seqs("yeast-upstream-1k-negative.fa")
neg_seqs_filtered = [x for x in neg_seqs if len(x) != 0]

# Make sure no positive sequences were grabbed as negative sequence
for seq in pos_seqs:
    if seq in neg_seqs_filtered:
        print("Uh-oh! Negative sequence is actually positive sequence!")

# Randomly choose 137 negative sequences to match number of positive sequences 1:1
neg_seqs_1 = list(np.random.choice(neg_seqs_filtered, 137))

# Give all negative sequences an output value of 0
y_neg = np.zeros([len(neg_seqs_1)])

# Create array of positive and negative sequences and convert to binary notation
all_seqs = np.append(pos_seqs,neg_seqs_1)
binary_matrix = utils.convert_to_binary(all_seqs)

# Create array of output values in same order of sequence array
all_ys = np.append(y_pos,y_neg)

# Randomly shuffle the input array so it's not in a particular order
shuf = list(range(0,len(all_seqs)))
np.random.shuffle(shuf)
M = binary_matrix[shuf,:]

# Put output values in same order as shuffled input values
ys = all_ys[shuf]

# Input layer size is set by the DNA encoding strategy. 
input_layer_size = 68

hidden_layer_size = 20

# Output layer size is defined by the output being a real number
output_layer_size = 1

number_samples = 274

alphas = [1,10,100] #10 best out of 5,10,15
lam = 0

color_map = {0:'r',1:'b',2:'c',3:'m',4:'g',5:'k'} 

# Split data in 5 different sections for cross-validation
skf = KFold(n_splits=5)
# For each train and test set, train, then test
plt_index = 1
AUCs ={}
for alpha in alphas:
    AUCs[alpha] = []
for train, test in skf.split(M, ys):  
    #print("%s %s" % (train, test))
    print(len(train),len(test))
    # Set values to training values
    M_train=M[train] 
    y_train = ys[train]
    M_test = M[test]
    y_test = ys[test]
    color_index = 0
    mean_error = []
    x_values = []
    # allows for testing different alphas
    for alpha in alphas:

        label = "alpha: " + str(alpha)
        print("alpha:",alpha)
        np.random.seed(1)
        
        # Initialize deltaW and delta_B matrices
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
                       
        # setup for mini batch gradient descent
        batches = list(range(10,len(y_train),10))
        mean_error = []
        x_values = []
        overall_mean_error = []
        for iterations in range(501):
            # Break the training set and output into mini-batches
            for index,batch in enumerate(batches):
                if index == 0:
                    M_train_b = M_train[:batch,:]
                    y_train_b = np.matrix(y_train[:batch]).T
                elif batch == batches[-1]:
                    M_train_b = M_train[batch:,:]
                    y_train_b = np.matrix(y_train[batch:]).T
                else:
                    M_train_b = M_train[batches[index-1]:batches[index],:]
                    y_train_b = np.matrix(y_train[batches[index-1]:batches[index]]).T
                                          
                # Train
                W1, W2, B1, B2, mean_error, x_values = training(M_train_b,y_train_b,W1,W2,B1,B2, mean_error, x_values)
    
            if mean_error != []:
                # Calculate the overall mean error from iteration through all of the sequences
                # Note the mean error is only recorded for certain iteration numbers set in function. This can be altered as desired.
                overall_mean_error.append(np.mean(mean_error))
                mean_error = []
        
        # Plot the mean error over iterations for each set of k-fold cross-validation
        plt.figure(plt_index)
        plt.plot(x_values,overall_mean_error,color=color_map[color_index],label=label)
        plt.legend()
        plt.xlabel("Iterations")
        plt.ylabel("Mean error")
        color_index += 1
     
        # after training, test on test set:
        a2_test = sigmoid(np.dot(M_test,W1)+B1)
        a3_test = sigmoid(np.dot(a2_test,W2)+B2)
        
        # Calculate AUC using scikitlearn ROC curve function
        ROC_ID = y_test
        ROC_values = a3_test
        AUC = metrics.roc_auc_score(ROC_ID, ROC_values)
        print(AUC)
        AUCs[alpha].append(AUC)
    plt_index += 1 
#%%
# Plot barplot of AUC results of each test set for each alpha
# code adapted from: http://maheshakya.github.io/miscellaneous/2015/06/04/a-quick-guide-plotting-with-python-and-matplotlib-2.html 
x = np.array([0,1,2,5,6,7,10,11,12,15,16,17,20,21,22 ]) + 1
AUC1 = []
AUC2 = []
AUC3 = []
# Generate lists of AUC values (1/cross validation set) for each alpha tested
for key,sublist in AUCs.items():
    for value in sublist:
        if key == alphas[0]:
            AUC1.append(value)
        elif key == alphas[1]:
            AUC2.append(value)
        elif key == alphas[2]:
            AUC3.append(value)
# Put all values together for plotting
AUC_all = []
for index,value in enumerate(AUC1):
    AUC_all.append(value)
    AUC_all.append(AUC2[index])
    AUC_all.append(AUC3[index])

labels = ["","test set 1","","","test set 2","","","test set 3","","","test set 4","","","test set 5","","","","","","","","",""]

plt.figure()
p1 = plt.Rectangle((0, 0), 0.1, 0.1, fc = 'r')
p2 = plt.Rectangle((0, 0), 0.1, 0.1, fc = 'g')
p3 = plt.Rectangle((0, 0), 0.1, 0.1, fc = 'b')
plt.legend((p1, p2, p3), ('alpha = 1','alpha = 10','alpha = 100'), loc='upper left')

plt.bar(left = x, height=AUC_all, color=['r', 'g', 'b'])
plt.ylim([0,1.4])
plt.xticks(x, labels)
plt.ylabel("AUC")
plt.legend