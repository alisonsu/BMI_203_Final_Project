#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is the code for my 3 layer ANN (1 hidden layer) that uses a sigmoidal 
activation function. It predicts the probability of a DNA sequence being a
Rap1 binding site. This is the final code for my ANN that tests and writes
predictions for the test data from rap1-lieb-test.txt

For more extensive commenting, see "...try_alphas.py" code, as the code
is essentially the same

@author: alisonsu
"""

from final_project import *
import numpy as np
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn import metrics
import sys

def sigmoid(x):
    return(1/(1+np.exp(-x)))

def training(M_train_b,y_train_b, W1,W2,B1,B2, mean_error, x_values):
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
    if iterations % 1000 == 0:
        print_string = "Error after " + str(iterations) + ":"
        error = np.mean(np.abs(error_L3))
        # collect the error from each mini-batch
        mean_error.append(error)
        if index == 0:
            # for the first mini-batch for batch, print error so can track
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
output_layer_size = 1
number_samples = 274

alphas = [10] #10 best out of 5,10,15
lam = 0

color_map = {0:'r',1:'b',2:'c',3:'m',4:'g',5:'k'} 

# This will split data into 5 different sets for cross validation
# can make this into for loop to try different numbers of splits
# may be better to do K-fold
skf = KFold(n_splits=5)
# For each train and test set, train, then test
plt_index = 1
for train, test in skf.split(M, ys):  
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
        for iterations in range(100001):
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
                
                W1, W2, B1, B2, mean_error, x_values = training(M_train_b,y_train_b,W1,W2,B1,B2, mean_error, x_values)
            if mean_error != []:
                # calculate the overall mean error over the full set of sequences
                overall_mean_error.append(np.mean(mean_error))
                mean_error = []
            # test for convergence, set by the difference in error being <1E-6
            if len(overall_mean_error) > 1:
                if (overall_mean_error[-2]-overall_mean_error[-1]) < 1E-6:
                    break
            if iterations > 2000 and overall_mean_error[-1] > 0.01:
                print("training error not converging.")
                sys.exit(1)
        plt.figure(plt_index)
        plt.plot(x_values,overall_mean_error,color=color_map[color_index],label=label)
        plt.legend()
        plt.xlabel("Iterations")
        plt.ylabel("Mean error")
        color_index += 1
     
        # after training, try on test set:
        a2_test = sigmoid(np.dot(M_test,W1)+B1)
        a3_test = sigmoid(np.dot(a2_test,W2)+B2)
        
        ROC_ID = y_test
        ROC_values = a3_test
        AUC = metrics.roc_auc_score(ROC_ID, ROC_values)
        print(AUC)
    plt_index += 1 
#%%

test_seqs = utils.read_pos_seqs("rap1-lieb-test.txt")
test_matrix = utils.convert_to_binary(test_seqs)

outfile = open("rap-lieb-test-predictions.txt","w")
for i,sequence in enumerate(test_seqs):
    a2_test = sigmoid(np.dot(test_matrix[i],W1)+B1)
    a3_test = sigmoid(np.dot(a2_test,W2)+B2)
    a3_output = float(a3_test)
    a3_output_cut = "%.3f" %a3_output
    print_seq = sequence + "\t" + a3_output_cut + "\n"
    outfile.write(print_seq)

outfile.close()

