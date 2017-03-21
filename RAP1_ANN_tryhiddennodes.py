#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This code is edited from the "..._tryalphas" code in order to test the effect
of hidden layer size on performance

For more extensive commenting, see "...tryalphas"

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

pos_seqs = utils.read_pos_seqs("rap1-lieb-positives.txt")
#pos_matrix = utils.convert_to_binary(pos_seqs)
y_pos = np.ones([len(pos_seqs)])

neg_seqs = utils.read_neg_seqs("yeast-upstream-1k-negative.fa")
neg_seqs_filtered = [x for x in neg_seqs if len(x) != 0]

# Make sure no positive sequences were grabbed as negative sequence
for seq in pos_seqs:
    if seq in neg_seqs_filtered:
        print("Uh-oh! Negative sequence is actually positive sequence!")

neg_seqs_1 = list(np.random.choice(neg_seqs_filtered, 137))
y_neg = np.zeros([len(neg_seqs_1)])


all_seqs = np.append(pos_seqs,neg_seqs_1)
binary_matrix = utils.convert_to_binary(all_seqs)
all_ys = np.append(y_pos,y_neg)

shuf = list(range(0,len(all_seqs)))
np.random.shuffle(shuf)
M = binary_matrix[shuf,:]
ys = all_ys[shuf]

# Input layer size is set by the DNA encoding strategy. 
input_layer_size = 68
# Now, try multiple hidden layer sizes
hidden_layer_sizes = [200,20,2]
output_layer_size = 1
number_samples = 274

alpha = 10 #10 best out of 5,10,15
lam = 0

color_map = {0:'r',1:'b',2:'c',3:'m',4:'g',5:'k'} 

# This will split data into 5 different sets for cross validation
# can make this into for loop to try different numbers of splits
# may be better to do K-fold
skf = KFold(n_splits=5)
# For each train and test set, train, then test
plt_index = 1

AUCs ={}
for hidden_layer in hidden_layer_sizes:
    AUCs[hidden_layer] = []

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
    for hidden_layer_size in hidden_layer_sizes:

        label = "hidden layer size: " + str(hidden_layer_size)
        print(label)
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
        for iterations in range(501):
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
                overall_mean_error.append(np.mean(mean_error))
                mean_error = []
        
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
        #ROC_values = np.array(np.matrix(a3_test).T)
        ROC_values = a3_test
        AUC = metrics.roc_auc_score(ROC_ID, ROC_values)
        print(AUC)
        AUCs[hidden_layer_size].append(AUC)
    #fig, ax = plt.subplots()
    #rects1 = ax.bar(ind, AUCs, color='r')
    plt_index += 1 
#%%

# Plot barplot of AUC results of each test set for each alpha
# code adapted from: http://maheshakya.github.io/miscellaneous/2015/06/04/a-quick-guide-plotting-with-python-and-matplotlib-2.html 
x = np.array([0,1,2,5,6,7,10,11,12,15,16,17,20,21,22 ]) + 1
AUC1 = []
AUC2 = []
AUC3 = []
for key,sublist in AUCs.items():
    for value in sublist:
        if key == hidden_layer_sizes[0]:
            AUC1.append(value)
        elif key == hidden_layer_sizes[1]:
            AUC2.append(value)
        elif key == hidden_layer_sizes[2]:
            AUC3.append(value)
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
plt.legend((p1, p2, p3), ('HLS = 200','HLS = 20','HLS = 2'), loc='upper left')

plt.bar(left = x, height=AUC_all, color=['r', 'g', 'b'])
plt.ylim([0,1.4])
plt.xticks(x, labels)
plt.ylabel("AUC")
plt.legend

