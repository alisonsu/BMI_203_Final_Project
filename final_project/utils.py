#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 10:27:42 2017

@author: alisonsu
"""
import numpy as np
from Bio import SeqIO

def read_pos_seqs(filename):
    # read positive sequences
    pos_seqs = []
    file = open(filename,"r")
    
    for line in file:
        line=line.strip("\n")
        pos_seqs.append(line)
    file.close()
    return(pos_seqs)

def convert_to_binary(seq_list):
    # convert to binary
    A = "1000"
    T = "0100"
    C = "0010"
    G = "0001"
    
    matrix = np.zeros([len(seq_list),len(seq_list[0])*4])
    for index,seq in enumerate(seq_list):
        binary = ""
        for letter in seq:
            if letter == "A":
                binary += A
            elif letter == "T":
                binary += T
            elif letter == "C":
                binary += C
            elif letter == "G":
                binary += G
        binary = list(binary)
        matrix[index,:] = binary 
    return(matrix)

def read_neg_seqs(filename):
# read negative sequences
    all_names = []
    all_neg_seqs = []
    for record in SeqIO.parse(filename,"fasta"):
        name = record.description
        all_names.append(name)
        seq = str(record.seq)
        cut_seq = seq[700:717]
        all_neg_seqs.append(cut_seq)
    return(all_neg_seqs)