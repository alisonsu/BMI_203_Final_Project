#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 15:08:09 2017

@author: alisonsu
"""
from final_project import utils
import os
import numpy as np
import math


def test_binary_encoder():
    sequences = ["A","T","C","G"]
    M = np.matrix([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
    binary = utils.convert_to_binary(sequences)
    assert(np.array_equal(M,binary))

def test_binary_encoder_multiple_letters():
    sequences = ["AT","CG"]
    M = np.matrix([[1,0,0,0,0,1,0,0],[0,0,1,0,0,0,0,1]])
    binary = utils.convert_to_binary(sequences)
    assert(np.array_equal(M,binary))

