#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 07:48:09 2020

@author: abda
"""
import numpy as np
import matplotlib.pyplot as plt

#%%
f = open('t10k-images-idx3-ubyte', 'r')
a = np.fromfile(f, dtype='>i4', count=4)
images = np.fromfile(f, dtype='u1')

im = np.reshape(images[0:(28*28)],(28,28))
fig, ax = plt.subplots(1,1)
ax.imshow(im)
plt.title('MNST')
fig.show
#%%
f = open('t10k-labels-idx1-ubyte', 'r')
t = np.fromfile(f, count = 2, dtype='>i4');
labels = np.fromfile(f, dtype='u1')
