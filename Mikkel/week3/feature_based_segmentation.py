#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 20:11:26 2020

@author: vand
"""

import skimage.io
import numpy as np
import matplotlib.pyplot as plt
import sklearn.cluster
import scipy.ndimage


def im2col(A, BSZ, stepsize=1):
    # Parameters
    m,n = A.shape
    s0, s1 = A.strides    
    nrows = m-BSZ[0]+1
    ncols = n-BSZ[1]+1
    shp = BSZ[0],BSZ[1],nrows,ncols
    strd = s0,s1,s0,s1

    out_view = np.lib.stride_tricks.as_strided(A, shape=shp, strides=strd)
    return out_view.reshape(BSZ[0]*BSZ[1],-1)[:,::stepsize]


def ndim2col(A, BSZ, stepsize=1):
    if(A.ndim == 2):
        return im2col(A, BSZ, stepsize)
    else:
        r,c,l = A.shape
        patches = np.zeros((l*BSZ[0]*BSZ[1],(r-BSZ[0]+1)*(c-BSZ[1]+1)))
        for i in range(l):
            patches[i*BSZ[0]*BSZ[1]:(i+1)*BSZ[0]*BSZ[1],:] = im2col(A[:,:,i],BSZ,stepsize)
        return patches


def get_gauss_feat_im(im, s):

    x = np.arange(-np.ceil(4*s),np.ceil(4*s)+1)
    g = np.exp(-x**2/(2*s**2))
    g = g/np.sum(g)
    dg = -x/(s**2)*g
    ddg = -1/(s**2)*g - x/(s**2)*dg
    dddg = -2/(s**2)*dg - x/(s**2)*ddg
    ddddg = -2/(s**2)*ddg - 1/(s**2)*ddg - x/(s**2)*dddg
    
    
    r,c = im.shape
    imfeat = np.zeros((r,c,15))
    imfeat[:,:,0] = scipy.ndimage.convolve1d(scipy.ndimage.convolve1d(im,g,axis=1), g, axis=0)
    imfeat[:,:,1] = scipy.ndimage.convolve1d(scipy.ndimage.convolve1d(im,dg,axis=1), g, axis=0)
    imfeat[:,:,2] = scipy.ndimage.convolve1d(scipy.ndimage.convolve1d(im,g,axis=1), dg, axis=0)
    imfeat[:,:,3] = scipy.ndimage.convolve1d(scipy.ndimage.convolve1d(im,ddg,axis=1), g, axis=0)
    imfeat[:,:,4] = scipy.ndimage.convolve1d(scipy.ndimage.convolve1d(im,g,axis=1), ddg, axis=0)
    imfeat[:,:,5] = scipy.ndimage.convolve1d(scipy.ndimage.convolve1d(im,dg,axis=1), dg, axis=0)
    imfeat[:,:,6] = scipy.ndimage.convolve1d(scipy.ndimage.convolve1d(im,dddg,axis=1), g, axis=0)
    imfeat[:,:,7] = scipy.ndimage.convolve1d(scipy.ndimage.convolve1d(im,g,axis=1), dddg, axis=0)
    imfeat[:,:,8] = scipy.ndimage.convolve1d(scipy.ndimage.convolve1d(im,ddg,axis=1), dg, axis=0)
    imfeat[:,:,9] = scipy.ndimage.convolve1d(scipy.ndimage.convolve1d(im,dg,axis=1), ddg, axis=0)
    imfeat[:,:,10] = scipy.ndimage.convolve1d(scipy.ndimage.convolve1d(im,g,axis=1), ddddg, axis=0)
    imfeat[:,:,11] = scipy.ndimage.convolve1d(scipy.ndimage.convolve1d(im,dg,axis=1), dddg, axis=0)
    imfeat[:,:,12] = scipy.ndimage.convolve1d(scipy.ndimage.convolve1d(im,ddg,axis=1), ddg, axis=0)
    imfeat[:,:,13] = scipy.ndimage.convolve1d(scipy.ndimage.convolve1d(im,dddg,axis=1), dg, axis=0)
    imfeat[:,:,14] = scipy.ndimage.convolve1d(scipy.ndimage.convolve1d(im,ddddg,axis=1), g, axis=0)

    return imfeat
#%%

if __name__ == '__main__':
    
    filename = '../preparing_data/out/test_C_image.png'
    I = skimage.io.imread(filename)
    I = I.astype(np.float)
    
    s = 1;
    gf = get_gauss_feat_im(I, s)
    
    fig,ax = plt.subplots(3,5)
    for j in range(5):
        for i in range(3):
            ax[i][j].imshow(gf[:,:,5*i+j])
            ax[i][j].set_title(f'layer{5*i+j}')
    
