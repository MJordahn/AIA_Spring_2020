#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Anders Bjorholm Dahl
abda@dtu.dk
2020
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
import scipy.ndimage.filters
import skimage.io
import skimage.feature

#%% Computing Gaussian and its second order derivative

im = skimage.io.imread('EX_2_data/test_blob_uniform.png').astype(np.float)

fig, ax = plt.subplots(1,1,figsize=(10,10),sharex=True,sharey=True)
ax.imshow(im,cmap='plasma')

def getGaussDerivative(t):
    # Computes kernels of Gaussian and its derivatives.
    # Input: Vairance - t
    # Output: Gaussian and derivatives - g, dg, ddg, dddg
    # Anders Bjorholm Dahl
    # abda@dtu.dk
    # 2020
    kSize = 5
    s = np.sqrt(t)
    x = np.arange(int(-np.ceil(s*kSize)), int(np.ceil(s*kSize))+1)
    x = np.reshape(x,(-1,1))
    g = np.exp(-x**2/(2*t))
    g = g/np.sum(g)
    dg = -x/t*g
    ddg = -g/t - x/t*dg
    dddg = -2*dg/t - x/t*ddg
    return g, dg, ddg, dddg

g, dg, ddg, dddg = getGaussDerivative(3)
fig, ax = plt.subplots(1,1,figsize=(10,10),sharex=True,sharey=True)
ax.plot(g)
ax.plot(dg)
ax.plot(ddg)
ax.plot(dddg)

#%% Convolve an image

t = 325
g, dg, ddg, dddg = getGaussDerivative(t)

Lg = scipy.ndimage.filters.convolve(scipy.ndimage.filters.convolve(im, g), g.T)
fig, ax = plt.subplots(1,1,figsize=(10,10),sharex=True,sharey=True)
ax.imshow(Lg,cmap='gray')


#%% Detecting blobs on one scale

im = skimage.io.imread('EX_2_data/test_blob_uniform.png').astype(np.float)

Lxx = scipy.ndimage.filters.convolve(scipy.ndimage.filters.convolve(im, g), ddg.T)
Lyy = scipy.ndimage.filters.convolve(scipy.ndimage.filters.convolve(im, ddg), g.T)

L_blob = t*(Lxx + Lyy)

# how blob response
fig, ax = plt.subplots(1,1,figsize=(10,10),sharex=True,sharey=True)
pos = ax.imshow(L_blob)
fig.colorbar(pos)



#%% Find regional maximum in Laplacian
magnitudeThres = 50

coord_pos = skimage.feature.peak_local_max(L_blob, threshold_abs=magnitudeThres)
coord_neg = skimage.feature.peak_local_max(-L_blob, threshold_abs=magnitudeThres)
coord = np.concatenate((coord_pos, coord_neg), axis = 0)

# Show circles
theta = np.arange(0, 2*np.pi, step=np.pi/100)
theta = np.append(theta, 0)
circ = np.array((np.cos(theta),np.sin(theta)))
n = coord.shape[0]
m = circ.shape[1]

fig, ax = plt.subplots(1,1,figsize=(10,10),sharex=True,sharey=True)
ax.imshow(im)
plt.plot(coord[:,1], coord[:,0], '.r')
circ_y = np.sqrt(2*t)*np.reshape(circ[0,:],(1,-1)).T*np.ones((1,n)) + np.ones((m,1))*np.reshape(coord[:,0],(-1,1)).T
circ_x = np.sqrt(2*t)*np.reshape(circ[1,:],(1,-1)).T*np.ones((1,n)) + np.ones((m,1))*np.reshape(coord[:,1],(-1,1)).T
plt.plot(circ_x, circ_y, 'r')



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Detecting blobs on multiple scales

im = skimage.io.imread('EX_2_data/test_blob_uniform.png').astype(np.float)

t = 15
g, dg, ddg, dddg = getGaussDerivative(t)

r,c = im.shape
n = 100
L_blob_vol = np.zeros((r,c,n))
tStep = np.zeros(n)

Lg = im
for i in range(0,n):
    tStep[i] = t*i
    L_blob_vol[:,:,i] = t*i*(scipy.ndimage.filters.convolve(scipy.ndimage.filters.convolve(Lg, g), ddg.T) +
        scipy.ndimage.filters.convolve(scipy.ndimage.filters.convolve(Lg, ddg), g.T))
    Lg = scipy.ndimage.filters.convolve(scipy.ndimage.filters.convolve(Lg, g), g.T)


#%% find maxima in scale-space
thres = 40.0
coord_pos = skimage.feature.peak_local_max(L_blob_vol, threshold_abs = thres)
coord_neg = skimage.feature.peak_local_max(-L_blob_vol, threshold_abs = thres)
coord = np.concatenate((coord_pos,coord_neg),axis = 0)

# Show circles
theta = np.arange(0, 2*np.pi, step=np.pi/100)
theta = np.append(theta, 0)
circ = np.array((np.cos(theta),np.sin(theta)))
n = coord.shape[0]
m = circ.shape[1]

fig, ax = plt.subplots(1,1,figsize=(10,10),sharex=True,sharey=True)
ax.imshow(im)
plt.plot(coord[:,1], coord[:,0], '.r')
scale = tStep[coord[:,2]]
circ_y = np.sqrt(2*scale)*np.reshape(circ[0,:],(1,-1)).T*np.ones((1,n)) + np.ones((m,1))*np.reshape(coord[:,0],(-1,1)).T
circ_x = np.sqrt(2*scale)*np.reshape(circ[1,:],(1,-1)).T*np.ones((1,n)) + np.ones((m,1))*np.reshape(coord[:,1],(-1,1)).T
plt.plot(circ_x, circ_y, 'r')

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Detecting blobs in real data (scale space)

# diameter interval and steps
d = np.arange(10, 24.5, step = 0.4)
tStep = np.sqrt(0.5)*((d/2)**2) # convert to scale

# read image and take out a small part
im = skimage.io.imread('EX_2_data/SEM.png').astype(np.float)
im = im[200:500,200:500]

fig, ax = plt.subplots(1,1,figsize=(10,10),sharex=True,sharey=True)
ax.imshow(im)

#%% Compute scale space

r,c = im.shape
n = d.shape[0]
L_blob_vol = np.zeros((r,c,n))

for i in range(0,n):
    g, dg, ddg, dddg = getGaussDerivative(tStep[i])
    L_blob_vol[:,:,i] = tStep[i]*(scipy.ndimage.filters.convolve(scipy.ndimage.filters.convolve(im, g), ddg.T) +
        scipy.ndimage.filters.convolve(scipy.ndimage.filters.convolve(im, ddg), g.T))


#%% Find maxima in scale space

thres = 30
coord = skimage.feature.peak_local_max(-L_blob_vol, threshold_abs = thres)

# Show circles
def getCircles(coord, scale):
    theta = np.arange(0, 2*np.pi, step=np.pi/100)
    theta = np.append(theta, 0)
    circ = np.array((np.cos(theta),np.sin(theta)))
    n = coord.shape[0]
    m = circ.shape[1]
    circ_y = np.sqrt(2*scale)*np.reshape(circ[0,:],(1,-1)).T*np.ones((1,n)) + np.ones((m,1))*np.reshape(coord[:,0],(-1,1)).T
    circ_x = np.sqrt(2*scale)*np.reshape(circ[1,:],(1,-1)).T*np.ones((1,n)) + np.ones((m,1))*np.reshape(coord[:,1],(-1,1)).T
    return circ_x, circ_y

scale = tStep[coord[:,2]]
circ_x, circ_y = getCircles(coord[:,0:2], scale)
fig, ax = plt.subplots(1,1,figsize=(10,10),sharex=True,sharey=True)
ax.imshow(im)
plt.plot(coord[:,1], coord[:,0], '.r')
plt.plot(circ_x, circ_y, 'r')

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Localize blobs - Example high resolution lab X-ray CT - find the coordinates
# using Gaussian smoothing and use the scale space to find the scale

im = skimage.io.imread('EX_2_data/CT_lab_high_res.png').astype(np.float)/255

fig, ax = plt.subplots(1,1,figsize=(10,10),sharex=True,sharey=True)
ax.imshow(im)

# %% Set parameters
def detectFibers(im, diameterLimit, stepSize, tCenter, thresMagnitude):
    # Detects fibers in images by finding maxima of Gaussian smoothed image
    # Input: image - im, 2 x 1 vector of limits of diameters of the fibers (in
    #   pixels) - diameterLimit, step size in pixels - stepSize, scale of the
    #   Gaussian for center detection - tCenter, threshold on the blob
    #   magnitude - thresMagnitude.
    # Output: n x 2 matrix of coordinates, n x 1 vector of scales
    # Anders Bjorholm Dahl
    # abda@dtu.dk
    # 2020

    radiusLimit = diameterLimit/2
    radiusSteps = np.arange(radiusLimit[0], radiusLimit[1], stepSize)
    tStep = radiusSteps**2/np.sqrt(2)

    r,c = im.shape
    n = tStep.shape[0]
    L_blob_vol = np.zeros((r,c,n))
    for i in range(0,n):
        g, dg, ddg, dddg = getGaussDerivative(tStep[i])
        L_blob_vol[:,:,i] = tStep[i]*(scipy.ndimage.filters.convolve(scipy.ndimage.filters.convolve(im, g), ddg.T) +
                                      scipy.ndimage.filters.convolve(scipy.ndimage.filters.convolve(im, ddg), g.T))

    # Detect fibre centers
    g, dg, ddg, dddg = getGaussDerivative(tCenter)
    Lg = scipy.ndimage.filters.convolve(scipy.ndimage.filters.convolve(im, g), g.T)

    coord = skimage.feature.peak_local_max(Lg, threshold_abs = thres)

    # Find coordinates and size (scale) of fibres
    magnitudeIm = np.min(L_blob_vol, axis = 2)
    scaleIm = np.argmin(L_blob_vol, axis = 2)
    scales = scaleIm[coord[:,0], coord[:,1]]
    magnitudes = -magnitudeIm[coord[:,0], coord[:,1]]
    idx = np.where(magnitudes > thresMagnitude)
    coord = coord[idx[0],:]
    scale = np.sqrt(2)*tStep[scales[idx[0]]]
    return coord, scale

#%% Set parameters

# Radius limit
diameterLimit = np.array([10,25])
stepSize = 0.3

# Parameter for Gaussian to detect center point
tCenter = 20

# Parameter for finding maxima over Laplacian in scale-space
thresMagnitude = 8

# Detect fibres
coord, scale = detectFibers(im, diameterLimit, stepSize, tCenter, thresMagnitude)

# Plot detected fibres
fig, ax = plt.subplots(1,1,figsize=(10,10),sharex=True,sharey=True)
ax.imshow(im)
ax.plot(coord[:,1], coord[:,0], 'r.')
circ_x, circ_y = getCircles(coord, scale)
plt.plot(circ_x, circ_y, 'r')
