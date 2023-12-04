import numpy as np
import math as ma
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#plt.rc('text', usetex=True)
#plt.rcParams.update({'font.size': 15})
#plt.rcParams["font.family"] = "serif"
#plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]

plt.rcParams.update({'font.family':'sans-serif'})
plt.rcParams.update({'font.sans-serif':'Helvetica'})

import random
import h5py

from matplotlib import ticker
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import colors as mcolors

import easydict
from utils import standardize, get_m_score, bootstrap_mean, spca, merge_scores
from utils import skewness_flatness, generate_balances, merge_clusters
import time


cluster_method = 'KMeans'  #'HDBSCAN'#
data_set_name = 'synth'#'ECCO' #'tumor_angiogenesis' #'tumor_dynamics' #
path = '/Users/bkaiser/Documents/data/m_score/' + data_set_name + '_' + cluster_method + '/'
read_path = path + 'results/'
features_path = path + 'features/'
figure_path = path + 'figures/'

# SPCA: a) needs large cluster sizes
#       b) is sensitive to variances. Must be non-zero and small.
#       c) taking the log of the features is necessary if the cluster
#          distribution is skewed.
# CHS:  a) bias towards fewer leading terms
#       b) is sensitive to variances. Finite variances can lead to more clusters.

# grid resolution
N = 100 # 50,100,200, 500 # must be even
nf = 8

# parameters
theta = N*np.pi*np.ones([nf])*4.  # N*np.pi*4.0 for step function slope factor, N*np.pi/20 for noisy gradient.
alpha = np.ones([nf])*10. #1e10 # upper value magnitude
delta = np.ones([nf])*0.1 #1e-10 # lower value magnitude
beta = np.ones([nf]) # intercept
eta = 0.1 # noise # eta<1 for closed eqn
offset = np.ones([nf])*0.5
#offset = np.array([0.4,0.4,0.4,0.4,0.6,0.6,0.6,0.6])
noise_flag = 'uniform' # 'normal' #


# overfit definition:
# too many clusters with many terms and therefore a lower score, compared to
# a signal with few terms and a higher score. Need to make a signal that has
# noisy extra terms and just a leading-order terms. If I make a gradient between
# two balances that is filled with multivariate noise, that won't work.
# If I make non-dominant noise that occasionally is the same magnitude as the
# dominant signal, this will be over clustered. Two terms are just noise?

# theta = slope factor: 0.0<theta<=4.0*np.pi*N,
# set to upper limit 4.0*np.pi*N for discontinuity

if nf == 4:
    eqn_labels = [r'$a$', r'$b$', r'$c$',
                  r'$d$']
    legend_fontsize = 18
    title_fontsize = 18
elif nf == 8:
    #eqn_labels = [r'$a$', r'$b$', r'$c$',r'$d$',
    #              r'$e$', r'$f$', r'$g$',r'$h$']
    eqn_labels = [r'$e_1$', r'$e_2$', r'$e_3$',r'$e_4$',
                  r'$e_5$', r'$e_6$', r'$e_7$',r'$e_8$']
    legend_fontsize = 16
    title_fontsize = 16
elif nf == 9:
    eqn_labels = [r'$a$', r'$b$', r'$c$',r'$d$',
                  r'$e$', r'$f$', r'$g$',r'$h$',r'$i$']
    legend_fontsize = 16
    title_fontsize = 16
elif nf == 10:
    #eqn_labels = [r'$a$', r'$b$', r'$c$',r'$d$',
    #              r'$e$', r'$f$', r'$g$',r'$h$']
    eqn_labels = [r'$x_1$', r'$x_2$', r'$x_3$',r'$x_4$',
                  r'$x_5$', r'$x_6$', r'$x_7$',r'$x_8$',
                  r'$x_9$',r'$x_{10}$']
    legend_fontsize = 16
    title_fontsize = 16
elif nf == 12:
    eqn_labels = [r'$a$', r'$b$', r'$c$',r'$d$',
                  r'$e$', r'$f$', r'$g$',r'$h$',
                  r'$i$', r'$j$', r'$k$',r'$l$']
    legend_fontsize = 12
    title_fontsize = 12
elif nf == 14:
    eqn_labels = [r'$a$', r'$b$', r'$c$',r'$d$',
                  r'$e$', r'$f$', r'$g$',r'$h$',
                  r'$i$', r'$j$', r'$k$',r'$l$',
                  r'$m$', r'$n$']
    legend_fontsize = 12
    title_fontsize = 12
elif nf == 16:
    eqn_labels = [r'$a$', r'$b$', r'$c$',r'$d$',
                  r'$e$', r'$f$', r'$g$',r'$h$',
                  r'$i$', r'$j$', r'$k$',r'$l$',
                  r'$m$', r'$n$', r'$o$',r'$p$']
    legend_fontsize = 12
    title_fontsize = 12
elif nf == 24:
    eqn_labels = [r'$a$', r'$b$', r'$c$',r'$d$',
                  r'$e$', r'$f$', r'$g$',r'$h$',
                  r'$i$', r'$j$', r'$k$',r'$l$']
    legend_fontsize = 12
    title_fontsize = 12

#===============================================================================
# cell centered grid

x = np.linspace(1./(2.*N),1.-1./(2.*N),num=N,endpoint=True) # cell edges
y = np.linspace(1./(2.*N),1.-1./(2.*N),num=N,endpoint=True) # cell edges
X,Y = np.meshgrid(x,y)
dx = 1./N
dy = 1./N
omg = 2.*np.pi*5.

#===============================================================================
# data

"""
# closure by even/odd pairing:
noise = np.zeros([N,N,nf])
for f in range(0,nf,2): # for each feature
    for j in range(0,N):
        if noise_flag == 'uniform':
            rn = np.random.uniform(low=-1., high=1., size=N)
        elif noise_flag == 'normal':
            rn = np.random.normal(loc=0., scale=1., size=N)
        noise[:,j,f] = (rn-np.amin(rn))/(np.amax(rn-np.amin(rn)))*2.*eta+(1.-eta)
        noise[:,j,f+1] = np.copy(noise[:,j,f])
        #print(np.amin(noise[:,j,f]),np.amax(noise[:,j,f]))
features = np.zeros([N,N,nf])
for f in range(0,nf): # for each feature
    for j in range(0,N): # for each row
        if f >= int(nf/2):
            features[:,j,f] = ((-1.)**f)*( (np.tanh((y-offset[f])*(theta[f]*np.pi)) + beta[f])*alpha[f]/2. + delta[f] ) * noise[:,j,f]
        else:
            features[:,j,f] = ((-1.)**f)*( (np.tanh(-(y-offset[f])*(theta[f]*np.pi)) + beta[f])*alpha[f]/2. + delta[f] ) * noise[:,j,f]
"""

# try clustering on:
# - standardized
# - log10(abs())
# - raw

# CHS needs additive noise because it can be quite sensitive: check this
# with scoring [1e-10,1e-10,1e10,5e10,5e10]

# closure by additional term:
noise = np.zeros([N,N,nf])
#for f in range(0,nf,2): # for each feature
#    for j in range(0,N):
#        if noise_flag == 'uniform':
#            rn = np.random.uniform(low=-1., high=1., size=N)
#        elif noise_flag == 'normal':
#            rn = np.random.normal(loc=0., scale=1., size=N)
#        noise[:,j,f] = (rn-np.amin(rn))/(np.amax(rn-np.amin(rn)))*2.*eta+(1.-eta)
#        noise[:,j,f+1] = np.copy(noise[:,j,f])
#        #print(np.amin(noise[:,j,f]),np.amax(noise[:,j,f]))


features = np.zeros([N,N,nf])
for f in range(0,nf): # for each feature # do to nf-1 for nf = odd number
    for j in range(0,N): # for each row
        # sinusoidal multiplicative noise: leading balances all are coherent.
        wave = np.ones([N]) + eta*np.sin(omg*x)
        #if f >= int(nf/2): # for nf = odd number
        #    features[:,j,f] = ((-1.)**f)*( (np.tanh((y-offset[f])*(theta[f]*np.pi)) + beta[f])*alpha[f]/2. + delta[f] ) * wave
        #else:
        #    features[:,j,f] = ((-1.)**f)*( (np.tanh(-(y-offset[f])*(theta[f]*np.pi)) + beta[f])*alpha[f]/2. + delta[f] ) * wave

        if f >= int(nf/2): # for nf = odd number
            features[:,j,f] = ((-1.)**f)*( np.heaviside(y-offset[f],y)*alpha[f] + delta[f] ) * wave
        else:
            features[:,j,f] = ((-1.)**f)*( np.heaviside(-(y-offset[f]),y)*alpha[f] + delta[f] ) * wave

        """
        # random multiplicative noise: harder for brute force, it finds the smaller balances.
        noise = np.ones([N]) + np.random.uniform(low=-eta, high=eta, size=N)
        if f >= int((nf-1)/2): # for nf = odd number
            features[:,j,f] = ((-1.)**f)*( (np.tanh((y-offset[f])*(theta[f]*np.pi)) + beta[f])*alpha[f]/2. + delta[f] ) * noise
        else:
            features[:,j,f] = ((-1.)**f)*( (np.tanh(-(y-offset[f])*(theta[f]*np.pi)) + beta[f])*alpha[f]/2. + delta[f] ) * noise
        """

        """
        # random additive noise: harder for c
        noise = np.random.uniform(low=-eta, high=eta, size=N)
        for i in range(0,N):
            if y[i] >= 0.5:
                noise[i] = 2.*noise[i]
        if f >= int((nf-1)/2):
            features[:,j,f] = ((-1.)**f)*( (np.tanh((y-offset[f])*(theta[f]*np.pi)) + beta[f])*alpha[f]/2. + delta[f] ) + noise
            #print('np.mean(features[0:25,j,f]) = ',np.mean(features[0:25,j,f]))
            #print('np.mean(features[25:50,j,f]) = ',np.mean(features[25:50,j,f]))
            #print('np.std(noise[0:25]) = ',np.std(noise[0:25]))
            #print('np.std(noise[25:50]) = ',np.std(noise[25:50]))
        else:
            features[:,j,f] = ((-1.)**f)*( (np.tanh(-(y-offset[f])*(theta[f]*np.pi)) + beta[f])*alpha[f]/2. + delta[f] ) + np.flip(noise)
            #print('np.mean(features[0:25,j,f]) = ',np.mean(features[0:25,j,f]))
            #print('np.mean(features[25:50,j,f]) = ',np.mean(features[25:50,j,f]))
            #print('np.std(np.flip(noise)[0:25]) = ',np.std(np.flip(noise)[0:25]))
            #print('np.std(np.flip(noise)[25:50]) = ',np.std(np.flip(noise)[25:50]))
        """

# # if nf = odd number:
#for i in range(0,N):
#    for j in range(0,N):
#        features[i,j,nf-1] = -np.sum(features[i,j,:])



# the additive noise helps the CHS algorithm find the correct balances.
# the multiplicative noise disrupts that...

#print(np.shape(features))
features = np.reshape(features, [int(N*N),nf], order='F')
#print(np.shape(features))
features = np.reshape(features, [N,N,nf], order='F')
#print(np.shape(features))


if nf == 4:
    figure_name = 'features.png'
    plotname = figure_path + figure_name
    fig = plt.figure(figsize=(12, 10))
    ax=plt.subplot(2,2,1)
    cs = plt.contourf(X, Y, np.abs(features[:,:,0]), locator=ticker.LogLocator(), cmap='gist_yarg_r') #
    plt.xlabel(r'$x$',fontsize=20)
    plt.ylabel(r'$y$',fontsize=20)
    plt.title(r'$|a|$',fontsize=20)
    cbar=plt.colorbar(cs) #, ticks=np.arange(results.nc))
    ax=plt.subplot(2,2,2)
    cs = plt.contourf(X, Y, np.abs(features[:,:,1]), locator=ticker.LogLocator(), cmap='gist_yarg_r') #
    plt.xlabel(r'$x$',fontsize=20)
    plt.ylabel(r'$y$',fontsize=20)
    plt.title(r'$|b|$',fontsize=20)
    cbar=plt.colorbar(cs) #, ticks=np.arange(results.nc))
    ax=plt.subplot(2,2,3)
    cs = plt.contourf(X, Y, np.abs(features[:,:,2]), locator=ticker.LogLocator(), cmap='gist_yarg_r') #
    plt.xlabel(r'$x$',fontsize=20)
    plt.ylabel(r'$y$',fontsize=20)
    plt.title(r'$|c|$',fontsize=20)
    cbar=plt.colorbar(cs) #, ticks=np.arange(results.nc))
    ax=plt.subplot(2,2,4)
    cs = plt.contourf(X, Y, np.abs(features[:,:,3]), locator=ticker.LogLocator(), cmap='gist_yarg_r') #
    plt.xlabel(r'$x$',fontsize=20)
    plt.ylabel(r'$y$',fontsize=20)
    plt.title(r'$|d|$',fontsize=20)
    cbar=plt.colorbar(cs) #, ticks=np.arange(results.nc))
    plt.subplots_adjust(top=0.95, bottom=0.075, left=0.075, right=0.975, hspace=0.3, wspace=0.3)
    plt.savefig(plotname,format="png"); plt.close(fig);


if nf == 8:
    figure_name = 'features.png'
    plotname = figure_path + figure_name
    fig = plt.figure(figsize=(20, 10))
    ax=plt.subplot(2,4,1)
    cs = plt.contourf(X, Y, np.abs(features[:,:,0]), locator=ticker.LogLocator(), cmap='gist_yarg_r') #
    plt.xlabel(r'$x$',fontsize=20)
    plt.ylabel(r'$y$',fontsize=20)
    plt.title(r'$|a|$',fontsize=20)
    cbar=plt.colorbar(cs) #, ticks=np.arange(results.nc))
    ax=plt.subplot(2,4,2)
    cs = plt.contourf(X, Y, np.abs(features[:,:,1]), locator=ticker.LogLocator(), cmap='gist_yarg_r') #
    plt.xlabel(r'$x$',fontsize=20)
    plt.ylabel(r'$y$',fontsize=20)
    plt.title(r'$|b|$',fontsize=20)
    cbar=plt.colorbar(cs) #, ticks=np.arange(results.nc))
    ax=plt.subplot(2,4,3)
    cs = plt.contourf(X, Y, np.abs(features[:,:,2]), locator=ticker.LogLocator(), cmap='gist_yarg_r') #
    plt.xlabel(r'$x$',fontsize=20)
    plt.ylabel(r'$y$',fontsize=20)
    plt.title(r'$|c|$',fontsize=20)
    cbar=plt.colorbar(cs) #, ticks=np.arange(results.nc))
    ax=plt.subplot(2,4,4)
    cs = plt.contourf(X, Y, np.abs(features[:,:,3]), locator=ticker.LogLocator(), cmap='gist_yarg_r') #
    plt.xlabel(r'$x$',fontsize=20)
    plt.ylabel(r'$y$',fontsize=20)
    plt.title(r'$|d|$',fontsize=20)
    cbar=plt.colorbar(cs) #, ticks=np.arange(results.nc))
    ax=plt.subplot(2,4,5)
    cs = plt.contourf(X, Y, np.abs(features[:,:,4]), locator=ticker.LogLocator(), cmap='gist_yarg_r') #
    plt.xlabel(r'$x$',fontsize=20)
    plt.ylabel(r'$y$',fontsize=20)
    plt.title(r'$|e|$',fontsize=20)
    cbar=plt.colorbar(cs) #, ticks=np.arange(results.nc))
    ax=plt.subplot(2,4,6)
    cs = plt.contourf(X, Y, np.abs(features[:,:,5]), locator=ticker.LogLocator(), cmap='gist_yarg_r') #
    plt.xlabel(r'$x$',fontsize=20)
    plt.ylabel(r'$y$',fontsize=20)
    plt.title(r'$|f|$',fontsize=20)
    cbar=plt.colorbar(cs) #, ticks=np.arange(results.nc))
    ax=plt.subplot(2,4,7)
    cs = plt.contourf(X, Y, np.abs(features[:,:,6]), locator=ticker.LogLocator(), cmap='gist_yarg_r') #
    plt.xlabel(r'$x$',fontsize=20)
    plt.ylabel(r'$y$',fontsize=20)
    plt.title(r'$|g|$',fontsize=20)
    cbar=plt.colorbar(cs) #, ticks=np.arange(results.nc))
    ax=plt.subplot(2,4,8)
    cs = plt.contourf(X, Y, np.abs(features[:,:,7]), locator=ticker.LogLocator(), cmap='gist_yarg_r') #
    plt.xlabel(r'$x$',fontsize=20)
    plt.ylabel(r'$y$',fontsize=20)
    plt.title(r'$|h|$',fontsize=20)
    cbar=plt.colorbar(cs) #, ticks=np.arange(results.nc))
    plt.subplots_adjust(top=0.95, bottom=0.075, left=0.075, right=0.975, hspace=0.3, wspace=0.3)
    plt.savefig(plotname,format="png"); plt.close(fig);

if nf == 12:
    figure_name = 'features.png'
    plotname = figure_path + figure_name
    fig = plt.figure(figsize=(30, 10))
    ax=plt.subplot(2,6,1)
    cs = plt.contourf(X, Y, np.abs(features[:,:,0]), levels=100, locator=ticker.LogLocator(), cmap='gist_yarg_r') #
    plt.xlabel(r'$x$',fontsize=20)
    plt.ylabel(r'$y$',fontsize=20)
    plt.title(r'$|a|$',fontsize=20)
    cbar=plt.colorbar(cs) #, ticks=np.arange(results.nc))
    ax=plt.subplot(2,6,2)
    cs = plt.contourf(X, Y, np.abs(features[:,:,1]), levels=100, locator=ticker.LogLocator(), cmap='gist_yarg_r') #
    plt.xlabel(r'$x$',fontsize=20)
    plt.ylabel(r'$y$',fontsize=20)
    plt.title(r'$|b|$',fontsize=20)
    cbar=plt.colorbar(cs) #, ticks=np.arange(results.nc))
    ax=plt.subplot(2,6,3)
    cs = plt.contourf(X, Y, np.abs(features[:,:,2]), levels=100, locator=ticker.LogLocator(), cmap='gist_yarg_r') #
    plt.xlabel(r'$x$',fontsize=20)
    plt.ylabel(r'$y$',fontsize=20)
    plt.title(r'$|c|$',fontsize=20)
    cbar=plt.colorbar(cs) #, ticks=np.arange(results.nc))
    ax=plt.subplot(2,6,4)
    cs = plt.contourf(X, Y, np.abs(features[:,:,3]), levels=100, locator=ticker.LogLocator(), cmap='gist_yarg_r') #
    plt.xlabel(r'$x$',fontsize=20)
    plt.ylabel(r'$y$',fontsize=20)
    plt.title(r'$|d|$',fontsize=20)
    cbar=plt.colorbar(cs) #, ticks=np.arange(results.nc))
    ax=plt.subplot(2,6,5)
    cs = plt.contourf(X, Y, np.abs(features[:,:,4]), levels=100, locator=ticker.LogLocator(), cmap='gist_yarg_r') #
    plt.xlabel(r'$x$',fontsize=20)
    plt.ylabel(r'$y$',fontsize=20)
    plt.title(r'$|e|$',fontsize=20)
    cbar=plt.colorbar(cs) #, ticks=np.arange(results.nc))
    ax=plt.subplot(2,6,6)
    cs = plt.contourf(X, Y, np.abs(features[:,:,5]), levels=100, locator=ticker.LogLocator(), cmap='gist_yarg_r') #
    plt.xlabel(r'$x$',fontsize=20)
    plt.ylabel(r'$y$',fontsize=20)
    plt.title(r'$|f|$',fontsize=20)
    cbar=plt.colorbar(cs) #, ticks=np.arange(results.nc))
    ax=plt.subplot(2,6,7)
    cs = plt.contourf(X, Y, np.abs(features[:,:,6]), levels=100, locator=ticker.LogLocator(), cmap='gist_yarg_r') #
    plt.xlabel(r'$x$',fontsize=20)
    plt.ylabel(r'$y$',fontsize=20)
    plt.title(r'$|g|$',fontsize=20)
    cbar=plt.colorbar(cs) #, ticks=np.arange(results.nc))
    ax=plt.subplot(2,6,8)
    cs = plt.contourf(X, Y, np.abs(features[:,:,7]), levels=100, locator=ticker.LogLocator(), cmap='gist_yarg_r') #
    plt.xlabel(r'$x$',fontsize=20)
    plt.ylabel(r'$y$',fontsize=20)
    plt.title(r'$|h|$',fontsize=20)
    cbar=plt.colorbar(cs) #, ticks=np.arange(results.nc))
    ax=plt.subplot(2,6,9)
    cs = plt.contourf(X, Y, np.abs(features[:,:,8]), levels=100, locator=ticker.LogLocator(), cmap='gist_yarg_r') #
    plt.xlabel(r'$x$',fontsize=20)
    plt.ylabel(r'$y$',fontsize=20)
    plt.title(r'$|i|$',fontsize=20)
    cbar=plt.colorbar(cs) #, ticks=np.arange(results.nc))
    ax=plt.subplot(2,6,10)
    cs = plt.contourf(X, Y, np.abs(features[:,:,9]), levels=100, locator=ticker.LogLocator(), cmap='gist_yarg_r') #
    plt.xlabel(r'$x$',fontsize=20)
    plt.ylabel(r'$y$',fontsize=20)
    plt.title(r'$|j|$',fontsize=20)
    cbar=plt.colorbar(cs) #, ticks=np.arange(results.nc))
    ax=plt.subplot(2,6,11)
    cs = plt.contourf(X, Y, np.abs(features[:,:,10]), levels=100, locator=ticker.LogLocator(), cmap='gist_yarg_r') #
    plt.xlabel(r'$x$',fontsize=20)
    plt.ylabel(r'$y$',fontsize=20)
    plt.title(r'$|k|$',fontsize=20)
    cbar=plt.colorbar(cs) #, ticks=np.arange(results.nc))
    ax=plt.subplot(2,6,12)
    cs = plt.contourf(X, Y, np.abs(features[:,:,11]), levels=100, locator=ticker.LogLocator(), cmap='gist_yarg_r') #
    plt.xlabel(r'$x$',fontsize=20)
    plt.ylabel(r'$y$',fontsize=20)
    plt.title(r'$|l|$',fontsize=20)
    cbar=plt.colorbar(cs) #, ticks=np.arange(results.nc))
    plt.subplots_adjust(top=0.95, bottom=0.075, left=0.075, right=0.975, hspace=0.3, wspace=0.3)
    plt.savefig(plotname,format="png"); plt.close(fig);





# 1D plot:
#fx = (np.tanh((x-0.5)*(theta[0]*np.pi)) + beta[0])*alpha[0]/2. + delta[0]
#print(theta[0]*np.pi)
plotname = figure_path + 'tanh.png'
#fig = plt.figure(figsize=(6, 5))
fig = plt.figure(figsize=(5.5,4.5)) #4.8, 4))
ax=plt.subplot(1,1,1)
ax.tick_params(axis = 'both', which = 'major', labelsize = 24)
plt.plot(x, np.abs(features[:,0,0]), color='royalblue',alpha=1.0,linestyle='solid',linewidth=2,label=r'$|e_1|$') #
plt.plot(x, np.abs(features[:,0,1]), color='goldenrod',alpha=1.0,linestyle='dashed',linewidth=2,label=r'$|e_2|$') #
plt.plot(x, np.abs(features[:,0,2]), color='crimson',alpha=1.0,linestyle='solid',linewidth=2,label=r'$|e_3|$') #
plt.plot(x, np.abs(features[:,0,3]), color='rebeccapurple',alpha=1.0,linestyle='dashed',linewidth=2,label=r'$|e_4|$') #
if nf >= 8:
    plt.plot(x, np.abs(features[:,0,4]), color='red',alpha=1.0,linestyle='solid',linewidth=2,label=r'$|e_5|$') #
    plt.plot(x, np.abs(features[:,0,5]), color='saddlebrown',alpha=1.0,linestyle='dashed',linewidth=2,label=r'$|e_6|$') #
    plt.plot(x, np.abs(features[:,0,6]), color='olivedrab',alpha=1.0,linestyle='solid',linewidth=2,label=r'$|e_7|$') #
    plt.plot(x, np.abs(features[:,0,7]), color='cyan',alpha=1.0,linestyle='dashed',linewidth=2,label=r'$|e_8|$') #
if nf >= 9:
    plt.plot(x, np.abs(features[:,0,8]), color='lime',alpha=1.0,linestyle='solid',linewidth=2,label=r'$|x_9|$') #
if nf >= 10:
    plt.plot(x, np.abs(features[:,0,8]), color='blue',alpha=1.0,linestyle='solid',linewidth=2,label=r'$|x_{10}|$') #
if nf >= 12:
    plt.plot(x, np.abs(features[:,0,10]), color='fuchsia',alpha=1.0,linestyle='solid',linewidth=2,label=r'$|k|$') #
    plt.plot(x, np.abs(features[:,0,11]), color='midnightblue',alpha=1.0,linestyle='dashed',linewidth=2,label=r'$|l|$') #
if nf >= 16:
    plt.plot(x, np.abs(features[:,0,12]), color='blue',alpha=1.0,linestyle='dashed',linewidth=2,label=r'$|m|$') #
    plt.plot(x, np.abs(features[:,0,13]), color='fuchsia',alpha=1.0,linestyle='solid',linewidth=2,label=r'$|n|$') #
    plt.plot(x, np.abs(features[:,0,14]), color='midnightblue',alpha=1.0,linestyle='dashed',linewidth=2,label=r'$|o|$') #
    plt.plot(x, np.abs(features[:,0,15]), color='deeppink',alpha=1.0,linestyle='dashed',linewidth=2,label=r'$|p|$') #
ax.set_yscale('log')
plt.ylabel(r'Magnitude',fontsize=26)
plt.xlabel(r'${y}$',fontsize=26)
if nf == 4:
    plt.title(r'$a+b+c+d=0$',fontsize=26)
elif nf == 8:
    #plt.title(r'$a+b+c+d+e+f+g+h=0$',fontsize=title_fontsize)
    #plt.title(r'$\sum_{i=1}^{8} e_i(\hat{y})=0$',fontsize=title_fontsize)
    plt.title(r'$\sum_{i=1}^{D=8} e_i({y})=0$',fontsize=20,pad=12)
elif nf == 9:
    plt.title(r'$a+b+c+d+e+f+g+h+i=0$',fontsize=title_fontsize)
elif nf == 12:
    plt.title(r'$a+b+c+d+e+f+g+h+i+j+k+l=0$',fontsize=title_fontsize)
elif nf == 16:
    plt.title(r'$\sum_{i=1}^{16} x_i=0$',fontsize=title_fontsize)
elif nf == 24:
    plt.title(r'$\sum_{i=1}^{24} x_i=0$',fontsize=title_fontsize)
plt.legend(loc=7,framealpha=1.,fontsize=15)
plt.grid()
#plt.ylim([5e9,5e10])
#plt.subplots_adjust(top=0.86, bottom=0.175, left=0.2, right=0.95, hspace=0.3, wspace=0.3)
plt.subplots_adjust(top=0.82, bottom=0.175, left=0.2, right=0.95, hspace=0.3, wspace=0.3)
plt.savefig(plotname,format="png"); plt.close(fig);

#===============================================================================
# plot standardization:

#print('\n  1) np.amax(features),np.amin(features) = ',np.amax(features),np.amin(features))
features_std = np.reshape(np.copy(features), [int(N*N),nf], order='F')
for i in range(0,np.shape(features_std)[1]):
    features_std[:,i] = standardize( features_std[:,i] )
features_std = np.reshape(features_std, [N,N,nf], order='F')
print('\n raw max/min = ',np.amax(features),np.amin(features))
print(' raw mean,std = ',np.mean(features),np.std(features))
print(' standardized max/min = ',np.amax(features_std),np.amin(features_std))
print(' standardized mean,std = ',np.mean(features_std),np.std(features_std))

plotname = figure_path + 'tanh_standardized.png'
fig = plt.figure(figsize=(6, 5))
ax=plt.subplot(1,1,1)
plt.plot(x, features_std[:,0,0], color='royalblue',alpha=1.0,linestyle='solid',linewidth=2,label=r'$a$') #
plt.plot(x, features_std[:,0,1], color='goldenrod',alpha=1.0,linestyle='dashed',linewidth=2,label=r'$b$') #
plt.plot(x, features_std[:,0,2], color='crimson',alpha=1.0,linestyle='solid',linewidth=2,label=r'$c$') #
plt.plot(x, features_std[:,0,3], color='rebeccapurple',alpha=1.0,linestyle='dashed',linewidth=2,label=r'$d$') #
if nf >= 8:
    plt.plot(x, features_std[:,0,4], color='red',alpha=1.0,linestyle='solid',linewidth=2,label=r'$e$') #
    plt.plot(x, features_std[:,0,5], color='saddlebrown',alpha=1.0,linestyle='dashed',linewidth=2,label=r'$f$') #
    plt.plot(x, features_std[:,0,6], color='olivedrab',alpha=1.0,linestyle='solid',linewidth=2,label=r'$g$') #
    plt.plot(x, features_std[:,0,7], color='cyan',alpha=1.0,linestyle='dashed',linewidth=2,label=r'$h$') #
if nf >= 9:
    plt.plot(x, features_std[:,0,8], color='lime',alpha=1.0,linestyle='solid',linewidth=2,label=r'$i$') #
if nf >= 12:
    plt.plot(x, features_std[:,0,9], color='blue',alpha=1.0,linestyle='dashed',linewidth=2,label=r'$j$') #
    plt.plot(x, features_std[:,0,10], color='fuchsia',alpha=1.0,linestyle='solid',linewidth=2,label=r'$k$') #
    plt.plot(x, features_std[:,0,11], color='midnightblue',alpha=1.0,linestyle='dashed',linewidth=2,label=r'$l$') #
plt.xlabel(r'$y$',fontsize=18)
if nf == 4:
    plt.title(r'standardized, $a+b+c+d=0$',fontsize=title_fontsize)
elif nf == 8:
    plt.title(r'standardized, $a+b+c+d+e+f+g+h=0$',fontsize=title_fontsize)
elif nf == 9:
    plt.title(r'standardized, $a+b+c+d+e+f+g+h+i=0$',fontsize=title_fontsize)
elif nf == 12:
    plt.title(r'standardized, $a+b+c+d+e+f+g+h+i+j+k+l=0$',fontsize=title_fontsize)
elif nf == 24:
    plt.title(r'$\sum_{i=1}^{24} x_i=0$',fontsize=title_fontsize)
plt.legend(loc=7,framealpha=1.,fontsize=legend_fontsize)
plt.grid()
plt.subplots_adjust(top=0.925, bottom=0.15, left=0.1, right=0.975, hspace=0.3, wspace=0.3)
plt.savefig(plotname,format="png"); plt.close(fig);

"""
# scatter plot standardized features:
plotname = figure_path + 'features_std_scatter.png'
fig = plt.figure(figsize=(6, 5))
ax=plt.subplot(1,1,1)
plt.plot(features_std[:,0,0], features_std[:,0,2], color='royalblue',alpha=1.0,marker='o',linestyle='None') #
plt.xlabel(r'$a$',fontsize=18)
plt.ylabel(r'$c$',fontsize=18)
plt.grid()
plt.subplots_adjust(top=0.925, bottom=0.15, left=0.15, right=0.975, hspace=0.3, wspace=0.3)
plt.savefig(plotname,format="png"); plt.close(fig);

plotname = figure_path + 'features_scatter.png'
fig = plt.figure(figsize=(6, 5))
ax=plt.subplot(1,1,1)
plt.plot(features[:,0,0], features[:,0,2], color='royalblue',alpha=1.0,marker='o',linestyle='None') #
ax.set_xscale('log')
ax.set_yscale('log')
plt.xlabel(r'$a$',fontsize=18)
plt.ylabel(r'$c$',fontsize=18)
plt.grid()
plt.subplots_adjust(top=0.925, bottom=0.15, left=0.15, right=0.975, hspace=0.3, wspace=0.3)
plt.savefig(plotname,format="png"); plt.close(fig);
"""
#===============================================================================

# save
features = np.reshape(features, [int(N*N),nf], order='F')
features_std = np.reshape(features_std, [int(N*N),nf], order='F')
#print('\n raw max/min = ',np.amax(features),np.amin(features))
#print(' np.shape(np.sum(features,axis=1)) = ',np.shape(np.sum(features,axis=1)))
print('\n raw closure = ',np.mean(np.sum(features,axis=1)))
print(' standardized closure = ',np.mean(np.sum(features,axis=1)))
filename = features_path + 'features.npy'
np.save( filename, features )
filename = features_path + 'x.npy'
np.save( filename, x )
filename = features_path + 'y.npy'
np.save( filename, y )


#features_std = np.copy(features)
#for i in range(0,np.shape(features_std)[1]):
#    features_std[:,i] = standardize( features_std[:,i] )
#print('\n standardized max/min = ',np.amax(features_std),np.amin(features_std))
#print(' standardized mean,std = ',np.mean(features_std),np.std(features_std))
print('\n This standardization works! \n')

# SPCA needs large cluster sizes and variance in the clusters.
# CHS expects there to be fewer leading terms (bias) and can be susceptible to noise.


# score test:

# spca_features_mean, spca_features_95 = bootstrap_mean( spca_features , Nbootstrap=5000 )

yflat = np.reshape(np.copy(Y), [int(N*N)], order='F')
locs = np.argwhere(yflat>=0.5)[:,0]
locs2 = np.argwhere(yflat<0.5)[:,0]
feature_mean, feature_mean_95 = bootstrap_mean( features[locs,:] , 0 )
print('  feature_mean = ',feature_mean)
print('  feature_std = ',np.std(features[locs,:],axis=0))
#print('  np.log10(feature_mean) = ',np.log10(feature_mean))
print('\n  np.mean(np.log10(np.abs(features[locs,:])) = ',np.mean(np.log10(np.abs(features[locs,:])),axis=0))
print('  np.std(np.log10(np.abs(features[locs,:]))) = ',np.std(np.log10(np.abs(features[locs,:])),axis=0))
skew = np.zeros([nf]); flat = np.zeros([nf])
for i in range(0,nf):
    skew[i],flat[i] = skewness_flatness( features[locs,i] )
print('\n  skew = ',skew)
print('  flat = ',flat)


# area
dx = x[1]-x[0]; dy = y[1:]-y[:-1]
nx = len(x); ny = len(y)
# data is cell centered
dx = x[1:nx]-x[0:nx-1]
dxe = np.append(dx,dx[0]) # grid is uniform in x
dy = y[1:ny]-y[0:ny-1]
yedges = np.append(0.,y[0:ny-1]+dy/2.)
ytop = y[ny-1]+dy[ny-2]/2.
yedges = np.append(yedges,ytop)
nedges = len(yedges)
dye = yedges[1:nedges]-yedges[0:nedges-1]
DXe,DYe = np.meshgrid(dxe,dye)
area = DXe*DYe
area = (area).flatten('F')
area = area/np.sum(area)

"""
# brute force calculation
start_total_time = time.time()
balance_combinations = generate_balances(nf)
ng = int(N*N)
closure = np.zeros([ng])
score = np.zeros([ng])
balance = np.zeros([ng,nf])
ngb = np.shape(balance_combinations)[0] # number of combinations
print('\n  Number of possible balances = ',ngb)
print('  np.amax(np.abs(features)),np.amin(np.abs(features)) = ',np.amax(np.abs(features)),np.amin(np.abs(features)))

for nn in range(0,ng): # loop over grid points
    score_mm = np.zeros([ngb])
    closure_mm = np.zeros([ngb])
    for mm in range(0,ngb): # loop over all possible balances for the cluster
        score_mm[mm], closure_mm[mm] = get_m_score( balance_combinations[mm,:] , features[nn,:] )
    score[nn] = np.amax(score_mm) # maximum score for the active term choice for a cluster
    loc_max = (np.argwhere(score_mm==np.amax(score[nn]))[:,0])[0] #np.argwhere(score_mm==score[nn])[:,0]
    #loc_max = (np.argwhere(score_mm==score[nn])[:,0])[1] # WHY DID THIS NEED TO CHANGE?
    balance[nn,:] = balance_combinations[loc_max,:]
    closure[nn] = closure_mm[loc_max]
    #print('  np.amax(score_mm) = ',np.amax(score_mm))

# agglomerate:
cluster_balance, cluster_labels, label_matching_index = merge_clusters( balance , np.arange(ng) )
merged_scores, merged_closures, merged_area_weights = merge_scores( score, closure, area, np.unique(cluster_labels), label_matching_index )
total_time = time.time() - start_total_time

print('  np.sum(label_matching_index-cluster_labels) = ',np.sum(label_matching_index-cluster_labels))
print('  np.shape(cluster_balance) = ',np.shape(cluster_balance))
print('  np.unique(cluster_labels) = ',np.unique(cluster_labels))
print('  np.shape(cluster_labels) = ',np.shape(cluster_labels))
print('  np.shape(merged_scores) = ',np.shape(merged_scores))
print('  merged_areas = ',merged_area_weights)
print('  global score = ',np.sum(merged_area_weights * merged_scores))

score_file_name =  features_path + 'no_cluster_scores.npy'
np.save(score_file_name, merged_scores)
closure_file_name =  features_path + 'no_cluster_area_weights.npy'
np.save(closure_file_name, merged_area_weights)
balance_file_name =  features_path + 'no_cluster_balance.npy'
np.save(balance_file_name, cluster_balance)
labels_file_name =  features_path + 'no_cluster_labels.npy'
np.save(labels_file_name, cluster_labels)
file_name =  features_path + 'no_cluster_time.npy'
np.save(file_name, total_time)

"""













"""
if nf == 4:
    balance1 = np.array([0.,0.,1.,1.])
    score_max, closure_max = get_m_score( balance1 , feature_mean )
    print('\n  balance 1 = ',balance1)
    print('  score_max 1 = ',score_max)
    balance2 = np.array([1.,1.,0.,0.])
    score_max, closure_max = get_m_score( balance2 , feature_mean )
    print('\n  balance 2 = ',balance2)
    print('  score_max 2 = ',score_max)
    balance3 = np.array([1.,1.,1.,1.])
    score_max, closure_max = get_m_score( balance3 , feature_mean )
    print('\n  balance 3 = ',balance3)
    print('  score_max 3 = ',score_max)

    cluster_idx = np.zeros(np.shape(features[locs,:])[0],dtype=int)
    nc = 1
    alpha = 1.0 # 0.5
    #balance = spca( nc , cluster_idx, features[locs,:], alpha )
    balance = spca( nc , cluster_idx, np.log10(np.abs(features[locs,:])), alpha, False )
    #print('\n  log spca balance = ',balance)
    #print('\n  log abs(spca balance - 1) = ',np.abs(balance-1))

    balance = spca( nc , cluster_idx, features[locs,:], alpha, True )
    print('\n  spca balance, skew_flag on = ',balance)

    balance = spca( nc , cluster_idx, features[locs,:], alpha, False )
    print('\n  spca balance, skew_flag off = ',balance)

    # outlier removal:
    #spca_features_mean, spca_features_95 = bootstrap_mean( features[locs,:] , Nbootstrap=5000 )
    #print('\n  bootstrap feature mean = ',spca_features_mean)
    #print('  bootstrap feature 95% = ',spca_features_95)
    ## remove outliers: (in log space?)
    #Nl = np.shape(features[locs,:])[0]
    #for i in range(0,Nl):
    #    j = locs[i]
    #    for k in range(0,np.shape(features[locs,:])[1]):
    #        if features[j,k] <= spca_features_mean[k] - spca_features_95[k]:
    #            #print('\n  features[j,k] = ',features[j,k])
    #            #print('  spca_features_mean[k] - spca_features_95[k]/2. = ',spca_features_mean[k] - spca_features_95[k]/2.)
    #            features[j,k] = np.nan
    #        elif features[j,k] >= spca_features_mean[k] + spca_features_95[k]:
    #            features[j,k] = np.nan
    #print('\n  outliers removed feature_mean = ',np.nanmean(features[locs,:],axis=0))
    #print('  outliers removed feature_std = ',np.nanstd(features[locs,:],axis=0))

    ##print(np.shape(features[locs,:]))
    #new_cluster_idx = np.zeros([0])
    #new_features = np.zeros([0,nf])
    #for i in range(0,len(locs)):
    #    if np.any(np.isnan(features[locs[i],:])) == False: # no NaNs for a sample
    #        tmp = np.zeros([1,nf])
    #        tmp[0,:] = features[locs[i],:]
    #        new_features = np.concatenate((new_features,tmp),axis=0)
    #        new_cluster_idx = np.append(new_cluster_idx,cluster_idx[i])

    #print('\n  outliers removed feature_mean = ',np.mean(new_features,axis=0))
    #print('  outliers removed feature_std = ',np.std(new_features,axis=0))
    #print('\n  outliers removed len(new_cluster_idx) = ',len(new_cluster_idx))
    #print('  outliers removed np.shape(features) = ', np.shape(new_features))
    #print('  np.shape(features[locs,:]) = ', np.shape(features[locs,:]))
    #print('\n  outliers removed np.mean(np.log10(np.abs(features[locs,:])) = ',np.mean(np.log10(np.abs(new_features)),axis=0))
    #print('  outliers removed np.std(np.log10(np.abs(features[locs,:]))) = ',np.std(np.log10(np.abs(new_features)),axis=0))
    ##alphas = np.logspace(-2,2.,num=101,endpoint=True)
    ##for i in range(0,len(alphas)):
    ##    balance = spca( nc , new_cluster_idx, new_features, alphas[i] )
    ##    print('\n  spca balance = ',balance)
    ##    print('  alpha = ',alphas[i])

    #balance = spca( nc , new_cluster_idx, new_features, 0.1 )
    #print('\n  spca balance = ',balance)

    # area
    dx = x[1]-x[0]; dy = y[1:]-y[:-1]
    nx = len(x); ny = len(y)
    # data is cell centered
    dx = x[1:nx]-x[0:nx-1]
    dxe = np.append(dx,dx[0]) # grid is uniform in x
    dy = y[1:ny]-y[0:ny-1]
    yedges = np.append(0.,y[0:ny-1]+dy/2.)
    ytop = y[ny-1]+dy[ny-2]/2.
    yedges = np.append(yedges,ytop)
    nedges = len(yedges)
    dye = yedges[1:nedges]-yedges[0:nedges-1]
    DXe,DYe = np.meshgrid(dxe,dye)
    area = DXe*DYe
    area = (area).flatten('F')
    area = area/np.sum(area)

    # calculate the truth!
    start_total_time = time.time()
    balance_combinations = generate_balances(nf)
    ng = int(N*N)
    closure = np.zeros([ng])
    score = np.zeros([ng])
    balance = np.zeros([ng,nf])
    ngb = np.shape(balance_combinations)[0] # number of combinations
    print('\n  Number of possible balances = ',ngb)
    print('  np.amax(np.abs(features)),np.amin(np.abs(features)) = ',np.amax(np.abs(features)),np.amin(np.abs(features)))

    for nn in range(0,ng): # loop over grid points
        score_mm = np.zeros([ngb])
        closure_mm = np.zeros([ngb])
        for mm in range(0,ngb): # loop over all possible balances for the cluster
            score_mm[mm], closure_mm[mm] = get_m_score( balance_combinations[mm,:] , features[nn,:] )
        score[nn] = np.amax(score_mm) # maximum score for the active term choice for a cluster
        loc_max = (np.argwhere(score_mm==np.amax(score[nn]))[:,0])[0] #np.argwhere(score_mm==score[nn])[:,0]
        #loc_max = (np.argwhere(score_mm==score[nn])[:,0])[1] # WHY DID THIS NEED TO CHANGE?
        balance[nn,:] = balance_combinations[loc_max,:]
        closure[nn] = closure_mm[loc_max]
        #print('  np.amax(score_mm) = ',np.amax(score_mm))

    ##print('  balance = ',balance)
    #cluster_balance = np.unique(balance,axis=0)
    #print('  cluster_balance = ',cluster_balance)
    #nc = np.shape(cluster_balance)[0]
    #print('  nc = ',nc)
    #cluster_labels = np.zeros([ng])

    # agglomerate:
    cluster_balance, cluster_labels, label_matching_index = merge_clusters( balance , np.arange(ng) )
    merged_scores, merged_closures, merged_area_weights = merge_scores( score, closure, area, np.unique(cluster_labels), label_matching_index )
    total_time = time.time() - start_total_time

    print('  np.sum(label_matching_index-cluster_labels) = ',np.sum(label_matching_index-cluster_labels))
    print('  np.shape(cluster_balance) = ',np.shape(cluster_balance))
    print('  np.unique(cluster_labels) = ',np.unique(cluster_labels))
    print('  np.shape(cluster_labels) = ',np.shape(cluster_labels))
    print('  np.shape(merged_scores) = ',np.shape(merged_scores))
    print('  merged_areas = ',merged_area_weights)
    print('  global score = ',np.sum(merged_area_weights * merged_scores))

    score_file_name =  features_path + 'no_cluster_scores.npy'
    np.save(score_file_name, merged_scores)
    closure_file_name =  features_path + 'no_cluster_area_weights.npy'
    np.save(closure_file_name, merged_area_weights)
    balance_file_name =  features_path + 'no_cluster_balance.npy'
    np.save(balance_file_name, cluster_balance)
    labels_file_name =  features_path + 'no_cluster_labels.npy'
    np.save(labels_file_name, cluster_labels)
    file_name =  features_path + 'no_cluster_time.npy'
    np.save(file_name, total_time)

elif nf == 8:
    balance1 = np.array([0.,0.,0.,0.,1.,1.,1.,1.])
    score_max, closure_max = get_m_score( balance1 , feature_mean )
    print('\n  balance 1 = ',balance1)
    print('  score_max 1 = ',score_max)
    balance2 = np.array([0.,0.,0.,0.,0.,0.,1.,1.])
    score_max, closure_max = get_m_score( balance2 , feature_mean )
    print('\n  balance 2 = ',balance2)
    print('  score_max 2 = ',score_max)
    balance3 = np.array([0.,0.,0.,0.,1.,1.,0.,0.])
    score_max, closure_max = get_m_score( balance3 , feature_mean )
    print('\n  balance 3 = ',balance3)
    print('  score_max 3 = ',score_max)
    balance4 = np.array([1.,1.,1.,1.,0.,0.,0.,0.])
    score_max, closure_max = get_m_score( balance4 , feature_mean )
    print('\n  balance 4 = ',balance4)
    print('  score_max 4 = ',score_max)
    balance5 = np.array([1.,1.,1.,1.,1.,1.,1.,1.])
    score_max, closure_max = get_m_score( balance5 , feature_mean )
    print('\n  balance 5 = ',balance5)
    print('  score_max 5 = ',score_max)

    # area
    dx = x[1]-x[0]; dy = y[1:]-y[:-1]
    nx = len(x); ny = len(y)
    # data is cell centered
    dx = x[1:nx]-x[0:nx-1]
    dxe = np.append(dx,dx[0]) # grid is uniform in x
    dy = y[1:ny]-y[0:ny-1]
    yedges = np.append(0.,y[0:ny-1]+dy/2.)
    ytop = y[ny-1]+dy[ny-2]/2.
    yedges = np.append(yedges,ytop)
    nedges = len(yedges)
    dye = yedges[1:nedges]-yedges[0:nedges-1]
    DXe,DYe = np.meshgrid(dxe,dye)
    area = DXe*DYe
    area = (area).flatten('F')
    area = area/np.sum(area)

    # calculate the truth!
    start_total_time = time.time()
    balance_combinations = generate_balances(nf)
    ng = int(N*N)
    closure = np.zeros([ng])
    score = np.zeros([ng])
    balance = np.zeros([ng,nf])
    ngb = np.shape(balance_combinations)[0] # number of combinations
    print('\n  Number of possible balances = ',ngb)
    print('  np.amax(np.abs(features)),np.amin(np.abs(features)) = ',np.amax(np.abs(features)),np.amin(np.abs(features)))

    for nn in range(0,ng): # loop over grid points
        score_mm = np.zeros([ngb])
        closure_mm = np.zeros([ngb])
        for mm in range(0,ngb): # loop over all possible balances for the cluster
            score_mm[mm], closure_mm[mm] = get_m_score( balance_combinations[mm,:] , features[nn,:] )
        score[nn] = np.amax(score_mm) # maximum score for the active term choice for a cluster
        #loc_max = np.argwhere(score_mm==score[nn])[:,0]
        loc_max = (np.argwhere(score_mm==np.amax(score[nn]))[:,0])[0]
        balance[nn,:] = balance_combinations[loc_max,:]
        closure[nn] = closure_mm[loc_max]
        #print('  np.amax(score_mm) = ',np.amax(score_mm))

    #print('  balance = ',balance)
    #cluster_balance = np.unique(balance,axis=0)
    #print('  cluster_balance = ',cluster_balance)
    #nc = np.shape(cluster_balance)[0]
    #print('  nc = ',nc)
    #cluster_labels = np.zeros([ng])

    # agglomerate:
    cluster_balance, cluster_labels, label_matching_index = merge_clusters( balance , np.arange(ng) )
    merged_scores, merged_closures, merged_area_weights = merge_scores( score, closure, area, np.unique(cluster_labels), label_matching_index )
    total_time = time.time() - start_total_time


    print('  np.sum(label_matching_index-cluster_labels) = ',np.sum(label_matching_index-cluster_labels))
    print('  np.shape(cluster_balance) = ',np.shape(cluster_balance))
    print('  np.unique(cluster_labels) = ',np.unique(cluster_labels))
    print('  np.shape(cluster_labels) = ',np.shape(cluster_labels))
    print('  np.shape(merged_scores) = ',np.shape(merged_scores))
    print('  merged_areas = ',merged_area_weights)
    print('  global score = ',np.sum(merged_area_weights * merged_scores))

    score_file_name =  features_path + 'no_cluster_scores.npy'
    np.save(score_file_name, merged_scores)
    closure_file_name =  features_path + 'no_cluster_area_weights.npy'
    np.save(closure_file_name, merged_area_weights)
    balance_file_name =  features_path + 'no_cluster_balance.npy'
    np.save(balance_file_name, cluster_balance)
    labels_file_name =  features_path + 'no_cluster_labels.npy'
    np.save(labels_file_name, cluster_labels)
    file_name =  features_path + 'no_cluster_time.npy'
    np.save(file_name, total_time)


elif nf == 9:
    balance1 = np.array([0.,0.,0.,0.,1.,1.,1.,1.,0.])
    score_max, closure_max = get_m_score( balance1 , feature_mean )
    print('\n  balance 1 = ',balance1)
    print('  score_max 1 = ',score_max)
    balance2 = np.array([0.,0.,0.,0.,0.,0.,1.,1.,0.])
    score_max, closure_max = get_m_score( balance2 , feature_mean )
    print('\n  balance 2 = ',balance2)
    print('  score_max 2 = ',score_max)
    balance3 = np.array([0.,0.,0.,0.,1.,1.,0.,0.,0.])
    score_max, closure_max = get_m_score( balance3 , feature_mean )
    print('\n  balance 3 = ',balance3)
    print('  score_max 3 = ',score_max)
    balance4 = np.array([1.,1.,1.,1.,0.,0.,0.,0.,0.])
    score_max, closure_max = get_m_score( balance4 , feature_mean )
    print('\n  balance 4 = ',balance4)
    print('  score_max 4 = ',score_max)
    balance5 = np.array([1.,1.,1.,1.,1.,1.,1.,1.,0.])
    score_max, closure_max = get_m_score( balance5 , feature_mean )
    print('\n  balance 5 = ',balance5)
    print('  score_max 5 = ',score_max)
    balance6 = np.array([1.,1.,1.,1.,1.,1.,1.,1.,1.])
    score_max, closure_max = get_m_score( balance6 , feature_mean )
    print('\n  balance 6 = ',balance6)
    print('  score_max 6 = ',score_max)

    features0 = np.array([1e-10,1e-10,1e10,5e10,5e10])
    balance0 = np.array([0.,0.,1.,1.,1.])
    score_max, closure_max = get_m_score( balance0 , features0 )
    print('\n  balance = ',balance0)
    print('  features test 0 = ',features0)
    print('  score_max test 0 = ',score_max)

    balance1 = np.array([0.,0.,0.,1.,1.])
    score_max, closure_max = get_m_score( balance1 , features0 )
    print('\n  balance = ',balance1)
    print('  features test 0 = ',features0)
    print('  score_max test 0 = ',score_max)

    features1 = np.array([1e-10,1e-10,4.9e10,5e10,5e10])
    balance1 = np.array([0.,0.,1.,1.,1.])
    score_max, closure_max = get_m_score( balance1 , features1 )
    print('\n  balance = ',balance1)
    print('  features test 1 = ',features1)
    print('  score_max test 1 = ',score_max)

    balance2 = np.array([0.,0.,0.,1.,1.])
    score_max, closure_max = get_m_score( balance2 , features1 )
    print('\n  balance = ',balance2)
    print('  features test 1 = ',features1)
    print('  score_max test 1 = ',score_max)


    # area
    dx = x[1]-x[0]; dy = y[1:]-y[:-1]
    nx = len(x); ny = len(y)
    # data is cell centered
    dx = x[1:nx]-x[0:nx-1]
    dxe = np.append(dx,dx[0]) # grid is uniform in x
    dy = y[1:ny]-y[0:ny-1]
    yedges = np.append(0.,y[0:ny-1]+dy/2.)
    ytop = y[ny-1]+dy[ny-2]/2.
    yedges = np.append(yedges,ytop)
    nedges = len(yedges)
    dye = yedges[1:nedges]-yedges[0:nedges-1]
    DXe,DYe = np.meshgrid(dxe,dye)
    area = DXe*DYe
    area = (area).flatten('F')
    area = area/np.sum(area)


    start_total_time = time.time()
    #  the brute force calculation
    balance_combinations = generate_balances(nf)
    ng = int(N*N)
    closure = np.zeros([ng])
    score = np.zeros([ng])
    balance = np.zeros([ng,nf])
    ngb = np.shape(balance_combinations)[0] # number of combinations
    #print('\n  Number of possible balances = ',ngb)
    #print('  np.amax(np.abs(features)),np.amin(np.abs(features)) = ',np.amax(np.abs(features)),np.amin(np.abs(features)))

    for nn in range(0,ng): # loop over grid points
        score_mm = np.zeros([ngb])
        closure_mm = np.zeros([ngb])
        for mm in range(0,ngb): # loop over all possible balances for the cluster
            score_mm[mm], closure_mm[mm] = get_m_score( balance_combinations[mm,:] , features[nn,:] )
        score[nn] = np.amax(score_mm) # maximum score for the active term choice for a cluster
        #loc_max = np.argwhere(score_mm==score[nn])[:,0]
        loc_max = (np.argwhere(score_mm==np.amax(score[nn]))[:,0])[0]
        balance[nn,:] = balance_combinations[loc_max,:]
        closure[nn] = closure_mm[loc_max]
        #print('  np.amax(score_mm) = ',np.amax(score_mm))



    # agglomerate:
    cluster_balance, cluster_labels, label_matching_index = merge_clusters( balance , np.arange(ng) )
    merged_scores, merged_closures, merged_area_weights = merge_scores( score, closure, area, np.unique(cluster_labels), label_matching_index )
    total_time = time.time() - start_total_time

    print('  np.sum(label_matching_index-cluster_labels) = ',np.sum(label_matching_index-cluster_labels))
    print('  np.shape(cluster_balance) = ',np.shape(cluster_balance))
    print('  np.unique(cluster_labels) = ',np.unique(cluster_labels))
    print('  np.shape(cluster_labels) = ',np.shape(cluster_labels))
    print('  np.shape(merged_scores) = ',np.shape(merged_scores))
    print('  merged_areas = ',merged_area_weights)
    print('  global score = ',np.sum(merged_area_weights * merged_scores))

    score_file_name =  features_path + 'no_cluster_scores.npy'
    np.save(score_file_name, merged_scores)
    closure_file_name =  features_path + 'no_cluster_area_weights.npy'
    np.save(closure_file_name, merged_area_weights)
    balance_file_name =  features_path + 'no_cluster_balance.npy'
    np.save(balance_file_name, cluster_balance)
    labels_file_name =  features_path + 'no_cluster_labels.npy'
    np.save(labels_file_name, cluster_labels)
    file_name =  features_path + 'no_cluster_time.npy'
    np.save(file_name, total_time)
"""
