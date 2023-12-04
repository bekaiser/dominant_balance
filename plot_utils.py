
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rcParams.update({'font.size': 15})
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]


import numpy as np
import math as ma
import random
import h5py

from scipy import sparse, linalg
from scipy.optimize import curve_fit, root
from scipy.integrate import odeint
from scipy.interpolate import interp1d

from numpy.random import randint
import sklearn as sk
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import SparsePCA

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

from itertools import combinations

from matplotlib import ticker
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import colors as mcolors

from scipy import stats

import easydict

from utils import merge_clusters, get_scores, get_max_scores_labels, combine_clusters

#===============================================================================

def get_colors():

    set_colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
    #print(mcolors.to_rgb(set_colors['goldenrod']))
    colors = [mcolors.to_rgb(set_colors['gray']), mcolors.to_rgb(set_colors['lightsteelblue']),
         mcolors.to_rgb(set_colors['royalblue']), mcolors.to_rgb(set_colors['blue']),
         mcolors.to_rgb(set_colors['mediumblue']), mcolors.to_rgb(set_colors['darkblue']),
         mcolors.to_rgb(set_colors['midnightblue']), mcolors.to_rgb(set_colors['black']),
         mcolors.to_rgb(set_colors['indigo']), mcolors.to_rgb(set_colors['rebeccapurple']),
         mcolors.to_rgb(set_colors['darkviolet']), mcolors.to_rgb(set_colors['mediumorchid']),
         mcolors.to_rgb(set_colors['orchid']), mcolors.to_rgb(set_colors['plum']),
         mcolors.to_rgb(set_colors['thistle']), mcolors.to_rgb(set_colors['lavenderblush'])]
    n_bins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]  # Discretizes the interpolation into bins
    #cmap_name = 'my_list'
    #cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=len(n_bins))

    return colors


def reduced_balance_plot( input ):

    if input.cluster_method == 'GMM':
        nc = input.j
        i = input.i
    nx = input.nx
    ny = input.ny
    x = input.x
    y = input.y
    score_max = input.score_max
    closure_max = input.closure_max
    reduced_balance = input.reduced_balance
    labels = input.reduced_labels
    Kr = len(np.unique(labels))
    eqn_labels = input.eqn_labels
    colors = get_colors()

    # Plot:
    xc = np.linspace(1,np.shape(reduced_balance)[1],num=np.shape(reduced_balance)[1],endpoint=True)
    yc = np.linspace(0,np.shape(reduced_balance)[0]-1,num=np.shape(reduced_balance)[0],endpoint=True)
    Xc,Y = np.meshgrid(xc,yc)
    ind0 = 0
    for jj in range(0,np.shape(reduced_balance)[0]):
        reduced_balance[jj,:] = reduced_balance[jj,:]*(yc[jj]+1)
        if sum(reduced_balance[jj,:]) == 0.:
            ind0 = jj

    #print('reduced_balance = ',reduced_balance)
    titlename = r'GMM, K = %i, OMS score = %.3f, closure score = %.3f' %(nc,np.nanmean(score_max),np.nanmean(closure_max))

    cmap_nameSC = 'my_blues'
    cmSC = LinearSegmentedColormap.from_list(cmap_nameSC, colors[0:Kr], N=Kr)
    figure_name = 'reduced_balance_K%i_n%i.png' %(nc,i)
    plotname = input.figure_path + figure_name
    fig = plt.figure(figsize=(15, 3))
    plt.subplot(1,2,1)
    #print('here1')
    labelmap = np.reshape(labels, [ny, nx], order='F')
    #print('here2')
    cs = plt.pcolor(x, y, labelmap, cmap=cmSC, vmin=-0.5, vmax=(Kr-0.5), alpha=1, edgecolors='face')
    plt.xlabel('$x$',fontsize=18)
    plt.ylabel('$y$',fontsize=18)
    plt.title(titlename,fontsize=16)
    cbar=plt.colorbar(cs)
    cbar.ax.get_yaxis().labelpad = 20
    cbar.set_label(r'cluster', rotation=270,fontsize=16)
    #plt.clim(0.5,max_cluster+0.5)
    ax=plt.subplot(1,2,2)
    for yy in range(0,Kr):
        plt.scatter(Xc[yy,:],reduced_balance[yy,:]-1,marker='s',s=800,color=colors[yy]) #'black')
    #plt.scatter(Xc[ind0,:],balance_models[ind0,:],marker='s',s=500,color='white')
    plt.xticks([1,2,3,4,5,6],eqn_labels)
    ax.tick_params(axis = 'both', which = 'major', labelsize = 18)
    plt.ylabel(r'cluster',fontsize=16)
    if np.shape(reduced_balance)[0] == 3:
        plt.yticks([0,1,2],[r'$0$',r'$1$',r'$2$'])
    if np.shape(reduced_balance)[0] == 4:
        plt.yticks([0,1,2,3],[r'$0$',r'$1$',r'$2$',r'$3$'])
    if np.shape(reduced_balance)[0] == 5:
        plt.yticks([0,1,2,3,4],[r'$0$',r'$1$',r'$2$',r'$3$',r'$4$'])
    if np.shape(reduced_balance)[0] == 6:
        plt.yticks([0,1,2,3,4,5],[r'$0$',r'$1$',r'$2$',r'$3$',r'$4$',r'$5$'])
    if np.shape(reduced_balance)[0] == 7:
        plt.yticks([0,1,2,3,4,5,6],[r'$0$',r'$1$',r'$2$',r'$3$',r'$4$',r'$5$',r'$6$'])
    #plt.yticks(yc)
    #plt.xlim([0.5,ncb+0.5])
    plt.title(titlename,fontsize=16)
    plt.axis([0.5,np.shape(reduced_balance)[1]+0.5,-0.5,np.shape(reduced_balance)[0]-0.5])
    plt.subplots_adjust(top=0.9, bottom=0.175, left=0.05, right=0.985, hspace=0.15, wspace=0.1)
    plt.savefig(plotname,format="png"); plt.close(fig);

    return

def plot_processed_data( data , plot_specifics , metrics_flag, data_set_name ):

    NE = int(data.NE)

    eqn_labels = data.eqn_labels
    mean_score = data.mean_score
    std_score = data.std_score
    mean_closure = data.mean_closure
    std_closure = data.std_closure
    features = data.features
    reduced_balance = data.balance
    for jj in range(0,np.shape(reduced_balance)[0]):
        for kk in range(0,np.shape(reduced_balance)[1]):
            if reduced_balance[jj,kk] == 0.:
                reduced_balance[jj,kk] = np.nan
    labels = data.labels
    x = data.x; y = data.y; nx = len(x); ny = len(y)
    nf = data.nf
    nc = data.nc # the reduced number of clusters

    # make meshgrid
    # get nan_locs
    # get feature_locs

    max_area_weights = data.max_area_weights
    max_scores = data.max_scores
    max_closure = data.max_closure
    max_scores_labels = get_max_scores_labels( max_scores )

    feature_mean = data.feature_mean
    feature_mean_95 = data.feature_mean_95

    mycolors = data.mycolors
    mymarkers = data.mymarkers
    mymarkersize = data.mymarkersize


    if data.cluster_method == 'GMM' or data.cluster_method == 'KMeans':

        if data.reduction_method == 'SPCA':

            Kr = data.Kr; K = data.K; alphas = data.alphas
            Kx,Ay = np.meshgrid(K,alphas)
            nk = data.nk; ne = data.ne; na = data.na
            nc = data.nc; Xc = data.Xc

            #if (data.read_path).split('_')[1] == 'GMM':
            #    figure_name = 'metrics_GMM_SPCA_3.png'
            #elif (data.read_path).split('_')[1] == 'KMeans':
            #    figure_name = 'metrics_GMM_SPCA_3.png'
            figure_name = 'metrics_' + (data.read_path).split('_')[1] + '_SPCA_3.png'
            plotname = data.figure_path + figure_name
            #fig = plt.figure(figsize=(12.5, 11)) # for 2x2
            fig = plt.figure(figsize=(17, 5)) # for 3x2
            ax=plt.subplot(1,3,1) #2,2,1)
            #CS=plt.contour(Kx,Ay, np.transpose(mean_score[:,ne,:]), levels=[(mean_score[nk,ne,na]-std_score[nk,ne,na])],colors=('royalblue',),linestyles=('--',),linewidths=(2,)) #, cmap='gist_yarg') #vmin=-0.5, vmax=cm.N-0.5 cmap=cm)
            CS=plt.contour(Kx,Ay, np.transpose(mean_score[:,ne,:]), levels=[nf/(2.*(nf-1))-0.00001],colors=('royalblue',),linestyles=('-',),linewidths=(2,)) #, cmap='gist_yarg') #vmin=-0.5, vmax=cm.N-0.5 cmap=cm)
            CSF=plt.contourf(Kx,Ay, np.transpose(mean_score[:,ne,:]), 100, cmap='gist_yarg') #vmin=-0.5, vmax=cm.N-0.5 cmap=cm)
            #max_label = r'$%.4f$, $K=%i,\alpha=%.4f$' %(mean_score[nk,ne,na],Kx[na,nk],Ay[na,nk])
            max_label = r'$%.4f$, $K=%i,\alpha=%.4f$' %(np.transpose(mean_score[:,ne,:])[na,nk],Kx[na,nk],Ay[na,nk])
            plt.plot(Kx[na,nk],Ay[na,nk],color='crimson',linestyle='None',marker='*',markersize=12,label=max_label)
            cbar=plt.colorbar(CSF) #boundaries=np.arange(-0.5, ncb+1.5), ticks=np.arange(0, ncb+1))
            cbar.ax.get_yaxis().labelpad = 20
            cbar.add_lines(CS) #plt.colorbar(CS,colors='royalblue')
            plt.xticks([5,10,15,20],[r"$5$",r"$10$",r"$15$",r"$20$"])
            plt.yscale('log')
            #plt.yticks([1e-5,1e-4,1e-3,1e-2,1e-1,1e-0,1e1,1e2,1e3],[r"$10^{-5}$",r"$10^{-4}$",r"$10^{-3}$",r"$10^{-2}$",r"$10^{-1}$",r"$10^{0}$",r"$10^{1}$",r"$10^{2}$",r"$10^{3}$"])
            plt.yticks([1e-3,1e-2,1e-1,1e-0,1e1,1e2,1e3,1e4],[r"$10^{-3}$",r"$10^{-2}$",r"$10^{-1}$",r"$10^{0}$",r"$10^{1}$",r"$10^{2}$",r"$10^{3}$",r"$10^{4}$"])
            plt.xlabel(r'$K$',fontsize=18)
            plt.ylabel(r'$\alpha$',fontsize=20)
            plt.title(r'area-weighted cluster mean $M$ score',fontsize=14)
            plt.legend(loc=4,framealpha=1.)

            ax=plt.subplot(1,3,2) #2,2,3)
            plt.contourf(Kx,Ay, np.transpose(mean_closure[:,ne,:]), 100, cmap='gist_yarg') #vmin=-0.5, vmax=cm.N-0.5 cmap=cm)
            cbar=plt.colorbar() #boundaries=np.arange(-0.5, ncb+1.5), ticks=np.arange(0, ncb+1))
            cbar.ax.get_yaxis().labelpad = 20
            max_label = r'$%.4f$' %(mean_closure[nk,ne,na])
            plt.plot(Kx[na,nk],Ay[na,nk],color='crimson',linestyle='None',marker='*',markersize=12,label=max_label)
            plt.xticks([5,10,15,20],[r"$5$",r"$10$",r"$15$",r"$20$"])
            plt.yscale('log')
            #plt.yticks([1e-5,1e-4,1e-3,1e-2,1e-1,1e-0,1e1,1e2,1e3],[r"$10^{-5}$",r"$10^{-4}$",r"$10^{-3}$",r"$10^{-2}$",r"$10^{-1}$",r"$10^{0}$",r"$10^{1}$",r"$10^{2}$",r"$10^{3}$"])
            plt.yticks([1e-3,1e-2,1e-1,1e-0,1e1,1e2,1e3,1e4],[r"$10^{-3}$",r"$10^{-2}$",r"$10^{-1}$",r"$10^{0}$",r"$10^{1}$",r"$10^{2}$",r"$10^{3}$",r"$10^{4}$"])
            plt.xlabel(r'$K$',fontsize=18)
            #plt.ylabel(r'$\alpha$',fontsize=20)
            plt.title(r'area-weighted cluster mean closure score',fontsize=14)
            plt.legend(loc=4,framealpha=1.)

            ax=plt.subplot(1,3,3) #2,2,4)
            plt.contourf(Kx,Ay, np.transpose(Kr[:,ne,:]), 100, cmap='gist_yarg') #vmin=-0.5, vmax=cm.N-0.5 cmap=cm)
            cbar=plt.colorbar() #boundaries=np.arange(-0.5, ncb+1.5), ticks=np.arange(0, ncb+1))
            cbar.ax.get_yaxis().labelpad = 20
            max_label = r'$%i$' %(Kr[nk,ne,na])
            plt.plot(Kx[na,nk],Ay[na,nk],color='crimson',linestyle='None',marker='*',markersize=12,label=max_label)
            plt.xticks([5,10,15,20],[r"$5$",r"$10$",r"$15$",r"$20$"])
            plt.yscale('log')
            #plt.yticks([1e-5,1e-4,1e-3,1e-2,1e-1,1e-0,1e1,1e2,1e3],[r"$10^{-5}$",r"$10^{-4}$",r"$10^{-3}$",r"$10^{-2}$",r"$10^{-1}$",r"$10^{0}$",r"$10^{1}$",r"$10^{2}$",r"$10^{3}$"])
            plt.yticks([1e-3,1e-2,1e-1,1e-0,1e1,1e2,1e3,1e4],[r"$10^{-3}$",r"$10^{-2}$",r"$10^{-1}$",r"$10^{0}$",r"$10^{1}$",r"$10^{2}$",r"$10^{3}$",r"$10^{4}$"])
            plt.xlabel(r'$K$',fontsize=18)
            plt.ylabel(r'$\alpha$',fontsize=20)
            plt.title(r'number of refined clusters',fontsize=14)
            cbar.set_ticklabels(['1','2','3','4','5','6','7','8','9','10','11'])
            plt.legend(loc=4,framealpha=1.)

            plt.subplots_adjust(top=0.95, bottom=0.15, left=0.075, right=0.975, hspace=0.3, wspace=0.3)
            plt.savefig(plotname,format="png"); plt.close(fig);


            #figure_name = 'clusters_GMM_SPCA.png'
            figure_name = 'metrics_' + data.cluster_method + '_SPCA.png'
            plotname = data.figure_path + figure_name
            fig = plt.figure(figsize=(12,10)) #7, 5)) # for 3x2
            #ax=plt.subplot(2,2,1)
            ax1 = plt.subplot2grid((2, 2), (0, 0))
            for jj in range(0,np.shape(feature_mean)[0]):
                #print('np.shape(feature_mean[jj,:]) = ',np.shape(feature_mean[jj,:]))
                #print('np.shape(mycolors[jj,:]) = ',np.shape(mycolors[jj,:]))
                #print('np.shape(mymarkers[jj]) = ',np.shape(mymarkers[jj]))
                #print('np.shape(mymarkersize[jj]) = ',np.shape(mymarkersize[jj]))
                plt.plot(np.arange(nf),np.abs(feature_mean[jj,:]),color=mycolors[jj,:],linestyle='None',marker=mymarkers[jj],markersize=mymarkersize[jj],label=r"cluster %i" %jj) #,color=(data.colors)[jj,:]) #,linestyle='None',marker='*',markersize=12,label=max_label)
                plt.fill_between(np.arange(nf),np.abs(feature_mean[jj,:]-feature_mean_95[jj,:]),np.abs(feature_mean[jj,:]+feature_mean_95[jj,:]),alpha=0.2,color=mycolors[jj,:])
                #plt.errorbar(np.arange(nf),np.abs(feature_mean[jj,:]),yerr=feature_mean_95[jj,:])
                plt.yscale('log')
            plt.legend(loc=3,framealpha=1.,fontsize=14)
            plt.title(r'cluster mean features',fontsize=16)
            plt.ylabel(r'absolute value',fontsize=16)
            xtick_labels = [r'$\bar{u} \bar{u}_x$', r'$\bar{v}\bar{u}_y$', r'$\rho^{-1} \bar{p}_x$',
                r'$\nu \nabla^2 \bar{u}$', r'$\overline{(u^\prime v^\prime)}_y$', r'$\overline{({u^\prime} ^2)}_x$']
            plt.xticks([0,1,2,3,4,5],xtick_labels,fontsize=18)
            #plt.grid()
            plt.ylim([3e-6,6e0])

            #ax = plt.axes([0.5, 1., 0.5, 1.])
            #ax = plt.subplot(2,2,2)
            ax2 = plt.subplot2grid((2, 2), (0, 1))
            for yy in range(0,nc):
                plt.scatter(Xc[yy,:],reduced_balance[yy,:]-1,marker=mymarkers[yy],s=800,color=data.colors[yy]) #'black')
            #plt.scatter(Xc[ind0,:],balance_models[ind0,:],marker='s',s=500,color='white')
            xticks_labels = [r'$\bar{u} \bar{u}_x$', r'$\bar{v}\bar{u}_y$', r'$\rho^{-1} \bar{p}_x$',
                        r'$\nu \nabla^2 \bar{u}$', r'$\overline{(u^\prime v^\prime)}_y$', r'$\overline{({u^\prime} ^2)}_x$']
            plt.xticks([1,2,3,4,5,6],xticks_labels)
            ax2.tick_params(axis = 'both', which = 'major', labelsize = 18)
            plt.ylabel(r'cluster',fontsize=18)
            if np.shape(reduced_balance)[0] == 3:
                plt.yticks([0,1,2],[r'$0$',r'$1$',r'$2$'])
            if np.shape(reduced_balance)[0] == 4:
                plt.yticks([0,1,2,3],[r'$0$',r'$1$',r'$2$',r'$3$'])
            if np.shape(reduced_balance)[0] == 5:
                plt.yticks([0,1,2,3,4],[r'$0$',r'$1$',r'$2$',r'$3$',r'$4$'])
            if np.shape(reduced_balance)[0] == 6:
                plt.yticks([0,1,2,3,4,5],[r'$0$',r'$1$',r'$2$',r'$3$',r'$4$',r'$5$'])
            if np.shape(reduced_balance)[0] == 7:
                plt.yticks([0,1,2,3,4,5,6],[r'$0$',r'$1$',r'$2$',r'$3$',r'$4$',r'$5$',r'$6$'])
            plt.title(r'active terms',fontsize=16)
            plt.axis([0.5,np.shape(reduced_balance)[1]+0.5,-0.5,np.shape(reduced_balance)[0]-0.5])
            ax2b=ax2.twinx()
            ax2b.set_yticks(np.arange(nc)) # FIX <<<<---------------------------------------!
            ax2b.set_yticklabels(max_scores_labels,fontsize=18)
            ax2b.set_ylim([-0.5,nc-0.5]) # FIX <<<<---------------------------------------!
            ax2b.set_ylabel(r'$M$ score',fontsize=18,rotation=270,labelpad=18)

            #ax = plt.axes([0, 1.25, 0, 0.5])
            #ax=plt.subplot(2,1,2)
            ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2)
            ax3.set_anchor('S')
            #titlename = r'$K$ = %i, $\alpha$ = %.3f' %(int(K[nk]),alphas[na])
            titlename = r'area-weighted cluster mean: closure score = %.3f, $M$ score = %.3f' %(np.sum(max_closure*max_area_weights),np.sum(max_scores*max_area_weights))
            cmap_nameSC = 'my_blues'
            cmSC = LinearSegmentedColormap.from_list(cmap_nameSC, data.colors[0:nc], N=nc)
            #print('np.amax(x),np.amin(x),x[10]-x[9] = ',np.amax(x),np.amin(x),x[10]-x[9])
            #print('np.amax(y),np.amin(y),y[10]-y[9] = ',np.amax(y),np.amin(y),y[10]-y[9])
            labelmap = np.reshape(labels, [ny, nx], order='F')
            cs = plt.pcolor(x, y, labelmap, cmap=cmSC, vmin=-0.5, vmax=(nc-0.5), alpha=1, edgecolors='face')
            #nu = 1.25e-3; U = 1. # d(x) = 0.37*x/Rex**(1/5) (Schlicting book)
            #plt.plot(x,0.37*(x-0.)**(4./5.)*(nu/U)**(-1./5.),color='k',linewidth=2,linestyle='dashed',label=r'$\delta\sim{}x^{4/5}$')
            #plt.legend(loc=1,framealpha=1.)
            plt.xlabel('$x$',fontsize=18)
            plt.ylabel('$y$',fontsize=18)
            plt.title(titlename,fontsize=16)
            cbar=plt.colorbar(cs)
            cbar.ax.get_yaxis().labelpad = 20
            if np.shape(reduced_balance)[0] == 2:
                cbar.set_ticks([0.,1.],['0','1'])
            elif np.shape(reduced_balance)[0] == 3:
                cbar.set_ticks([0.,1.,2.],['0','1','2'])
            elif np.shape(reduced_balance)[0] == 4:
                cbar.set_ticks([0.,1.,2.,3.],['0','1','2','3'])
            elif np.shape(reduced_balance)[0] == 5:
                cbar.set_ticks([0.,1.,2.,3.,4.],['0','1','2','3','4'])
            elif np.shape(reduced_balance)[0] == 6:
                cbar.set_ticks([0.,1.,2.,3.,4.,5.],['0','1','2','3','4','5'])
            elif np.shape(reduced_balance)[0] == 7:
                cbar.set_ticks([0.,1.,2.,3.,4.,5.,6.],['0','1','2','3','4','5','6'])
            plt.ylim([0.,np.amax(y)])
            #ax3.set_aspect(0.5) #,anchor='C') #'equal')


            plt.tight_layout()
            plt.subplots_adjust(top=0.95, bottom=0.1, left=0.1, right=0.875, hspace=0.3, wspace=0.2)
            plt.savefig(plotname,format="png"); plt.close(fig);



        elif data.reduction_method == 'Mscore':

            #max_area_weights = data.max_area_weights
            nc = data.nc; Xc = data.Xc
            #max_scores = data.max_scores
            #max_closure = data.max_closure
            #print(max_scores)

            if metrics_flag == True:

                nk = data.nk; ne = data.ne
                Kr = data.Kr; K = data.K

                figure_name = 'metrics_' + data.cluster_method + '_Mscore_1.png'
                plotname = data.figure_path + figure_name
                fig = plt.figure(figsize=(6.5, 5))
                ax=plt.subplot(1,1,1)
                for i in range(0,NE):
                    plt.plot(K[:], mean_score[:,i], color='royalblue',alpha=0.1,linewidth=2)
                max_label = r'$%.4f$, $K=%i$' %(mean_score[nk,ne],K[nk])
                plt.plot(K[nk], mean_score[nk,ne], color='crimson',linestyle='None',marker='*',markersize=12,label=max_label)
                #textstr = r'full set score: $N_f/(2(N_f-1)) = %.2f$' %(nf/(2*(nf-1)))
                #textstr = '\n'.join((r'cluster 0: ' + reduced_labels[0],
                #                    r'cluster 1: ' + reduced_labels[1],
                #                    r'cluster 2: ' + reduced_labels[2]))
                #props = dict(boxstyle='round', facecolor='white', alpha=1)
                # place a text box in upper left in axes coords
                #ax.text(0.15, 0.35, textstr, transform=ax.transAxes, fontsize=14,verticalalignment='top', bbox=props)
                #plt.xticks([4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25],[r"$4$",r"$5$",r"$6$",r"$7$",r"$8$",r"$9$",r"$10$",r"$11$",r"$12$",r"$13$",r"$14$",r"$15$",r"$16$",r"$17$",r"$18$",r"$19$",r"$20$",r"$21$",r"$22$",r"$23$",r"$24$",r"$25$"])
                #plt.xticks([5,10,15,20,25],[r"$5$",r"$10$",r"$15$",r"$20$",r"$25$"])
                plt.xticks([5,10,15,20],[r"$5$",r"$10$",r"$15$",r"$20$"])
                plt.xlabel(r'$K$, number of clusters',fontsize=18)
                plt.xlim([3.,20.])
                plt.ylabel(r'area-weighted cluster mean magnitude score',fontsize=15)
                plt.legend(loc=4)

                plt.subplots_adjust(top=0.95, bottom=0.15, left=0.15, right=0.975, hspace=0.3, wspace=0.3)
                plt.savefig(plotname,format="png"); plt.close(fig);


                figure_name = 'metrics_' + data.cluster_method + '_Mscore_3.png'
                plotname = data.figure_path + figure_name
                fig = plt.figure(figsize=(17, 5))
                ax=plt.subplot(1,3,1)
                for i in range(0,NE):
                    plt.plot(K[:], mean_score[:,i], color='royalblue',alpha=0.1,linewidth=2)
                max_label = r'$%.4f$, $K=%i$' %(mean_score[nk,ne],K[nk])
                plt.plot(K[nk], mean_score[nk,ne], color='crimson',linestyle='None',marker='*',markersize=12,label=max_label)
                #textstr = r'full set score: $N_f/(2(N_f-1)) = %.2f$' %(nf/(2*(nf-1)))
                #props = dict(boxstyle='round', facecolor='white', alpha=1)
                ## place a text box in upper left in axes coords
                #ax.text(0.15, 0.35, textstr, transform=ax.transAxes, fontsize=14,verticalalignment='top', bbox=props)
                plt.xlabel(r'$K$',fontsize=18)
                #plt.xlim([3.,20.])
                plt.ylabel(r'area-weighted global magnitude score',fontsize=14)
                plt.legend(loc=4)

                plt.subplot(1,3,2)
                for i in range(0,NE):
                    plt.plot(K[:], mean_closure[:,i], color='royalblue',alpha=0.1,linewidth=2)
                plt.plot(K[nk], mean_closure[nk,ne], color='crimson',linestyle='None',marker='*',markersize=12,label=max_label)
                #plt.xticks([4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25],[r"$4$",r"$5$",r"$6$",r"$7$",r"$8$",r"$9$",r"$10$",r"$11$",r"$12$",r"$13$",r"$14$",r"$15$",r"$16$",r"$17$",r"$18$",r"$19$",r"$20$",r"$21$",r"$22$",r"$23$",r"$24$",r"$25$"])
                #plt.xticks([5,10,15,20,25],[r"$5$",r"$10$",r"$15$",r"$20$",r"$25$"])
                #plt.xticks([5,10,15,20],[r"$5$",r"$10$",r"$15$",r"$20$"])
                plt.xlabel(r'$K$',fontsize=18)
                #plt.xlim([3.,20.])
                plt.title(r'area-weighted cluster mean closure score',fontsize=14)

                plt.subplot(1,3,3)
                for i in range(0,NE):
                    plt.plot(K[:], Kr[:,i], color='royalblue',alpha=0.1,linewidth=2)
                plt.plot(K[nk], Kr[nk,ne], color='crimson',linestyle='None',marker='*',markersize=12,label=max_label)
                #plt.xticks([4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25],[r"$4$",r"$5$",r"$6$",r"$7$",r"$8$",r"$9$",r"$10$",r"$11$",r"$12$",r"$13$",r"$14$",r"$15$",r"$16$",r"$17$",r"$18$",r"$19$",r"$20$",r"$21$",r"$22$",r"$23$",r"$24$",r"$25$"])
                #plt.xticks([5,10,15,20,25],[r"$5$",r"$10$",r"$15$",r"$20$",r"$25$"])
                #plt.xticks([5,10,15,20],[r"$5$",r"$10$",r"$15$",r"$20$"])
                plt.xlabel(r'$K$',fontsize=18)
                #plt.xlim([3.,20.])
                plt.ylabel(r'unique clusters',fontsize=14)

                plt.subplots_adjust(top=0.95, bottom=0.15, left=0.075, right=0.975, hspace=0.3, wspace=0.3)
                plt.savefig(plotname,format="png"); plt.close(fig);

            #print('\neqn_labels = ',eqn_labels)


            """
            #figure_name = 'clusters_GMM_Mscore.png'
            figure_name = 'metrics_' + data.cluster_method + '_Mscore' + plot_specifics + '.png'
            plotname = data.figure_path + figure_name
            #fig = plt.figure(figsize=(12,10)) #7, 5)) # for 3x2
            fig = plt.figure(figsize=(20,10)) #7, 5)) # for 3x2
            #ax=plt.subplot(2,2,1)
            ax1 = plt.subplot2grid((2, 2), (0, 0))
            print('np.shape(feature_mean) = ',np.shape(feature_mean))
            print('np.shape(mymarkers) = ',np.shape(mymarkers))
            for jj in range(0,np.shape(feature_mean)[0]):
                print('jj = ',jj)
                print('np.shape(feature_mean[jj,:]) = ',np.shape(feature_mean[jj,:]))
                print('np.shape(mymarkers[jj]) = ',np.shape(mymarkers[jj]))
                print('np.shape(mymarkersize[jj]) = ',np.shape(mymarkersize[jj]))
                print('np.shape(mycolors[jj,:]) = ',np.shape(mycolors[jj,:]))
                plt.plot(np.arange(nf),np.abs(feature_mean[jj,:]),color=mycolors[jj,:],linestyle='None',marker=mymarkers[jj],markersize=mymarkersize[jj],label=r"cluster %i" %jj) #,color=(data.colors)[jj,:]) #,linestyle='None',marker='*',markersize=12,label=max_label)
                plt.fill_between(np.arange(nf),np.abs(feature_mean[jj,:]-feature_mean_95[jj,:]),np.abs(feature_mean[jj,:]+feature_mean_95[jj,:]),alpha=0.2,color=mycolors[jj,:])
                #plt.errorbar(np.arange(nf),np.abs(feature_mean[jj,:]),yerr=feature_mean_95[jj,:])
                plt.yscale('log')
            plt.legend(loc=3,framealpha=1.,fontsize=14)
            plt.title(r'cluster mean features',fontsize=16)
            plt.ylabel(r'absolute value',fontsize=16)
            plt.xlabel(r'features',fontsize=16)
            textstr = r'a'
            props = dict(boxstyle='round', facecolor='white', alpha=1)
            ax1.text(0.9, 0.15, textstr, transform=ax1.transAxes, fontsize=14,verticalalignment='top', bbox=props)
            #xtick_labels = [r'$\bar{u} \bar{u}_x$', r'$\bar{v}\bar{u}_y$', r'$\rho^{-1} \bar{p}_x$',
            #    r'$\nu \nabla^2 \bar{u}$', r'$\overline{(u^\prime v^\prime)}_y$', r'$\overline{({u^\prime} ^2)}_x$']
            #
            #xtick_labels = [r'$\overline{u} \frac{\partial \overline{u} }{\partial x}$',
            #    r'$\overline{v} \frac{\partial \overline{u} }{\partial y}$',
            #    r'$\frac{1}{\rho} \frac{\partial \overline{p} }{\partial x}$',
            #    r'$\nu\nabla^2\overline{u}$',
            #    r'$\frac{\partial \overline{u^\prime v^\prime} }{\partial y}$',
            #    r'$\frac{\partial \overline{{u^\prime}^2} }{\partial x}$']
            #plt.xticks([0,1,2,3,4,5],xtick_labels,fontsize=18)
            plt.xticks(np.arange(len(eqn_labels)),eqn_labels,fontsize=18)
            plt.xlim([-0.5,len(eqn_labels)-0.5])
            #
            #plt.grid()
            #
            #plt.ylim([3e-6,6e0])

            #ax = plt.axes([0.5, 1., 0.5, 1.])
            #ax = plt.subplot(2,2,2)
            ax2 = plt.subplot2grid((2, 2), (0, 1))
            #print('reduced_balance-1 = ',reduced_balance-1.)
            #print(Xc)
            for yy in range(0,nc):
                plt.scatter(Xc[yy,:],reduced_balance[yy,:]-1,marker=mymarkers[yy],s=800,color=data.colors[yy]) #'black')
            #plt.scatter(Xc[ind0,:],balance_models[ind0,:],marker='s',s=500,color='white')
            #xticks_labels = [r'$\bar{u} \bar{u}_x$', r'$\bar{v}\bar{u}_y$', r'$\rho^{-1} \bar{p}_x$',
            #            r'$\nu \nabla^2 \bar{u}$', r'$\overline{(u^\prime v^\prime)}_y$', r'$\overline{({u^\prime} ^2)}_x$']
            #
            #xticks_labels = [r'$\overline{u} \frac{\partial \overline{u} }{\partial x}$',
            #    r'$\overline{v} \frac{\partial \overline{u} }{\partial y}$',
            #    r'$\frac{1}{\rho} \frac{\partial \overline{p} }{\partial x}$',
            #    r'$\nu\nabla^2\overline{u}$',
            #    r'$\frac{\partial \overline{u^\prime v^\prime} }{\partial y}$',
            #    r'$\frac{\partial \overline{{u^\prime}^2} }{\partial x}$']
            #plt.xticks([1,2,3,4,5,6],xticks_labels)
            ax2.tick_params(axis = 'both', which = 'major', labelsize = 18)
            plt.ylabel(r'cluster',fontsize=18)
            plt.xlabel(r'features',fontsize=16)
            textstr = r'b'
            props = dict(boxstyle='round', facecolor='white', alpha=1)
            if data_set_name == 'tumor dynamics':
                ax2.text(0.9, 0.9, textstr, transform=ax2.transAxes, fontsize=14,verticalalignment='top', bbox=props)
            else:
                ax2.text(0.9, 0.15, textstr, transform=ax2.transAxes, fontsize=14,verticalalignment='top', bbox=props)
            if np.shape(reduced_balance)[0] == 3:
                plt.yticks([0,1,2],[r'$0$',r'$1$',r'$2$'])
            if np.shape(reduced_balance)[0] == 4:
                plt.yticks([0,1,2,3],[r'$0$',r'$1$',r'$2$',r'$3$'])
            if np.shape(reduced_balance)[0] == 5:
                plt.yticks([0,1,2,3,4],[r'$0$',r'$1$',r'$2$',r'$3$',r'$4$'])
            if np.shape(reduced_balance)[0] == 6:
                plt.yticks([0,1,2,3,4,5],[r'$0$',r'$1$',r'$2$',r'$3$',r'$4$',r'$5$'])
            if np.shape(reduced_balance)[0] == 7:
                plt.yticks([0,1,2,3,4,5,6],[r'$0$',r'$1$',r'$2$',r'$3$',r'$4$',r'$5$',r'$6$'])
            #plt.title(r'active terms',fontsize=16)
            plt.title(r'hypotheses',fontsize=16)
            plt.axis([0.5,np.shape(reduced_balance)[1]+0.5,-0.5,np.shape(reduced_balance)[0]-0.5])
            ax2b=ax2.twinx()
            ax2b.set_yticks(np.arange(nc)) # FIX <<<<---------------------------------------!
            ax2b.set_yticklabels(max_scores_labels,fontsize=18)
            ax2b.set_ylim([-0.5,nc-0.5]) # FIX <<<<---------------------------------------!
            ax2b.set_ylabel(r'magnitude score',fontsize=18,rotation=270,labelpad=18)
            #print(np.arange(len(eqn_labels)),eqn_labels)
            plt.xticks(np.arange(len(eqn_labels))+1.,eqn_labels,fontsize=18)
            plt.xlim([0.5,len(eqn_labels)+0.5])


            if data_set_name == 'tumor_invasion' or data_set_name == 'tumor_angiogenesis':

                ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=1)
                ax3.set_anchor('S')
                cmap_nameSC = 'my_blues'
                cmSC = LinearSegmentedColormap.from_list(cmap_nameSC, data.colors[0:nc], N=nc)
                labelmap = np.reshape(labels, [ny, nx], order='F')
                cs = plt.pcolor(x, y, labelmap, cmap=cmSC, vmin=-0.5, vmax=(nc-0.5), alpha=1, edgecolors='face')
                textstr = r'c'
                props = dict(boxstyle='round', facecolor='white', alpha=1)
                ax3.text(0.9, 0.9, textstr, transform=ax3.transAxes, fontsize=14,verticalalignment='top', bbox=props)
                plt.xlabel('$x$',fontsize=18)
                plt.ylabel('$y$',fontsize=18)

                #bar=plt.colorbar(cs)
                #cbar.ax.set_ylabel('cluster')
                #if np.shape(reduced_balance)[0] == 2:
                #    cbar.set_ticks([0.,1.],['0','1'])
                #elif np.shape(reduced_balance)[0] == 3:
                #    cbar.set_ticks([0.,1.,2.],['0','1','2'])
                #elif np.shape(reduced_balance)[0] == 4:
                #    cbar.set_ticks([0.,1.,2.,3.],['0','1','2','3'])
                #elif np.shape(reduced_balance)[0] == 5:
                #    cbar.set_ticks([0.,1.,2.,3.,4.],['0','1','2','3','4'])
                #elif np.shape(reduced_balance)[0] == 6:
                #    cbar.set_ticks([0.,1.,2.,3.,4.,5.],['0','1','2','3','4','5'])
                #elif np.shape(reduced_balance)[0] == 7:
                #    cbar.set_ticks([0.,1.,2.,3.,4.,5.,6.],['0','1','2','3','4','5','6'])

                plt.ylim([0.,np.amax(y)])

                ax4 = plt.subplot2grid((2, 2), (1, 1), colspan=1)
                ax4.set_anchor('S')
                cmap_nameSC = 'my_blues'
                cmSC = LinearSegmentedColormap.from_list(cmap_nameSC, data.colors[0:nc], N=nc)
                labelmap = np.reshape(labels, [ny, nx], order='F')
                cs = plt.pcolor(x, y, labelmap, cmap=cmSC, vmin=-0.5, vmax=(nc-0.5), alpha=1, edgecolors='face')
                textstr = r'd'
                props = dict(boxstyle='round', facecolor='white', alpha=1)
                ax4.text(0.7, 0.9, textstr, transform=ax4.transAxes, fontsize=14,verticalalignment='top', bbox=props)
                plt.xlabel('$x$',fontsize=18)
                plt.ylabel('$y$',fontsize=18)
                cbar=plt.colorbar(cs)
                cbar.ax.set_ylabel('cluster')
                if np.shape(reduced_balance)[0] == 2:
                    cbar.set_ticks([0.,1.],['0','1'])
                elif np.shape(reduced_balance)[0] == 3:
                    cbar.set_ticks([0.,1.,2.],['0','1','2'])
                elif np.shape(reduced_balance)[0] == 4:
                    cbar.set_ticks([0.,1.,2.,3.],['0','1','2','3'])
                elif np.shape(reduced_balance)[0] == 5:
                    cbar.set_ticks([0.,1.,2.,3.,4.],['0','1','2','3','4'])
                elif np.shape(reduced_balance)[0] == 6:
                    cbar.set_ticks([0.,1.,2.,3.,4.,5.],['0','1','2','3','4','5'])
                elif np.shape(reduced_balance)[0] == 7:
                    cbar.set_ticks([0.,1.,2.,3.,4.,5.,6.],['0','1','2','3','4','5','6'])
                plt.axis([0.275,0.325,0.075,0.125])


            else:

                print( '\ndata.read_path @ = ', data.read_path)
                print('np.amin(x),np.amax(x) = ',np.amin(x),np.amax(x))
                print('np.amin(y),np.amax(y) = ',np.amin(y),np.amax(y))
                nan_locs = np.load( data.read_path + 'nan_locs.npy' )
                feature_locs = np.load( data.read_path + 'feature_locs.npy' )
                labelmap = np.zeros([len(nan_locs)+len(feature_locs)])*np.nan
                labelmap[feature_locs] = labels
                labelmap = np.reshape(labelmap, [nx, ny], order='F') # <-----

                #area = (area).flatten('F')
                #area = area[feature_locsCB]

                #labelmap = np.reshape(labels, [ny, nx], order='F')
                ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2)
                ax3.set_anchor('S')
                cmap_nameSC = 'my_blues'
                cmSC = LinearSegmentedColormap.from_list(cmap_nameSC, data.colors[0:nc], N=nc)
                cs = plt.pcolor(x, y, np.transpose(labelmap), cmap=cmSC, vmin=-0.5, vmax=(nc-0.5), alpha=1, edgecolors='face')
                textstr = r'c'
                props = dict(boxstyle='round', facecolor='white', alpha=1)
                ax3.text(1.6, 0.9, textstr, transform=ax.transAxes, fontsize=14,verticalalignment='top', bbox=props)
                plt.xlabel('$x$',fontsize=18)
                plt.ylabel('$y$',fontsize=18)
                cbar=plt.colorbar(cs)
                cbar.ax.set_ylabel('cluster')
                if np.shape(reduced_balance)[0] == 2:
                    cbar.set_ticks([0.,1.],['0','1'])
                elif np.shape(reduced_balance)[0] == 3:
                    cbar.set_ticks([0.,1.,2.],['0','1','2'])
                elif np.shape(reduced_balance)[0] == 4:
                    cbar.set_ticks([0.,1.,2.,3.],['0','1','2','3'])
                elif np.shape(reduced_balance)[0] == 5:
                    cbar.set_ticks([0.,1.,2.,3.,4.],['0','1','2','3','4'])
                elif np.shape(reduced_balance)[0] == 6:
                    cbar.set_ticks([0.,1.,2.,3.,4.,5.],['0','1','2','3','4','5'])
                elif np.shape(reduced_balance)[0] == 7:
                    cbar.set_ticks([0.,1.,2.,3.,4.,5.,6.],['0','1','2','3','4','5','6'])
                #plt.ylim([0.,np.amax(y)])

            plt.tight_layout()
            plt.subplots_adjust(top=0.95, bottom=0.1, left=0.1, right=0.875, hspace=0.3, wspace=0.3)
            plt.savefig(plotname,format="png"); plt.close(fig);

            """

    if data.cluster_method == 'DBSCAN':

        nc = data.nc; Xc = data.Xc

        if data.reduction_method == 'Mscore':

            mean_score = data.mean_score
            std_score = data.std_score
            mean_closure = data.mean_closure
            std_closure = data.std_closure
            Kr = data.Kr
            ms = data.min_samples
            eps = data.epsilon
            nf = data.nf
            Ex,My = np.meshgrid(eps,ms)
            #print(len(ms),len(eps),np.shape(Ex),np.shape(mean_score))
            loc_max = (np.argwhere(mean_score[:,:]==np.amax(mean_score[:,:])))[0,:] # 0:NK
            #print(loc_max)
            nm = loc_max[0] # minimum number of samples with the best score
            ne = loc_max[1] # epsilon with the best score
            #na = loc_max[2] # alpha[na] = alpha value for maximum score
            #print(np.transpose(mean_score)[ne,nm])
            #print(Ex[ne,nm])


            figure_name = 'metrics_DBSCAN_Mscore_3.png'
            plotname = data.figure_path + figure_name
            #fig = plt.figure(figsize=(16, 5))
            fig = plt.figure(figsize=(17, 5))
            ax=plt.subplot(1,3,1) # (1,3,1)
            #CS=plt.contour(Ex,My, np.transpose(mean_score), levels=[nf/(2.*(nf-1))-0.00001],colors=('royalblue',),linestyles=('-',),linewidths=(2,)) #, cmap='gist_yarg') #vmin=-0.5, vmax=cm.N-0.5 cmap=cm)
            CSF=plt.contourf(Ex,My, np.transpose(mean_score), 100, cmap='gist_yarg') #vmin=-0.5, vmax=cm.N-0.5 cmap=cm)
            max_label = r'$%.4f$, $\varepsilon=%.3f,$min. samples$=%.i$' %(np.transpose(mean_score)[ne,nm],Ex[ne,nm],int(My[ne,nm]))
            plt.plot(Ex[ne,nm],My[ne,nm],color='crimson',linestyle='None',marker='*',markersize=12,label=max_label)
            cbar=plt.colorbar(CSF) #boundaries=np.arange(-0.5, ncb+1.5), ticks=np.arange(0, ncb+1))
            cbar.ax.get_yaxis().labelpad = 20
            #cbar.add_lines(CS) #plt.colorbar(CS,colors='royalblue')
            #plt.xticks([5,10,15,20],[r"$5$",r"$10$",r"$15$",r"$20$"])
            #plt.yscale('log')
            #plt.yticks([1e-5,1e-4,1e-3,1e-2,1e-1,1e-0,1e1,1e2,1e3],[r"$10^{-5}$",r"$10^{-4}$",r"$10^{-3}$",r"$10^{-2}$",r"$10^{-1}$",r"$10^{0}$",r"$10^{1}$",r"$10^{2}$",r"$10^{3}$"])
            #plt.xlabel(r'$K$',fontsize=20)
            #plt.ylabel(r'$\alpha$',fontsize=22)
            plt.title(r'$M$ score, cluster mean',fontsize=16)
            plt.legend(loc=4,framealpha=1.)

            ax=plt.subplot(1,3,2) #(1,3,3)
            plt.contourf(Ex,My, np.transpose(mean_closure), 100, cmap='gist_yarg') #vmin=-0.5, vmax=cm.N-0.5 cmap=cm)
            cbar=plt.colorbar() #boundaries=np.arange(-0.5, ncb+1.5), ticks=np.arange(0, ncb+1))
            cbar.ax.get_yaxis().labelpad = 20
            max_label = r'$%.4f$' %(np.transpose(mean_closure)[ne,nm])
            plt.plot(Ex[ne,nm],My[ne,nm],color='crimson',linestyle='None',marker='*',markersize=12,label=max_label)
            #plt.xticks([5,10,15,20],[r"$5$",r"$10$",r"$15$",r"$20$"])
            #plt.yscale('log')
            #plt.yticks([1e-5,1e-4,1e-3,1e-2,1e-1,1e-0,1e1,1e2,1e3],[r"$10^{-5}$",r"$10^{-4}$",r"$10^{-3}$",r"$10^{-2}$",r"$10^{-1}$",r"$10^{0}$",r"$10^{1}$",r"$10^{2}$",r"$10^{3}$"])
            #plt.xlabel(r'$K$',fontsize=20)
            #plt.ylabel(r'$\alpha$',fontsize=22)
            plt.title(r'closure score, cluster mean',fontsize=16)
            plt.legend(loc=4,framealpha=1.)

            ax=plt.subplot(1,3,3)
            plt.contourf(Ex,My, np.transpose(Kr), 100, cmap='gist_yarg') #vmin=-0.5, vmax=cm.N-0.5 cmap=cm)
            cbar=plt.colorbar() #boundaries=np.arange(-0.5, ncb+1.5), ticks=np.arange(0, ncb+1))
            cbar.ax.get_yaxis().labelpad = 20
            max_label = r'$%i$' %(np.transpose(Kr)[ne,nm])
            plt.plot(Ex[ne,nm],My[ne,nm],color='crimson',linestyle='None',marker='*',markersize=12,label=max_label)
            #plt.xticks([5,10,15,20],[r"$5$",r"$10$",r"$15$",r"$20$"])
            #plt.yscale('log')
            #plt.yticks([1e-5,1e-4,1e-3,1e-2,1e-1,1e-0,1e1,1e2,1e3],[r"$10^{-5}$",r"$10^{-4}$",r"$10^{-3}$",r"$10^{-2}$",r"$10^{-1}$",r"$10^{0}$",r"$10^{1}$",r"$10^{2}$",r"$10^{3}$"])
            #plt.xlabel(r'$K$',fontsize=20)
            #plt.ylabel(r'$\alpha$',fontsize=22)
            plt.title(r'number of refined clusters',fontsize=16)
            cbar.set_ticklabels(['1','2','3','4','5','6','7','8','9','10','11'])
            plt.legend(loc=4,framealpha=1.)

            plt.subplots_adjust(top=0.95, bottom=0.15, left=0.075, right=0.975, hspace=0.3, wspace=0.3)
            plt.savefig(plotname,format="png"); plt.close(fig);


            #figure_name = 'clusters_GMM_Mscore.png'
            figure_name = 'clusters_DBSCAN_Mscore' + plot_specifics + '.png'
            plotname = data.figure_path + figure_name
            fig = plt.figure(figsize=(12,10)) #7, 5)) # for 3x2
            #ax=plt.subplot(2,2,1)
            ax1 = plt.subplot2grid((2, 2), (0, 0))
            #print('np.shape(feature_mean[:,:]) = ',np.shape(feature_mean[:,:]))
            #print('np.shape(feature_mean_95[:,:]) = ',np.shape(feature_mean_95[:,:]))
            #print('np.shape(mycolors[jj,:]) = ',np.shape(mycolors[jj,:]))
            for jj in range(0,np.shape(feature_mean)[0]):
                plt.plot(np.arange(nf),np.abs(feature_mean[jj,:]),color=mycolors[jj,:],linestyle='None',marker=mymarkers[jj],markersize=mymarkersize[jj],label=r"cluster %i" %jj) #,color=(data.colors)[jj,:]) #,linestyle='None',marker='*',markersize=12,label=max_label)
                plt.fill_between(np.arange(nf),np.abs(feature_mean[jj,:]-feature_mean_95[jj,:]),np.abs(feature_mean[jj,:]+feature_mean_95[jj,:]),alpha=0.2,color=mycolors[jj,:])
                #plt.errorbar(np.arange(nf),np.abs(feature_mean[jj,:]),yerr=feature_mean_95[jj,:])
                plt.yscale('log')
            plt.legend(loc=3,framealpha=1.,fontsize=14)
            plt.title(r'cluster mean features',fontsize=16)
            plt.ylabel(r'absolute value',fontsize=16)
            xtick_labels = [r'$\bar{u} \bar{u}_x$', r'$\bar{v}\bar{u}_y$', r'$\rho^{-1} \bar{p}_x$',
                r'$\nu \nabla^2 \bar{u}$', r'$\overline{(u^\prime v^\prime)}_y$', r'$\overline{({u^\prime} ^2)}_x$']
            plt.xticks([0,1,2,3,4,5],xtick_labels,fontsize=18)
            #plt.grid()
            plt.ylim([3e-6,6e0])

            #ax = plt.axes([0.5, 1., 0.5, 1.])
            #ax = plt.subplot(2,2,2)
            ax2 = plt.subplot2grid((2, 2), (0, 1))
            print('nc = ',nc)
            print('np.shape(mymarkers),np.shape(reduced_balance),np.shape(Xc),np.shape(colors) = ',np.shape(mymarkers),np.shape(reduced_balance),np.shape(Xc),np.shape(data.colors))
            for yy in range(0,nc):
                plt.scatter(Xc[yy,:],reduced_balance[yy,:]-1,marker=mymarkers[yy],s=800,color=mycolors[yy]) #'black')
            #plt.scatter(Xc[ind0,:],balance_models[ind0,:],marker='s',s=500,color='white')
            xticks_labels = [r'$\bar{u} \bar{u}_x$', r'$\bar{v}\bar{u}_y$', r'$\rho^{-1} \bar{p}_x$',
                        r'$\nu \nabla^2 \bar{u}$', r'$\overline{(u^\prime v^\prime)}_y$', r'$\overline{({u^\prime} ^2)}_x$']
            plt.xticks([1,2,3,4,5,6],xticks_labels)
            ax2.tick_params(axis = 'both', which = 'major', labelsize = 18)
            plt.ylabel(r'cluster',fontsize=18)
            if np.shape(reduced_balance)[0] == 3:
                plt.yticks([0,1,2],[r'$0$',r'$1$',r'$2$'])
            if np.shape(reduced_balance)[0] == 4:
                plt.yticks([0,1,2,3],[r'$0$',r'$1$',r'$2$',r'$3$'])
            if np.shape(reduced_balance)[0] == 5:
                plt.yticks([0,1,2,3,4],[r'$0$',r'$1$',r'$2$',r'$3$',r'$4$'])
            if np.shape(reduced_balance)[0] == 6:
                plt.yticks([0,1,2,3,4,5],[r'$0$',r'$1$',r'$2$',r'$3$',r'$4$',r'$5$'])
            if np.shape(reduced_balance)[0] == 7:
                plt.yticks([0,1,2,3,4,5,6],[r'$0$',r'$1$',r'$2$',r'$3$',r'$4$',r'$5$',r'$6$'])
            #plt.title(r'active terms',fontsize=16)
            plt.title(r'sparsity patterns',fontsize=16)
            plt.axis([0.5,np.shape(reduced_balance)[1]+0.5,-0.5,np.shape(reduced_balance)[0]-0.5])
            ax2b=ax2.twinx()
            ax2b.set_yticks(np.arange(nc)) # FIX <<<<---------------------------------------!
            ax2b.set_yticklabels(max_scores_labels,fontsize=18)
            ax2b.set_ylim([-0.5,nc-0.5]) # FIX <<<<---------------------------------------!
            ax2b.set_ylabel(r'$M$ score',fontsize=18,rotation=270,labelpad=18)

            #print('max_scores_labels = ',max_scores_labels)
            #print('max_area_weights = ',max_area_weights)
            #print('np.sum(max_area_weights) = ',np.sum(max_area_weights))
            #print('np.sum(max_scores_labels*max_area_weights) = ',np.sum(max_scores_labels*max_area_weights))
            #print('np.sum(max_area_weights/np.sum(max_area_weights)) = ',np.sum(max_area_weights/np.sum(max_area_weights)))
            max_area_weights =  max_area_weights/np.sum(max_area_weights) # <----------------- area weights hack to normal by area that is labeled, not total area

            #ax = plt.axes([0, 1.25, 0, 0.5])
            #ax=plt.subplot(2,1,2)
            ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2)
            ax3.set_anchor('S')
            #titlename = r'$K$ = %i' %(int(K[nk]))
            #titlename = r'mean closure score = %.3f, mean $M$ score = %.3f, $K$ = %i' %(np.mean(max_closure),np.mean(max_scores),int(K[nk]))
            titlename = r'area-weighted cluster mean closure score = %.3f, $M$ score = %.3f' %(np.nansum(max_closure*max_area_weights),np.sum(max_scores*max_area_weights))
            cmap_nameSC = 'my_blues'
            cmSC = LinearSegmentedColormap.from_list(cmap_nameSC, data.colors[0:nc], N=nc)
            labelmap = np.reshape(labels, [ny, nx], order='F')
            #print('np.amax(x),np.amin(x),x[10]-x[9] = ',np.amax(x),np.amin(x),x[10]-x[9])
            #print('np.amax(y),np.amin(y),y[10]-y[9] = ',np.amax(y),np.amin(y),y[10]-y[9])
            #cs = plt.pcolor(x, y, np.isnan(labelmap), alpha=1, edgecolors='face')
            cs = plt.pcolor(x, y, labelmap, cmap=cmSC, vmin=-0.5, vmax=(nc-0.5), alpha=1, edgecolors='face')
            #nu = 1.25e-3; U = 1. # d(x) = 0.37*x/Rex**(1/5) (Schlicting book)
            #plt.plot(x,0.37*(x-0.)**(4./5.)*(nu/U)**(-1./5.),color='k',linewidth=2,linestyle='dashed',label=r'$\delta\sim{}x^{4/5}$')
            #plt.legend(loc=1,framealpha=1.)
            plt.xlabel('$x$',fontsize=18)
            plt.ylabel('$y$',fontsize=18)
            plt.title(titlename,fontsize=16)
            cbar=plt.colorbar(cs)
            cbar.ax.get_yaxis().labelpad = 20
            if np.shape(reduced_balance)[0] == 2:
                cbar.set_ticks([0.,1.],['0','1'])
            elif np.shape(reduced_balance)[0] == 3:
                cbar.set_ticks([0.,1.,2.],['0','1','2'])
            elif np.shape(reduced_balance)[0] == 4:
                cbar.set_ticks([0.,1.,2.,3.],['0','1','2','3'])
            elif np.shape(reduced_balance)[0] == 5:
                cbar.set_ticks([0.,1.,2.,3.,4.],['0','1','2','3','4'])
            elif np.shape(reduced_balance)[0] == 6:
                cbar.set_ticks([0.,1.,2.,3.,4.,5.],['0','1','2','3','4','5'])
            elif np.shape(reduced_balance)[0] == 7:
                cbar.set_ticks([0.,1.,2.,3.,4.,5.,6.],['0','1','2','3','4','5','6'])
            plt.ylim([0.,np.amax(y)])
            #ax3.set_aspect(0.5) #,anchor='C') #'equal')

            plt.tight_layout()
            plt.subplots_adjust(top=0.95, bottom=0.1, left=0.1, right=0.875, hspace=0.3, wspace=0.2)
            plt.savefig(plotname,format="png"); plt.close(fig);


    if data.cluster_method == 'HDBSCAN':

            #print('here!>')
            #max_area_weights = data.max_area_weights
            nc = data.nc; Xc = data.Xc
            ms = data.min_samples

            if metrics_flag == True:

                ne = data.ne
                nms = data.nms
                Kr = data.Kr
                #print(np.shape(data.mean_score))

                figure_name = 'metrics_HDBSCAN_Mscore_3.png'
                plotname = data.figure_path + figure_name
                fig = plt.figure(figsize=(17, 5))
                ax=plt.subplot(1,3,1)
                for i in range(0,NE):
                    plt.plot(ms[:], mean_score[i,:], color='royalblue',alpha=1,linewidth=2,linestyle='None',marker='o')
                max_label = r'$%.4f$, $m.s.=%i$' %(mean_score[ne,nms],ms[nms])
                plt.plot(ms[nms], mean_score[ne,nms], color='crimson',linestyle='None',marker='*',markersize=12,label=max_label)
                textstr = r'full set score: $N_f/(2(N_f-1)) = %.2f$' %(nf/(2*(nf-1)))
                #textstr = '\n'.join((r'cluster 0: ' + reduced_labels[0],
                #                    r'cluster 1: ' + reduced_labels[1],
                #                    r'cluster 2: ' + reduced_labels[2]))
                props = dict(boxstyle='round', facecolor='white', alpha=1)
                # place a text box in upper left in axes coords
                ax.text(0.15, 0.35, textstr, transform=ax.transAxes, fontsize=14,verticalalignment='top', bbox=props)
                #plt.xticks([5,10,15,20],[r"$5$",r"$10$",r"$15$",r"$20$"])
                plt.xlabel(r'minimum number of samples',fontsize=18)
                #plt.xlim([3.,20.])
                plt.title(r'area-weighted cluster mean $M$ score',fontsize=14)
                plt.legend(loc=4)

                plt.subplot(1,3,2)
                for i in range(0,NE):
                    plt.plot(ms[:], mean_closure[i,:], color='royalblue',alpha=0.1,linewidth=2)
                plt.plot(ms[nms], mean_closure[ne,nms], color='crimson',linestyle='None',marker='*',markersize=12,label=max_label)
                #plt.xticks([4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25],[r"$4$",r"$5$",r"$6$",r"$7$",r"$8$",r"$9$",r"$10$",r"$11$",r"$12$",r"$13$",r"$14$",r"$15$",r"$16$",r"$17$",r"$18$",r"$19$",r"$20$",r"$21$",r"$22$",r"$23$",r"$24$",r"$25$"])
                #plt.xticks([5,10,15,20,25],[r"$5$",r"$10$",r"$15$",r"$20$",r"$25$"])
                #plt.xticks([5,10,15,20],[r"$5$",r"$10$",r"$15$",r"$20$"])
                plt.xlabel(r'minimum number of samples',fontsize=18)
                #plt.xlim([3.,20.])
                plt.title(r'area-weighted cluster mean closure score',fontsize=14)

                plt.subplot(1,3,3)
                for i in range(0,NE):
                    plt.plot(ms[:], Kr[i,:], color='royalblue',alpha=0.1,linewidth=2)
                plt.plot(ms[nms], Kr[ne,nms], color='crimson',linestyle='None',marker='*',markersize=12,label=max_label)
                #plt.xticks([4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25],[r"$4$",r"$5$",r"$6$",r"$7$",r"$8$",r"$9$",r"$10$",r"$11$",r"$12$",r"$13$",r"$14$",r"$15$",r"$16$",r"$17$",r"$18$",r"$19$",r"$20$",r"$21$",r"$22$",r"$23$",r"$24$",r"$25$"])
                #plt.xticks([5,10,15,20,25],[r"$5$",r"$10$",r"$15$",r"$20$",r"$25$"])
                #plt.xticks([5,10,15,20],[r"$5$",r"$10$",r"$15$",r"$20$"])
                plt.xlabel(r'minimum number of samples',fontsize=18)
                #plt.xlim([3.,20.])
                plt.title(r'number of refined clusters',fontsize=14)

                plt.subplots_adjust(top=0.95, bottom=0.15, left=0.075, right=0.975, hspace=0.3, wspace=0.3)
                plt.savefig(plotname,format="png"); plt.close(fig);


            #figure_name = 'clusters_GMM_Mscore.png'
            figure_name = 'clusters_HDBSCAN_Mscore' + plot_specifics + '.png'
            plotname = data.figure_path + figure_name
            fig = plt.figure(figsize=(12,10)) #7, 5)) # for 3x2
            #ax=plt.subplot(2,2,1)
            ax1 = plt.subplot2grid((2, 2), (0, 0))
            for jj in range(0,np.shape(feature_mean)[0]):
                plt.plot(np.arange(nf),np.abs(feature_mean[jj,:]),color=mycolors[jj,:],linestyle='None',marker=mymarkers[jj],markersize=mymarkersize[jj],label=r"cluster %i" %jj) #,color=(data.colors)[jj,:]) #,linestyle='None',marker='*',markersize=12,label=max_label)
                plt.fill_between(np.arange(nf),np.abs(feature_mean[jj,:]-feature_mean_95[jj,:]),np.abs(feature_mean[jj,:]+feature_mean_95[jj,:]),alpha=0.2,color=mycolors[jj,:])
                #plt.errorbar(np.arange(nf),np.abs(feature_mean[jj,:]),yerr=feature_mean_95[jj,:])
                plt.yscale('log')
            #plt.legend(loc=3,framealpha=1.,fontsize=14)
            plt.title(r'cluster mean features',fontsize=16)
            plt.ylabel(r'absolute value',fontsize=16)
            xtick_labels = [r'$\bar{u} \bar{u}_x$', r'$\bar{v}\bar{u}_y$', r'$\rho^{-1} \bar{p}_x$',
                r'$\nu \nabla^2 \bar{u}$', r'$\overline{(u^\prime v^\prime)}_y$', r'$\overline{({u^\prime} ^2)}_x$']
            plt.xticks([0,1,2,3,4,5],xtick_labels,fontsize=18)
            #plt.grid()
            plt.ylim([3e-6,6e0])

            #ax = plt.axes([0.5, 1., 0.5, 1.])
            #ax = plt.subplot(2,2,2)
            ax2 = plt.subplot2grid((2, 2), (0, 1))
            #print('np.shape(Xc),np.shape(reduced_balance),np.shape(mymarkers) = ',np.shape(Xc),np.shape(reduced_balance),np.shape(mymarkers))
            for yy in range(0,nc):
                plt.scatter(Xc[yy,:],reduced_balance[yy,:]-1,marker=mymarkers[yy],s=150,color=mycolors[yy]) #'black')
            #plt.scatter(Xc[ind0,:],balance_models[ind0,:],marker='s',s=500,color='white')
            xticks_labels = [r'$\bar{u} \bar{u}_x$', r'$\bar{v}\bar{u}_y$', r'$\rho^{-1} \bar{p}_x$',
                        r'$\nu \nabla^2 \bar{u}$', r'$\overline{(u^\prime v^\prime)}_y$', r'$\overline{({u^\prime} ^2)}_x$']
            plt.xticks([1,2,3,4,5,6],xticks_labels)
            ax2.tick_params(axis = 'both', which = 'major', labelsize = 18)
            plt.ylabel(r'cluster',fontsize=18)
            if np.shape(reduced_balance)[0] == 3:
                plt.yticks([0,1,2],[r'$0$',r'$1$',r'$2$'])
            if np.shape(reduced_balance)[0] == 4:
                plt.yticks([0,1,2,3],[r'$0$',r'$1$',r'$2$',r'$3$'])
            if np.shape(reduced_balance)[0] == 5:
                plt.yticks([0,1,2,3,4],[r'$0$',r'$1$',r'$2$',r'$3$',r'$4$'])
            if np.shape(reduced_balance)[0] == 6:
                plt.yticks([0,1,2,3,4,5],[r'$0$',r'$1$',r'$2$',r'$3$',r'$4$',r'$5$'])
            if np.shape(reduced_balance)[0] == 7:
                plt.yticks([0,1,2,3,4,5,6],[r'$0$',r'$1$',r'$2$',r'$3$',r'$4$',r'$5$',r'$6$'])
            #plt.title(r'active terms',fontsize=16)
            plt.title(r'sparsity patterns',fontsize=16)
            plt.axis([0.5,np.shape(reduced_balance)[1]+0.5,-0.5,np.shape(reduced_balance)[0]-0.5])
            ax2b=ax2.twinx()
            ax2b.set_yticks(np.arange(nc)) # FIX <<<<---------------------------------------!
            ax2b.set_yticklabels(max_scores_labels,fontsize=18)
            ax2b.set_ylim([-0.5,nc-0.5]) # FIX <<<<---------------------------------------!
            ax2b.set_ylabel(r'$M$ score',fontsize=18,rotation=270,labelpad=18)

            #ax = plt.axes([0, 1.25, 0, 0.5])
            #ax=plt.subplot(2,1,2)
            ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2)
            ax3.set_anchor('S')
            #titlename = r'$K$ = %i' %(int(K[nk]))
            #titlename = r'mean closure score = %.3f, mean $M$ score = %.3f, $K$ = %i' %(np.mean(max_closure),np.mean(max_scores),int(K[nk]))
            titlename = r'area-weighted cluster mean closure score = %.3f, $M$ score = %.3f' %(np.sum(max_closure*max_area_weights),np.sum(max_scores*max_area_weights))
            #titlename = r'area-weighted cluster mean closure score = %.3f, $M$ score = %.3f' %(np.sum(max_closure),np.sum(max_scores))
            cmap_nameSC = 'my_blues'
            cmSC = LinearSegmentedColormap.from_list(cmap_nameSC, data.colors[0:nc], N=nc)
            labelmap = np.reshape(labels, [ny, nx], order='F')
            cs = plt.pcolor(x, y, labelmap, cmap=cmSC, vmin=-0.5, vmax=(nc-0.5), alpha=1, edgecolors='face')
            #nu = 1.25e-3; U = 1. # d(x) = 0.37*x/Rex**(1/5) (Schlicting book)
            #plt.plot(x,0.37*(x-0.)**(4./5.)*(nu/U)**(-1./5.),color='k',linewidth=2,linestyle='dashed',label=r'$\delta\sim{}x^{4/5}$')
            #plt.legend(loc=1,framealpha=1.)
            plt.xlabel('$x$',fontsize=18)
            plt.ylabel('$y$',fontsize=18)
            plt.title(titlename,fontsize=16)
            cbar=plt.colorbar(cs)
            cbar.ax.get_yaxis().labelpad = 20
            if np.shape(reduced_balance)[0] == 2:
                cbar.set_ticks([0.,1.],['0','1'])
            elif np.shape(reduced_balance)[0] == 3:
                cbar.set_ticks([0.,1.,2.],['0','1','2'])
            elif np.shape(reduced_balance)[0] == 4:
                cbar.set_ticks([0.,1.,2.,3.],['0','1','2','3'])
            elif np.shape(reduced_balance)[0] == 5:
                cbar.set_ticks([0.,1.,2.,3.,4.],['0','1','2','3','4'])
            elif np.shape(reduced_balance)[0] == 6:
                cbar.set_ticks([0.,1.,2.,3.,4.,5.],['0','1','2','3','4','5'])
            elif np.shape(reduced_balance)[0] == 7:
                cbar.set_ticks([0.,1.,2.,3.,4.,5.,6.],['0','1','2','3','4','5','6'])
            plt.ylim([0.,np.amax(y)])
            #ax3.set_aspect(0.5) #,anchor='C') #'equal')

            plt.tight_layout()
            plt.subplots_adjust(top=0.95, bottom=0.1, left=0.1, right=0.875, hspace=0.3, wspace=0.2)
            plt.savefig(plotname,format="png"); plt.close(fig);




    return


def plot_recursive_data( data ):

    #figure_path = data.figure_path
    #read_path = data.read_path
    #cluster_method = (read_path).split('_')[1]
    #reduction_method = (read_path).split('_')[2]
    NE = int(data.NE)

    mean_score = data.mean_score
    std_score = data.std_score
    mean_closure = data.mean_closure
    std_closure = data.std_closure
    features = data.features
    #print('plot np.shape(features) = ',np.shape(features))

    reduced_balance = data.balance # check read data <-------------------------!!
    labels = data.labels
    x = data.x; y = data.y; nx = len(x); ny = len(y)
    nf = data.nf
    nc = data.nc

    locs = data.locs
    full_labels = data.full_labels
    full_balance = data.full_balance

    feature_mean = data.feature_mean
    feature_mean_95 = data.feature_mean_95

    max_scores = data.max_scores
    Nmax = len(max_scores)
    max_scores_labels = np.empty(Nmax)
    for uu in range(0,len(max_scores_labels)):
        max_scores_labels[uu] = '%.3f' %max_scores[uu]
    #print(max_scores_labels)

    mycolors = data.mycolors
    mymarkers = data.mymarkers
    mymarkersize = data.mymarkersize

    """
    mycolors= np.zeros(np.shape(data.colors))
    mycolors[:,:] = data.colors
    if int(nc) == 3:
        mymarkers = np.array(['o','*','P'])
        mymarkersize = np.array([8,10,8])
    elif int(nc) == 4:
        mymarkers = np.array(['o','*','P','s'])
        mymarkersize = np.array([8,10,8,7])
    elif int(nc) == 5:
        mymarkers = np.array(['o','*','P','s','^'])
        mymarkersize = np.array([8,10,8,7,8])
    """

    if (data.read_path).split('_')[1] == 'GMM':

        if (data.read_path).split('_')[2] == 'SPCA':

            Kr = data.Kr; K = data.K; alphas = data.alphas
            Kx,Ay = np.meshgrid(K,alphas)
            nk = data.nk; ne = data.ne; na = data.na
            nc = data.nc; Xc = data.Xc

        elif (data.read_path).split('_')[2] == 'Mscore':

            Kr = data.Kr; K = data.K
            nk = data.nk; ne = data.ne
            nc = data.nc; Xc = data.Xc

            """
            titlename = r'cluster k = %i' %(int(data.recursive_k))
            figure_name = 'balance_GMM_Mscore_k%i.png' %(int(data.recursive_k))
            plotname = data.figure_path + figure_name
            fig = plt.figure(figsize=(6, 5))
            ax=plt.subplot(1,1,1)
            for yy in range(0,nc):
                plt.scatter(Xc[yy,:],reduced_balance[yy,:]-1,marker='s',s=800,color=data.colors[yy]) #'black')
            #plt.scatter(Xc[ind0,:],balance_models[ind0,:],marker='s',s=500,color='white')
            labels = [r'$\bar{u} \bar{u}_x$', r'$\bar{v}\bar{u}_y$', r'$\rho^{-1} \bar{p}_x$',
                        r'$\nu \nabla^2 \bar{u}$', r'$\overline{(u^\prime v^\prime)}_y$', r'$\overline{({u^\prime} ^2)}_x$']
            plt.xticks([1,2,3,4,5,6],labels)
            ax.tick_params(axis = 'both', which = 'major', labelsize = 18)
            #ax2 = ax.twinx()
            plt.ylabel(r'subcluster',fontsize=18)
            if np.shape(reduced_balance)[0] == 3:
                plt.yticks([0,1,2],[r'$0$',r'$1$',r'$2$'])
            if np.shape(reduced_balance)[0] == 4:
                plt.yticks([0,1,2,3],[r'$0$',r'$1$',r'$2$',r'$3$'])
            if np.shape(reduced_balance)[0] == 5:
                plt.yticks([0,1,2,3,4],[r'$0$',r'$1$',r'$2$',r'$3$',r'$4$'])
            if np.shape(reduced_balance)[0] == 6:
                plt.yticks([0,1,2,3,4,5],[r'$0$',r'$1$',r'$2$',r'$3$',r'$4$',r'$5$'])
            if np.shape(reduced_balance)[0] == 7:
                plt.yticks([0,1,2,3,4,5,6],[r'$0$',r'$1$',r'$2$',r'$3$',r'$4$',r'$5$',r'$6$'])
            #ax2.yticks(max_scores)
            plt.title(titlename,fontsize=16)
            plt.axis([0.5,np.shape(reduced_balance)[1]+0.5,-0.5,np.shape(reduced_balance)[0]-0.5])

            ax2=ax.twinx()
            ax2.set_yticks([0,1,2,3])
            #ax2.set_yticklabels([r'$0$',r'$1$',r'$2$',r'$3$'],fontsize=18)
            ax2.set_yticklabels(max_scores_labels,fontsize=18)
            ax2.set_ylim([-0.5,3.5])
            ax2.set_ylabel(r'$M$ score',fontsize=18,rotation=270,labelpad=18)

            plt.subplots_adjust(top=0.95, bottom=0.1, left=0.1, right=0.85, hspace=0.3, wspace=0.2)
            plt.savefig(plotname,format="png"); plt.close(fig);
            """

            """
            #mycolors= np.zeros(np.shape(data.colors))
            #mycolors[:,:] = data.colors
            #if int(nc) == 4:
            #    mymarkers = np.array(['o','*','P','s'])
            #    mymarkersize = np.array([8,10,8,7])
            #elif int(nc) == 5:
            #    mymarkers = np.array(['o','*','P','s','^'])
            #    mymarkersize = np.array([8,10,8,7,8])
            figure_name = 'clusters_GMM_Mscore_k%i.png' %(int(data.recursive_k))
            plotname = data.figure_path + figure_name
            fig = plt.figure(figsize=(12,10)) #7, 5)) # for 3x2
            #ax=plt.subplot(2,2,1)
            ax1 = plt.subplot2grid((2, 2), (0, 0))
            for jj in range(0,np.shape(feature_mean)[0]):
                plt.plot(np.arange(nf),np.abs(feature_mean[jj,:]),color=mycolors[jj,:],linestyle='None',marker=mymarkers[jj],markersize=mymarkersize[jj],label=r"cluster %i" %jj) #,color=(data.colors)[jj,:]) #,linestyle='None',marker='*',markersize=12,label=max_label)
                plt.fill_between(np.arange(nf),np.abs(feature_mean[jj,:]-feature_mean_95[jj,:]),np.abs(feature_mean[jj,:]+feature_mean_95[jj,:]),alpha=0.2,color=mycolors[jj,:])
                #plt.errorbar(np.arange(nf),np.abs(feature_mean[jj,:]),yerr=feature_mean_95[jj,:])
                plt.yscale('log')
            plt.legend(loc=3,framealpha=1.,fontsize=14)
            titlename = r'cluster mean features within cluster k = %i' %(int(data.recursive_k))
            plt.title(titlename,fontsize=16)
            plt.ylabel(r'absolute value',fontsize=16)
            xtick_labels = [r'$\bar{u} \bar{u}_x$', r'$\bar{v}\bar{u}_y$', r'$\rho^{-1} \bar{p}_x$',
                r'$\nu \nabla^2 \bar{u}$', r'$\overline{(u^\prime v^\prime)}_y$', r'$\overline{({u^\prime} ^2)}_x$']
            plt.xticks([0,1,2,3,4,5],xtick_labels,fontsize=18)
            #plt.grid()
            plt.ylim([3e-6,6e0])

            #ax = plt.axes([0.5, 1., 0.5, 1.])
            #ax = plt.subplot(2,2,2)
            ax2 = plt.subplot2grid((2, 2), (0, 1))
            for yy in range(0,nc):
                plt.scatter(Xc[yy,:],reduced_balance[yy,:]-1,marker=mymarkers[yy],s=800,color=data.colors[yy]) #'black')
            #plt.scatter(Xc[ind0,:],balance_models[ind0,:],marker='s',s=500,color='white')
            xticks_labels = [r'$\bar{u} \bar{u}_x$', r'$\bar{v}\bar{u}_y$', r'$\rho^{-1} \bar{p}_x$',
                        r'$\nu \nabla^2 \bar{u}$', r'$\overline{(u^\prime v^\prime)}_y$', r'$\overline{({u^\prime} ^2)}_x$']
            plt.xticks([1,2,3,4,5,6],xticks_labels)
            ax2.tick_params(axis = 'both', which = 'major', labelsize = 18)
            plt.ylabel(r'cluster',fontsize=18)
            if np.shape(reduced_balance)[0] == 3:
                plt.yticks([0,1,2],[r'$0$',r'$1$',r'$2$'])
            if np.shape(reduced_balance)[0] == 4:
                plt.yticks([0,1,2,3],[r'$0$',r'$1$',r'$2$',r'$3$'])
            if np.shape(reduced_balance)[0] == 5:
                plt.yticks([0,1,2,3,4],[r'$0$',r'$1$',r'$2$',r'$3$',r'$4$'])
            if np.shape(reduced_balance)[0] == 6:
                plt.yticks([0,1,2,3,4,5],[r'$0$',r'$1$',r'$2$',r'$3$',r'$4$',r'$5$'])
            if np.shape(reduced_balance)[0] == 7:
                plt.yticks([0,1,2,3,4,5,6],[r'$0$',r'$1$',r'$2$',r'$3$',r'$4$',r'$5$',r'$6$'])
            titlename = r'active terms within cluster k = %i' %(int(data.recursive_k))
            plt.title(titlename,fontsize=16)
            plt.axis([0.5,np.shape(reduced_balance)[1]+0.5,-0.5,np.shape(reduced_balance)[0]-0.5])
            ax2b=ax2.twinx()
            ax2b.set_yticks(np.arange(nc)) # FIX <<<<---------------------------------------!
            ax2b.set_yticklabels(max_scores_labels,fontsize=18)
            ax2b.set_ylim([-0.5,nc-0.5]) # FIX <<<<---------------------------------------!
            ax2b.set_ylabel(r'$M$ score',fontsize=18,rotation=270,labelpad=18)


            #ax = plt.axes([0, 1.25, 0, 0.5])
            #ax=plt.subplot(2,1,2)
            ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2)
            ax3.set_anchor('S')
            titlename = r'$K$ = %i' %(int(K[nk]))
            cmap_nameSC = 'my_blues'
            cmSC = LinearSegmentedColormap.from_list(cmap_nameSC, data.colors[0:nc], N=nc)
            full_labels_nans = full_labels*np.nan
            full_labels_nans[locs] = labels
            labelmap = np.reshape(full_labels_nans, [ny, nx], order='F')
            cs = plt.pcolor(x, y, labelmap, cmap=cmSC, vmin=-0.5, vmax=(nc-0.5), alpha=1, edgecolors='face')
            #nu = 1.25e-3; U = 1. # d(x) = 0.37*x/Rex**(1/5) (Schlicting book)
            #plt.plot(x,0.37*(x-0.)**(4./5.)*(nu/U)**(-1./5.),color='k',linewidth=2,linestyle='dashed',label=r'$\delta\sim{}x^{4/5}$')
            #plt.legend(loc=1,framealpha=1.)
            plt.xlabel('$x$',fontsize=18)
            plt.ylabel('$y$',fontsize=18)
            plt.title(titlename,fontsize=16)
            cbar=plt.colorbar(cs)
            cbar.ax.get_yaxis().labelpad = 20
            if np.shape(reduced_balance)[0] == 2:
                cbar.set_ticks([0.,1.],['0','1'])
            elif np.shape(reduced_balance)[0] == 3:
                cbar.set_ticks([0.,1.,2.],['0','1','2'])
            elif np.shape(reduced_balance)[0] == 4:
                cbar.set_ticks([0.,1.,2.,3.],['0','1','2','3'])
            elif np.shape(reduced_balance)[0] == 5:
                cbar.set_ticks([0.,1.,2.,3.,4.],['0','1','2','3','4'])
            elif np.shape(reduced_balance)[0] == 6:
                cbar.set_ticks([0.,1.,2.,3.,4.,5.],['0','1','2','3','4','5'])
            elif np.shape(reduced_balance)[0] == 7:
                cbar.set_ticks([0.,1.,2.,3.,4.,5.,6.],['0','1','2','3','4','5','6'])
            plt.ylim([0.,np.amax(y)])
            #ax3.set_aspect(0.5) #,anchor='C') #'equal')

            plt.tight_layout()

            plt.subplots_adjust(top=0.95, bottom=0.1, left=0.1, right=0.875, hspace=0.3, wspace=0.2)
            plt.savefig(plotname,format="png"); plt.close(fig);
            """



            figure_name = 'metrics_GMM_Mscore_k%i.png' %(int(data.recursive_k))
            plotname = data.figure_path + figure_name
            fig = plt.figure(figsize=(12.5, 11))
            ax=plt.subplot(2,2,1)
            for i in range(0,NE):
                plt.plot(K[:], mean_score[:,i], color='royalblue',alpha=0.1,linewidth=2)
            max_label = r'$%.4f$, $K=%i$' %(mean_score[nk,ne],K[nk])
            plt.plot(K[nk], mean_score[nk,ne], color='crimson',linestyle='None',marker='*',markersize=12,label=max_label)
            textstr = r'full set score: $N_f/(2(N_f-1)) = %.3f$' %(nf/(2*(nf-1)))
            #textstr = '\n'.join((r'cluster 0: ' + reduced_labels[0],
            #                    r'cluster 1: ' + reduced_labels[1],
            #                    r'cluster 2: ' + reduced_labels[2]))
            props = dict(boxstyle='round', facecolor='white', alpha=1)
            # place a text box in upper left in axes coords
            ax.text(0.3, 0.95, textstr, transform=ax.transAxes, fontsize=14,verticalalignment='top', bbox=props)
            #plt.xticks([4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25],[r"$4$",r"$5$",r"$6$",r"$7$",r"$8$",r"$9$",r"$10$",r"$11$",r"$12$",r"$13$",r"$14$",r"$15$",r"$16$",r"$17$",r"$18$",r"$19$",r"$20$",r"$21$",r"$22$",r"$23$",r"$24$",r"$25$"])
            #plt.xticks([5,10,15,20,25],[r"$5$",r"$10$",r"$15$",r"$20$",r"$25$"])
            plt.xticks([5,10,15,20],[r"$5$",r"$10$",r"$15$",r"$20$"])
            #plt.xlabel(r'$K$',fontsize=18)
            plt.xlim([3.,20.])
            plt.title(r'$M$ score, cluster mean',fontsize=16)
            plt.legend(loc=4)

            plt.subplot(2,2,2)
            for i in range(0,NE):
                plt.plot(K[:], std_score[:,i], color='royalblue',alpha=0.1,linewidth=2)
            plt.plot(K[nk], std_score[nk,ne], color='crimson',linestyle='None',marker='*',markersize=12,label=max_label)
            #plt.xticks([4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25],[r"$4$",r"$5$",r"$6$",r"$7$",r"$8$",r"$9$",r"$10$",r"$11$",r"$12$",r"$13$",r"$14$",r"$15$",r"$16$",r"$17$",r"$18$",r"$19$",r"$20$",r"$21$",r"$22$",r"$23$",r"$24$",r"$25$"])
            #plt.xticks([5,10,15,20,25],[r"$5$",r"$10$",r"$15$",r"$20$",r"$25$"])
            plt.xticks([5,10,15,20],[r"$5$",r"$10$",r"$15$",r"$20$"])
            #plt.xlabel(r'$K$',fontsize=18)
            plt.xlim([3.,20.])
            #plt.title(r'domain standard deviation of OMS score',fontsize=18)
            plt.title(r'$M$ score, cluster std. dev.',fontsize=16)

            plt.subplot(2,2,3)
            for i in range(0,NE):
                plt.plot(K[:], mean_closure[:,i], color='royalblue',alpha=0.1,linewidth=2)
            plt.plot(K[nk], mean_closure[nk,ne], color='crimson',linestyle='None',marker='*',markersize=12,label=max_label)
            #plt.xticks([4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25],[r"$4$",r"$5$",r"$6$",r"$7$",r"$8$",r"$9$",r"$10$",r"$11$",r"$12$",r"$13$",r"$14$",r"$15$",r"$16$",r"$17$",r"$18$",r"$19$",r"$20$",r"$21$",r"$22$",r"$23$",r"$24$",r"$25$"])
            #plt.xticks([5,10,15,20,25],[r"$5$",r"$10$",r"$15$",r"$20$",r"$25$"])
            plt.xticks([5,10,15,20],[r"$5$",r"$10$",r"$15$",r"$20$"])
            plt.xlabel(r'$K$',fontsize=18)
            plt.xlim([3.,20.])
            plt.title(r'closure score, cluster mean',fontsize=16)

            plt.subplot(2,2,4)
            for i in range(0,NE):
                plt.plot(K[:], Kr[:,i], color='royalblue',alpha=0.1,linewidth=2)
            plt.plot(K[nk], Kr[nk,ne], color='crimson',linestyle='None',marker='*',markersize=12,label=max_label)
            #plt.xticks([4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25],[r"$4$",r"$5$",r"$6$",r"$7$",r"$8$",r"$9$",r"$10$",r"$11$",r"$12$",r"$13$",r"$14$",r"$15$",r"$16$",r"$17$",r"$18$",r"$19$",r"$20$",r"$21$",r"$22$",r"$23$",r"$24$",r"$25$"])
            #plt.xticks([5,10,15,20,25],[r"$5$",r"$10$",r"$15$",r"$20$",r"$25$"])
            plt.xticks([5,10,15,20],[r"$5$",r"$10$",r"$15$",r"$20$"])
            plt.xlabel(r'$K$',fontsize=18)
            plt.xlim([3.,20.])
            plt.title(r'number of refined clusters',fontsize=16)

            plt.subplots_adjust(top=0.95, bottom=0.1, left=0.075, right=0.975, hspace=0.3, wspace=0.2)
            plt.savefig(plotname,format="png"); plt.close(fig);







            """
            mycolors= np.zeros(np.shape(data.colors))
            mycolors[:,:] = data.colors
            if int(nc) == 4:
                mymarkers = np.array(['o','*','P','s'])
                mymarkersize = np.array([8,10,8,7])
            elif int(nc) == 5:
                mymarkers = np.array(['o','*','P','s','^'])
                mymarkersize = np.array([8,10,8,7,8])
            figure_name = 'clusters_GMM_Mscore.png'
            plotname = data.figure_path + figure_name
            fig = plt.figure(figsize=(12,10)) #7, 5)) # for 3x2
            #ax=plt.subplot(2,2,1)
            ax1 = plt.subplot2grid((2, 2), (0, 0))
            for jj in range(0,np.shape(feature_mean)[0]):
                plt.plot(np.arange(nf),np.abs(feature_mean[jj,:]),color=mycolors[jj,:],linestyle='None',marker=mymarkers[jj],markersize=mymarkersize[jj],label=r"cluster %i" %jj) #,color=(data.colors)[jj,:]) #,linestyle='None',marker='*',markersize=12,label=max_label)
                plt.fill_between(np.arange(nf),np.abs(feature_mean[jj,:]-feature_mean_95[jj,:]),np.abs(feature_mean[jj,:]+feature_mean_95[jj,:]),alpha=0.2,color=mycolors[jj,:])
                #plt.errorbar(np.arange(nf),np.abs(feature_mean[jj,:]),yerr=feature_mean_95[jj,:])
                plt.yscale('log')
            plt.legend(loc=3,framealpha=1.,fontsize=14)
            plt.title(r'cluster mean features',fontsize=16)
            plt.ylabel(r'absolute value',fontsize=16)
            xtick_labels = [r'$\bar{u} \bar{u}_x$', r'$\bar{v}\bar{u}_y$', r'$\rho^{-1} \bar{p}_x$',
                r'$\nu \nabla^2 \bar{u}$', r'$\overline{(u^\prime v^\prime)}_y$', r'$\overline{({u^\prime} ^2)}_x$']
            plt.xticks([0,1,2,3,4,5],xtick_labels,fontsize=18)
            #plt.grid()
            plt.ylim([3e-6,6e0])

            #ax = plt.axes([0.5, 1., 0.5, 1.])
            #ax = plt.subplot(2,2,2)
            ax2 = plt.subplot2grid((2, 2), (0, 1))
            for yy in range(0,nc):
                plt.scatter(Xc[yy,:],reduced_balance[yy,:]-1,marker=mymarkers[yy],s=800,color=data.colors[yy]) #'black')
            #plt.scatter(Xc[ind0,:],balance_models[ind0,:],marker='s',s=500,color='white')
            xticks_labels = [r'$\bar{u} \bar{u}_x$', r'$\bar{v}\bar{u}_y$', r'$\rho^{-1} \bar{p}_x$',
                        r'$\nu \nabla^2 \bar{u}$', r'$\overline{(u^\prime v^\prime)}_y$', r'$\overline{({u^\prime} ^2)}_x$']
            plt.xticks([1,2,3,4,5,6],xticks_labels)
            ax2.tick_params(axis = 'both', which = 'major', labelsize = 18)
            plt.ylabel(r'cluster',fontsize=18)
            if np.shape(reduced_balance)[0] == 3:
                plt.yticks([0,1,2],[r'$0$',r'$1$',r'$2$'])
            if np.shape(reduced_balance)[0] == 4:
                plt.yticks([0,1,2,3],[r'$0$',r'$1$',r'$2$',r'$3$'])
            if np.shape(reduced_balance)[0] == 5:
                plt.yticks([0,1,2,3,4],[r'$0$',r'$1$',r'$2$',r'$3$',r'$4$'])
            if np.shape(reduced_balance)[0] == 6:
                plt.yticks([0,1,2,3,4,5],[r'$0$',r'$1$',r'$2$',r'$3$',r'$4$',r'$5$'])
            if np.shape(reduced_balance)[0] == 7:
                plt.yticks([0,1,2,3,4,5,6],[r'$0$',r'$1$',r'$2$',r'$3$',r'$4$',r'$5$',r'$6$'])
            plt.title(r'active terms',fontsize=16)
            plt.axis([0.5,np.shape(reduced_balance)[1]+0.5,-0.5,np.shape(reduced_balance)[0]-0.5])
            ax2b=ax2.twinx()
            ax2b.set_yticks(np.arange(nc)) # FIX <<<<---------------------------------------!
            ax2b.set_yticklabels(max_scores_labels,fontsize=18)
            ax2b.set_ylim([-0.5,nc-0.5]) # FIX <<<<---------------------------------------!
            ax2b.set_ylabel(r'$M$ score',fontsize=18,rotation=270,labelpad=18)

            #ax = plt.axes([0, 1.25, 0, 0.5])
            #ax=plt.subplot(2,1,2)
            ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2)
            ax3.set_anchor('S')
            titlename = r'$K$ = %i' %(int(K[nk]))
            cmap_nameSC = 'my_blues'
            cmSC = LinearSegmentedColormap.from_list(cmap_nameSC, data.colors[0:nc], N=nc)
            labelmap = np.reshape(labels, [ny, nx], order='F')
            cs = plt.pcolor(x, y, labelmap, cmap=cmSC, vmin=-0.5, vmax=(nc-0.5), alpha=1, edgecolors='face')
            #nu = 1.25e-3; U = 1. # d(x) = 0.37*x/Rex**(1/5) (Schlicting book)
            #plt.plot(x,0.37*(x-0.)**(4./5.)*(nu/U)**(-1./5.),color='k',linewidth=2,linestyle='dashed',label=r'$\delta\sim{}x^{4/5}$')
            #plt.legend(loc=1,framealpha=1.)
            plt.xlabel('$x$',fontsize=18)
            plt.ylabel('$y$',fontsize=18)
            plt.title(titlename,fontsize=16)
            cbar=plt.colorbar(cs)
            cbar.ax.get_yaxis().labelpad = 20
            if np.shape(reduced_balance)[0] == 2:
                cbar.set_ticks([0.,1.],['0','1'])
            elif np.shape(reduced_balance)[0] == 3:
                cbar.set_ticks([0.,1.,2.],['0','1','2'])
            elif np.shape(reduced_balance)[0] == 4:
                cbar.set_ticks([0.,1.,2.,3.],['0','1','2','3'])
            elif np.shape(reduced_balance)[0] == 5:
                cbar.set_ticks([0.,1.,2.,3.,4.],['0','1','2','3','4'])
            elif np.shape(reduced_balance)[0] == 6:
                cbar.set_ticks([0.,1.,2.,3.,4.,5.],['0','1','2','3','4','5'])
            elif np.shape(reduced_balance)[0] == 7:
                cbar.set_ticks([0.,1.,2.,3.,4.,5.,6.],['0','1','2','3','4','5','6'])
            plt.ylim([0.,np.amax(y)])
            #ax3.set_aspect(0.5) #,anchor='C') #'equal')

            plt.tight_layout()
            plt.subplots_adjust(top=0.95, bottom=0.1, left=0.1, right=0.875, hspace=0.3, wspace=0.2)
            plt.savefig(plotname,format="png"); plt.close(fig);
            """





            # how to merge with the full labels? Need to combine, then rescore:

            #balance1, labels1 = merge_clusters( reduced_balance , full_labels )
            #print(balance1)
            #print(np.unique(labels1))


            # merge two separate sets of clusters based on balances:
            # 1) make k2 labels numbered higher than full labels
            """
            # labels:
            #print(np.unique(full_labels))
            nc_full = len(np.unique(full_labels))
            #print(nc_full)
            labels = labels + nc_full
            #print(np.unique(labels))
            #print(np.unique(labels)+nc_full)
            full_labels[locs] = labels
            #print(np.unique(full_labels))
            # balances:
            #print(np.shape(reduced_balance))
            #print(np.shape(full_balance))
            for uu in range(0,np.shape(reduced_balance)[0]):
                for kk in range(0,np.shape(reduced_balance)[1]):
                    if reduced_balance[uu,kk] > 0.:
                        reduced_balance[uu,kk] = reduced_balance[uu,kk]+nc_full

            #print()
            #print(reduced_balance)
            #print()
            #print(full_balance)
            #print()
            full_balance = np.concatenate((full_balance,reduced_balance),axis=0)
            active_terms = np.zeros(np.shape(full_balance))
            for uu in range(0,np.shape(full_balance)[0]):
                for kk in range(0,np.shape(full_balance)[1]):
                    if full_balance[uu,kk] > 0.:
                        active_terms[uu,kk] = 1.
            """
            combo_labels, combo_balance = combine_clusters( full_labels, full_balance, labels, reduced_balance, locs )

            #balance, label_matching_index = np.unique(active_terms, axis=0, return_inverse=True) # MOVE TO M score section ????? <---!
            #labels = np.array([label_matching_index[ii] for ii in full_labels])
            #print(active_terms)
            #print(balance)
            #print(label_matching_index)
            #print(np.unique(labels))
            #nc = len(np.unique(labels))

            balance, labels = merge_clusters( combo_balance , combo_labels )
            #balance, labels = merge_clusters( active_terms , full_labels )
            #print(balance)
            #print(np.unique(labels))
            nc = len(np.unique(labels))



            combo_data = easydict.EasyDict({
                "nf":data.nf,
                "nc":nc,
                "refine_flag": True,
                "features":data.full_features,
                "refined_balance":balance,
                "refined_labels":labels,
                "Nbootstrap":1000,
                })

            # now get M scores:
            combo_data = get_scores( combo_data , preassigned_balance_flag=True )
            #print(combo_data.refined_balance)
            #print(np.unique(combo_data.refined_labels))
            closure_max = combo_data.closure_max  # closure score for max. M score.
            score_max = combo_data.score_max
            feature_mean = combo_data.feature_mean
            feature_mean_95 = combo_data.feature_mean_95



            max_scores_labels = get_max_scores_labels( score_max )

            #print('mean(score_max) = ',np.mean(score_max))
            #print('mean(closure_max) = ',np.mean(closure_max))

            mycolors= np.zeros(np.shape(data.colors))
            mycolors[:,:] = data.colors
            if int(nc) == 4:
                mymarkers = np.array(['o','*','P','s'])
                mymarkersize = np.array([8,10,8,7])
            elif int(nc) == 5:
                mymarkers = np.array(['o','*','P','s','^'])
                mymarkersize = np.array([8,10,8,7,8])


            # prep for plotting the max. M score reduced balance:
            xc = np.linspace(1,np.shape(balance)[1],num=np.shape(balance)[1],endpoint=True)
            yc = np.linspace(0,np.shape(balance)[0]-1,num=np.shape(balance)[0],endpoint=True)
            Xc,Y = np.meshgrid(xc,yc)
            ind0 = 0
            for jj in range(0,np.shape(balance)[0]):
                balance[jj,:] = balance[jj,:]*(yc[jj]+1)
                if sum(balance[jj,:]) == 0.:
                    ind0 = jj


            reduced_balance = balance
            #print(balance)

            # loop over k2 clusters, if the balance matches the full label balance,
            # relabel it as the full label.
            # if not, label it as a number greater than the full label balance.

            figure_name = 'clusters_GMM_Mscore_k%i.png' %(int(data.recursive_k))
            plotname = data.figure_path + figure_name
            fig = plt.figure(figsize=(12,10)) #7, 5)) # for 3x2
            #ax=plt.subplot(2,2,1)
            ax1 = plt.subplot2grid((2, 2), (0, 0))
            for jj in range(0,np.shape(feature_mean)[0]):
                plt.plot(np.arange(nf),np.abs(feature_mean[jj,:]),color=mycolors[jj,:],linestyle='None',marker=mymarkers[jj],markersize=mymarkersize[jj],label=r"cluster %i" %jj) #,color=(data.colors)[jj,:]) #,linestyle='None',marker='*',markersize=12,label=max_label)
                plt.fill_between(np.arange(nf),np.abs(feature_mean[jj,:]-feature_mean_95[jj,:]),np.abs(feature_mean[jj,:]+feature_mean_95[jj,:]),alpha=0.2,color=mycolors[jj,:])
                #plt.errorbar(np.arange(nf),np.abs(feature_mean[jj,:]),yerr=feature_mean_95[jj,:])
                plt.yscale('log')
            plt.legend(loc=3,framealpha=1.,fontsize=14)
            titlename = r'cluster mean features within cluster k = %i' %(int(data.recursive_k))
            plt.title(titlename,fontsize=16)
            plt.ylabel(r'absolute value',fontsize=16)
            xtick_labels = [r'$\bar{u} \bar{u}_x$', r'$\bar{v}\bar{u}_y$', r'$\rho^{-1} \bar{p}_x$',
                r'$\nu \nabla^2 \bar{u}$', r'$\overline{(u^\prime v^\prime)}_y$', r'$\overline{({u^\prime} ^2)}_x$']
            plt.xticks([0,1,2,3,4,5],xtick_labels,fontsize=18)
            #plt.grid()
            plt.ylim([3e-6,6e0])

            #ax = plt.axes([0.5, 1., 0.5, 1.])
            #ax = plt.subplot(2,2,2)
            ax2 = plt.subplot2grid((2, 2), (0, 1))
            for yy in range(0,nc):
                plt.scatter(Xc[yy,:],reduced_balance[yy,:]-1,marker=mymarkers[yy],s=800,color=data.colors[yy]) #'black')
            #plt.scatter(Xc[ind0,:],balance_models[ind0,:],marker='s',s=500,color='white')
            xticks_labels = [r'$\bar{u} \bar{u}_x$', r'$\bar{v}\bar{u}_y$', r'$\rho^{-1} \bar{p}_x$',
                        r'$\nu \nabla^2 \bar{u}$', r'$\overline{(u^\prime v^\prime)}_y$', r'$\overline{({u^\prime} ^2)}_x$']
            plt.xticks([1,2,3,4,5,6],xticks_labels)
            ax2.tick_params(axis = 'both', which = 'major', labelsize = 18)
            plt.ylabel(r'cluster',fontsize=18)
            if np.shape(reduced_balance)[0] == 3:
                plt.yticks([0,1,2],[r'$0$',r'$1$',r'$2$'])
            if np.shape(reduced_balance)[0] == 4:
                plt.yticks([0,1,2,3],[r'$0$',r'$1$',r'$2$',r'$3$'])
            if np.shape(reduced_balance)[0] == 5:
                plt.yticks([0,1,2,3,4],[r'$0$',r'$1$',r'$2$',r'$3$',r'$4$'])
            if np.shape(reduced_balance)[0] == 6:
                plt.yticks([0,1,2,3,4,5],[r'$0$',r'$1$',r'$2$',r'$3$',r'$4$',r'$5$'])
            if np.shape(reduced_balance)[0] == 7:
                plt.yticks([0,1,2,3,4,5,6],[r'$0$',r'$1$',r'$2$',r'$3$',r'$4$',r'$5$',r'$6$'])
            titlename = r'active terms within cluster k = %i' %(int(data.recursive_k))
            plt.title(titlename,fontsize=16)
            plt.axis([0.5,np.shape(reduced_balance)[1]+0.5,-0.5,np.shape(reduced_balance)[0]-0.5])
            ax2b=ax2.twinx()
            ax2b.set_yticks(np.arange(nc)) # FIX <<<<---------------------------------------!
            ax2b.set_yticklabels(max_scores_labels,fontsize=18)
            ax2b.set_ylim([-0.5,nc-0.5]) # FIX <<<<---------------------------------------!
            ax2b.set_ylabel(r'$M$ score',fontsize=18,rotation=270,labelpad=18)


            #ax = plt.axes([0, 1.25, 0, 0.5])
            #ax=plt.subplot(2,1,2)
            ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2)
            ax3.set_anchor('S')
            titlename = r'mean closure score = %.3f, mean $M$ score = %.3f' %(np.mean(closure_max),np.mean(score_max))
            cmap_nameSC = 'my_blues'
            cmSC = LinearSegmentedColormap.from_list(cmap_nameSC, data.colors[0:nc], N=nc)
            labelmap = np.reshape(labels, [ny, nx], order='F')
            cs = plt.pcolor(x, y, labelmap, cmap=cmSC, vmin=-0.5, vmax=(nc-0.5), alpha=1, edgecolors='face')
            #nu = 1.25e-3; U = 1. # d(x) = 0.37*x/Rex**(1/5) (Schlicting book)
            #plt.plot(x,0.37*(x-0.)**(4./5.)*(nu/U)**(-1./5.),color='k',linewidth=2,linestyle='dashed',label=r'$\delta\sim{}x^{4/5}$')
            #plt.legend(loc=1,framealpha=1.)
            plt.xlabel('$x$',fontsize=18)
            plt.ylabel('$y$',fontsize=18)
            plt.title(titlename,fontsize=16)
            cbar=plt.colorbar(cs)
            cbar.ax.get_yaxis().labelpad = 20
            if np.shape(reduced_balance)[0] == 2:
                cbar.set_ticks([0.,1.],['0','1'])
            elif np.shape(reduced_balance)[0] == 3:
                cbar.set_ticks([0.,1.,2.],['0','1','2'])
            elif np.shape(reduced_balance)[0] == 4:
                cbar.set_ticks([0.,1.,2.,3.],['0','1','2','3'])
            elif np.shape(reduced_balance)[0] == 5:
                cbar.set_ticks([0.,1.,2.,3.,4.],['0','1','2','3','4'])
            elif np.shape(reduced_balance)[0] == 6:
                cbar.set_ticks([0.,1.,2.,3.,4.,5.],['0','1','2','3','4','5'])
            elif np.shape(reduced_balance)[0] == 7:
                cbar.set_ticks([0.,1.,2.,3.,4.,5.,6.],['0','1','2','3','4','5','6'])
            plt.ylim([0.,np.amax(y)])
            #ax3.set_aspect(0.5) #,anchor='C') #'equal')

            plt.tight_layout()

            plt.subplots_adjust(top=0.95, bottom=0.1, left=0.1, right=0.875, hspace=0.3, wspace=0.2)
            plt.savefig(plotname,format="png"); plt.close(fig);


    return



def plot_true_balance_comparison( data1 , data2 , data3 , truth , n_top ):

    balance_truth = truth.balance
    balance_truth_unique = np.unique(balance_truth,axis=0)
    balance1 = data1.balance
    balance2 = data2.balance
    balance3 = data3.balance

    # clean up the input:
    print('\n\nplot histogram')
    print('balance1 = ',balance1)
    print('balance2 = ',balance2)
    print('balance_truth_unique = ',balance_truth_unique)
    for i in range(0,np.shape(balance_truth)[0]):
        for j in range(0,np.shape(balance_truth)[1]):
            if np.isnan(balance_truth[i,j]) == True:
                balance_truth[i,j] = 0.
            elif balance_truth[i,j] > 0.:
                balance_truth[i,j] = 1

    for i in range(0,np.shape(balance1)[0]):
        for j in range(0,np.shape(balance1)[1]):
            if np.isnan(balance1[i,j]) == True:
                balance1[i,j] = 0.
            elif balance1[i,j] > 0.:
                balance1[i,j] = 1

    for i in range(0,np.shape(balance2)[0]):
        for j in range(0,np.shape(balance2)[1]):
            if np.isnan(balance2[i,j]) == True:
                balance2[i,j] = 0.
            elif balance2[i,j] > 0.:
                balance2[i,j] = 1

    for i in range(0,np.shape(balance3)[0]):
        for j in range(0,np.shape(balance3)[1]):
            if np.isnan(balance3[i,j]) == True:
                balance3[i,j] = 0.
            elif balance3[i,j] > 0.:
                balance3[i,j] = 1

    balance_truth_unique = np.unique(balance_truth,axis=0)
    print('balance1 = ',balance1)
    print('balance2 = ',balance2)
    print('balance3 = ',balance3)
    print('balance_truth_unique = ',balance_truth_unique)


    # get the labels:
    ng = np.shape(balance_truth)[0]
    label_truth = np.zeros([ng])*np.nan
    labels1_match = np.zeros([ng])*np.nan
    labels2_match = np.zeros([ng])*np.nan
    labels3_match = np.zeros([ng])*np.nan
    labels1 = data1.labels
    labels2 = data2.labels
    labels3 = data3.labels
    print('np.shape(labels1) = ',np.shape(labels1))
    print('np.shape(labels2) = ',np.shape(labels2))
    print('np.shape(labels3) = ',np.shape(labels3))
    print('np.unique(labels1) = ',np.unique(labels1))
    print('np.unique(labels2) = ',np.unique(labels2))
    print('np.unique(labels3) = ',np.unique(labels3))
    for ii in range(0,ng):
        local_balance_truth = balance_truth[ii,:]
        loc_truth = (np.where((balance_truth_unique==local_balance_truth).all(axis=1)))
        label_truth[ii] = (loc_truth[0])[0] # label of the truth balance
        local_balance1 = balance1[int(labels1[ii]),:]
        local_balance2 = balance2[int(labels2[ii]),:]
        local_balance3 = balance3[int(labels3[ii]),:]
        if np.sum(np.abs(local_balance1-local_balance_truth)) == 0:
            labels1_match[ii] = label_truth[ii]
        else:
            labels1_match[ii] = -1
        if np.sum(np.abs(local_balance2-local_balance_truth)) == 0:
            labels2_match[ii] = label_truth[ii]
        else:
            labels2_match[ii] = -1
        if np.sum(np.abs(local_balance3-local_balance_truth)) == 0:
            labels3_match[ii] = label_truth[ii]
        else:
            labels3_match[ii] = -1

    # get complete, unsorted histogram
    dataH = np.zeros([ng,4])
    dataH[:,0] = label_truth
    dataH[:,1] = labels1_match
    dataH[:,2] = labels2_match
    dataH[:,3] = labels3_match
    nbins = np.arange(len(np.unique(label_truth)))-0.5
    figure_name = 'histogram_all_bins.png'
    plotname = './Figures/' + figure_name
    fig = plt.figure(figsize=(8, 5)) # for 3x2
    ax=plt.subplot(1,1,1) #2,2,1)
    colorsH = ['black','darkorange','darkorchid','crimson']
    labelsH = [r'truth',r'GMM$+M$','GMM$+$SPCA',r'K-means$+M$']
    n, bins, patches = plt.hist(dataH, bins=nbins, color=colorsH, label=labelsH,histtype='bar') #density=True,
    plt.legend(loc=1,framealpha=1.)
    plt.subplots_adjust(top=0.95, bottom=0.1, left=0.125, right=0.9, hspace=0.3, wspace=0.15)
    plt.savefig(plotname,format="png"); plt.close(fig);


    # get n_top values
    ind = (n[0,:]).argsort()[-int(n_top+1):][::-1] # label names of most-full bins
    print('\n\npercent of all grid points (truth) = ',np.sum(n[0,ind])/ng)
    print('\n')
    sorted_bins_truth = bins[ind]-0.5
    n_sort = len(sorted_bins_truth)


    # now sort the actual labels:
    label_truth_sort = np.array([])
    labels1_sort0 = np.array([])
    labels2_sort0 = np.array([])
    labels3_sort0 = np.array([])
    for i in range(0,ng):
        for j in range(0,n_sort):
            if label_truth[i] == ind[j]:
                label_truth_sort = np.append(label_truth_sort,j)
            if labels1_match[i] == ind[j]:
                labels1_sort0 = np.append(labels1_sort0,j)
            if labels2_match[i] == ind[j]:
                labels2_sort0 = np.append(labels2_sort0,j)
            if labels3_match[i] == ind[j]:
                labels3_sort0 = np.append(labels3_sort0,j)

    # pad with -1s
    print('len(label_truth_sort) = ',len(label_truth_sort))
    print('len(labels1_sort0) = ',len(labels1_sort0))
    print('len(labels2_sort0) = ',len(labels2_sort0))
    print('len(labels3_sort0) = ',len(labels3_sort0))

    labels1_sort = -np.ones([len(label_truth_sort)])
    labels2_sort = -np.ones([len(label_truth_sort)])
    labels3_sort = -np.ones([len(label_truth_sort)])
    labels1_sort[0:len(labels1_sort0)] = labels1_sort0
    labels2_sort[0:len(labels2_sort0)] = labels2_sort0
    labels3_sort[0:len(labels3_sort0)] = labels3_sort0
    print('len(labels1_sort) = ',len(labels1_sort))
    print('len(labels2_sort) = ',len(labels2_sort))
    print('len(labels3_sort) = ',len(labels3_sort))

    # get checkboard of truth balances for plotting:
    balance_truth_sort = np.zeros([n_sort,np.shape(balance_truth)[1]])
    for j in range(0,n_sort):
        #print(ind[j])
        balance_truth_sort[j,:] = balance_truth_unique[int(ind[j]),:]
    #print('balance_truth_sort = ',balance_truth_sort)
    checkboard = np.copy(balance_truth_sort)
    #print('np.shape(checkboard_truth) = ',np.shape(checkboard))
    for i in range(0,np.shape(checkboard)[0]):
        for j in range(0,np.shape(checkboard)[1]):
            if checkboard[i,j] == 0.:
                checkboard[i,j] = np.nan
            elif checkboard[i,j] == 1.:
                checkboard[i,j] = i
    xc = np.linspace(1,np.shape(checkboard)[1],num=np.shape(checkboard)[1],endpoint=True)
    yc = np.linspace(0,np.shape(checkboard)[0]-1,num=np.shape(checkboard)[0],endpoint=True)
    Xc,Y = np.meshgrid(xc,yc)


    # final histogram plot:
    nbins = np.arange(len(np.unique(label_truth_sort)))-0.5
    dataH = np.zeros([len(label_truth_sort),4])
    dataH[:,0] = label_truth_sort
    dataH[:,1] = labels1_sort
    dataH[:,2] = labels2_sort
    dataH[:,3] = labels3_sort
    figure_name = 'histogram_top_%i_bins.png' %(n_top)
    plotname = './Figures/' + figure_name
    fig = plt.figure(figsize=(11, 5)) # for 3x2
    ax=plt.subplot(1,2,1) #2,2,1)
    colorsH = ['black','darkorange','darkorchid','crimson']
    labelsH = [r'truth',r'GMM$+M$','GMM$+$SPCA',r'K-means$+M$']
    #colorsH = ['black','darkorange','darkorchid']
    #labelsH = [r'CHS',r'GMM$+$CHS','GMM$+$SPCA']
    n, bins, patches = plt.hist(dataH, bins=nbins, color=colorsH, label=labelsH,histtype='bar') #density=True,
    plt.legend(loc=1,framealpha=1.)
    plt.ylabel(r'number of samples',fontsize=16)
    #plt.xlabel(r'cluster')
    plt.xlabel(r'binned hypotheses',fontsize=16)
    plt.xticks(np.arange(n_top))
    textstr = r'a'
    props = dict(boxstyle='round', facecolor='white', alpha=1)
    ax.text(0.9, 0.5, textstr, transform=ax.transAxes, fontsize=14,verticalalignment='top', bbox=props)
    ax=plt.subplot(1,2,2) #2,2,1)
    plt.scatter(Xc-1.,checkboard,marker='s',s=800,color='black')
    plt.ylim([-0.5,n_top-0.5])
    plt.ylabel(r'binned hypotheses',fontsize=16)
    plt.xlabel(r'features',fontsize=16)
    #xtick_labels = [r'$\bar{u} \bar{u}_x$', r'$\bar{v}\bar{u}_y$', r'$\rho^{-1} \bar{p}_x$',
    #r'$\nu \nabla^2 \bar{u}$', r'$\overline{(u^\prime v^\prime)}_y$', r'$\overline{({u^\prime} ^2)}_x$']
    xticks_labels = [r'$\overline{u} \frac{\partial \overline{u} }{\partial x}$',
    r'$\overline{v} \frac{\partial \overline{u} }{\partial y}$',
    r'$\frac{1}{\rho} \frac{\partial \overline{p} }{\partial x}$',
    r'$\nu\nabla^2\overline{u}$',
    r'$\frac{\partial \overline{u^\prime v^\prime} }{\partial y}$',
    r'$\frac{\partial \overline{{u^\prime}^2} }{\partial x}$']
    plt.xticks([0,1,2,3,4,5],xticks_labels,fontsize=18)
    textstr = r'b'
    props = dict(boxstyle='round', facecolor='white', alpha=1)
    ax.text(0.9, 0.9, textstr, transform=ax.transAxes, fontsize=14,verticalalignment='top', bbox=props)
    plt.subplots_adjust(top=0.95, bottom=0.15, left=0.125, right=0.9, hspace=0.3, wspace=0.25)
    plt.savefig(plotname,format="png"); plt.close(fig);



    """
    checkboard = balance_truth_unique
    print('np.shape(checkboard_truth) = ',np.shape(checkboard))
    for i in range(0,np.shape(checkboard)[0]):
        for j in range(0,np.shape(checkboard)[1]):
            if checkboard[i,j] == 0.:
                checkboard[i,j] = np.nan
            elif checkboard[i,j] == 1.:
                checkboard[i,j] = i
    xc = np.linspace(1,np.shape(checkboard)[1],num=np.shape(checkboard)[1],endpoint=True)
    yc = np.linspace(0,np.shape(checkboard)[0]-1,num=np.shape(checkboard)[0],endpoint=True)
    Xc,Y = np.meshgrid(xc,yc)


    # plot histogram with of the labels in the plots matching
    dataH = np.zeros([ng,3])
    dataH[:,0] = label_truth
    dataH[:,1] = labels1_match
    dataH[:,2] = labels2_match
    #print(np.shape(dataH))
    nbins = np.arange(len(np.unique(data1.labels_truth)))-0.5
    #print('nbins = ',nbins)
    figure_name = 'histogram.png'
    plotname = './Figures/' + figure_name
    fig = plt.figure(figsize=(12, 5)) # for 3x2
    ax=plt.subplot(1,2,1) #2,2,1)
    colorsH = ['black','darkorange','darkorchid']
    labelsH = [r'truth',r'GMM$+M$','GMM$+$SPCA']
    n, bins, patches = plt.hist(dataH, bins=nbins, color=colorsH, label=labelsH,rwidth=5) #density=True,
    #print(n)
    #print(bins)
    plt.legend(loc=1,framealpha=1.)

    ax=plt.subplot(1,2,2) #2,2,1)
    plt.scatter(Xc,checkboard,marker='s',s=800,color='black')
    xtick_labels = [r'$\bar{u} \bar{u}_x$', r'$\bar{v}\bar{u}_y$', r'$\rho^{-1} \bar{p}_x$',
                r'$\nu \nabla^2 \bar{u}$', r'$\overline{(u^\prime v^\prime)}_y$', r'$\overline{({u^\prime} ^2)}_x$']
    plt.xticks([0,1,2,3,4,5],xtick_labels,fontsize=18)

    plt.subplots_adjust(top=0.95, bottom=0.1, left=0.125, right=0.9, hspace=0.3, wspace=0.15)
    plt.savefig(plotname,format="png"); plt.close(fig);
    """

    """
    # now:
    label_locs1 = data1.labels_locs
    #print('len(label_locs1) = ',len(label_locs1))
    label_area1 = np.zeros(len(data1.area))
    label_area1[label_locs1] = np.ones(len(label_locs1))
    label_area1 = np.reshape(label_area1, [len(data1.y), len(data1.x)], order='F')
    print('np.shape(label_area1) = ',np.shape(label_area1))
    print('np.unique(label_area1) = ',np.unique(label_area1))
    label_area1_copy = np.copy(label_area1)

    from matplotlib import colors
    cmap1 = colors.ListedColormap(['white', 'darkorange','blue'])
    bounds=[-0.5,0.5,1.5,2.5]
    norm1 = colors.BoundaryNorm(bounds, cmap1.N)


    #labels_locs = data1.labels_locs

    balance_err_map1 = np.reshape(data1.balance_err, [len(data1.y), len(data1.x)], order='F')
    X1,Y1 = np.meshgrid(data1.x,data1.y)
    for i in range(0,len(data1.x)):
        for j in range(0,len(data1.y)):
            if balance_err_map1[j,i] == 1.:
                label_area1[j,i] = label_area1[j,i]+1.
    #balance_err_map1 = balance_err_map1 + np.ones(np.shape(balance_err_map1))
    #print('np.shape(balance_err_map1) = ',np.shape(balance_err_map1))
    print('np.unique(balance_err_map1) = ',np.unique(balance_err_map1))
    print('np.unique(label_area1) = ',np.unique(label_area1))


    balance_err_map2 = np.reshape(data2.balance_err, [len(data2.y), len(data2.x)], order='F')
    X2,Y2 = np.meshgrid(data2.x,data2.y)

    cluster1 = (data1.read_path).split('_')[1] #== 'GMM':
    refine1 = (data1.read_path).split('_')[2] #== 'SPCA':
    if refine1 == 'Mscore':
        refine1 = r'$M$ score'

    cluster2 = (data2.read_path).split('_')[1] #== 'GMM':
    refine2 = (data2.read_path).split('_')[2] #== 'SPCA':
    if refine2 == 'Mscore':
        refine2 = r'$M$ score'
    """
    """
    figure_name = 'correct_balance_comparison.png'
    plotname = './Figures/' + figure_name
    fig = plt.figure(figsize=(10, 10)) # for 3x2
    ax=plt.subplot(2,1,1)
    #cs = plt.contourf(X1, Y1, label_area1, cmap=cmap1) #, norm=norm1, alpha=1, edgecolors='face',snap=True)
    cs = plt.pcolor(data1.x, data1.y, label_area1, cmap=cmap1, norm=norm1, alpha=1, edgecolors='face',snap=True)
    #CSF=plt.contour(X1,Y1, balance_err_map1, 2, cmap='gist_yarg_r') #vmin=-0.5, vmax=cm.N-0.5 cmap=cm)
    titlename = r'%.3f percent total area labeled, %.3f percent of that labeled correctly ' %(100.*data1.labeled_area,100.*data1.correctly_labeled_area)
    #titlename = r', %.3f percent total area of correctly labeled, %.3f unlabeled area' %(100.*data1.correctly_labeled_area,100.*data1.unlabeled_area)
    #titlename = cluster1 + r', refinement by ' + refine1 + r', %.3f percent total area of correctly labeled, %.3f unlabeled area' %(100.*data1.correctly_labeled_area,100.*data1.unlabeled_area)
    #titlename = r'clustering by ' + cluster1 + ' refinement by ' + refine1
    plt.title(titlename,fontsize=16)
    plt.xlabel(r'$x$',fontsize=16)
    plt.ylabel(r'$y$',fontsize=16)
    #cbar=plt.colorbar(cs)
    ax=plt.subplot(2,1,2)
    cs = plt.pcolor(data1.x, data1.y, label_area1_copy, cmap='gist_yarg', vmin=0., vmax=1., alpha=1, edgecolors='face',snap=True)
    #cs = plt.pcolor(data2.x, data2.y, balance_err_map2, cmap='gist_yarg', vmin=0., vmax=1., alpha=1, edgecolors='face',snap=True)
    titlename = r'clustering by ' + cluster2 + r', refinement by ' + refine2 + r', %.1f percent correct' %(100.*data2.correct_balance_percent)
    plt.title(titlename,fontsize=16)
    plt.xlabel(r'$x$',fontsize=16)
    plt.ylabel(r'$y$',fontsize=16)
    #cbar=plt.colorbar(cs)
    plt.subplots_adjust(top=0.95, bottom=0.1, left=0.075, right=0.96, hspace=0.3, wspace=0.15)
    plt.savefig(plotname,format="png"); plt.close(fig);
    """
    return



def plot_balance_accuracy( data1 ):

    # check that labels_truth from data1 and data2 are the same:
    #print('np.sum(np.abs(data1.labels_truth-data2.labels_truth)) = ',np.sum(np.abs(data1.labels_truth-data2.labels_truth)))
    ng = data1.ng
    nc_truth = data1.nc_truth

    balance1 = data1.balance
    for bb in range(0,np.shape(balance1)[0]):
        balance1[bb,:] = balance1[bb,:]/(bb+1)
    balance_truth_unique = np.unique(data1.balance_truth,axis=0)

    #labels_truth = np.zeros([ng])*np.nan
    labels1_match = np.zeros([ng])*np.nan
    for ii in range(0,ng):
        local_balance_truth = data1.balance_truth[ii,:]
        loc_truth = (np.where((balance_truth_unique==local_balance_truth).all(axis=1)))
        loc_truth = (loc_truth[0])[0] # label of the truth balance

        loc1 = (np.where((balance1==local_balance_truth).all(axis=1)))
        if np.shape(loc1)[1] == 0: # no match
            labels1_match[ii] = -1
        elif np.shape(loc1)[1] > 1:
            print('ERROR: too many matches')
        elif np.shape(loc1)[1] == 1:
            labels1_match[ii] = loc_truth

        #loc2 = (np.where((balance2==local_balance_truth).all(axis=1)))
        #if np.shape(loc2)[1] == 0: # no match
        #    labels2_match[ii] = -1
        #elif np.shape(loc1)[1] > 1:
        #    print('ERROR: too many matches')
        #elif np.shape(loc1)[1] == 1:
        #    labels2_match[ii] = loc_truth

    """
    # plot histogram with of the labels in the plots matching
    #print(np.shape(data1.labels_truth),np.shape(labels1_match))
    dataH = np.zeros([ng,3])
    dataH[:,0] = data1.labels_truth
    dataH[:,1] = labels1_match
    dataH[:,2] = labels2_match
    #print(np.shape(dataH))
    nbins = np.arange(len(np.unique(data1.labels_truth)))-0.5
    #print('nbins = ',nbins)
    figure_name = 'histogram.png'
    plotname = './Figures/' + figure_name
    fig = plt.figure(figsize=(6, 5)) # for 3x2
    ax=plt.subplot(1,1,1) #2,2,1)
    colorsH = ['black','darkorange','darkorchid']
    labelsH = [r'truth',r'GMM$+M$','GMM$+$SPCA']
    n, bins, patches = plt.hist(dataH, bins=nbins, color=colorsH, label=labelsH,rwidth=5) #density=True,
    #print(n)
    #print(bins)
    plt.legend(loc=1,framealpha=1.)
    plt.subplots_adjust(top=0.95, bottom=0.1, left=0.125, right=0.9, hspace=0.3, wspace=0.15)
    plt.savefig(plotname,format="png"); plt.close(fig);
    """


    # now:
    #nan_locs = np.load( data.read_path + 'nan_locs.npy' )
    #feature_locs = np.load( data.read_path + 'feature_locs.npy' )
    #labelmap = np.zeros([len(nan_locs)+len(feature_locs)])*np.nan
    #labelmap[feature_locs] = labels
    #labelmap = np.reshape(labelmap, [ny, nx], order='F')
    # FIX HERE!!!!!!!!!!
    label_locs1 = data1.labels_locs
    label_area1 = np.zeros(len(data1.area))
    label_area1[label_locs1] = np.ones(len(label_locs1))
    label_area1 = np.reshape(label_area1, [len(data1.y), len(data1.x)], order='F')
    label_area1_copy = np.copy(label_area1)



    from matplotlib import colors
    cmap1 = colors.ListedColormap(['white', 'darkorange','blue'])
    bounds=[-0.5,0.5,1.5,2.5]
    norm1 = colors.BoundaryNorm(bounds, cmap1.N)


    balance_err_map1 = np.reshape(data1.balance_err, [len(data1.y), len(data1.x)], order='F')
    X1,Y1 = np.meshgrid(data1.x,data1.y)
    for i in range(0,len(data1.x)):
        for j in range(0,len(data1.y)):
            if balance_err_map1[j,i] == 1.:
                label_area1[j,i] = label_area1[j,i]+1.
    #print('np.unique(balance_err_map1) = ',np.unique(balance_err_map1))
    #print('np.unique(label_area1) = ',np.unique(label_area1))

    cluster1 = data1.cluster_method
    refine1 = data1.reduction_method
    figure_path = data1.figure_path

    figure_name = 'balance_accuracy_' + cluster1 + '_' + refine1 + '.png'
    plotname = figure_path + figure_name
    if refine1 == 'Mscore':
        refine1 = r'$M$ score'
    fig = plt.figure(figsize=(9, 5)) # for 3x2
    ax=plt.subplot(1,1,1)
    #cs = plt.contourf(X1, Y1, label_area1, cmap=cmap1) #, norm=norm1, alpha=1, edgecolors='face',snap=True)
    cs = plt.pcolor(data1.x, data1.y, label_area1, cmap=cmap1, norm=norm1, alpha=1, edgecolors='face',snap=True)
    #CSF=plt.contour(X1,Y1, balance_err_map1, 2, cmap='gist_yarg_r') #vmin=-0.5, vmax=cm.N-0.5 cmap=cm)
    titlename = r'%.3f percent total area labeled, %.3f percent of that labeled correctly ' %(100.*data1.labeled_area,100.*data1.correctly_labeled_area)
    #titlename = r', %.3f percent total area of correctly labeled, %.3f unlabeled area' %(100.*data1.correctly_labeled_area,100.*data1.unlabeled_area)
    #titlename = cluster1 + r', refinement by ' + refine1 + r', %.3f percent total area of correctly labeled, %.3f unlabeled area' %(100.*data1.correctly_labeled_area,100.*data1.unlabeled_area)
    #titlename = r'clustering by ' + cluster1 + ' refinement by ' + refine1
    plt.title(titlename,fontsize=16)
    plt.xlabel(r'$x$',fontsize=16)
    plt.ylabel(r'$y$',fontsize=16)

    plt.subplots_adjust(top=0.95, bottom=0.1, left=0.075, right=0.96, hspace=0.3, wspace=0.15)
    plt.savefig(plotname,format="png"); plt.close(fig);

    return
