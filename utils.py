
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#plt.rc('text', usetex=True)
#plt.rcParams.update({'font.size': 15})
#plt.rcParams["font.family"] = "serif"
#plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
plt.rcParams.update({'font.family':'sans-serif'})
plt.rcParams.update({'font.sans-serif':'Helvetica'})
import matplotlib.colors as colors

import os
import shutil

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

from sklearn.preprocessing import RobustScaler

from itertools import combinations

from matplotlib import ticker
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import colors as mcolors

from scipy import stats

import easydict

#===============================================================================


def refine( input ):

    if len(np.unique(input.cluster_labels)) == 1 and np.unique(input.cluster_labels)[0] == -1.:
        print('\nERROR: all points labeled as noise ')
        # Save refined labels & balances + corresponding features, scores, etc:
        if input.save_flag == True:
            input.refined_labels = np.array([np.nan])
            input.refined_balance = np.array([np.nan])
            input.feature_mean = np.array([np.nan])
            input.feature_mean_95 = np.array([np.nan])
            input.score_max = np.array([np.nan])  # "max" refers to the best score for the data.
            input.closure_max = np.array([np.nan])
            input.area_weights= np.array([np.nan])
            save_refined_labels( input )
            save_refined_data( input )
        return

    else:

        # maybe I should check if a feature is ignored by all clusters and then recluster w/o unused features?
        # global scores do not count full set regions? NO! Need bad results to be represented.

        # 1) identify identical clusters and merge to create "refined" clusters
        input = cluster_id_and_merge( input )
        #print('\n  1) Cluster identification and agglomeration: ')
        #print('  agglomerated balances = ',input.refined_balance)
        #print('  agglomerated scores = ',input.score_max)
        #print('  agglomerated areas = ',input.area_weights)
        #print('  sum areas = ',np.sum(input.area_weights) )
        #print('  global score = ',np.sum(input.area_weights * input.score_max))

        if input.verbose == True:
            print('\n  1) Cluster identification and agglomeration: ')
            print('  agglomerated balances = ',input.refined_balance)
            print('  agglomerated scores = ',input.score_max)
            print('  agglomerated areas = ',input.area_weights)
            print('  sum areas = ',np.sum(input.area_weights) )
            print('  global score = ',np.sum(input.area_weights * input.score_max))

        # 3) set clusters with scores less than the full set score as full sets
        if input.bias_flag == 'biased':
            input = full_set_clusters( input )
            if input.verbose == True:
                print('\n  3) Low score clusters converted to full set: ')
                print('  post full_set_clusters')
                print('  agglomerated + fs scores = ',input.score_max)
                print('  agglomerated + fs areas = ',input.area_weights)
                print('  retained_columns = ',input.retained_columns)
                print('  np.unique(input.refined_labels) = ',np.unique(input.refined_labels))
                if input.cluster_method == 'HDBSCAN':
                    print('  input.noise_locs = ',input.noise_locs)
                    print('  len(input.noise_locs) = ',len(input.noise_locs))
                    print('  len(input.noise_locs)+len(input.labels_locs) = ',len(input.noise_locs)+len(input.labels_locs))
                print('  input.ng = ',input.ng)
                print('  input.nc = ',input.nc)

        print('\n  final global score = ',np.sum(input.area_weights * input.score_max))
        print('  final sum of areas = ',np.sum(input.area_weights))
        print('  final balances = ',input.refined_balance)
        print('  final areas = ',input.area_weights)
        print('  final scores = ',input.score_max)
        print('\n')

        # Save refined labels & balances + corresponding features, scores, etc:
        if input.save_flag == True:
            save_refined_labels( input )
            save_refined_data( input )
        return


def spca( nc, cluster_idx, features, alpha_opt, skew_flag ):

    nf = np.shape(features)[1]
    #nc = len(np.unique(cluster_idx))
    #print('\n  nc = ',nc)

    # sparse principal component analysis
    spca_model = np.zeros([nc, nf])

    for i in range(nc):
        feature_idx = np.nonzero(cluster_idx==i)[0]
        cluster_features = features[feature_idx, :]
        #print('\n  np.shape(cluster_features) = ',np.shape(cluster_features))
        #print(' cluster label = ',i)

        if skew_flag == True:
            # test the skewness of each cluster, if it is skewed use log10(abs())
            skew = np.zeros([nf]); flat = np.zeros([nf])
            for j in range(0,nf):
                skew[j],flat[j] = skewness_flatness( cluster_features[:,j] )
            if np.any(np.abs(skew)>=1.) == True: # one or more skewed features exist
                cluster_features_tmp = np.log10(np.abs(np.copy(cluster_features)))
                if np.any(np.isnan(cluster_features_tmp)-1) == True: # no nans:
                    if np.any(np.isinf(cluster_features_tmp)) == False: # no infs:
                        cluster_features = np.copy(cluster_features_tmp)

        spca = SparsePCA(n_components=1, alpha=alpha_opt) #, normalize_components=True)
        spca.fit(cluster_features)
        active_terms = np.nonzero(spca.components_[0])[0]
        #spca_components[i,:] = spca.components_[0]
        feature_idx = []
        cluster_features = []
        if len(active_terms)>0:
            # loop over labels
            spca_model[i, active_terms] = 1  # Set to 1 for active terms in model

        if skew_flag == True:
            if np.any(np.abs(skew)>=1.) == True:
                spca_model[i,:] = np.abs(spca_model[i,:]-1)

    return spca_model


def get_scores( input , preassigned_balance_flag ):
    # find the M score for the input cluster features, either the best fit M
    # score and accompanying balances, or the M score for preassigned balances.

    # FIX 2/21: MAKE BOTH preassigned and unpreassigned data processed by the same function <-------------------- !!!!!!!

    if preassigned_balance_flag == True:
        # the balance for each cluster is already prescribed
        # (by SPCA or previous assignment)

        if input.refine_flag == True:
            labels = input.refined_labels
        else:
            labels = input.cluster_labels

        nf = input.nf
        nc = input.nc
        features = input.raw_features # unnormalized
        #print('\n  preassigned balance, raw features:')
        #print('  np.amin(raw_features),np.amax(raw_features) = ',np.amin(features),np.amax(features))
        #print('  np.amin(np.abs(raw_features)),np.amax(np.abs(raw_features)) = ',np.amin(np.abs(features)),np.amax(np.abs(features)))
        #print('  np.std(raw_features,axis=1) = ',np.std(features,axis=1))
        #print('  np.mean(raw_features,axis=1) = ',np.mean(features,axis=1))
        #print('  np.log10(np.amax(np.abs(raw_features))) = ',np.log10(np.amax(np.abs(features))))
        #print('  np.log10(np.amin(np.abs(raw_features))) = ',np.log10(np.amin(np.abs(features))))
        #print('  np.shape(np.argwhere(raw_features==0.0)) = ',np.shape(np.argwhere(features==0.0)))
        #print('\n')

        # "area" is the labeled area (excluding noise), not the total area:
        area = (input.area).flatten('F')
        if input.cluster_method == 'HDBSCAN':
            area_locs = input.labels_locs
            area = area[area_locs]
            #if input.verbose == True:
            #    print('\n get_scores: np.shape(area) = ',np.shape(area))

        area_weights = np.zeros([nc])
        closure_max = np.zeros([nc])
        score_max = np.zeros([nc])
        score_max_95 = np.zeros([nc])
        feature_mean = np.zeros([nc,nf])
        feature_mean_95 = np.zeros([nc,nf])

        # import balances:
        if input.refine_flag == True:
            balance = input.refined_balance
        else:
            balance = input.cluster_balance
        #if input.verbose == True:
        #    print('\n get_scores: balance = ',balance)

        if nc != np.shape(balance)[0] and input.reduction_method != 'SPCA':
            print('\n  ERROR: number of clusters (nc) does not match number of balances')
            print('  nc = ', nc)
            print('  balance = ', balance)
        elif nc != np.shape(balance)[0] and input.reduction_method == 'SPCA':
            nc = np.shape(balance)[0]
            input.nc = nc

        for nn in range(0,int(nc)): # loop over reduced set of clusters
            locs = np.argwhere(labels==nn)[:,0]
            area_weights[nn] = np.sum(area[locs]) #/np.sum(area) # for area-weighted average
            # # depreciated:
            #if input.bias_flag == 'biased':
            #    feature_mean[nn,:], feature_mean_95[nn,:] = bootstrap_mean( features[locs,:] , int(input.Nbootstrap) )
            #    score_max[nn], closure_max[nn] = get_m_score( balance[nn,:] , feature_mean[nn,:] , input )
            #elif input.bias_flag == 'unbiased':
            #    input.cluster_area = area[locs]
            #    score_max[nn], score_max_95[nn] = get_m_score( balance[nn,:] , features[locs,:] , input )

            # better way:
            feature_mean[nn,:], feature_mean_95[nn,:] = bootstrap_mean( features[locs,:] , int(input.Nbootstrap) )
            if input.bias_flag == 'biased':
                score_max[nn], closure_max[nn] = get_m_score( balance[nn,:] , feature_mean[nn,:] , input )
            elif input.bias_flag == 'unbiased':
                score_max[nn], score_max_95[nn] = get_m_score( balance[nn,:] , feature_mean[nn,:] , input )


        # only the refined closure, score, etc, are saved later:
        if input.bias_flag == 'biased':
            input.closure_max = closure_max # closure score for max. M score.
        elif input.bias_flag == 'unbiased':
            input.score_max_95 = score_max_95
        input.score_max = score_max
        input.feature_mean = feature_mean
        input.feature_mean_95 = feature_mean_95
        input.area_weights = area_weights

        return input

    else: # preassigned_balance_flag == False

        input = assign_balance_and_score( input ) #, np.arange(input.nf,dtype=int) )

        if input.refine_flag == True:
            balance = input.refined_balance
        else:
            balance = input.cluster_balance

        return input

def check_columns( balance ):
    retained_columns = np.zeros([1],dtype=int)
    for j in range(0,np.shape(balance)[1]):
        if np.sum(balance[:,j],axis=0) != 0.:
            retained_columns = np.append(retained_columns,int(j))
    retained_columns = retained_columns[1:len(retained_columns)]
    return retained_columns

def weighted_avg_and_std(values, weights):
    #Return the weighted average and standard deviation.
    #values, weights -- Numpy ndarrays with the same shape.
    normalized_weights = weights / np.sum(weights)
    average = np.average(values, weights=normalized_weights)
    #print('  np.sum(normalized_weights) = ',np.sum(normalized_weights))
    #print('  weights = ',weights)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights)
    return average, ma.sqrt(variance)

def rescore( input, retained_columns ):

    print('\n  Rescore retained_columns = ',retained_columns)

    nf = input.nf
    nc = input.nc
    features = input.raw_features
    if input.refine_flag == True:
        balance = input.refined_balance
    else:
        balance = input.cluster_balance
    feature_mean = input.feature_mean

    closure_max = np.zeros([nc])
    score_max = np.zeros([nc])
    score_max_95 = np.zeros([nc])
    for nn in range(0,nc): # loop over clusters
        locs = np.argwhere(labels==nn)[:,0]
        area_weights[nn] = np.sum(area[locs]) #/np.sum(area) # for area-weighted average
        # # depreciated:
        #if input.bias_flag == 'biased':
        #    feature_mean[nn,:], feature_mean_95[nn,:] = bootstrap_mean( features[locs,:] , int(input.Nbootstrap) )
        #    score_max[nn], closure_max[nn] = get_m_score( balance[nn,retained_columns] , feature_mean[nn,retained_columns] , input )
        #elif input.bias_flag == 'unbiased':
        #    input.cluster_area = area[locs]
        #    score_max[nn], score_max_95[nn] = get_m_score( balance[nn,retained_columns] , features[locs,retained_columns] , input )
        # better way:
        feature_mean[nn,:], feature_mean_95[nn,:] = bootstrap_mean( features[locs,:] , int(input.Nbootstrap) )
        if input.bias_flag == 'biased':
            score_max[nn], closure_max[nn] = get_m_score( balance[nn,:] , feature_mean[nn,:] , input )
        elif input.bias_flag == 'unbiased':
            score_max[nn], score_max_95[nn] = get_m_score( balance[nn,:] , feature_mean[nn,:] , input )

    # only the refined closure, score, etc, are saved later:
    if input.bias_flag == 'biased':
        input.closure_max = closure_max # closure score for max. M score.
    elif input.bias_flag == 'unbiased':
        input.score_max_95 = score_max_95

    input.score_max = score_max

    return input

def get_optimal_balance_from_score( score, feature_mean, balance_combinations ):
    maxabsval = np.amax(np.abs(feature_mean))
    loc_maxabsval = np.argwhere(np.abs(feature_mean)==maxabsval)
    #if np.any(np.isnan(score_max)) == True:
    score_max = np.nanmax(score)
    loc_matches = np.argwhere(score==np.amax(score))
    Nmatch = np.shape(np.argwhere(score==np.amax(score)))[0]
    #print('score_max  = ',score_max )
    #print('Nmatch = ',Nmatch)
    #print('balance_max = ',balance_max)
    if Nmatch > 1:
        # if there is more than one maximum score, find the balance that
        # includes the element with the maximum score.
        for i in range(0,Nmatch):
            balance = (balance_combinations[loc_matches[i],:])[0]
            if balance[loc_maxabsval] == 1.:
                balance_max = balance_combinations[loc_matches[i],:]
    elif Nmatch == 1:
        #print('here !')
        loc_max = (np.argwhere(score==np.amax(score))[:,0])[0]
        #print('loc_max = ',loc_max)
        balance_max = balance_combinations[loc_max,:]
        #print('balance_max = ',balance_max)
    elif Nmatch == 0:
        print('\n  ERROR: no maximum score found')
        print(' ')
        nf = np.shape(balance_combinations)[1]
        balance_max = np.ones([nf])
        score_max = nf / (2.*(nf-1.))
        #print('score = ',score)
        #print('np.amax(score) = ',np.amax(score))
    #print('balance_max = ',balance_max)
    return balance_max, score_max

def assign_balance_and_score( input ): #, retained_columns ):

    nf = input.nf
    nc = input.nc

    features = input.raw_features
    if input.bias_flag == 'unbiased':
        balance_combinations = input.balance_combinations_unbiased
        #print('\n  balance_combinations = \n',balance_combinations)
    elif input.bias_flag == 'biased':
        balance_combinations = input.balance_combinations_biased

    if input.refine_flag == True:
        labels = input.refined_labels
    else:
        labels = input.cluster_labels

    # "area" is the labeled area (excluding noise), not the total area:
    area = (input.area).flatten('F')
    #if input.cluster_method == 'HDBSCAN':
    #    area_locs = input.labels_locs
    #    noise_locs = input.noise_locs
    #    print('noise_locs = ',noise_locs)
    #    noise_area = area[noise_locs]
    #    area = area[area_locs]
        #nc = nc + 1 # add the noise as a "cluster"


    if input.cluster_method == 'HDBSCAN':
        #print('\n  np.unique(labels) = ',np.unique(labels))
        #print('  len(np.unique(labels)) = ',len(np.unique(labels)))
        #print('  nc = ',nc)
        #locsm1 = np.argwhere(labels==-1)[:,0]
        #locs0 = np.argwhere(labels==0)[:,0]
        #locsp1 = np.argwhere(labels==1)[:,0]
        #area_weightsm1 = np.sum(area[locsm1])
        #area_weights0 = np.sum(area[locs0])
        #area_weightsp1 = np.sum(area[locsp1])
        #print('  noise area weights = ',area_weightsm1)
        #print('  0 area weights = ',area_weights0)
        #print('  1 area weights = ',area_weightsp1)
        labels = labels + np.ones(np.shape(labels),dtype=int)
        if input.refine_flag == True:
            input.refined_labels = labels
        else:
            input.cluster_labels = labels
        #print('\n  np.unique(labels) = ',np.unique(labels))
        #print('  len(np.unique(labels)) = ',len(np.unique(labels)))
        #print('  nc = ',nc)

    area_weights = np.zeros([nc])
    closure_max = np.zeros([nc])
    score_max = np.zeros([nc])
    score_max_95 = np.zeros([nc])
    feature_mean = np.zeros([nc,nf])
    feature_mean_95 = np.zeros([nc,nf])
    balance = np.zeros([nc,nf])
    for nn in range(0,nc): # loop over clusters
        locs = np.argwhere(labels==nn)[:,0]
        if len(locs) == 1:
            feature_mean[nn,:] = features[locs,:]
            feature_mean_95[nn,:] = np.zeros([nf])
        else:
            feature_mean[nn,:], feature_mean_95[nn,:] = bootstrap_mean( features[locs,:] , int(input.Nbootstrap) )
        if input.verbose == True:
            print('  assign_balance_and_score: feature_mean[nn,:] = ',feature_mean[nn,:])
        ngb = np.shape(balance_combinations)[0] # number of combinations
        score = np.zeros([ngb])
        closure = np.zeros([ngb])
        score_95 = np.zeros([ngb])

        #print('\n  feature_mean[nn,:] = ',feature_mean[nn,:])

        # depreciated: cluster_area:
        #if input.bias_flag == 'unbiased':
        #    input.cluster_area = area[locs]

        if input.cluster_method == 'HDBSCAN' and nn == -1: # set nn == 0 to fix noise as full set
            # the "noise" cluster
            score_max[nn] = 0.
            score_max_95[nn] = 0.
            balance[nn,:] = np.ones([nf])

        else:

            for mm in range(0,ngb): # loop over all possible balances for the cluster

                if input.bias_flag == 'biased':
                    score[mm], closure[mm] = get_m_score( balance_combinations[mm,:] , feature_mean[nn,:] , input )
                elif input.bias_flag == 'unbiased':
                    score[mm], score_95[mm] = get_m_score( balance_combinations[mm,:] , feature_mean[nn,:] , input )

            if input.bias_flag == 'biased':
                balance[nn,:], score_max[nn] = get_optimal_balance_from_score( score, feature_mean[nn,:], balance_combinations )
            elif input.bias_flag == 'unbiased':
                # get the highest score:
                sorted_score_index = score.argsort()[-1]
                #print('\n  np.amax(score) = ',np.amax(score))
                #print('  score[sorted_score_index] = ',score[sorted_score_index])
                #print('  score_95[sorted_score_index] = ',score_95[sorted_score_index])
                if score[sorted_score_index] - score_95[sorted_score_index] <= 0.0:
                    balance[nn,:] = np.ones([nf])
                    score_max[nn] = 0.
                    score_max_95[nn] = 0.
                elif score[sorted_score_index] <= 0.0:
                    balance[nn,:] = np.ones([nf])
                    score_max[nn] = 0.
                    score_max_95[nn] = 0.
                else:
                    balance[nn,:] = balance_combinations[sorted_score_index,:]
                    score_max[nn] = score[sorted_score_index]
                    score_max_95[nn] = score_95[sorted_score_index]

        closure_max[nn] = np.nan #closure[loc_max]
        area_weights[nn] = np.sum(area[locs]) # checked, working correctly


    # if the balances are not preassigned, output the now-assigned balances:
    if input.refine_flag == True:
        input.refined_balance = balance # balances assigned by scoring the merged clusters
    else:
        input.cluster_balance = balance # balances assigned by scoring the clustering results

    # only the refined closure, score, etc, are saved later:
    input.score_max = score_max
    if input.bias_flag == 'biased':
        input.closure_max = closure_max # closure score for max. M score.
    elif input.bias_flag == 'unbiased':
        input.score_max_95 = score_max_95
    input.feature_mean = feature_mean
    input.feature_mean_95 = feature_mean_95
    input.area_weights = area_weights

    return input


def cluster( input , import_raw_data ):

    if import_raw_data == True: # fresh import of the feature data
        #print('   Import data anew:')
        cluster_data = get_feature_data( input.read_path, input.data_set_name, input.standardize_flag, input.residual_flag, input.verbose )
        input.features = cluster_data.features
        input.nf = cluster_data.nf
        input.balance_combinations_unbiased = cluster_data.balance_combinations_unbiased #generate_balances(nf)
        input.balance_combinations_biased = cluster_data.balance_combinations_biased #generate_balances(nf)
        input.raw_features = cluster_data.raw_features
        #print('   np.shape(input.features) = ',np.shape(input.features))

    features = input.features

    if input.cluster_method == 'GMM':

        # Fit Gaussian mixture model
        seed = 3696299933  #  Keep a seed for debugging/plotting
        model = GaussianMixture(n_components=int(input.K), random_state=seed) #,max_iter=200, n_init=2)

        # Permutation of the data
        mask = np.random.permutation(features.shape[0])[:int(input.sample_pct*features.shape[0])]
        model.fit(features[mask, :])

        # "Predict" labels for the entire domain
        labels = model.predict(features)
        #print('\n  np.unique(labels) = ',np.unique(labels))
        nc = len(np.unique(labels))
        #print('  nc = ',nc)
        #print('  input.K = ',input.K)

        # make labels consecutive
        labels_unique = np.unique(labels)
        labels_new = np.zeros(np.shape(labels),dtype=int)
        for i in range(0,len(labels)):
            for j in range(0,nc):
               if labels[i] == labels_unique[j]:
                   labels_new[i] = int(j)
        #for j in range(0,nc):
        #    print('  matching = ',np.sum(np.argwhere(labels==labels_unique[j]) -  np.argwhere(labels_new==j)))
        #print('\n  np.unique(labels_new) = ',np.unique(labels_new))

        input.cluster_labels = labels_new
        input.nc = len(np.unique(labels_new))
        #input.cluster_labels = labels
        #input.nc = input.K
        if input.ic_flag == True:
            input.aic = model.aic(features[mask, :])
            input.bic = model.bic(features[mask, :])
        #print('\n  np.unique(input.cluster_labels) = ',np.unique(input.cluster_labels))
        #print('  input.nc = ',input.nc)
        #print('  input.K = ',input.K)

        return input

    elif input.cluster_method == 'DBSCAN':
        print('\n\n  ERROR: DBSCAN is depreciated \n\n')
        model = DBSCAN(eps=input.eps, min_samples=input.ms)

        # Permutation of the data
        #mask = np.random.permutation(features.shape[0])[:int(input.sample_pct*features.shape[0])]
        #model.fit(features[mask, :])
        model.fit(features)
        #print('np.shape(features[mask, :]) = ',np.shape(features[mask, :]))

        # "Predict" labels for the entire domain
        #labels2 = model.fit_predict(features)
        #print('np.shape(labels2) = ',np.shape(labels2))

        core_samples_mask = np.zeros_like(model.labels_, dtype=bool)
        core_samples_mask[model.core_sample_indices_] = True
        labels = model.labels_
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)
        #print('> np.shape(labels) = ',np.shape(labels))
        #print('> np.shape(features) = ',np.shape(features))

        # # remove noise:
        #labels_locs = np.argwhere(labels!=-1.)[:,0]
        #cluster_labels = labels[labels_locs]
        #features_no_noise = features[labels_locs,:]
        #noise_locs = np.argwhere(labels==-1.)[:,0]
        #input.labels_locs = labels_locs
        #input.noise_locs = noise_locs
        #input.cluster_labels = cluster_labels
        #input.nc = len(np.unique(cluster_labels))
        # # "-1" label values in DBSCAN are "noise" and not clusters, therefore
        # # nc (number of clusters) is = number of unique labels minus one.

        # keep noise:
        labels_locs = np.argwhere(labels!=-2.)[:,0]
        cluster_labels = labels[labels_locs]
        features_no_noise = features[labels_locs,:]
        noise_locs = np.argwhere(labels==-2.)[:,0]
        input.labels_locs = labels_locs
        input.noise_locs = noise_locs
        input.cluster_labels = cluster_labels
        input.nc = len(np.unique(cluster_labels))
        # "-1" label values in DBSCAN are "noise" and not clusters, therefore
        # nc (number of clusters) is = number of unique labels minus one.

        #noise_ratio = n_noise_/input.ng
        input.features_no_noise = features_no_noise #features[mask, :]
        input.ng = np.shape(cluster_labels)[0] # number of labeled grid points (less than original)

        return input

    elif input.cluster_method == 'HDBSCAN':

        import hdbscan

        #print('\n cluster: np.shape(features) = ',np.shape(features))
        #print('cluster: input.nx*input.ny = ',input.nx*input.ny)

        # Train on permutation of the data
        mask = np.random.permutation(features.shape[0])[:int(input.sample_pct*features.shape[0])]
        #selection_epsilon = 0.001 #1.0 #1e-2 #5e-2 # 1e-1 #0.0 #1e-2
        #selection_method = 'eom' #'leaf' #
        if input.ms != 'None' and input.mcs != 'None':
            model = hdbscan.HDBSCAN(min_cluster_size=int(input.mcs),
                                    min_samples=int(input.ms),
                                    allow_single_cluster=True,
                                    cluster_selection_epsilon=input.selection_epsilon,
                                    cluster_selection_method=input.selection_method,
                                    metric=input.selection_metric)
        elif input.ms != 'None':
            model = hdbscan.HDBSCAN(min_cluster_size=int(input.mcs),
                                    allow_single_cluster=True,
                                    cluster_selection_epsilon=input.selection_epsilon,
                                    cluster_selection_method=input.selection_method,
                                    metric=input.selection_metric)
        else:
            model = hdbscan.HDBSCAN(allow_single_cluster=True,
                                    cluster_selection_epsilon=input.selection_epsilon,
                                    cluster_selection_method=input.selection_method,
                                    metric=input.selection_metric)
        model.fit(features[mask, :])
        labels = model.fit_predict(features)

        # remove noise:
        #labels_locs = np.argwhere(labels!=-1.)[:,0]
        #cluster_labels = labels[labels_locs]
        #features_no_noise = features[labels_locs,:]
        #noise_locs = np.argwhere(labels==-1.)[:,0]

        # keep noise:
        labels_locs = np.argwhere(labels!=-2.)[:,0]
        cluster_labels = labels[labels_locs]
        features_no_noise = features[labels_locs,:]
        noise_locs = np.argwhere(labels==-2.)[:,0]

        input.labels_locs = labels_locs
        input.noise_locs = noise_locs
        input.cluster_labels = cluster_labels
        #print('\n  HDBSCAN labels = ',np.unique(cluster_labels))
        input.nc = len(np.unique(cluster_labels))
        # "-1" label values in DBSCAN are "noise" and not clusters, therefore
        # nc (number of clusters) is = number of unique labels minus one.
        input.ng = np.shape(cluster_labels)[0] # number of labeled grid points (less than original)

        if input.ng == 0: # entire domain is noise!
            print('\n  CAUTION: entire domain labeled as noise\n')
            input.labels_locs = noise_locs
            input.noise_locs = labels_locs
            input.cluster_labels = np.zeros([len(input.labels_locs)],dtype=int)
            input.ng = np.shape(input.cluster_labels)[0]
            input.nc = len(np.unique(input.cluster_labels))

        if input.verbose == True:
            print('\n  cluster: np.unique(cluster_labels) = ',np.unique(cluster_labels))
            print('  len(input.labels_locs) = ',len(input.labels_locs))
            print('  len(input.noise_locs) = ',len(input.noise_locs))
            print('  len(input.feature_locs) = ',len(input.feature_locs))
            print('  len(input.noise_locs)+len(input.labels_locs) = ',len(input.noise_locs)+len(input.labels_locs))
            print('  input.ng = ',input.ng)
            print('  input.nc = ',input.nc)

        return input

    elif input.cluster_method == 'KMeans':

        from sklearn.cluster import KMeans

        #print('  np.amax(features), np.amin(features) = ',np.amax(features), np.amin(features))
        #print('  np.amax(input.raw_features), np.amin(input.raw_features) = ',np.amax(input.raw_features), np.amin(input.raw_features))

        # Fit Gaussian mixture model
        seed = 3696299933  #  Keep a seed for debugging/plotting
        model = KMeans(n_clusters=input.K, random_state=seed)

        # Permutation of the data
        #print('input.sample_pct = ',input.sample_pct)
        mask = np.random.permutation(features.shape[0])[:int(input.sample_pct*features.shape[0])]
        model.fit(features[mask, :])

        # "Predict" labels for the entire domain
        labels = model.predict(features)

        input.cluster_labels = labels
        input.nc = input.K
        print('\n  np.unique(input.cluster_labels) = ',np.unique(input.cluster_labels))
        print('  input.nc = ',input.nc)
        if input.ic_flag == True:
            gmm = GaussianMixture(n_components=input.K, random_state=seed, init_params='kmeans')
            gmm.fit(features[mask, :])
            input.aic = gmm.aic(features[mask, :])
            input.bic = gmm.bic(features[mask, :])
            if input.verbose == True:
                print('\n  aic = ',input.aic)
                print('  bic = ',input.bic)

        return input

    elif input.cluster_method == 'Spectral':

        from sklearn.cluster import SpectralClustering

        # Train on permutation of the data
        seed = 3696299933
        model = SpectralClustering(n_clusters=input.K,assign_labels="discretize",random_state=seed)
        mask = np.random.permutation(features.shape[0])[:int(input.sample_pct*features.shape[0])]
        model.fit(features[mask, :])
        labels = model.fit_predict(features)
        labels_locs = np.argwhere(labels!=-1.)[:,0]
        cluster_labels = labels[labels_locs]
        features_no_noise = features[labels_locs,:]
        noise_locs = np.argwhere(labels==-1.)[:,0]

        input.labels_locs = labels_locs
        input.noise_locs = noise_locs
        input.cluster_labels = cluster_labels
        input.nc = len(np.unique(cluster_labels))
        # "-1" label values in DBSCAN are "noise" and not clusters, therefore
        # nc (number of clusters) is = number of unique labels minus one.
        input.ng = np.shape(cluster_labels)[0] # number of labeled grid points (less than original)

        return input


def get_TBL_features( path, standardize_flag ):

    TBL_file_path = path + 'Transition_BL_Time_Averaged_Profiles.h5'

    file = h5py.File( TBL_file_path , 'r')
    #print(list(f.keys()))

    x = np.array(file['x_coor'])
    y = np.array(file['y_coor'])
    u = np.array(file['um'])
    v = np.array(file['vm'])
    p = np.array(file['pm'])
    Ruu = np.array(file['uum']) - u**2
    Ruv = np.array(file['uvm']) - u*v
    Rvv = np.array(file['uvm']) - v**2

    # Visualize by wall-normal Reynolds stress
    X, Y = np.meshgrid(x, y)

    # Include 99% isocontour of mean velocity
    U_inf = 1
    nu = 1/800  # Details in README-transition_bl.pdf
    Re = (U_inf/nu)*x

    # take derivatives

    dx = x[1]-x[0]
    dy = y[1:]-y[:-1]

    nx = len(x)
    ny = len(y)

    Dy = sparse.diags( [-1, 1], [-1, 1], shape=(ny, ny) ).toarray()

    # Second-order forward/backwards at boundaries
    Dy[0, :3] = np.array([-3, 4, -1])
    Dy[-1, -3:] = np.array([1, -4, 3])
    for i in range(ny-1):
        Dy[i, :] = Dy[i, :]/(2*dy[i])
    Dy[-1, :] = Dy[-1, :]/(2*dy[-1])

    # Repeat for each x-location
    Dy = sparse.block_diag([Dy for i in range(nx)])

    Dx = sparse.diags( [-1, 1], [-ny, ny], shape=(nx*ny, nx*ny))
    Dx = sparse.lil_matrix(Dx)
    # Second-order forwards/backwards at boundaries
    for i in range(ny):
        Dx[i, i] = -3
        Dx[i, ny+i] = 4
        Dx[i, 2*ny+i] = -1
        Dx[-(i+1), -(i+1)] = 3
        Dx[-(i+1), -(ny+i+1)] = -4
        Dx[-(i+1), -(2*ny+i+1)] = 1
    Dx = Dx/(2*dx)

    Dx = sparse.csr_matrix(Dx)
    Dy = sparse.csr_matrix(Dy)

    Dxx = 2*(Dx @ Dx)
    Dyy = 2*(Dy @ Dy)

    u = u.flatten('F')
    v = v.flatten('F')
    p = p.flatten('F')
    Ruu = Ruu.flatten('F')
    Ruv = Ruv.flatten('F')

    ux = Dx @ u
    uy = Dy @ u
    vx = Dx @ v
    vy = Dy @ v
    px = Dx @ p
    py = Dy @ p
    lap_u = (Dxx + Dyy) @ u
    Ruux = Dx @ Ruu
    Ruvy = Dy @ Ruv

    #print('budget closure = ',np.sum(np.array([u*ux, v*uy, px, nu*lap_u, Ruvy, Ruux])))

    # Advection, pressure, viscous, Reynolds stresses
    #raw_features = 1e3*np.vstack([u*ux, v*uy, px, nu*lap_u, Ruvy, Ruux]).T
    raw_features = np.vstack([u*ux, v*uy, px, nu*lap_u, Ruvy, Ruux]).T

    # Callaham multiplied by 1e3 for SPCA

    #print('\n  np.shape(raw_features) = ',np.shape(raw_features))
    #print('  np.mean(np.sum(raw_features,axis=1)) = ',np.mean(np.sum(raw_features,axis=1)))
    #print('  np.std(np.sum(raw_features,axis=1)) = ',np.std(np.sum(raw_features,axis=1)))
    #print('  np.mean(raw_features,axis=0) = ',np.mean(raw_features,axis=0))
    #print('  np.sum(np.mean(raw_features,axis=0)) = ',np.sum(np.mean(raw_features,axis=0)))


    # data is cell centered, get the
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

    # standardize:
    if standardize_flag == 'on':
        #features = np.zeros(np.shape(raw_features))
        #for i in range(0,raw_features.shape[1]):
        #    features[:,i] = standardize( raw_features[:,i] )
        transformer = RobustScaler().fit(raw_features)
        features = transformer.transform(raw_features)
    else:
        features = np.copy(raw_features)

    eqn_labels = [r'$\bar{u} \bar{u}_x$', r'$\bar{v}\bar{u}_y$', r'$\rho^{-1} \bar{p}_x$',
                  r'$\nu \nabla^2 \bar{u}$', r'$\overline{(u^\prime v^\prime)}_y$', r'$\overline{({u^\prime} ^2)}_x$']

    nan_locs = np.array([]) #np.empty([])
    feature_locs = np.arange(features.shape[0])

    area = (area).flatten('F')
    area = area[feature_locs]
    area = area/np.sum(area)

    data = make_feature_dictionary( x, y, area, features, raw_features, eqn_labels, nan_locs, feature_locs )

    return data


def get_m_score( balance, features, data ):
    # features need to be raw, i.e. not standardized.

    if data.bias_flag == 'biased':

        if sum(balance) == 0.:
            balance = np.ones(np.shape(balance))

        # 1) get the differences for the full set of features
        diffs_full,id_full = get_diff_vec( np.ones(np.shape(balance)) , features )
        #print('diffs_full = ',diffs_full)

        # 2) get the differences for the reduced set of features
        diffs_red,id_red = get_diff_vec( balance , features )
        #print('diffs_red = ',diffs_red)

        # if any difference is exactly zero:
        if np.any(diffs_red==0.) == True:
            loc0 = np.argwhere(diffs_red==0.)[:,0]
            diffs_red[loc0] = (1e-100)*np.ones(np.shape(loc0)[0])

        if np.any(diffs_full==0.) == True:
            loc0 = np.argwhere(diffs_full==0.)[:,0]
            diffs_full[loc0] = (1e-100)*np.ones(np.shape(loc0)[0])

        sum_score = np.sum(np.log10(diffs_red)) / np.sum(np.log10(diffs_full))

        # 5) bias
        Mactive = np.shape(np.nonzero(balance)[0])[0] # number of active features in subset
        Qactive = np.shape(diffs_red)[0] # number of active differences in subset
        if Qactive == 0:
            print('\n\n  ERROR: Nb = 0 \n\n')
        factor = (Qactive+1.)/(2.*Qactive) # (Nb+1)/(2*Nb)

        score = sum_score * factor
        closure_score = np.nan
        return score, closure_score

    elif data.bias_flag == 'unbiased':
        # data = [samples,features]

        if np.shape(np.shape(features))[0] == 1: # vector (single data sample)

            if np.sum(balance) == data.nf: # all ones
                #print(' balance = ', balance)
                score = 0.
                score_95 = 0.
            elif np.sum(balance) == 0: # all zeros
                #print(' balance 0 = ', balance)
                score = 0.
                score_95 = 0.
            else:
                #print('\n  features = ',features)
                #print('  balance = ', balance)
                if np.amin(np.abs(features)) == 0.:
                    nonzero_features = np.abs(features[np.nonzero(features)])
                    np.abs(features)/np.amin(np.abs(nonzero_features))
                    #features_normalized = np.abs(features)/np.array([1e-20])
                else:
                    features_normalized = np.abs(features)/np.amin(np.abs(features))
                #print(' features_normalized = ',features_normalized)
                select,remain = get_sets( balance, features_normalized  )
                select_max,select_min,select_imax,select_imin = find_maxmin( select )
                remain_max,remain_min,remain_imax,remain_imin = find_maxmin( remain )
                if select_min <= remain_max:
                    G = 0.
                else:
                    G = ( np.log(select_min - remain_max) ) / ( np.log(select_min+remain_max) )

                #print('  G = ',G)
                if G < 0.:
                    G = 0.

                if select_min == select_max:
                    P = 0.
                else:
                    P =  np.log10(select_max) - np.log10(select_min)

                score = G / (1. + P )
                score_95 = 0. #np.inf # fix this! propogate error from the standard error of the feature mean...
                #print('  score = ',score)

        elif np.shape(np.shape(features))[0] >= 2: # array (many samples, e.g. a cluster)
            #print('\n  unbiased, feature shape = 2 \n')
            # need to be the area-weighted averages:
            print('  ERROR: unbiased score must be calculated using the cluster-mean features, depreciated ')

            # mean(log(x)) score:
            """
            weights = data.cluster_area
            if data.score_confidence == 'SEM':

                score_rr = np.zeros([len(weights)])
                for rr in range(0,len(weights)):
                    if np.sum(balance) == data.nf: # all ones
                        score_rr[rr] = 0.
                    elif np.sum(balance) == 0: # all zeros
                        score_rr[rr] = 0.
                    else:
                        #features_normalized = features[rr,:]/np.amin(features[rr,:])
                        #select,remain = get_sets( balance, features[rr,:] )
                        #
                        #select_max,select_min,select_imax,select_imin = find_maxmin( select )
                        #remain_max,remain_min,remain_imax,remain_imin = find_maxmin( remain )
                        #if select_min < remain_max:
                        #    G = 0.
                        #else:
                        #    G = abs( ( abs(np.log10(np.abs(select_min))) - abs(np.log10(np.abs(remain_max))) ) / ( abs(np.log10(np.abs(select_min))) + abs(np.log10(np.abs(remain_max))) ) )
                        #P = abs( ( abs(np.log10(np.abs(select_min))) - abs(np.log10(np.abs(select_max))) ) / ( abs(np.log10(np.abs(select_min))) + abs(np.log10(np.abs(select_max))) ) )
                        #score[rr] = G / (1. + P )
                        if np.amin(np.abs(features[rr,:])) == 0.:
                            #features_normalized = np.array([1e-20])
                            features_normalized = np.abs(features[rr,:])/np.array([1e-20])
                        else:
                            features_normalized = np.abs(features[rr,:])/np.amin(np.abs(features[rr,:]))

                        #features_normalized = np.abs(features[rr,:])/np.amin(np.abs(features[rr,:]))
                        #print(' features_normalized = ',features_normalized)
                        select,remain = get_sets( balance, features_normalized  )
                        select_max,select_min,select_imax,select_imin = find_maxmin( select )
                        remain_max,remain_min,remain_imax,remain_imin = find_maxmin( remain )
                        if select_min <= remain_max:
                            G = 0.
                        else:
                            G = ( np.log(select_min - remain_max) ) / ( np.log(select_min+remain_max) )
                            #print('np.log(select_min - remain_max) = ',np.log(select_min - remain_max))
                            #print('np.log(select_min + remain_max) = ',np.log(select_min + remain_max))

                        if G < 0.:
                            G = 0.

                        if select_min == select_max:
                            P = 0.
                        else:
                            P =  np.log10(select_max) - np.log10(select_min)
                            #print('np.log10(select_max) - np.log10(select_min) = ',np.log10(select_max) - np.log10(select_min))
                        score_rr[rr] = G / (1. + P )


                score, stddev = weighted_avg_and_std( score_rr, weights )

                #print('  score,stddev  = ',score, stddev)
                score_95 = stddev / np.sqrt(len(weights)) # standard error of the mean, 95% confidence: std. dev / sqrt(N)
                #score = np.nanmean(score_boot,axis=0)
                #score_95 = 2.*np.nanstd(score_boot,axis=0) # 95% confidence intervals

                #print('  score = ',score)
                #print('  score_95 = ',score_95)

            elif data.score_confidence == 'bootstrap':
                print('  ERROR: bootstrapped unbiased score is not ready')
                # add error bars from bootstrapping instead:
                Nbootstrap = data.Nbootstrap
                NSAMPLE = np.shape(features)[0]
                score_boot = np.zeros([int(Nbootstrap),np.shape(features)[1]])
                for kk in range(0,int(Nbootstrap)):
                    locs = np.random.choice(range(NSAMPLE), NSAMPLE)
                    mean_features = np.nanmean( data[locs,:] , axis=0 )
                    select,remain = get_sets( balance, mean_features )
                    select_max,select_min,select_imax,select_imin = find_maxmin( select )
                    remain_max,remain_min,remain_imax,remain_imin = find_maxmin( remain )
                    if select_min < remain_max:
                        G = 0.
                    else:
                        G = abs( ( abs(np.log10(np.abs(select_min))) - abs(np.log10(np.abs(remain_max))) ) / ( abs(np.log10(np.abs(select_min))) + abs(np.log10(np.abs(remain_max))) ) )
                    P = abs( ( abs(np.log10(np.abs(select_min))) - abs(np.log10(np.abs(select_max))) ) / ( abs(np.log10(np.abs(select_min))) + abs(np.log10(np.abs(select_max))) ) )
                    score_boot[kk,:] = G / (1. + P )
                # standard error of the mean:
                score = np.nanmean(score_boot,axis=0)
                score_95 = 2.*np.nanstd(score_boot,axis=0) # 95% confidence intervals
            """
            score = np.nan
            score_95 = np.nan

        return score, score_95

def find_maxmin( vector ):
    # find the max/min absolute values
    max = np.amax(np.abs(vector))
    min = np.amin(np.abs(vector))
    imax = np.argwhere(np.abs(vector)==max)[0,0]
    imin = np.argwhere(np.abs(vector)==min)[0,0]
    return max,min,imax,imin

def get_sets( hypothesis, vector ):
    # find the absolute values of the selected and remainder subsets
    iselect = np.flatnonzero(hypothesis)
    iremain = np.flatnonzero(hypothesis-1.)
    select = np.abs(vector[iselect])
    remain = np.abs(vector[iremain])
    return select,remain




def get_diff_vec( grid_balance, grid_features ):
    #print('\n')
    locs = np.nonzero(grid_balance)[0] # grid balance is 0's and 1's for on/off terms
    if np.shape(locs)[0] == 0: # quiescent balance: trivial solution
        locs = np.nonzero(np.ones(np.shape(grid_balance)))[0] # 4 zeros = 4 ones in terms of differences
    abs_balance = np.abs(grid_features[locs]) # absolute values of 'on'  terms
    #print('abs_balance = ',abs_balance)
    N_balance = np.shape(abs_balance)[0] # number of 'on' terms
    max_abs_term = np.amax(np.abs(grid_features)) # maximum term, may or may not be in grid_balance!
    #print('max_abs_term = ',max_abs_term)

    max_loc = np.argwhere(abs_balance==max_abs_term)
    #print('abs_balance = ',abs_balance)
    #print('max_loc = ',max_loc)

    if np.shape(max_loc)[0] == 0:
        max_loc = np.nan
    else:
        max_loc = max_loc[0]
    diffs = np.zeros([N_balance])
    for ib in range(0,N_balance):
        if ib == max_loc:
            diffs[ib] = np.nan
        else:
            diffs[ib] = abs( max_abs_term - abs_balance[ib] ) /  abs( max_abs_term + abs_balance[ib] )

    id = locs[np.argwhere(np.isnan(diffs)-1)[:,0]] # difference indices
    #print('id = ',id)
    diffs = diffs[np.argwhere(np.isnan(diffs)-1)[:,0]] # differences
    #print('diffs = ',diffs)
    if np.shape(diffs)[0] == 0: # if no differences are found
        return np.array([1.]),0
    else:
        return diffs,id


def generate_balances( nf , bias_flag ):
    # nf = number of features
    #nf = 3
    nc = int(np.power(2,nf)-1)
    grid_balance = np.zeros([nc,nf])
    i = 0
    for vertex in generate_vertices(nf):
        if sum(vertex) == 0.:
            continue
        else:
            grid_balance[i,:] = vertex
            i = i+1
    # remove balances with only one term:
    grid_balance = grid_balance[nf:None,:]
    if bias_flag == 'unbiased':
        # remove full set:
        grid_balance = grid_balance[0:np.shape(grid_balance)[0]-1,:]
    return grid_balance

def generate_vertices(n):
    # n is the number of dimensions of the cube (3 for a 3d cube)
    for number_of_ones in range(0, n + 1):
        for location_of_ones in combinations(range(0, n), number_of_ones):
            result = [0] * n
            for location in location_of_ones:
                result[location] = 1
            yield result

def rows_uniq_elems(arr):
    #a_sorted = np.sort(a,axis=-1)
    #return a[(a_sorted[...,1:] != a_sorted[...,:-1]).all(-1)]
    _, indices = np.unique(arr[:, 0], return_index=True)
    return arr[indices, :]

def bootstrap_mean( A , Nbootstrap ):
    if Nbootstrap == 0: # bootstrapping is off
        SM = np.nanmean(A,axis=0)
        SD = 0.
        return SM,SD
    else: # bootstrapping is on
        # 95% confidence intervals from bootstrapping
        # A = [samples,features], average bootstrap over samples
        if np.shape(A)[0] == 1:
            SM = np.array([A],dtype=float)
            SD = np.array([0.],dtype=float)
            return SM,2*SD
        else:
            NSAMPLE = np.shape(A)[0]
            AMEAN = np.zeros([int(Nbootstrap),np.shape(A)[1]])
            for kk in range(0,int(Nbootstrap)):
                locs = np.random.choice(range(NSAMPLE), NSAMPLE)
                AMEAN[kk,:] = np.nanmean(A[locs,:],axis=0)
            # standard error of the mean:
            SM = np.nanmean(AMEAN,axis=0)
            #SM = np.nanmean(A,axis=0)
            SD = np.nanstd(AMEAN,axis=0)
            return SM,2*SD

def skewness_flatness( arr ):
    if np.shape(np.shape(arr))[0] >= 2:
        arr = np.ndarray.flatten(arr)
    n = np.shape(arr)[0]
    if n <= 20: # less than 20 samples
        return np.nan,np.nan
    else:
        mu = np.nanmean(arr)
        m2 = np.nansum((arr-mu)**2.)/n

        # skewness
        m3 = np.nansum((arr-mu)**3.)/n
        g1 = m3 / (m2**(3./2.)) # skewness
        #g1sp = stats.skew(arr, axis=0, bias=True) # scipy skewness (THE SAME)
        G1 = g1 * np.sqrt( n*(n-1.)/(n-2.) ) # sample skewness
        SES = np.sqrt( 6.*n*(n-1.) / ((n-2.)*(n+1.)*(n+3.)) ) # std. error of skewness
        Zg1 = G1 / SES # test statistic
        #Zg1sp,_ = stats.skewtest(arr, axis=0) # scipy test statistic
        G1_95 = np.array([G1+2.*SES,G1-2.*SES])
        # * if Zg1 < -2 the population is very likely skewed negatively (though you
        #   don’t know by how much).
        # * If Z is between −2 and +2, you can’t reach any conclusion about the
        #   skewness of the population: it might be symmetric or skewed either way
        # * if Zg1 > 2 the population is very likely skewed positively (though you
        #   don’t know by how much).

        # sk_test = sp.stats.skewtest(arr, axis=0)
        # sk_test tests the null hypothesis that the skewness of the population
        # that the sample was drawn from is the same as that of a corresponding
        # normal distribution. Computes z-score and 2-sided p-value

        # excess kurtosis
        m4 = np.nansum((arr-mu)**4.)/n
        g2 = m4 / (m2**(2.)) - 3. # excess kurtosis
        G2 = ((n+1.)*g2+6.) * (n-1.)/((n-2.)*(n-3.)) # sample excess kurtosis
        SEK = 2.*SES*np.sqrt( (n**2.-1.) / ((n-3.)*(n+5.)) )
        Zg2 = G2 / SEK # test statistic
        # * if Zg2 < -2 the population is very likely negative excess kurtosis
        #   (though you don’t know by how much).
        # * If Z is between −2 and +2, you can’t reach any conclusion about the
        #   kurtosis of the population
        # * if Zg2 > 2 the population is very likely skewed positive excess kurtosis
        #   (though you don’t know by how much).

        #return Zg1,Zg2 # inferred stats
        #return G1,G2 # sample stats
        return g1,g2 # stats

def generate_vertices(n):
    # n is the number of dimensions of the cube (3 for a 3d cube)
    for number_of_ones in range(0, n + 1):
        for location_of_ones in combinations(range(0, n), number_of_ones):
            result = [0] * n
            for location in location_of_ones:
                result[location] = 1
            yield result


def random_eqn( N , low_power, high_power , leading_order ):
    # N = number of features (equation terms)
    # low_power/high_power = lowest and highest power of 10 in equation

    if leading_order <= 1:
        print('ERROR: leading order must be >= 2')
        return np.array([np.nan])

    elif leading_order >= 2: # number of leading_order terms is even (>=2)

        eqn = []
        closure = 1.
        while closure != 0:
            normval = 2.
            while normval >= 1.025:
                following_order = int(N-leading_order)
                fuzz = np.random.uniform(low=0., high=1., size=1)*(10.**(low_power))
                if is_even(leading_order) == True:
                    nl1 = leading_order-1
                    nf1 = following_order-1
                    lead_part1 = np.random.uniform(low=0.5,high=1.,size=nl1)*(10.**(high_power))*np.random.choice([1., -1.],size=nl1)
                    if leading_order == int(N): # get rid of fuzz if number of leading order terms = N
                        lead_part2 = - (np.sum(lead_part1))*np.ones([1])
                    else:
                        lead_part2 = - (np.sum(lead_part1) - fuzz)*np.ones([1])
                    lead = np.append(lead_part1,lead_part2)
                    fuzz2 = np.sum(lead)
                    if leading_order == int(N):
                        follow = np.empty([0])
                    elif leading_order == int(N-1):
                        follow = - fuzz2
                    else:
                        follow_part1 = np.power( np.ones([nf1])*10. , np.random.uniform(low=low_power, high=(high_power-1), size=nf1) )
                        follow_part2 = - np.sum(follow_part1)*np.ones([1]) + fuzz2
                        follow = -np.append(follow_part1,follow_part2)
                else: # is_even(leading_order) == False: number of leading_order terms is odd (>=3)
                    nl1 = leading_order-2
                    nf1 = following_order-1
                    lead_part1 = np.random.uniform(low=0.5,high=1.,size=nl1)*(10.**(high_power))*np.random.choice([1., -1.],size=nl1)
                    if leading_order == int(N): # get rid of fuzz if number of leading order terms = N
                        lead_part2 = - np.longdouble(np.sum(lead_part1) )*np.array([0.49999999999999,0.50000000000001])
                    else:
                        lead_part2 = - np.longdouble(np.sum(lead_part1) - fuzz )*np.array([0.49999999999999,0.50000000000001])
                    lead = np.append(lead_part1,lead_part2)
                    fuzz2 = np.sum(lead)
                    if leading_order == int(N):
                        follow = np.empty([0])
                    elif leading_order == int(N-1):
                        follow = - fuzz2
                    else:
                        follow_part1 = np.power( np.ones([nf1])*10. , np.random.uniform(low=low_power, high=(high_power-1), size=nf1) )
                        follow_part2 = - ( np.sum(follow_part1) - fuzz2 )*np.ones([1])
                        follow = - np.append(follow_part1,follow_part2)
                eqn = np.append(lead,follow)
                normval = np.linalg.norm(eqn[0:leading_order]*10.**(-high_power))
                closure = abs(np.sum(eqn))
                if np.any(np.floor(np.log10(abs(lead))) !=  high_power-1): # if the high order terms don't match order
                    normval = 2.
                if normval <= 0.975: # if the high order terms aren't close enough in magnitude
                    normval = 2.
            if closure == 0.:
                return eqn


def is_even( num ):
    if num % 2 == 0:
        return True
    else:
        return False


def minus_plus():
    # returns -1 or +1 from a uniform distribution
    return 1.0 if random.random() < 0.5 else -1.0


def get_diff_err_example( grid_balance , grid_features ):

    # 1) get the differences for the full set of features
    diffs_full,id_full = get_diff_vec( np.ones(np.shape(grid_balance)) , grid_features )

    # 2) get the differences for the reduced set of features
    diffs_red,id_red = get_diff_vec( grid_balance , grid_features )

    # 4) sum score:
    if np.any(diffs_red==0.) == True:
        loc0 = np.argwhere(diffs_red==0.)[:,0]
        diffs_red[loc0] = 1e-100*np.ones(np.shape(loc0)[0])

    if np.any(diffs_full==0.) == True:
        loc0 = np.argwhere(diffs_full==0.)[:,0]
        diffs_full[loc0] = 1e-100*np.ones(np.shape(loc0)[0])

    sum_score = np.sum(np.log10(diffs_red)) / np.sum(np.log10(diffs_full))

    # 5) normalization
    Mactive = np.shape(np.nonzero(grid_balance)[0])[0] # number of active features in subset
    Qactive = np.shape(diffs_red)[0] # number of active differences in subset
    # basically we're multiplying 1/Qactive * Mactive/Mtotal
    factor = 1/Qactive * Mactive/2 # = 1/Qactive takes mean, Mactive/2 normalizes. Between 0.5 and 1.0
    score = sum_score * factor

    # closure_score:
    chosen_locs = np.nonzero(grid_balance)[0] # grid balance is 0's and 1's for on/off terms
    neglected_locs = np.nonzero(grid_balance-1.)[0] # grid balance is 0's and 1's for on/off terms

    sum_chosen_balance = np.sum(grid_features[chosen_locs])
    sum_neglected_balance = np.sum(grid_features[neglected_locs]) # absolute values of 'on'  terms

    # closure def 5:
    # ADD: if neglected_locs is empty
    if np.sum(abs(grid_features[chosen_locs])) == 0.:
        #closure_score = 1.
        closure_score = np.sum(abs(grid_features[neglected_locs])) #
    else:
        closure_score = np.sum(abs(grid_features[neglected_locs])) / np.sum(abs(grid_features[chosen_locs]))

    return score, closure_score, grid_balance #, np.sum(np.log10(diffs_red))


def get_tumor_dynamics_features( path, standardize_flag, data_set_name ):

    filename_features = path + 'features.npy'
    raw_features = np.load(filename_features)

    # standardize:
    if standardize_flag == 'on':
        features = np.zeros(np.shape(raw_features))
        for i in range(0,raw_features.shape[1]):
            features[:,i] = standardize( raw_features[:,i] )
            #print('standardized features[:,i] mu, sig = ', np.mean(features[:,i]),np.std(features[:,i]))
    else:
        features = raw_features

    filename_area = path + 'area.npy'
    area = np.load(filename_area)
    filename_x = path + 'x.npy'
    x = np.load(filename_x)
    filename_y = path + 'y.npy'
    y = np.load(filename_y)

    if data_set_name == 'tumor_dynamics':
        eqn_labels = [r'$\partial{n}/\partial{t}$', r'$d_n\nabla m \cdot \nabla n$', r'$d_n m \nabla^2n$',
                      r'$\rho \nabla n \cdot \nabla f$', r'$\rho n\nabla^2 f$', r'$\lambda n (1-n-f)$']
    elif data_set_name == 'tumor_angiogenesis':
        eqn_labels = [r'$\partial{n}/\partial{t}$', r'$D\nabla^2 n$', r'$\chi n \nabla^2 c$',
                      r'$\chi\nabla{n}\cdot\nabla{c}$', r'$n\nabla\chi\cdot\nabla{c}$', r'$\rho n \nabla^2 f$', r'$\rho \nabla{n}\cdot\nabla{f}$']

    dx = x[1]-x[0]; dy = y[1:]-y[:-1]
    nx = len(x); ny = len(y)
    # data is cell centered, get the
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

    nan_locs = np.array([]) #np.empty([])
    feature_locs = np.arange(features.shape[0])

    area = (area).flatten('F')
    area = area[feature_locs]
    area = area/np.sum(area)

    data = make_feature_dictionary( x, y, area, features, raw_features, eqn_labels, nan_locs, feature_locs )

    return data

def make_feature_dictionary( x, y, area, features, raw_features, eqn_labels, nan_locs, feature_locs ):

    nx = len(x); ny = len(y)
    ng = int(features.shape[0]) # number of grid points
    nf = int(features.shape[1]) # number of features
    balance_combinations_unbiased = generate_balances( nf, 'unbiased' ) # every possible combination of balances
    balance_combinations_biased = generate_balances( nf, 'biased' ) # every possible combination of balances

    data = easydict.EasyDict({
        "nf": nf, # number of features
        "ng": ng, # number of grid points
        "nx": nx,
        "ny": ny,
        "x": x,
        "y": y,
        "area": area,
        "features": features,
        "raw_features": raw_features,
        "balance_combinations_biased":balance_combinations_biased,
        "balance_combinations_unbiased":balance_combinations_unbiased,
        "eqn_labels":eqn_labels,
        "map_nan_locs":nan_locs,
        "feature_locs":feature_locs,
        })

    return data

def get_feature_data( path, data_set_name, standardize_flag, residual_flag, verbose ):

    # create data.raw_area ? For HDBSCAN...

    if data_set_name == 'tumor_dynamics' or data_set_name == 'tumor_angiogenesis':
        data = get_tumor_dynamics_features( path, standardize_flag, data_set_name )

    elif data_set_name == 'TBL': #'transitional_boundary_layer':
        data = get_TBL_features( path, standardize_flag )

    elif data_set_name == 'ECCO' or data_set_name == 'ECCO_residual':
        data = get_ECCO_features( path, standardize_flag, residual_flag, verbose )

    elif data_set_name == 'Stokes':
        data = get_Stokes_features( path, standardize_flag, residual_flag, verbose )

    elif data_set_name == 'synth':
        data = get_synth_features( path, standardize_flag, residual_flag, verbose )

    return data

def get_synth_features( file_path, standardize_flag, residual_flag, verbose ):

    # grid & area:
    x = np.load(file_path + 'x.npy')
    y = np.load(file_path + 'y.npy')
    Nx = len(x); Ny = len(y)

    # features:
    features = np.load(file_path + 'features.npy')
    raw_features = np.copy(features)
    #print(' np.shape(features) = ',np.shape(features))

    #print('  np.amin(raw_features),np.amax(raw_features) = ',np.amin(raw_features),np.amax(raw_features))
    #print('  np.amin(np.abs(raw_features)),np.amax(np.abs(raw_features)) = ',np.amin(np.abs(raw_features)),np.amax(np.abs(raw_features)))
    #print('  np.std(raw_features,axis=1) = ',np.std(raw_features,axis=1))
    #print('  np.mean(raw_features,axis=1) = ',np.mean(raw_features,axis=1))
    #print('  np.log10(np.amax(np.abs(raw_features))) = ',np.log10(np.amax(np.abs(raw_features))))
    #print('  np.log10(np.amin(np.abs(raw_features))) = ',np.log10(np.amin(np.abs(raw_features))))
    #print('  np.shape(np.argwhere(raw_features==0.0)) = ',np.shape(np.argwhere(raw_features==0.0)))
    #print('\n')

    # standardized, non-dimensional features
    if standardize_flag == 'on':
        for i in range(0,np.shape(features)[1]):
            features[:,i] = standardize( features[:,i] )

    eqn_labels = [r'$|a|$', r'$|b|$',
                  r'$|c|$', r'$|d|$']

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

    nan_locs = np.array([])
    feature_locs = np.arange(features.shape[0])

    area = (area).flatten('F')
    area = area[feature_locs]
    area = area/np.sum(area)

    data = make_feature_dictionary( x, y, area, features, raw_features, eqn_labels, nan_locs, feature_locs )

    return data


def get_Stokes_features( file_path, standardize_flag, residual_flag, verbose ):

    # grid & area:
    #T = np.load(file_path + 'T.npy')
    #Z = np.load(file_path + 'Z.npy')
    x = np.load(file_path + 't.npy')
    y = np.load(file_path + 'z.npy')
    Nx = len(x); Ny = len(y)

    # non-dimensional features:
    dtk = (np.load(file_path + 'dtk.npy')).flatten('F') # RHS, minus the time rate of change of KE
    dif = (np.load(file_path + 'dif.npy')).flatten('F') # RHS, diffusion of KE
    dis = (np.load(file_path + 'dis.npy')).flatten('F') # RHS, dissipation of KE
    prv = (np.load(file_path + 'prv.npy')).flatten('F') # RHS, forcing of KE
    residual = - ( dtk + dif + dis + prv )

    # unstandardized, dimensional features
    if residual_flag == 'off':
        raw_features = np.vstack([dtk, dif, dis, prv]).T
        if verbose == True:
            print('  raw closure = ',np.sum(np.sum(raw_features,axis=0)))
    else:
        raw_features = np.vstack([dtk , dif , dis , prv, residual]).T
        #print('raw closure with residual = ',np.sum(np.sum(raw_features,axis=0)))

    # standardized, non-dimensional features
    if standardize_flag == 'on':
        dtk = standardize( dtk )
        dif = standardize( dif )
        dis = standardize( dis )
        prv = standardize( prv )
        residual = standardize( residual )
        if residual_flag == 'off':
            features = np.vstack([dtk, dif , dis , prv]).T
        else:
            features = np.vstack([dtk , dif , dis , prv, residual]).T
    else:
        features = raw_features

    if residual_flag == 'off':
        eqn_labels = [r'$-\partial_t(|\mathbf{u}|^2/2)$', r'$\nu\nabla^2(|\mathbf{u}|^2/2)$',
                      r'$-\nu\nabla\mathbf{u}\nabla\mathbf{u}$', r'$-\mathbf{u}\cdot\nabla(p/\rho)$']
    else:
        eqn_labels = [r'$-\partial_t(|\mathbf{u}|^2/2)$', r'$\nu\nabla^2(|\mathbf{u}|^2/2)$',
                      r'$-\nu\nabla\mathbf{u}\nabla\mathbf{u}$', r'$-\mathbf{u}\cdot\nabla(p/\rho)$', r'residual']

    """
    dx = x[1]-x[0]; dy = y[1:]-y[:-1]
    nx = len(x); ny = len(y)
    # data is cell centered, get the
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
    """
    # z-direction:
    omg = 2.0*np.pi/44712.
    nu = 2.0e-6 #1.6e-6 # m^2/s, kinematic viscosity
    d = (2.*nu/omg)**(1./2.)
    H = d*20.
    Nz = len(y)
    z = -np.cos(((np.linspace(1., 2.*Nz, num=int(2*Nz)))*2.-1.)/(4.*Nz)*np.pi)*H+H # centers (input .npy)
    z = z[0:Nz] # cell centers
    dz = z[1:Nz] - z[0:Nz-1]
    zb = np.append(np.array([0.]),z[0:Nz-1]+dz/2.) # cell bottoms
    zt = np.append(z[0:Nz-1]+dz/2.,np.array([H])) # cell tops
    dzc = zt-zb # centered
    # time:
    tf = 44712.
    Nt = len(x)
    #t = np.linspace(0.,tf,num=Nt,endpoint=True)
    #print('t = ',t)
    tb = np.linspace(0.,tf-(tf/Nt),num=Nt,endpoint=True)
    t = np.linspace((tf/Nt)/2.,tf-(tf/Nt)/2.,num=Nt,endpoint=True) # centers (input .npy)
    tt = np.linspace(tf/Nt,tf,num=Nt,endpoint=True)
    dtc = tt-tb # centered
    DTc,DZc = np.meshgrid(dtc,dzc)
    area = DTc*DZc

    nan_locs = np.array([])
    feature_locs = np.arange(features.shape[0])

    area = (area).flatten('F')
    area = area[feature_locs]
    area = area/np.sum(area)

    data = make_feature_dictionary( x, y, area, features, raw_features, eqn_labels, nan_locs, feature_locs )

    return data


def remove_nans_infs( feature ):
    # features is 1d
    nan_locs = np.array([],dtype=int) # locations of infs or nans
    if np.sum(np.isinf(feature)) > 0:
        #print('here1')
        #print('np.sum(np.isinf(feature)) = ',np.sum(np.isinf(feature)))
        nan_locs = np.append(nan_locs , np.argwhere(np.isnan(feature))[:,0] )
    if np.sum(np.isnan(feature)) > 0:
        #rint('here2')
        #print('\nnp.sum(np.isnan(feature)) = ',np.sum(np.isnan(feature)))
        nan_locs = np.append(nan_locs , np.argwhere(np.isnan(feature))[:,0] )
    vals = np.ones([len(feature)])
    vals[nan_locs] = np.zeros([len(nan_locs)])
    feature_locs = np.argwhere(vals==1.)[:,0]
    if np.shape(feature_locs)[0]+np.shape(nan_locs)[0] != np.shape(feature)[0]:
        print('\n\nERROR: mismatched nan/inf locations and usable data locations\n\n')
    #print(np.shape(feature[:]))
    #print(np.shape(feature_locs)[0])
    #print(np.shape(feature_locs)[0]+np.shape(nan_locs)[0])
    feature = feature[feature_locs]
    #print(np.shape(feature[:]))
    #print('')
    return nan_locs, feature_locs, feature

def plot_ECCO_features( curlCori, BPT, curlTau, curlA, curlB, area ):

    Nx=720
    Ny=360
    x = np.linspace(0.5,float(Nx-0.5),num=Nx,endpoint=True)
    y = np.linspace(0.5,float(Ny-0.5),num=Ny,endpoint=True)
    x = x/2
    y = y/2-90.
    Y,X = np.meshgrid(y,x)
    ifontsize=18

    figure_name = '/Users/bkaiser/Documents/data/m_score/ECCO/figures/world.png'
    plotname = figure_name
    fig = plt.figure(figsize=(22, 10))
    plt.subplot(2,3,1)
    cs=plt.contourf(X,Y,area,levels=200,cmap='seismic')
    plt.xlabel('$x$',fontsize=ifontsize); plt.ylabel('$y$',fontsize=ifontsize)
    cbar=plt.colorbar(cs);  plt.title('ECCO grid cell area',fontsize=ifontsize)
    cbar.ax.get_yaxis().labelpad = 20

    plt.subplot(2,3,2)
    cs=plt.contourf(X,Y,BPT,levels=200,cmap='seismic')
    plt.xlabel('$x$',fontsize=ifontsize); plt.ylabel('$y$',fontsize=ifontsize)
    cbar=plt.colorbar(cs);  plt.title('bottom pressure torque',fontsize=ifontsize)
    cbar.ax.get_yaxis().labelpad = 20

    plt.subplot(2,3,3)
    cs=plt.contourf(X,Y,curlCori,levels=200,cmap='seismic')
    plt.xlabel('$x$',fontsize=ifontsize); plt.ylabel('$y$',fontsize=ifontsize)
    cbar=plt.colorbar(cs);  plt.title('advection of planetary vorticity',fontsize=ifontsize)
    cbar.ax.get_yaxis().labelpad = 20

    plt.subplot(2,3,4)
    cs=plt.contourf(X,Y,curlTau,levels=200,cmap='seismic')
    plt.xlabel('$x$',fontsize=ifontsize); plt.ylabel('$y$',fontsize=ifontsize)
    cbar=plt.colorbar(cs);  plt.title('wind \& bottom stress curl',fontsize=ifontsize)
    cbar.ax.get_yaxis().labelpad = 20

    plt.subplot(2,3,5)
    cs=plt.contourf(X,Y,curlA,levels=200,cmap='seismic')
    plt.xlabel('$x$',fontsize=ifontsize); plt.ylabel('$y$',fontsize=ifontsize)
    cbar=plt.colorbar(cs);  plt.title(r'$\nabla\times\mathbf{A}$',fontsize=ifontsize)
    cbar.ax.get_yaxis().labelpad = 20

    plt.subplot(2,3,6)
    cs=plt.contourf(X,Y,curlB,levels=200,cmap='seismic')
    plt.xlabel('$x$',fontsize=ifontsize); plt.ylabel('$y$',fontsize=ifontsize)
    cbar=plt.colorbar(cs);  plt.title(r'$\nabla\times\mathbf{B}$',fontsize=ifontsize)
    cbar.ax.get_yaxis().labelpad = 20

    plt.subplots_adjust(top=0.95, bottom=0.075, left=0.05, right=0.975, hspace=0.3, wspace=0.2)
    plt.savefig(plotname,format="png"); plt.close(fig);

    return

def check_location_matching( array1, array2 ):
    if len(array1) != len(array2):
        print('ERROR: location arrays do not match in length')
    else:
        if np.sum(array1-array2) != 0:
            print('ERROR: location array elements do not match')
    return

def check_locations( nan_locs, feature_locs ):
    N = len(nan_locs) + len(feature_locs)
    test = np.arange(N)
    locs = np.zeros([N])
    locs[nan_locs] = nan_locs
    locs[feature_locs] = feature_locs
    should_be_zero = np.sum(locs-test)
    if should_be_zero != 0:
        print('ERROR: number of feature locations and nan/inf locs does not agree')
    return

def standardize( var ):
    varmean = np.nanmean(var)
    varstd = np.nanstd(var)
    var = ( var - varmean ) / varstd
    return var

def normalize( var ):
    varmax = np.amax(var)
    varmin = np.amin(var)
    var = ( var - varmin ) / ( varmax - varmin )
    return var

def get_ECCO_features( file_path, standardize_flag, residual_flag, verbose ):

    # grid:
    import scipy.io
    grid = scipy.io.loadmat(file_path + 'gridVars.mat')
    land = grid['land'][:]
    #area = grid['vortCellArea'][:] #().flatten('F')
    #area = area/np.nansum(np.nansum(area))
    area = grid['vortCellArea'][:] #).flatten('F')
    #area = area/np.nansum(np.nansum(area))

    Nx = 720; Ny = 360
    x = np.linspace(0.5,float(Nx-0.5),num=Nx,endpoint=True)
    y = np.linspace(0.5,float(Ny-0.5),num=Ny,endpoint=True)
    x = x/2; y = y/2-90.;

    # features:
    curlCori = np.load(file_path + 'curlCori.npy') # planetary vorticity advection term
    BPT = np.load(file_path + 'BPT.npy')  # bottom pressure torque
    curlTau = np.load(file_path + 'curlTau.npy')  # the wind and bottom stress curl,
    curlA = np.load(file_path + 'curlA.npy')  # the nonlinear torque
    curlB = np.load(file_path + 'curlB.npy')  # the viscous torque

    # mask on residuals:
    #stdresidual = curlCori + BPT + curlTau + curlA + curlB
    #indResidual=np.where(np.logical_and(stdresidual<=1e-11, stdresidual>=-1e-11)) # Find the areas
    #maskedField=np.zeros(curlCori.shape[:]); maskedField[:]=np.NaN;
    #maskedField[indResidual]=1
    #noiseMask=maskedField #*land
    #print('np.shape(noiseMask) = ',np.shape(noiseMask))
    #print('np.unique(noiseMask) = ',np.unique(noiseMask))
    #print('np.nansum(np.isnan(noiseMask.flatten())) = ',np.nansum(np.isnan(noiseMask.flatten())) )

    nan_locsCC, feature_locsCC, curlCori = remove_nans_infs( (curlCori).flatten('F') ) # locations of nans / infs
    check_locations( nan_locsCC, feature_locsCC )
    nan_locsBP, feature_locsBP, BPT = remove_nans_infs( (BPT).flatten('F') )
    check_locations( nan_locsBP, feature_locsBP )
    check_location_matching( nan_locsCC, nan_locsBP )
    check_location_matching( feature_locsCC, feature_locsBP )
    nan_locsCT, feature_locsCT, curlTau = remove_nans_infs( (curlTau).flatten('F') )
    check_locations( nan_locsCT, feature_locsCT )
    check_location_matching( nan_locsBP , nan_locsCT )
    check_location_matching( feature_locsBP , feature_locsCT )
    nan_locsCA, feature_locsCA, curlA = remove_nans_infs( (curlA).flatten('F') )
    check_locations( nan_locsCA, feature_locsCA )
    check_location_matching( nan_locsCT , nan_locsCA )
    check_location_matching( feature_locsCT , feature_locsCA )
    nan_locsCB, feature_locsCB, curlB = remove_nans_infs( (curlB).flatten('F') )
    check_locations( nan_locsCB, feature_locsCB )
    check_location_matching( nan_locsCA , nan_locsCB )
    check_location_matching( feature_locsCA , feature_locsCB )

    residual = - (curlCori + BPT + curlTau + curlA + curlB)

    # unstandardized, dimensional features
    if residual_flag == 'off':
        raw_features = np.vstack([curlCori, BPT, curlTau, curlA, curlB]).T
        #print('raw closure = ',np.sum(np.sum(raw_features,axis=0)))
    else:
        raw_features = np.vstack([curlCori, BPT, curlTau, curlA, curlB, residual]).T
        #print('raw closure with residual = ',np.sum(np.sum(raw_features,axis=0)))

    # standardized, non-dimensional features
    if standardize_flag == 'on':
        curlCori = standardize( curlCori )
        BPT = standardize( BPT )
        curlTau = standardize( curlTau )
        curlA = standardize( curlA )
        curlB = standardize( curlB )
        residual = standardize( residual )
        if residual_flag == 'off':
            features = np.vstack([curlCori, BPT, curlTau, curlA, curlB]).T
        else:
            features = np.vstack([curlCori, BPT, curlTau, curlA, curlB, residual]).T
    else:
        features = raw_features

    if residual_flag == 'off':
        eqn_labels = [r'$\beta V$', r'$\frac{\nabla p_b \times \nabla H}{\rho}$', r'$\frac{\nabla\times\tau}{\rho}$',
                  r'$\nabla\times\mathbf{A}$', r'$\nabla\times\mathbf{B}$']
    else:
        eqn_labels = [r'$\beta V$', r'$\frac{\nabla p_b \times \nabla H}{\rho}$', r'$\frac{\nabla\times\tau}{\rho}$',
                  r'$\nabla\times\mathbf{A}$', r'$\nabla\times\mathbf{B}$', r'residual']

    area = (area).flatten('F')
    area = area[feature_locsCB]
    area = area/np.sum(area)

    if verbose == True:
        print('\n  get_ECCO_features: np.sum(area) = ',np.sum(area))
        print('  get_ECCO_features: np.shape(area) = ',np.shape(area)[0])
        print('  get_ECCO_features: np.shape(map_nan_locs) = ',np.shape(nan_locsCB)[0])
        print('  get_ECCO_features: np.shape(feature_locs) = ',np.shape(feature_locsCB)[0])
        print('  get_ECCO_features: np.shape(feature_locs)+np.shape(map_nan_locs) = ',np.shape(feature_locsCB)[0]+np.shape(nan_locsCB)[0])
        print('\n')

    #print('np.shape(nan_locsCB) = ',np.shape(nan_locsCB))
    data = make_feature_dictionary( x, y, area, features, raw_features, eqn_labels, nan_locsCB, feature_locsCB )

    data.land = land
    #labels_sonnewald = np.load( file_path + 'kCluster6.npy' )
    #data.labels_sonnewald = labels_sonnewald*noiseMask

    data.labels_sonnewald = np.load( file_path + 'kCluster6.npy' )

    return data

def read_processed_data( data, feature_path, data_set_name ):

    residual_flag = data.residual_flag
    read_path = data.read_path
    cluster_method = data.cluster_method
    reduction_method = data.reduction_method
    standardize_flag = data.standardize_flag
    NE = int(data.NE)

    feature_data = get_feature_data( feature_path, data_set_name, standardize_flag, residual_flag, data.verbose )
    features = feature_data.features
    nx = len(feature_data.x); ny = len(feature_data.y)
    data.area = (feature_data.area).flatten('F')
    data.features = feature_data.features
    data.x = feature_data.x; data.y = feature_data.y
    data.nan_locs = feature_data.map_nan_locs #<------------------ NaN loc problem
    data.feature_locs = feature_data.feature_locs

    ng = np.shape(features)[0]
    nf = np.shape(features)[1]
    data.ng = ng; data.nf = nf

    if cluster_method == 'GMM' or cluster_method == 'KMeans':
        NK = len(data.K)
        K = data.K

        if reduction_method == 'SPCA':

            index0 = data.K
            index1 = np.arange(int(data.NE))
            #index2 = np.arange(len(data.alphas))
            index2 = data.alphas*100
            index_name = 'K%i_n%i_a%i.npy'

            data = import_data( data, index_name, index0, index1, index2 )

            nk = (data.loc_max)[0] # K[nk] = K value for maximum score
            ne = (data.loc_max)[1] # ensemble realization with the best score
            na = (data.loc_max)[2] # alpha[na] = alpha value for maximum score
            data.nk = nk; data.ne = ne; data.na = na

            return data

        elif reduction_method == 'score':

            index0 = data.K
            index1 = np.arange(int(data.NE))
            index_name = 'K%i_n%i.npy'

            data = import_data( data, index_name, index0, index1 )

            nk = (data.loc_max)[0] # K[nk] = K value for maximum score
            ne = (data.loc_max)[1] # ensemble realization with the best score
            #print('ne,K[nk] = ',ne,data.K[nk])
            data.nk = nk; data.ne = ne
            #print('data.ne,data.nk = ',data.ne,data.nk)

            return data

    elif cluster_method == 'DBSCAN':
        print('\n\n  ERROR: DBSCAN depreciated \n \n')
        NM = len(data.min_samples)
        ms = data.min_samples

        Ne = len(data.epsilon)
        eps = data.epsilon

        if reduction_method == 'SPCA':

            NA = len(data.alphas)
            alphas = data.alphas

            # ADD read_3_indices here!!

        elif reduction_method == 'score':

            index0 = eps*1000
            index1 = ms
            index_name = 'eps%i_ms%i.npy'

            data = import_data( data, index_name, index0, index1 )

            neps = (data.loc_max)[0] # K[nk] = K value for maximum score
            nms = (data.loc_max)[1] # ensemble realization with the best score
            data.nms = nms; data.neps = neps

            return data

    elif cluster_method == 'HDBSCAN':

        ms = data.ms
        mcs = data.mcs
        E = np.arange(int(data.NE))

        if reduction_method == 'SPCA':

            alphas = data.alphas

            index0 = ms
            index1 = mcs
            #index2 = np.arange(len(alphas))
            index2 = data.alphas*100
            index_name = 'ms%i_mcs%i_a%i.npy'

            data = import_data( data, index_name, index0, index1, index2 )

            nms = (data.loc_max)[0] # ensemble realization with the best score
            nmcs = (data.loc_max)[1] # min samples
            na = (data.loc_max)[2] # minimum cluster size
            data.nms = nms; data.nmcs = nmcs; data.na = na

            return data

        elif reduction_method == 'score':

            index0 = E
            index1 = ms
            index2 = mcs
            index_name = 'n%i_ms%i_mcs%i.npy'

            data = import_data( data, index_name, index0, index1, index2 )

            ne = (data.loc_max)[0] # ensemble realization with the best score
            nms = (data.loc_max)[1] # min samples
            nmcs = (data.loc_max)[2] # minimum cluster size
            data.nms = nms
            data.nmcs = nmcs
            data.ne = ne

            return data



def import_data( data, index_name , *args ):

    if len(args) == 2:
        index0 = args[0]
        index1 = args[1]
        N0 = len(index0)
        N1 = len(index1)
        mean_score = np.zeros([N0,N1])
        std_score = np.zeros([N0,N1])
        mean_score_95 = np.zeros([N0,N1])
        std_score_95 = np.zeros([N0,N1])
        mean_closure = np.zeros([N0,N1])
        std_closure = np.zeros([N0,N1])
        Kr = np.zeros([N0,N1])
        fss = np.zeros([N0,N1])
        aic = np.zeros([N0,N1])
        bic = np.zeros([N0,N1])
    elif len(args) == 3:
        index0 = args[0]
        index1 = args[1]
        index2 = args[2]
        N0 = len(index0)
        N1 = len(index1)
        N2 = len(index2)
        mean_score = np.zeros([N0,N1,N2])
        std_score = np.zeros([N0,N1,N2])
        mean_score_95 = np.zeros([N0,N1,N2])
        std_score_95 = np.zeros([N0,N1,N2])
        mean_closure = np.zeros([N0,N1,N2])
        std_closure = np.zeros([N0,N1,N2])
        Kr = np.zeros([N0,N1,N2])
        fss = np.zeros([N0,N1,N2])
        aic = np.zeros([N0,N1,N2])
        bic = np.zeros([N0,N1,N2])
    elif len(args) == 4:
        index0 = args[0]
        index1 = args[1]
        index2 = args[2]
        index3 = args[3]
        N0 = len(index0)
        N1 = len(index1)
        N2 = len(index2)
        N3 = len(index3)
        mean_score = np.zeros([N0,N1,N2,N3])
        std_score = np.zeros([N0,N1,N2,N3])
        mean_score_95 = np.zeros([N0,N1,N2,N3])
        std_score_95 = np.zeros([N0,N1,N2,N3])
        mean_closure = np.zeros([N0,N1,N2,N3])
        std_closure = np.zeros([N0,N1,N2,N3])
        Kr = np.zeros([N0,N1,N2,N3])
        fss = np.zeros([N0,N1,N2,N3])
        aic = np.zeros([N0,N1,N2,N3])
        bic = np.zeros([N0,N1,N2,N3])
    else:
        print('ERROR: not the right number of arguments for import_data in utils')

    for i in range(0,N0):
        for j in range(0,N1):

            if len(args) == 2:

                area_weights_file_name = data.read_path + 'area_weights/area_weights_' + index_name %(index0[i],index1[j])

                area_weights = np.load(area_weights_file_name)
                if data.cluster_method == 'HDBSCAN':
                    ##area_weights = np.load(area_weights_file_name) / np.sum(np.load(area_weights_file_name)) # % cluster area
                    #area_weights = np.load(area_weights_file_name) / np.sum(data.area) # % total area
                    labels_locs_file_name = data.read_path + 'labels/labels_locs_' + index_name %(index0[i],index1[j])
                    data.labels_locs = np.load(labels_locs_file_name)
                    retained_columns_file_name = data.read_path + 'balance/retained_columns_' + index_name %(index0[i],index1[j])
                    data.retained_columns = np.load(retained_columns_file_name)
                else:
                    #area_weights = np.load(area_weights_file_name)
                    data.retained_columns = np.arange(data.nf)

                if np.sum(area_weights) <= 0.99999999 or np.sum(area_weights) >= 1.00000001:
                    print('\n\nERROR: sum area_weights = ',np.sum(area_weights))
                    print('\n')

                #print('\narea_weights = ',np.load(area_weights_file_name))
                #print('sum area_weights = ',np.sum(np.load(area_weights_file_name)))
                #print('np.sum(data.area) = ',np.sum(data.area))
                #print('sum area_weights / np.sum(data.area) = ',np.sum(np.load(area_weights_file_name) / np.sum(data.area)))
                #print('sum loaded area_weights = ',np.sum(area_weights))
                if data.bias_flag == 'biased':
                    score_file_name = data.read_path + 'M_score/M_score_' + index_name %(index0[i],index1[j])
                elif data.bias_flag == 'unbiased':
                    score_file_name = data.read_path + 'M_score/unbiased_score_' + index_name %(index0[i],index1[j])
                score = np.load(score_file_name)
                mean_score[i,j] = np.sum(score*area_weights) #np.nanmean(score,axis=0) # mean value over domain
                std_score[i,j] = np.nan #np.nanstd(score,axis=0) # std dev value over domain
                #print('mean_score[i,j] = ',mean_score[i,j])

                if data.bias_flag == 'biased':
                    closure_score_file_name = data.read_path + 'closure_score/closure_score_' + index_name %(index0[i],index1[j])
                    closure = np.load(closure_score_file_name)
                    mean_closure[i,j] = np.sum(closure*area_weights) #np.nanmean(closure,axis=0) # mean value over domain
                    std_closure[i,j] = np.nan #np.nanstd(closure,axis=0) # std dev value over domain
                elif data.bias_flag == 'unbiased':
                    score_95_file_name = data.read_path + 'M_score/unbiased_score_95_' + index_name %(index0[i],index1[j])
                    score_95 = np.load(score_95_file_name)
                    mean_score_95[i,j] = np.sum(score_95*area_weights) #np.nanmean(score,axis=0) # mean value over domain
                    std_score_95[i,j] = np.nan #np.nanstd(score,axis=0) # std dev value over domain

                balance_models_file_name = data.read_path + 'balance/reduced_balance_' + index_name %(index0[i],index1[j])
                Kr[i,j] = np.shape(np.load(balance_models_file_name))[0]
                lrc = np.shape(np.load(balance_models_file_name))[1]
                if data.bias_flag == 'biased':
                    if lrc == 1:
                        fss[i,j] = 0.5
                        print('\n\n ERROR: balance of one term selected')
                    else:
                        fss[i,j] = lrc / (2.*(lrc -1.)) # should actually be differences! PROBLEM
                elif data.bias_flag == 'unbiased':
                    fss[i,j] = 0.0

                if data.ic_flag == True:
                    aic_file_name = data.read_path + 'M_score/aic_' + index_name %(index0[i],index1[j])
                    aic[i,j] = np.load(aic_file_name)
                    bic_file_name = data.read_path + 'M_score/bic_' + index_name %(index0[i],index1[j])
                    bic[i,j] = np.load(bic_file_name)

            elif len(args) == 3:

                for k in range(0,N2):

                    area_weights_file_name = data.read_path + 'area_weights/area_weights_' + index_name %(index0[i],index1[j],index2[k])

                    area_weights = np.load(area_weights_file_name)
                    if data.cluster_method == 'HDBSCAN':
                        ##area_weights = np.load(area_weights_file_name) / np.sum(np.load(area_weights_file_name)) # % cluster area
                        #area_weights = np.load(area_weights_file_name) / np.sum(data.area) # % total area
                        labels_locs_file_name = data.read_path + 'labels/labels_locs_' + index_name %(index0[i],index1[j],index2[k])
                        data.labels_locs = np.load(labels_locs_file_name)
                        retained_columns_file_name = data.read_path + 'balance/retained_columns_' + index_name %(index0[i],index1[j],index2[k])
                        data.retained_columns = np.load(retained_columns_file_name)
                    else:
                        #area_weights = np.load(area_weights_file_name)
                        data.retained_columns = np.arange(data.nf)

                    if np.sum(area_weights) <= 0.99999999 or np.sum(area_weights) >= 1.00000001:
                        print('\n\nERROR: sum area_weights = ',np.sum(area_weights))
                        print('\n')

                    if data.bias_flag == 'biased':
                        score_file_name = data.read_path + 'M_score/M_score_' + index_name %(index0[i],index1[j],index2[k])
                    elif data.bias_flag == 'unbiased':
                        score_file_name = data.read_path + 'M_score/unbiased_score_' + index_name %(index0[i],index1[j],index2[k])
                    score = np.load(score_file_name)
                    mean_score[i,j,k] = np.sum(score*area_weights) #np.nanmean(score,axis=0) # mean value over domain
                    std_score[i,j,k] = np.nan #np.nanstd(score,axis=0) # std dev value over domain
                    #print('mean_score[i,j] = ',mean_score[i,j])

                    if data.bias_flag == 'biased':
                        closure_score_file_name = data.read_path + 'closure_score/closure_score_' + index_name %(index0[i],index1[j],index2[k])
                        closure = np.load(closure_score_file_name)
                        mean_closure[i,j,k] = np.sum(closure*area_weights) #np.nanmean(closure,axis=0) # mean value over domain
                        std_closure[i,j,k] = np.nan #np.nanstd(closure,axis=0) # std dev value over domain
                    elif data.bias_flag == 'unbiased':
                        score_95_file_name = data.read_path + 'M_score/unbiased_score_95_' + index_name %(index0[i],index1[j],index2[k])
                        score_95 = np.load(score_95_file_name)
                        mean_score_95[i,j,k] = np.sum(score_95*area_weights) #np.nanmean(score,axis=0) # mean value over domain
                        std_score_95[i,j,k] = np.nan #np.nanstd(score,axis=0) # std dev value over domain

                    balance_models_file_name = data.read_path + 'balance/reduced_balance_' + index_name %(index0[i],index1[j],index2[k])
                    Kr[i,j,k] = np.shape(np.load(balance_models_file_name))[0]
                    lrc = np.shape(np.load(balance_models_file_name))[1]
                    #print('np.shape(np.load(balance_models_file_name)) = ',np.shape(np.load(balance_models_file_name)))
                    if data.bias_flag == 'biased':
                        if lrc == 1:
                            fss[i,j,k] = 0.5
                            print('\n\n ERROR: balance of one term selected')
                        else:
                            fss[i,j,k] = lrc / (2.*(lrc -1.)) # should actually be differences! PROBLEM
                    elif data.bias_flag == 'unbiased':
                        fss[i,j,k] = 0.0

                    if data.ic_flag == True:
                        aic_file_name = data.read_path + 'M_score/aic_' + index_name %(index0[i],index1[j],index2[k])
                        aic[i,j,k] = np.load(aic_file_name)
                        bic_file_name = data.read_path + 'M_score/bic_' + index_name %(index0[i],index1[j],index2[k])
                        bic[i,j,k] = np.load(bic_file_name)

    # correction: if all points are labeled as noise, should be full set score:
    correction_flag = 0
    if len(np.shape(Kr)) == 2:
        for k in range(0,np.shape(Kr)[0]):
            for i in range(0,np.shape(Kr)[1]):
                if Kr[k,i] == 1.:
                    if data.bias_flag == 'biased':
                        mean_score[k,i] = data.nf / (2.0*(data.nf - 1.0)) #5./8.
                        #elif data.bias_flag == 'unbiased':
                        #    mean_score[k,i] = 0.0
                        correction_flag = correction_flag + 1

    elif len(np.shape(Kr)) == 3:
        for k in range(0,np.shape(Kr)[0]):
            for i in range(0,np.shape(Kr)[1]):
                for j in range(0,np.shape(Kr)[2]):
                    if Kr[k,i,j] == 1.:
                        if data.bias_flag == 'biased':
                            mean_score[k,i,j] = data.nf / (2.0*(data.nf - 1.0)) #5./8.
                            #elif data.bias_flag == 'unbiased':
                            #    mean_score[k,i,j] = 0.0
                            correction_flag = correction_flag + 1
    if correction_flag >= 1:
        print('\n\n\n   WARNING: CORRECTED SCORES FOR KR=1 \n\n\n')

    data.mean_score = mean_score
    data.std_score = std_score
    if data.bias_flag == 'biased':
        data.mean_closure = mean_closure
        data.std_closure = std_closure
    if data.bias_flag == 'unbiased':
        data.mean_score_95 = mean_score_95
        data.std_score_95 = std_score_95
    data.Kr = Kr
    data.fss = fss
    if data.ic_flag == True:
        data.aic = aic
        data.bic = bic

    idx0 = data.min_convergence_idx_0
    idx1 = data.min_convergence_idx_1
    idx2 = data.min_convergence_idx_2

    # get the labels (which cluster for which grid point) and
    # balance (which active terms per cluster) at max. score:

    #print('\n!! np.shape(mean_score) = ',np.shape(mean_score)) # (11, 1, 36) K,E,alphas
    print('\n np.amax(mean_score) = ',np.amax(mean_score))

    if len(np.shape(mean_score)) == 3:
        if data.bias_flag == 'biased':
            mean_score = mean_score - fss
        loc_max = (np.argwhere(mean_score==np.amax(mean_score[idx0:np.shape(mean_score)[0],idx1:np.shape(mean_score)[1],idx2:np.shape(mean_score)[2]])))[0,:]
        loc_max_Kr = (np.argwhere(Kr==np.amax(Kr[idx0:np.shape(Kr)[0],idx1:np.shape(Kr)[1],idx2:np.shape(Kr)[2]])))[0,:]
    elif len(np.shape(mean_score)) == 2:
        if data.bias_flag == 'biased':
            mean_score = mean_score - fss
        loc_max = (np.argwhere(mean_score==np.amax(mean_score[idx0:np.shape(mean_score)[0],idx1:np.shape(mean_score)[1]])))[0,:]
        loc_max_Kr = (np.argwhere(Kr==np.amax(Kr[idx0:np.shape(Kr)[0],idx1:np.shape(Kr)[1]])))[0,:]

    #loc_max = (np.argwhere(mean_score==np.amax(mean_score[idx0:np.shape(mean_score)[0],idx1:np.shape(mean_score)[1]])))[0,:]
    print(' index of np.amax(mean_score) = ',loc_max)
    #print('!! mean_score[loc_max[0],loc_max[1]] = ',mean_score[loc_max[0],loc_max[1]])
    #loc_max = np.array([0,3])

    # results for maximum score
    data.loc_max = loc_max
    max_data = get_optimal_results( data, index_name , loc_max, *args )
    data.balance = max_data.balance
    data.labels = max_data.labels
    data.max_scores = max_data.max_scores
    if data.bias_flag == 'biased':
        data.max_closure = max_data.max_closure
    elif data.bias_flag == 'unbiased':
        data.max_scores_95 = max_data.max_scores_95
    data.max_area_weights = max_data.max_area_weights
    data.nc = max_data.nc
    data.Xc = max_data.Xc
    data.feature_mean = max_data.feature_mean
    data.feature_mean_95 = max_data.feature_mean_95
    data.mycolors = max_data.mycolors
    data.mymarkers = max_data.mymarkers
    data.mymarkersize = max_data.mymarkersize

    return data


def get_optimal_results( data, index_name , loc_max, *args ):

    if len(args) == 2:
        index0 = args[0]
        index1 = args[1]
        N0 = len(index0)
        N1 = len(index1)
        mean_score = np.zeros([N0,N1])
        std_score = np.zeros([N0,N1])
        if data.bias_flag == 'biased':
            mean_closure = np.zeros([N0,N1])
            std_closure = np.zeros([N0,N1])
        elif data.bias_flag == 'unbiased':
            mean_score_95 = np.zeros([N0,N1])
            std_score_95 = np.zeros([N0,N1])
        Kr = np.zeros([N0,N1])
    elif len(args) == 3:
        index0 = args[0]
        index1 = args[1]
        index2 = args[2]
        N0 = len(index0)
        N1 = len(index1)
        N2 = len(index2)
        mean_score = np.zeros([N0,N1,N2])
        std_score = np.zeros([N0,N1,N2])
        if data.bias_flag == 'biased':
            mean_closure = np.zeros([N0,N1,N2])
            std_closure = np.zeros([N0,N1,N2])
        elif data.bias_flag == 'unbiased':
            mean_score_95 = np.zeros([N0,N1,N2])
            std_score_95 = np.zeros([N0,N1,N2])
        Kr = np.zeros([N0,N1,N2])
    elif len(args) == 4:
        index0 = args[0]
        index1 = args[1]
        index2 = args[2]
        index3 = args[3]
        N0 = len(index0)
        N1 = len(index1)
        N2 = len(index2)
        N3 = len(index3)
        mean_score = np.zeros([N0,N1,N2,N3])
        std_score = np.zeros([N0,N1,N2,N3])
        if data.bias_flag == 'biased':
            mean_closure = np.zeros([N0,N1,N2,N3])
            std_closure = np.zeros([N0,N1,N2,N3])
        elif data.bias_flag == 'unbiased':
            mean_score_95 = np.zeros([N0,N1,N2,N3])
            std_score_95 = np.zeros([N0,N1,N2,N3])
        Kr = np.zeros([N0,N1,N2,N3])
    else:
        print('ERROR: not the right number of arguments for import_data in utils')

    if len(loc_max) == 2:
        area_weights_file_name = data.read_path + 'area_weights/area_weights_' + index_name %(index0[loc_max[0]],index1[loc_max[1]])
        #if (data.read_path).split('_')[1] == 'HDBSCAN':
        if data.cluster_method == 'HDBSCAN':
            #max_area_weights = np.load(area_weights_file_name) / np.sum(np.load(area_weights_file_name)) # % cluster area
            max_area_weights = np.load(area_weights_file_name) / np.sum(data.area) # % total area

            retained_columns_file_name = data.read_path + 'balance/retained_columns_' + index_name %(index0[loc_max[0]],index1[loc_max[1]])
            retained_columns = np.load(retained_columns_file_name)

        else:
            #print('DEPRECIATED: new GMM data will need MAX_AREA_WEIGHTS definition fixed')
            max_area_weights = np.load(area_weights_file_name)
            retained_columns = np.arange(data.nf)

        if np.sum(max_area_weights) <= 0.99999999 or np.sum(max_area_weights) >= 1.00000001:
            print('\n\nERROR: max_sum area_weights = ',np.sum(amax_area_weights))
            print('\n')

        balance_file_name = data.read_path + 'balance/reduced_balance_' + index_name %(index0[loc_max[0]],index1[loc_max[1]])
        balance = np.load(balance_file_name)
        labels_file_name = data.read_path + 'labels/reduced_labels_' + index_name %(index0[loc_max[0]],index1[loc_max[1]])
        labels = np.load(labels_file_name)
        feature_mean_file_name = data.read_path + 'feature_mean/feature_mean_' + index_name %(index0[loc_max[0]],index1[loc_max[1]])
        feature_mean = np.load(feature_mean_file_name)
        feature_mean_95_file_name = data.read_path + 'feature_mean_95/feature_mean_95_' + index_name %(index0[loc_max[0]],index1[loc_max[1]])
        feature_mean_95 = np.load(feature_mean_95_file_name)
        if data.bias_flag == 'biased':
            max_scores_file_name = data.read_path + 'M_score/M_score_' + index_name %(index0[loc_max[0]],index1[loc_max[1]])
            max_closure_file_name = data.read_path + 'closure_score/closure_score_' + index_name %(index0[loc_max[0]],index1[loc_max[1]])
            max_closure = np.load(max_closure_file_name)
        elif data.bias_flag == 'unbiased':
            max_scores_file_name = data.read_path + 'M_score/unbiased_score_' + index_name %(index0[loc_max[0]],index1[loc_max[1]])
            max_scores_95_file_name = data.read_path + 'M_score/unbiased_score_95_' + index_name %(index0[loc_max[0]],index1[loc_max[1]])
            max_scores_95 = np.load(max_scores_95_file_name)
        max_scores = np.load(max_scores_file_name)

    elif len(loc_max) == 3:
        area_weights_file_name = data.read_path + 'area_weights/area_weights_' + index_name %(index0[loc_max[0]],index1[loc_max[1]],index2[loc_max[2]])
        if data.cluster_method == 'HDBSCAN':
            #max_area_weights = np.load(area_weights_file_name) / np.sum(np.load(area_weights_file_name)) # % cluster area
            max_area_weights = np.load(area_weights_file_name) / np.sum(data.area) # % total area

            retained_columns_file_name = data.read_path + 'balance/retained_columns_' + index_name %(index0[loc_max[0]],index1[loc_max[1]],index2[loc_max[2]])
            retained_columns = np.load(retained_columns_file_name)

        else:
            #print('DEPRECIATED: new GMM data will need MAX_AREA_WEIGHTS definition fixed')
            max_area_weights = np.load(area_weights_file_name)
            retained_columns = np.arange(data.nf)

        if np.sum(max_area_weights) <= 0.99999999 or np.sum(max_area_weights) >= 1.00000001:
            print('\n\nERROR: sum max_area_weights = ',np.sum(amax_area_weights))
            print('\n')

        balance_file_name = data.read_path + 'balance/reduced_balance_' + index_name %(index0[loc_max[0]],index1[loc_max[1]],index2[loc_max[2]])
        balance = np.load(balance_file_name)
        labels_file_name = data.read_path + 'labels/reduced_labels_' + index_name %(index0[loc_max[0]],index1[loc_max[1]],index2[loc_max[2]])
        labels = np.load(labels_file_name)
        feature_mean_file_name = data.read_path + 'feature_mean/feature_mean_' + index_name %(index0[loc_max[0]],index1[loc_max[1]],index2[loc_max[2]])
        feature_mean = np.load(feature_mean_file_name)
        feature_mean_95_file_name = data.read_path + 'feature_mean_95/feature_mean_95_' + index_name %(index0[loc_max[0]],index1[loc_max[1]],index2[loc_max[2]])
        feature_mean_95 = np.load(feature_mean_95_file_name)
        if data.bias_flag == 'biased':
            max_scores_file_name = data.read_path + 'M_score/M_score_' + index_name %(index0[loc_max[0]],index1[loc_max[1]],index2[loc_max[2]])
            max_scores = np.load(max_scores_file_name)
            max_closure_file_name = data.read_path + 'closure_score/closure_score_' + index_name %(index0[loc_max[0]],index1[loc_max[1]],index2[loc_max[2]])
            max_closure = np.load(max_closure_file_name)
        elif data.bias_flag == 'unbiased':
            max_scores_file_name = data.read_path + 'M_score/unbiased_score_' + index_name %(index0[loc_max[0]],index1[loc_max[1]],index2[loc_max[2]])
            max_scores = np.load(max_scores_file_name)
            max_scores_95_file_name = data.read_path + 'M_score/unbiased_score_95_' + index_name %(index0[loc_max[0]],index1[loc_max[1]],index2[loc_max[2]])
            max_scores_95 = np.load(max_scores_95_file_name)
        """
        if data.cluster_method == 'DBSCAN' or data.cluster_method == 'HDBSCAN':
            labels_locs_file_name = data.read_path + 'labels/labels_locs_' + index_name %(index0[loc_max[0]],index1[loc_max[1]],index2[loc_max[2]])
            data.labels_locs = np.load(labels_locs_file_name)
            print('np.shape(labels) = ',np.shape(labels))
            print('np.shape(data.labels_locs) = ',np.shape(data.labels_locs))
            print('int(len(data.x)*len(data.y)) = ',int(len(data.x)*len(data.y)))
            labels_all = np.zeros([int(len(data.x)*len(data.y))])*np.nan
            labels_all[data.labels_locs] = labels
        else:
            data.labels_locs = np.arange(len((data.area))) # all points
        """

    # prep for plotting the max. M score reduced balance:
    nc = np.shape(np.unique(labels))[0]
    #print('>>> nc = ',nc)

    #Z = np.zeros([nc,nf,2])
    #for k in range(0,nc):
    #    locs_k = np.argwhere(labels==float(k))[:,0]
    #    for h in range(0,nf):
    #        Z[k,h,:] = skewness_flatness( features[locs_k,h] )
    xc = np.linspace(1,np.shape(balance)[1],num=np.shape(balance)[1],endpoint=True)
    yc = np.linspace(0,np.shape(balance)[0]-1,num=np.shape(balance)[0],endpoint=True)
    Xc,Y = np.meshgrid(xc,yc)
    for jj in range(0,np.shape(balance)[0]):
        balance[jj,:] = balance[jj,:]*(yc[jj]+1)

    mycolors,mymarkers,mymarkersize = get_markers_and_colors( nc )

    max_data = easydict.EasyDict({
        "nc":nc,
        "Xc":Xc,
        "balance":balance,
        "max_scores":max_scores,
        "max_area_weights":max_area_weights,
        "feature_mean":feature_mean,
        "feature_mean_95":feature_mean_95,
        "max_area_weights":max_area_weights,
        "retained_columns":retained_columns,
        })
    if data.bias_flag == 'biased':
        max_data.max_closure = max_closure
    elif data.bias_flag == 'unbiased':
        max_data.max_scores_95 = max_scores_95

    """
    if data.cluster_method == 'DBSCAN' or data.cluster_method == 'HDBSCAN':
        max_data.labels = labels_all # includes nans where noise is
    else:
        max_data.labels = labels
    """
    max_data.labels = labels
    #data.labels_locs = np.arange(len((data.area))) # all points

    max_data.mycolors = mycolors
    max_data.mymarkers = mymarkers
    max_data.mymarkersize = mymarkersize

    return max_data


def save_refined_labels( input ):
    # save the labels and balances of the the refined clusters
    # refined: clusters have been identified and merging performed where necessary
    # labels: a vector of length ng with each element a cluster name (index)
    # balances: an array of size [nc,nf], where the num. of rows correspond to
    # the cluster names in labels and the number of columns correspond to the
    # number of features at each grid point. Each row in balances is a set of
    # active (ones) and inactive (zeros) terms, a unique combination for each
    # cluster.

    refined_balance = input.refined_balance
    refined_labels = input.refined_labels
    retained_columns = input.retained_columns
    alpha_factor = input.alpha_factor

    if input.reduction_method == 'score':

        if input.cluster_method == 'GMM' or input.cluster_method == 'KMeans':
            file_name1 = input.write_path + 'balance/reduced_balance_K%i_n%i.npy' %(int(input.K),int(input.E))
            file_name2 = input.write_path + 'labels/reduced_labels_K%i_n%i.npy' %(int(input.K),int(input.E))
            file_name_columns = input.write_path + 'balance/retained_columns_K%i_n%i.npy' %(int(input.K),int(input.E))

        elif input.cluster_method == 'HDBSCAN':
            file_name1 = input.write_path + 'balance/reduced_balance_n%i_ms%i_mcs%i.npy' %(int(input.E),input.ms,input.mcs)
            file_name2 = input.write_path + 'labels/reduced_labels_n%i_ms%i_mcs%i.npy' %(int(input.E),input.ms,input.mcs)
            file_name3 = input.write_path + 'labels/labels_locs_n%i_ms%i_mcs%i.npy' %(int(input.E),input.ms,input.mcs)
            file_name4 = input.write_path + 'labels/noise_locs_n%i_ms%i_mcs%i.npy' %(int(input.E),input.ms,input.mcs)
            file_name_columns = input.write_path + 'balance/retained_columns_n%i_ms%i_mcs%i.npy' %(int(input.E),input.ms,input.mcs)

        elif input.cluster_method == 'DBSCAN':
            print('\n   ERROR: save_refined_labels: DBSCAN results not saved for EHS')

    elif input.reduction_method == 'SPCA':
        if input.cluster_method == 'GMM' or input.cluster_method == 'KMeans':
            file_name1 = input.write_path + 'balance/reduced_balance_K%i_n%i_a%i.npy' %(int(input.K),int(input.E),int(input.alpha_opt*alpha_factor ))
            file_name2 = input.write_path + 'labels/reduced_labels_K%i_n%i_a%i.npy' %(int(input.K),int(input.E),int(input.alpha_opt*alpha_factor ))
            file_name_columns = input.write_path + 'balance/retained_columns_K%i_n%i_a%i.npy' %(int(input.K),int(input.E),int(input.alpha_opt*alpha_factor ))
        else:
            if input.cluster_method == 'HDBSCAN':
                file_name1 = input.write_path + 'balance/reduced_balance_ms%i_mcs%i_a%i.npy' %(input.ms,input.mcs,int(input.alpha_opt*alpha_factor ))
                file_name2 = input.write_path + 'labels/reduced_labels_ms%i_mcs%i_a%i.npy' %(input.ms,input.mcs,int(input.alpha_opt*alpha_factor ))
                file_name3 = input.write_path + 'labels/labels_locs_ms%i_mcs%i_a%i.npy' %(input.ms,input.mcs,int(input.alpha_opt*alpha_factor ))
                file_name4 = input.write_path + 'labels/noise_locs_ms%i_mcs%i_a%i.npy' %(input.ms,input.mcs,int(input.alpha_opt*alpha_factor ))
                file_name_columns = input.write_path + 'balance/retained_columns_ms%i_mcs%i_a%i.npy' %(input.ms,input.mcs,int(input.alpha_opt*alpha_factor))

            elif input.cluster_method == 'DBSCAN':
                print('\n   ERROR: save_refined_labels: DBSCAN results not saved for SPCA')

    np.save(file_name1,refined_balance)
    np.save(file_name2,refined_labels)
    np.save(file_name_columns,retained_columns)

    if input.cluster_method == 'HDBSCAN':
        np.save(file_name3,input.labels_locs)
        np.save(file_name4,input.noise_locs)
    elif input.cluster_method == 'DBSCAN':
        print('\n   ERROR: save_refined_labels: DBSCAN locs results not saved')

    return


def save_refined_data( input ):

    feature_mean = input.feature_mean
    feature_mean_95 = input.feature_mean_95
    score_max = input.score_max # "max" refers to the best score for the data.
    if input.bias_flag == 'biased':
        closure_max = input.closure_max
    elif input.bias_flag == 'unbiased':
        score_max_95 = input.score_max_95
    area_weights = input.area_weights
    alpha_factor = input.alpha_factor

    if input.reduction_method == 'score':

        if input.cluster_method == 'GMM' or input.cluster_method == 'KMeans':
            file_name1 =  input.write_path + 'feature_mean_95/feature_mean_95_K%i_n%i.npy' %(int(input.K),int(input.E))
            file_name2 =  input.write_path + 'feature_mean/feature_mean_K%i_n%i.npy' %(int(input.K),int(input.E))
            if input.bias_flag == 'biased':
                file_name3 =  input.write_path + 'M_score/biased_score_K%i_n%i.npy' %(int(input.K),int(input.E))
                file_name4 =  input.write_path + 'closure_score/closure_score_K%i_n%i.npy' %(int(input.K),int(input.E))
            elif input.bias_flag == 'unbiased':
                file_name3 =  input.write_path + 'M_score/unbiased_score_K%i_n%i.npy' %(int(input.K),int(input.E))
                file_name4 =  input.write_path + 'M_score/unbiased_score_95_K%i_n%i.npy' %(int(input.K),int(input.E))
            file_name5 =  input.write_path + 'area_weights/area_weights_K%i_n%i.npy' %(int(input.K),int(input.E))
            if input.ic_flag == True:
                file_name6 =  input.write_path + 'M_score/aic_K%i_n%i.npy' %(int(input.K),int(input.E))
                file_name7 =  input.write_path + 'M_score/bic_K%i_n%i.npy' %(int(input.K),int(input.E))

        elif input.cluster_method == 'DBSCAN':
            print('\n\n  ERROR: DBSCAN has depreciated output: fix!')
            file_name1 =  input.write_path + 'feature_mean_95/feature_mean_95_eps%i_ms%i.npy' %(int(input.eps*1000),input.ms)
            file_name2 =  input.write_path + 'feature_mean/feature_mean_eps%i_ms%i.npy' %(int(input.eps*1000),input.ms)
            file_name3 =  input.write_path + 'M_score/M_score_eps%i_ms%i.npy' %(int(input.eps*1000),input.ms)
            file_name4 =  input.write_path + 'closure_score/closure_score_eps%i_ms%i.npy' %(int(input.eps*1000),input.ms)
            file_name5 =  input.write_path + 'area_weights/area_weights_eps%i_ms%i.npy' %(int(input.eps*1000),input.ms)

        elif input.cluster_method == 'HDBSCAN':
            file_name1 =  input.write_path + 'feature_mean_95/feature_mean_95_n%i_ms%i_mcs%i.npy' %(int(input.E),input.ms,input.mcs)
            file_name2 =  input.write_path + 'feature_mean/feature_mean_n%i_ms%i_mcs%i.npy' %(int(input.E),input.ms,input.mcs)
            if input.bias_flag == 'biased':
                file_name3 =  input.write_path + 'M_score/biased_score_n%i_ms%i_mcs%i.npy' %(int(input.E),input.ms,input.mcs)
                file_name4 =  input.write_path + 'closure_score/closure_score_n%i_ms%i_mcs%i.npy' %(int(input.E),input.ms,input.mcs)
            elif input.bias_flag == 'unbiased':
                file_name3 =  input.write_path + 'M_score/unbiased_score_n%i_ms%i_mcs%i.npy' %(int(input.E),input.ms,input.mcs)
                file_name4 =  input.write_path + 'M_score/unbiased_score_95_n%i_ms%i_mcs%i.npy' %(int(input.E),input.ms,input.mcs)
            file_name5 =  input.write_path + 'area_weights/area_weights_n%i_ms%i_mcs%i.npy' %(int(input.E),input.ms,input.mcs)

    elif input.reduction_method == 'SPCA':

        if input.cluster_method == 'GMM' or input.cluster_method == 'KMeans':
            file_name1 =  input.write_path + 'feature_mean_95/feature_mean_95_K%i_n%i_a%i.npy' %(int(input.K),int(input.E),int(input.alpha_opt*alpha_factor))
            file_name2 =  input.write_path + 'feature_mean/feature_mean_K%i_n%i_a%i.npy' %(int(input.K),int(input.E),int(input.alpha_opt*alpha_factor))
            if input.bias_flag == 'biased':
                file_name3 =  input.write_path + 'M_score/biased_score_K%i_n%i_a%i.npy' %(int(input.K),int(input.E),int(input.alpha_opt*alpha_factor))
                file_name4 =  input.write_path + 'closure_score/closure_score_K%i_n%i_a%i.npy' %(int(input.K),int(input.E),int(input.alpha_opt*alpha_factor))
            elif input.bias_flag == 'unbiased':
                file_name3 =  input.write_path + 'M_score/unbiased_score_K%i_n%i_a%i.npy' %(int(input.K),int(input.E),int(input.alpha_opt*alpha_factor))
                file_name4 =  input.write_path + 'M_score/unbiased_score_95_K%i_n%i_a%i.npy' %(int(input.K),int(input.E),int(input.alpha_opt*alpha_factor))
            file_name5 =  input.write_path + 'area_weights/area_weights_K%i_n%i_a%i.npy' %(int(input.K),int(input.E),int(input.alpha_opt*alpha_factor))
            if input.ic_flag == True:
                file_name6 =  input.write_path + 'M_score/aic_K%i_n%i_a%i.npy' %(int(input.K),int(input.E),int(input.alpha_opt*alpha_factor))
                file_name7 =  input.write_path + 'M_score/bic_K%i_n%i_a%i.npy' %(int(input.K),int(input.E),int(input.alpha_opt*alpha_factor))

        elif input.cluster_method == 'DBSCAN':
            print('\n\n  ERROR: DBSCAN has depreciated output: fix!')
            file_name1 =  input.write_path + 'feature_mean_95/feature_mean_95_eps%i_ms%i_a%i.npy' %(int(input.eps*1000),input.ms,int(input.alpha_opt*alpha_factor))
            file_name2 =  input.write_path + 'feature_mean/feature_mean_eps%i_ms%i_a%i.npy' %(int(input.eps*1000),input.ms,int(input.alpha_opt*alpha_factor))
            file_name3 =  input.write_path + 'M_score/M_score_eps%i_ms%i_a%i.npy' %(int(input.eps*1000),input.ms,int(input.alpha_opt*alpha_factor))
            file_name4 =  input.write_path + 'closure_score/closure_score_eps%i_ms%i_a%i.npy' %(int(input.eps*1000),input.ms,int(input.alpha_opt*alpha_factor))
            file_name5 =  input.write_path + 'area_weights/area_weights_eps%i_ms%i_a%i.npy' %(int(input.eps*1000),input.ms,int(input.alpha_opt*alpha_factor))

        elif input.cluster_method == 'HDBSCAN':
            file_name1 =  input.write_path + 'feature_mean_95/feature_mean_95_ms%i_mcs%i_a%i.npy' %(input.ms,input.mcs,int(input.alpha_opt*alpha_factor))
            file_name2 =  input.write_path + 'feature_mean/feature_mean_ms%i_mcs%i_a%i.npy' %(input.ms,input.mcs,int(input.alpha_opt*alpha_factor))
            if input.bias_flag == 'biased':
                file_name3 =  input.write_path + 'M_score/biased_score_ms%i_mcs%i_a%i.npy' %(input.ms,input.mcs,int(input.alpha_opt*alpha_factor))
                file_name4 =  input.write_path + 'closure_score/closure_score_ms%i_mcs%i_a%i.npy' %(input.ms,input.mcs,int(input.alpha_opt*alpha_factor))
            elif input.bias_flag == 'unbiased':
                file_name3 =  input.write_path + 'M_score/unbiased_score_ms%i_mcs%i_a%i.npy' %(input.ms,input.mcs,int(input.alpha_opt*alpha_factor))
                file_name4 =  input.write_path + 'M_score/unbiased_score_95_ms%i_mcs%i_a%i.npy' %(input.ms,input.mcs,int(input.alpha_opt*alpha_factor))
            file_name5 =  input.write_path + 'area_weights/area_weights_ms%i_mcs%i_a%i.npy' %(input.ms,input.mcs,int(input.alpha_opt*alpha_factor))

    np.save(file_name1, feature_mean_95)
    np.save(file_name2, feature_mean)
    np.save(file_name3, score_max)
    if input.bias_flag == 'biased':
        np.save(file_name4, closure_max)
    elif input.bias_flag == 'unbiased':
        np.save(file_name4, score_max_95)

    np.save(file_name5, area_weights)
    if input.ic_flag == True:
        np.save(file_name6, input.aic)
        np.save(file_name7, input.bic)

    return

def cluster_id_and_merge( input ):
    # identify the dominant or leading order dynamics of clusters, then
    # merge identical clusters.
    nf = input.nf # number of features

    # 1) get balances & scores *************************************************
    input = remove_unused_features( input ) # should be called get_balances
    balance = input.cluster_balance
    if input.reduction_method == 'SPCA': # scores have not been assigned
        # check for balance with only one element, or all zeros.
        nc = np.shape(balance)[0]
        for i in range(0,nc):
            if np.sum(balance[i,:]) == 1.: # only one element
                balance[i,:] = np.ones([nf])
            elif np.sum(balance[i,:]) == 0.: # all zeros
                balance[i,:] = np.ones([nf])
        input.refine_flag = False
        input.cluster_balance = balance
        input.nc = nc
        input = get_scores( input , preassigned_balance_flag = True )

    print('\n  cluster and id results:')
    print('  remove_unused_features: input.score_max = ',input.score_max)
    print('  remove_unused_features: input.score_max_95 = ',input.score_max_95)
    print('  remove_unused_features: input.area_weights = ',input.area_weights)
    print('  remove_unused_features: sum areas = ',np.sum(input.area_weights) )
    print('  remove_unused_features: avg score = ',np.sum(input.score_max*input.area_weights) )
    print('  remove_unused_features: balance = ',balance)

    #if input.verbose == True:
    #    print('\n  cluster_id_and_merge: input.score_max = ',input.score_max)
    #    print('  cluster_id_and_merge: input.area_weights = ',input.area_weights)
    #    print('  cluster_id_and_merge: balance = ',balance)


    # 2) aggolmeration *********************************************************
    refined_balance,refined_labels,label_matching_index = merge_clusters( input.cluster_balance , input.cluster_labels )
    if input.bias_flag == 'biased':
        input.score_max, input.closure_max, input.area_weights = merge_scores( input.score_max, input.closure_max, input.area_weights, np.unique(input.cluster_labels), label_matching_index )
    elif input.bias_flag == 'unbiased':
        input.score_max, input.score_max_95, input.area_weights = merge_scores( input.score_max, input.score_max_95, input.area_weights, np.unique(input.cluster_labels), label_matching_index )

    print('\n  merged results:')
    print('  merge_clusters: scores = ',input.score_max)
    print('  merge_clusters: input.score_max_95 = ',input.score_max_95)
    print('  merge_clusters: area_weights = ',input.area_weights)
    print('  merge_clusters: sum areas = ',np.sum(input.area_weights) )
    print('  merge_clusters: avg score = ',np.sum(input.score_max*input.area_weights) )
    print('  merge_clusters: balance = ',refined_balance)

    input.refined_balance = refined_balance
    input.refined_labels = refined_labels
    input.nc = np.shape(refined_balance)[0] # reduced number of clusters
    input.refine_flag = True # flag for scoring the "refined" output labels & balances

    return input

def remove_unused_features( input ):
    # should be called get_balances

    nf = input.nf # number of features

    # assign balances
    if input.reduction_method == 'score':
        input.refine_flag = False # score the clustered data (not refined data)
        input = get_scores( input , preassigned_balance_flag = False ) # get cluster_balance + scores
        balance = input.cluster_balance # best balance for each cluster (from get_scores)
    elif input.reduction_method == 'SPCA':
        # the amplitude matters for SPCA, so normalized is usually better:
        #spca_features = input.raw_features
        #spca_features = input.features

        #transformer = RobustScaler().fit(raw_features)
        #features = transformer.transform(raw_features)

        # TBL, what works:  no-standard, 1e3*features, alpha=10
        # TBL, what doesn't work: no-standard, 1e0*features, alpha=10
        # TBL, what doesn't work: standardized score is higher but the results looks bad.

        # try: GMM+SPCA, standardize for outliers, regular standardize, standardize SPCA
        # multiply raw_features by something that makes it about 1 in amplitude?
        # or used the standardized features?

        # 1) try robust standardization, regular, and none + normalize features for spca by dividing by smallest non-zero term.

        # spca normalization:
        spca_features = input.raw_features
        nonzero_raw_features = np.abs(input.raw_features[np.nonzero(input.raw_features)])
        #print('\n  np.shape(nonzero_raw_features) = ',np.shape(nonzero_raw_features))
        #print('  np.amin(nonzero_raw_features),np.amax(nonzero_raw_features) = ', np.amin(nonzero_raw_features),np.amax(nonzero_raw_features))
        #print('  np.mean(nonzero_raw_features),np.std(nonzero_raw_features) = ', np.mean(nonzero_raw_features),np.std(nonzero_raw_features))
        spca_features = spca_features / np.mean(nonzero_raw_features)

        nc = len(np.unique(input.cluster_labels))
        spca_labels = input.cluster_labels
        alpha_opt = input.alpha_opt
        skew_flag = input.skew_flag
        balance = spca( nc, spca_labels, spca_features, alpha_opt, skew_flag )
        input.cluster_balance = balance

    # DEPRECIATED: check for unused features:
    retained_columns = check_columns( balance )
    input.retained_columns = retained_columns
    #print('   1) retained_columns = ',retained_columns)

    # DEPRECIATED:
    if input.feature_removal == 'on':
        print('\n  WARNING: FEATURE REMOVAL IS DEPRECIATED')

        # reassign scores balances and re-check for unused features:
        if len(retained_columns) < nf and len(retained_columns) > 1: # if balance is missing a feature, repeat clustering!
            # 1) shrink the data:
            input.features = input.features[:,retained_columns]
            input.raw_features = input.raw_features[:,retained_columns]
            nf = np.shape(input.features)[1]
            input.nf = nf
            input.balance_combinations = generate_balances(nf, input.bias_flag)
            # 2) re-cluster:
            input = cluster( input , False )
            # 3) re-assign balances
            if input.reduction_method == 'score':
                input.refine_flag = False # score the clustered data (not refined data)
                input = get_scores( input , preassigned_balance_flag = False ) # get cluster_balance + scores
                balance = input.cluster_balance # best balance for each cluster (from get_scores)
            elif input.reduction_method == 'SPCA':
                balance = spca( len(np.unique(input.cluster_labels)) , input.cluster_labels, input.features, input.alpha_opt, input.skew_flag )
                input.cluster_balance = balance
                print('\n  feature removal: SPCA balance = ',balance)
                print('  feature removal: SPCA standard dev of raw features = ',np.std(spca_features))
        return input

    elif input.feature_removal == 'off':

        return input

def merge_scores( scores, closures, weights, unique_labels, matching ):
    # unique_labels = the unmerged labels.

    merged_scores = np.zeros([len(np.unique(matching))])
    merged_closures = np.zeros([len(np.unique(matching))])
    merged_weights = np.zeros([len(np.unique(matching))])

    for i in range(0,len(np.unique(matching))): # loop over unique clusters:
        if len(np.argwhere( matching == i )) > 1:
            # if there are multiple clusters with index "i", combine:
            locs = np.argwhere( matching == i )[:,0]
            merged_weights[i] = np.sum(weights[locs])
            rel_weights = weights[locs] / merged_weights[i]
            merged_scores[i] = np.sum( scores[locs]*rel_weights )
            merged_closures[i] = np.sum( closures[locs]*rel_weights )
        else:
            locs = np.argwhere( matching == i )[:,0]
            # only one cluster with index "i"
            merged_scores[i] = scores[locs]
            merged_closures[i] = closures[locs]
            merged_weights[i] = weights[locs]

    #if np.sum(merged_scores*merged_weights) != np.sum(scores*weights):
    if np.abs( np.sum(merged_scores*merged_weights) - np.sum(scores*weights) ) > 1e-8:
        print('   ERROR: problem with agglomerated scores:')
        print('   np.abs( np.sum(merged_scores*merged_weights) - np.sum(scores*weights) ) = ',np.abs( np.sum(merged_scores*merged_weights) - np.sum(scores*weights) ))
        print('   unmerged_scores = ',scores)
        print('   merged_scores = ',merged_scores)
        print('   np.sum(unmerged_scores*weights) = ',np.sum(scores*weights))
        print('   np.sum(merged_scores*merged_weights) = ',np.sum(merged_scores*merged_weights))
        print('   merged_weights = ',merged_weights)
        print('   np.sum(merged_weights) = ',np.sum(merged_weights))

    return merged_scores, merged_closures, merged_weights

def full_set_clusters( input ):
    # change the balances of clusters with scores lower than
    # the full set to the full set.
    # If a clustering algorithm with noise is chosen, convert noise to full balances.

    score_max = input.score_max
    balance = input.refined_balance
    labels = input.refined_labels
    nc = input.nc
    nf = input.nf

    if input.feature_removal == True:
        retained_columns = check_columns( balance )
        if len(retained_columns) == 1:
            fss = 0.5
            print('\n\n ERROR: balance of only 1 term selected \n\n')
        else:
            fss = len(retained_columns) / (2.*(len(retained_columns)-1.))
    else:
        fss = nf / (2.*(nf-1.))

    # if a cluster scores below the full set score, make it the full set:
    score_flag = 0
    for i in range(0,nc):
        if score_max[i] < fss:
            score_flag = 1
            if input.feature_removal == True:
                balance[i,retained_columns] = np.ones([len(retained_columns)])
                score_max[i] = fss
            else:
                balance[i,:] = np.ones([nf])
                score_max[i] = fss
    input.score_max = score_max

    if input.cluster_method == 'HDBSCAN':
        # convert the noise to a cluster with the full set score areas for noise
        noise_area = np.sum(input.area[input.noise_locs])
        #print('\n noise_area = ',noise_area)
        #print(' len(input.noise_locs) = ',len(input.noise_locs))
        input.area_weights = np.append(input.area_weights,noise_area)
        # scores/closure for noise
        input.score_max= np.append(input.score_max,fss)
        input.closure_max= np.append(input.closure_max,0.)
        # balances for noise
        if len(input.noise_locs) >= 1: # no noise
            noise_balance = np.ones([1,nf])
            input.refined_balance = np.concatenate((input.refined_balance,noise_balance),axis=0)
        # labels
        input.nc = np.shape(input.refined_balance)[0]
        noise_label = len(np.unique(input.refined_labels))
        all_labels = np.zeros([len(input.area)],dtype=int)
        all_labels[input.labels_locs] = input.refined_labels
        all_labels[input.noise_locs] = np.ones([len(input.noise_locs)],dtype=int)*noise_label
        input.refined_labels = all_labels
        balance = input.refined_balance
        labels = input.refined_labels
        score_flag = 1 # merge fss clusters

    if score_flag == 1: # agglomerate the full sets
        refined_balance, refined_labels, label_matching_index = merge_clusters( balance , labels )
        input.refined_balance = refined_balance
        input.refined_labels = refined_labels
        input.nc = refined_balance.shape[0] # number of reduced clusters
        # need to rescore only in order to make the scores match the order of the balances
        input.score_max, input.closure_max, input.area_weights = merge_scores( input.score_max, input.closure_max, input.area_weights, np.unique(labels), label_matching_index )

    if input.verbose == True:
        print('\n full_set_clusters c): np.unique(input.refined_labels) = ',np.unique(input.refined_labels))
        print(' full_set_clusters c): input.refined_balance = ',input.refined_balance)
        print(' full_set_clusters: np.shape(input.refined_balance) = ',np.shape(input.refined_balance))
        if input.cluster_method == 'HDBSCAN':
            print(' full_set_clusters: np.shape(input.noise_locs) = ',np.shape(input.noise_locs))

    return input

def merge_clusters( balance , labels ):
    # balance = array size [nc,nf] with 1's for active terms and 0's for
    # inactive terms.
    # labels = array size [ng] with numbers corresponding to the rows of
    # balance (each number is a cluster id)
    # output: balances with the same sparsity pattern are combined, as are
    # the corresponding labels.
    #print('merge_clusters: np.shape(balance) = ',np.shape(balance))
    refined_balance, label_matching_index = np.unique(balance, axis=0, return_inverse=True)
    refined_labels = np.array([label_matching_index[ii] for ii in labels])
    return refined_balance, refined_labels, label_matching_index

def get_max_scores_strings( max_scores ):
    # labels for plots
    Nmax = len(max_scores)
    max_scores_strings = np.empty(Nmax)
    for uu in range(0,len(max_scores_strings)):
        max_scores_strings[uu] = '%.3f' %max_scores[uu]
    return max_scores_strings

def get_max_scores_and_areas_strings( max_scores, max_area_weights ):
    # labels for plots
    Nmax = len(max_scores)
    max_scores_strings = np.empty(Nmax,dtype=object)
    for uu in range(0,len(max_scores_strings)):
        string = '%.4f,%.4f' %(max_scores[uu],max_area_weights[uu])
        max_scores_strings[uu] = string
    return max_scores_strings

def combine_clusters( full_labels, full_balance, recursive_labels, recursive_balance, locs ):
    # full_ = the full domain results
    # recursive_ = the results for re-clustering a particular cluster
    # locs = the locations within the full domain of the recursive cluster

    nc_full = len(np.unique(full_labels))
    recursive_labels = recursive_labels + nc_full
    full_labels[locs] = recursive_labels
    for uu in range(0,np.shape(recursive_balance)[0]):
        for kk in range(0,np.shape(recursive_balance)[1]):
            if recursive_balance[uu,kk] > 0.:
                recursive_balance[uu,kk] = recursive_balance[uu,kk]+nc_full

    full_balance = np.concatenate((full_balance,recursive_balance),axis=0)
    active_terms = np.zeros(np.shape(full_balance))
    for uu in range(0,np.shape(full_balance)[0]):
        for kk in range(0,np.shape(full_balance)[1]):
            if full_balance[uu,kk] > 0.:
                active_terms[uu,kk] = 1.
    return full_labels, active_terms


def combine_processed_data( data_dicts ):
    # combine two sets of processed data for a given

    full_data = data_dicts[0]
    recursive_data = data_dicts[1]

    full_labels = full_data.labels
    full_balance = full_data.balance
    full_features = full_data.features

    recursive_labels = recursive_data.labels
    recursive_balance = recursive_data.balance
    locs = recursive_data.locs

    # combine data sets:
    combo_labels, combo_balance = combine_clusters( full_labels, full_balance, recursive_labels, recursive_balance, locs )
    #print(np.unique(combo_labels))
    #print(combo_balance)
    #print(np.shape(combo_labels)) #<<<<<

    # merge balances with same sparsity pattern:
    balance, labels, label_matching_index = merge_clusters( combo_balance , combo_labels )
    nc = len(np.unique(labels))
    xc = np.linspace(1,np.shape(balance)[1],num=np.shape(balance)[1],endpoint=True)
    yc = np.linspace(0,np.shape(balance)[0]-1,num=np.shape(balance)[0],endpoint=True)
    Xc,Y = np.meshgrid(xc,yc)

    combo_data = easydict.EasyDict({
        "nf":full_data.nf,
        "nc":nc,
        "refine_flag": True,
        "features":recursive_data.full_features,
        "refined_balance":balance,
        "refined_labels":labels,
        "balance":balance,
        "labels":labels,
        "Nbootstrap":1000,
        })

    # now get M scores:
    combo_data = get_scores( combo_data , preassigned_balance_flag=True )

    mycolors,mymarkers,mymarkersize = get_markers_and_colors( nc )
    combo_data.mycolors = mycolors
    combo_data.mymarkers = mymarkers
    combo_data.mymarkersize = mymarkersize

    combo_data.NE = int(full_data.NE)
    combo_data.mean_score = np.nanmean(combo_data.score_max)
    combo_data.std_score = np.nanstd(combo_data.score_max)
    combo_data.mean_closure = np.nanmean(combo_data.closure_max)
    combo_data.std_closure = np.nanstd(combo_data.closure_max)
    combo_data.features = full_data.features
    combo_data.x = full_data.x
    combo_data.y = full_data.y

    combo_data.max_scores = combo_data.score_max # FIX THE REDUNDANCY HERE
    combo_data.max_closure = combo_data.closure_max # FIX THE REDUNDANCY HERE

    combo_data.read_path = full_data.read_path
    combo_data.figure_path = full_data.figure_path

    combo_data.Xc = Xc

    return combo_data

def get_markers_and_colors( nc ):
    set_colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
    #print(mcolors.to_rgb(set_colors['goldenrod']))
    list_colors = [mcolors.to_rgb(set_colors['gray']), mcolors.to_rgb(set_colors['lightsteelblue']),
        mcolors.to_rgb(set_colors['royalblue']), mcolors.to_rgb(set_colors['blue']),
        mcolors.to_rgb(set_colors['mediumblue']), mcolors.to_rgb(set_colors['darkblue']),
        mcolors.to_rgb(set_colors['midnightblue']), mcolors.to_rgb(set_colors['black']),
        mcolors.to_rgb(set_colors['indigo']), mcolors.to_rgb(set_colors['rebeccapurple']),
        mcolors.to_rgb(set_colors['darkviolet']), mcolors.to_rgb(set_colors['mediumorchid']),
        mcolors.to_rgb(set_colors['orchid']), mcolors.to_rgb(set_colors['plum']),
        mcolors.to_rgb(set_colors['thistle']), mcolors.to_rgb(set_colors['lavenderblush']),
        mcolors.to_rgb(set_colors['black']), mcolors.to_rgb(set_colors['black']),
        mcolors.to_rgb(set_colors['black']), mcolors.to_rgb(set_colors['black']),
        mcolors.to_rgb(set_colors['black']), mcolors.to_rgb(set_colors['black'])]
    #n_bins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]  # Discretizes the interpolation into bins

    mycolors= np.zeros(np.shape(list_colors))
    mycolors[:,:] = list_colors
    if int(nc) == 1:
        mymarkers = np.array(['o'])
        mymarkersize = np.array([8])
    elif int(nc) == 2:
        mymarkers = np.array(['o','o'])
        mymarkersize = np.array([8,8])
    elif int(nc) == 3:
        mymarkers = np.array(['o','o','o'])
        mymarkersize = np.array([8,8,8])
    elif int(nc) == 4:
        mymarkers = np.array(['o','o','o','o'])
        mymarkersize = np.array([8,8,8,8])
    elif int(nc) == 5:
        mymarkers = np.array(['o','o','o','o','o'])
        mymarkersize = np.array([8,8,8,8,8])
    elif int(nc) == 6:
        mymarkers = np.array(['o','o','o','o','o','o'])
        mymarkersize = np.array([8,8,8,8,8,8])
    elif int(nc) == 7:
        #mymarkers = np.array(['o','*','P','s','^','>','v'])
        #mymarkersize = np.array([8,10,8,7,8,8,8])
        mymarkers = np.array(['o','o','o','o','o','o','o'])
        mymarkersize = np.array([8,8,8,8,8,8,8])
    elif int(nc) == 8:
        #mymarkers = np.array(['o','*','P','s','^','>','v','<'])
        #mymarkersize = np.array([8,10,8,7,8,8,8,8])
        mymarkers = np.array(['o','o','o','o','o','o','o','o'])
        mymarkersize = np.array([8,8,8,8,8,8,8,8])
    elif int(nc) == 9:
        mymarkers = np.array(['o','o','o','o','o','o','o','o','o'])
        mymarkersize = np.array([8,8,8,8,8,8,8,8,8])
    elif int(nc) == 10:
        mymarkers = np.array(['o','o','o','o','o','o','o','o','o','o'])
        mymarkersize = np.array([8,8,8,8,8,8,8,8,8,8])
        #mymarkers = np.array(['o','*','P','s','^','>','v','<','D','H'])
        #mymarkersize = np.array([8,10,8,7,8,8,8,8,8,9])
    elif int(nc) == 11:
        mymarkers = np.array(['o','o','o','o','o','o','o','o','o','o','o'])
        mymarkersize = np.array([8,8,8,8,8,8,8,8,8,8,8])
        #mymarkers = np.array(['o','*','P','s','^','>','v','<','D','H','<'])
        #mymarkersize = np.array([8,10,8,7,8,8,8,8,8,9,9])
    elif int(nc) == 12:
        mymarkers = np.array(['o','o','o','o','o','o','o','o','o','o','o','o'])
        mymarkersize = np.array([8,8,8,8,8,8,8,8,8,8,8,8])
        #mymarkers = np.array(['o','*','P','s','^','>','v','<','D','H','<','D'])
        #mymarkersize = np.array([8,10,8,7,8,8,8,8,8,9,9,9])
    elif int(nc) == 13:
        mymarkers = np.array(['o','o','o','o','o','o','o','o','o','o','o','o','o'])
        mymarkersize = np.array([8,8,8,8,8,8,8,8,8,8,8,8,8])
        #mymarkers = np.array(['o','*','P','s','^','>','v','<','D','H','<','D','H'])
        #mymarkersize = np.array([8,10,8,7,8,8,8,8,8,9,9,9,9])
    elif int(nc) == 14:
        mymarkers = np.array(['o','o','o','o','o','o','o','o','o','o','o','o','o','o'])
        mymarkersize = np.array([8,8,8,8,8,8,8,8,8,8,8,8,8,8])
        #mymarkers = np.array(['o','*','P','s','^','>','v','<','D','H','H','H','H','H'])
        #mymarkersize = np.array([8,10,8,7,8,8,8,8,8,9,9,9,9,9])
    elif int(nc) == 15:
        mymarkers = np.array(['o','o','o','o','o','o','o','o','o','o','o','o','o','o','o'])
        mymarkersize = np.array([8,8,8,8,8,8,8,8,8,8,8,8,8,8,8])
        #mymarkers = np.array(['o','*','P','s','^','>','v','<','D','H','H','H','H','H'])
        #mymarkersize = np.array([8,10,8,7,8,8,8,8,8,9,9,9,9,9])
    elif int(nc) == 16:
        mymarkers = np.array(['o','*','P','s','^','>','v','<','D','H','H','H','H','H','H','H'])
        mymarkersize = np.array([8,10,8,7,8,8,8,8,8,9,9,9,9,9,9,9])
    elif int(nc) == 17:
        mymarkers = np.array(['o','*','P','s','^','>','v','<','D','H','H','H','H','H','H','H','H'])
        mymarkersize = np.array([8,10,8,7,8,8,8,8,8,9,9,9,9,9,9,9,9])
    elif int(nc) == 18:
        mymarkers = np.array(['o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o'])
        mymarkersize = np.array([8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8])
    elif int(nc) == 19:
        mymarkers = np.array(['o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o'])
        mymarkersize = np.array([8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8])
        #mymarkers = np.array(['o','*','P','s','^','>','v','<','D','H','H','H','H','H','H','H','H','H','H'])
        #mymarkersize = np.array([8,10,8,7,8,8,8,8,8,9,9,9,9,9,9,9,9,9,9])
    elif int(nc) == 20:
        mymarkers = np.array(['o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o'])
        mymarkersize = np.array([8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8])
        #mymarkers = np.array(['o','*','P','s','^','>','v','<','D','H','H','H','H','H','H','H','H','H','H'])
        #mymarkersize = np.array([8,10,8,7,8,8,8,8,8,9,9,9,9,9,9,9,9,9,9])
    elif int(nc) == 21:
        mymarkers = np.array(['o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o'])
        mymarkersize = np.array([8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8])
    elif int(nc) == 22:
        mymarkers = np.array(['o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o'])
        mymarkersize = np.array([8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8])
    elif int(nc) == 23:
        mymarkers = np.array(['o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o'])
        mymarkersize = np.array([8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8])
    elif int(nc) == 24:
        mymarkers = np.array(['o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o'])
        mymarkersize = np.array([8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8])
    elif int(nc) == 25:
        mymarkers = np.array(['o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o'])
        mymarkersize = np.array([8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8])
    elif int(nc) == 26:
        mymarkers = np.array(['o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o'])
        mymarkersize = np.array([8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8])
    elif int(nc) == 28:
        mymarkers = np.array(['o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o'])
        mymarkersize = np.array([8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8])
    elif int(nc) == 32:
        mymarkers = np.array(['o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o'])
        mymarkersize = np.array([8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8])
    elif int(nc) == 35:
        mymarkers = np.array(['o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o'])
        mymarkersize = np.array([8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8])
    elif int(nc) == 36:
        mymarkers = np.array(['o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o'])
        mymarkersize = np.array([8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8])
    elif int(nc) == 48:
        mymarkers = np.array(['o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o'])
        mymarkersize = np.array([8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8])
    elif int(nc) == 49:
        mymarkers = np.array(['o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o'])
        mymarkersize = np.array([8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8])
    elif int(nc) == 54:
        mymarkers = np.array(['o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o'])
        mymarkersize = np.array([8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8])
    elif int(nc) == 55:
        mymarkers = np.array(['o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o'])
        mymarkersize = np.array([8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8])
    #print(' nc = ',nc)
    return mycolors,mymarkers,mymarkersize


def calculate_accuracy( data , truth ):

    # import data:
    read_path = data.read_path
    cluster_method = data.cluster_method
    reduction_method = data.reduction_method
    #cluster_method = (data.read_path).split('_')[1]
    #refinement_method = (data.read_path).split('_')[2]

    area = data.area #).flatten('F')
    max_area_weights = data.max_area_weights # normalized
    labels_locs = data.labels_locs
    #print(np.shape(labels_locs))
    #print(np.shape(np.unique(labels_locs)))
    #area_weights_file_name = data.read_path + 'area_weights/area_weights_' + index_name %(index0[loc_max[0]],index1[loc_max[1]])
    #max_area_weights_unnormalized = np.load(area_weights_file_name)
    max_scores = data.max_scores # one scores for each cluster
    max_closure = data.max_closure
    #features = data.features
    balance = data.balance
    labels = data.labels
    ng = len(labels)
    #print('np.shape(features) = ',np.shape(features))
    #print('np.shape(balance) = ',np.shape(balance))
    print(balance)
    print('np.shape(labels) = ',np.shape(labels))
    print('np.unique(labels) = ',np.unique(labels)) # <---- do w/o nans!
    x = data.x; y = data.y; nx = len(x); ny = len(y)
    nf = data.nf
    nc = data.nc # the reduced number of clusters

    # import the "truth"
    balance_truth = truth.balance
    score_truth = truth.scores
    balance_truth_unique = np.unique(balance_truth,axis=0)
    nc_truth = np.shape(balance_truth_unique)[0]

    # get the labels for the "truth"
    labels_truth = np.zeros([ng])
    for ii in range(0,ng):
        local_balance_truth = balance_truth[ii,:]
        loc = (np.where((balance_truth_unique==local_balance_truth).all(axis=1)))
        if np.shape(loc)[0] > 1:
            print('ERROR: too many labels for the correct balances')
        if np.shape(loc)[1] > 1:
            print('ERROR: too many labels for the correct balances')
        loc = (loc[0])[0]
        if np.sum(np.abs(balance_truth_unique[loc,:]-local_balance_truth)) > 0.:
            print('ERROR: incorrect balance and label match')
        labels_truth[ii] = loc

    """
    # histogram of how many grid points are labeled what
    nbins = np.arange(len(np.unique(labels_truth)))-0.5
    #print('nbins = ',nbins)
    figure_name = 'histogram_truth_only.png'
    plotname = './Figures/' + figure_name
    fig = plt.figure(figsize=(6, 5)) # for 3x2
    ax=plt.subplot(1,1,1) #2,2,1)
    n, bins, patches = plt.hist(labels_truth, bins=nbins, facecolor='b', alpha=0.75) #density=True,
    plt.subplots_adjust(top=0.95, bottom=0.1, left=0.125, right=0.9, hspace=0.3, wspace=0.15)
    plt.savefig(plotname,format="png"); plt.close(fig);
    """

    # calculate where the "truth" balances matched the cluster+refinement balances:
    score_mean = np.zeros([ng])
    closure_mean = np.zeros([ng])
    balance_err = np.zeros([ng]) # percentage of domain where balances are correct
    balance_full = np.zeros([ng]) # percentage of domain that all terms are active
    incorrect_area = 0.
    for g in range(0,ng):

        # if labeled noise, set balances as nans:
        if np.isnan(labels[g]) == True:
            local_balance = np.zeros(np.shape(balance)[1])*np.nan
            local_score = np.nan
            local_closure = np.nan
        else:
            id = int(labels[g])
            if np.amax(balance[id,:]) > 1.:
                #print('ERROR: local balance is not described by 0s and 1s')
                local_balance = balance[id,:]/np.amax(balance[id,:])
            else:
                local_balance = balance[id,:]
            local_score = max_scores[id]
            local_closure = max_closure[id]

        #print(local_balance)

        # if truth or cluster balances have only 1 term, don't count:
        if np.nansum(local_balance) == 1.:
            local_balance = np.zeros([nf])*np.nan
        if np.nansum(balance_truth[g,:]) == 1.:
            balance_truth[g,:] = np.zeros([nf])*np.nan

        # if the balance has nans, don't count the data point:
        if np.sum(np.isnan(local_balance)) > 0:
            balance_full[g] = np.nan
            balance_err[g] = np.nan
            score_mean[g] = np.nan
            closure_mean[g] = np.nan

        # do I need this part?
        #elif np.sum(np.isnan(local_score)) > 0:
        #    balance_full[g] = np.nan
        #    balance_err[g] = np.nan
        #    score_mean[g] = np.nan
        #    closure_mean[g] = np.nan

        else:
            #print('****here')
            # if local_balance or balance_truth = zeros, convert to all ones:
            if np.nansum(local_balance) == 0.:
                local_balance = np.ones([nf])
            if np.nansum(balance_truth[g,:]) == 0.:
                balance_truth[g,:] = np.ones([nf])

            # diagnostics:
            score_mean[g] = local_score
            closure_mean[g] = local_closure
            if np.sum(np.abs(local_balance - balance_truth[g,:])) > 0.: # if the balance is incorrect
                if np.shape(np.argwhere(labels_locs==g))[0] == 1.: # if the point is a labeled location (not noise)
                    balance_err[g] = 1 # local_balance is wrong
                    incorrect_area = incorrect_area + area[g]
            if np.nansum(balance_truth[g,:]) == nf:
                balance_full[g] = 1. # the true balance is the full set

    #count = 0
    #for i in range(0,ng):
    #    if np.shape(np.argwhere(labels_locs==i))[0] == 1.:
    #        count = count+1
    #print('count = ', count)
    print('len(labels_locs) = ', len(labels_locs))
    print('np.sum(area) = ',np.sum(area))
    print('np.sum(balance_err) = ',np.nansum(balance_err))
    percent_correctly_labeled_area = 1.-incorrect_area/np.sum(area[labels_locs])
    print('percent_correctly_labeled_area = ',percent_correctly_labeled_area)
    labeled_area = np.sum(area[labels_locs])/np.sum(area)
    print('labeled_area = ',labeled_area)
    print()

    # diagnostics:
    area_fraction = area/np.nansum(area)
    #print('np.sum(area_fraction) = ',np.nansum(area_fraction))
    #print('area-weighted cluster mean M score = ',np.nansum(score_mean*area_fraction))
    #print('area-weighted cluster mean closure score = ',np.nansum(closure_mean*area_fraction))
    #print('area-weighted truth mean M score = ',np.nansum(score_truth*area_fraction))
    #print('percentage of the domain the cluster balance is correct = ',1.-np.nansum(balance_err*area_fraction)) #/ng)
    #print('percentage of the domain the true balance is the full set = ',np.nansum(balance_full*area_fraction)) #/ng)
    #print('number of unique balances in data = ',nc_truth)

    data.nc_truth = nc_truth
    data.balance_truth = balance_truth
    data.labels_truth = labels_truth
    data.balance_err = balance_err
    #data.unique_balances_truth = nc_truth
    #data.comparable_samples = ng
    #data.cluster_mean_m_score = np.nanmean(score_mean)
    #data.truth_mean_m_score = np.nanmean(score_truth)
    #data.correct_balance_percent = 1.-np.nansum(balance_err)/ng
    #data.correct_balance_percent = 1.-np.nansum(balance_err)/len(labels_locs) # percentage of labeled points labeled correctly
    #print('np.sum(balance_err) = ',np.nansum(balance_err))
    #print('np.sum(balance_err)/len(labels_locs) = ',np.nansum(balance_err)/len(labels_locs))
    #data.full_balance_percent = np.nansum(balance_full)/ng

    # the two that matter:
    data.correctly_labeled_area = percent_correctly_labeled_area # out of total area
    #print('100.*data.correctly_labeled_area = ',100.*data.correctly_labeled_area)
    data.labeled_area = labeled_area # out of total area
    #print('data.labeled_area = ',data.labeled_area)
    return data

def coarsen_truth( balance0, score0, closure0, x0, y0, reduction_factor ):

    nf = np.shape(balance0)[1]
    nx = len(x0)
    ny = len(y0)
    balance0 = np.reshape(balance0, [ny, nx, nf], order='F')
    score0 = np.reshape(score0, [ny, nx], order='F')
    closure0 = np.reshape(closure0, [ny, nx], order='F')
    idx = np.arange(0,nx,reduction_factor)
    nx = len(idx)

    balance = np.zeros([ny,nx,nf])
    score = np.zeros([ny,nx])
    closure = np.zeros([ny,nx])
    x = np.zeros([len(idx)])
    y = np.copy(y0)
    for k in range(0,nx):
        idx0 = int(idx[k])
        x[k] = x0[idx0]
        for j in range(0,ny):
            balance[j,k,:] = balance0[j,idx0,:]
            score[j,k] = score0[j,idx0]
            closure[j,k] = closure0[j,idx0]
    balance = np.reshape(balance, [int(ny*nx),nf], order='F')
    score = np.reshape(score, [int(ny*nx)], order='F')
    closure = np.reshape(closure, [int(ny*nx)], order='F')

    return balance, score, closure

def coarsen_features( features0 , x0 , y0 , reduction_factor ):

    nf = np.shape(features0)[1]
    nx = len(x0)
    ny = len(y0)
    features0 = np.reshape(features0, [ny, nx, nf], order='F')
    idx = np.arange(0,nx,reduction_factor)
    #print('nx = ',nx)
    nx = len(idx)
    #print('nx = ',nx)

    features = np.zeros([ny,nx,nf])
    x = np.zeros([len(idx)])
    y = np.copy(y0)
    for k in range(0,nx):
        idx0 = int(idx[k])
        x[k] = x0[idx0]
        for j in range(0,ny):
            features[j,k,:] = features0[j,idx0,:]
    features = np.reshape(features, [int(ny*nx),nf], order='F')
    #features = features.flatten('F')

    # get the area (data is cell centered):
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

    return features, x, y, area

def make_output_folders():

    # detect the current working directory and print it
    path = os.getcwd()

    # delete the old Output folder
    remove_path = path +'/Output'
    try:
        shutil.rmtree(remove_path )
    except OSError:
        print ("Deletion of the directory %s failed" %(remove_path))
    else:
        print ("Successfully deleted the directory %s" %(remove_path))

    # create the new
    path1 = path + '/Output'
    path2 = path + '/Output/area_weights'
    path3 = path + '/Output/balance'
    path4 = path + '/Output/closure_score'
    path5 = path + '/Output/feature_mean'
    path6 = path + '/Output/feature_mean_95'
    path7 = path + '/Output/labels'
    path8 = path + '/Output/M_score'
    make_path( path1 )
    make_path( path2 )
    make_path( path3 )
    make_path( path4 )
    make_path( path5 )
    make_path( path6 )
    make_path( path7 )
    make_path( path8 )
    return

def make_path( path ):
    try:
        os.mkdir(path)
    except OSError:
        print ("Creation of the directory %s failed" % path)
    else:
        print ("Successfully created the directory %s " % path)
    return

def test_DBSCAN( path, ieps , ims ):

    balance = np.load(path + 'balance/reduced_balance_eps%i_ms%i.npy' %(ieps,ims))
    labels = np.load(path + 'labels/reduced_labels_eps%i_ms%i.npy' %(ieps,ims))
    print('np.shape(labels) = ',np.shape(labels))
    print('np.shape(balance) = ',np.shape(balance))
    for i in range(0,np.shape(balance)[0]):
        print('cluster balance = ',balance[i,:])
        print('number of grid points in cluster = ',len(np.argwhere(labels==i)[:,0]))

    return

def load_tumor_density( path, q ):

    filename_n = path + 'n_%i.npy' %(q)
    n = np.load(filename_n)

    filename_n_ddt = path + 'n_ddt_%i.npy' %(q)
    n_ddt = np.load(filename_n_ddt)

    return n, n_ddt

class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, vcenter=None, clip=False):
        self.vcenter = vcenter
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.vcenter, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


def get_convergence_and_optimal( data, write_path, feature_path, data_set_name ):

    # 1) convergence data (meta-data for all parameters, realizations)
    data = read_processed_data( data, feature_path, data_set_name )

    # 2) save convergence:
    np.save(write_path + 'conv_mean_score.npy', data.mean_score)
    np.save(write_path + 'conv_std_score.npy', data.std_score)
    if data.bias_flag == 'biased':
        np.save(write_path + 'conv_mean_closure.npy', data.mean_closure)
        np.save(write_path + 'conv_std_closure.npy', data.std_closure)
    elif data.bias_flag == 'unbiased':
        np.save(write_path + 'conv_mean_score_95.npy', data.mean_score_95)
    np.save(write_path + 'conv_E.npy', data.E)
    np.save(write_path + 'x.npy', data.x)
    np.save(write_path + 'y.npy', data.y)
    np.save(write_path + 'conv_fss.npy', data.fss)
    np.save(write_path + 'nan_locs.npy', data.nan_locs)
    np.save(write_path + 'feature_locs.npy', data.feature_locs)
    np.save(write_path + 'conv_Kr.npy', data.Kr) # Kr[nk] = Kr for the best score
    np.save(write_path + 'conv_loc_max.npy', data.loc_max)
    if data.cluster_method != 'HDBSCAN': # and data.reduction_method != 'SPCA':
        np.save(write_path + 'conv_ne.npy', data.ne)

    if data.cluster_method == 'KMeans' or data.cluster_method == 'GMM':
        np.save(write_path + 'conv_nk.npy', data.nk) # index for for the best score
        np.save(write_path + 'conv_K.npy', data.K) # K[nk] = K for the best score

    elif data.cluster_method == 'HDBSCAN':
        np.save(write_path + 'conv_nms.npy', data.nms) # index for for the best score
        np.save(write_path + 'conv_nmcs.npy', data.nmcs) # index for for the best score
        np.save(write_path + 'conv_ms.npy', data.ms)
        np.save(write_path + 'conv_mcs.npy', data.mcs)

    if data.reduction_method == 'SPCA':
        np.save(write_path + 'conv_alphas.npy', data.alphas)
        np.save(write_path + 'conv_na.npy', data.na) # index for for the best score
        #print('na = ',data.na)

    if data.ic_flag == True:
        np.save(write_path + 'conv_aic.npy', data.aic)
        np.save(write_path + 'conv_bic.npy', data.bic)

    # 3) save optimal:
    np.save(write_path + 'optimal_scores.npy', data.max_scores)
    if data.bias_flag == 'biased':
        np.save(write_path + 'optimal_closure.npy', data.max_closure)
    elif data.bias_flag == 'unbiased':
        np.save(write_path + 'optimal_scores_95.npy', data.max_scores_95)
    np.save(write_path + 'optimal_max_area_weights.npy', data.max_area_weights)
    np.save(write_path + 'optimal_labels.npy', data.labels)
    np.save(write_path + 'optimal_balance.npy', data.balance)
    if data.cluster_method == 'HDBSCAN':
        np.save(write_path + 'optimal_retained_columns.npy', data.retained_columns)

    # 4) get statistics from the optimal
    feature_data = get_feature_data( data.feature_path, data.data_set_name, data.standardize_flag, data.residual_flag, data.verbose )
    #print('np.shape(feature_data.raw_features) = ',np.shape(feature_data.raw_features))
    skew = np.zeros([data.nc,data.nf])
    flat = np.zeros([data.nc,data.nf])
    for i in range(0,data.nc):
        for j in range(0,data.nf):
            loc = np.argwhere(data.labels==i)[:,0] # cluster
            population = feature_data.raw_features[loc,j] # feature population
            skew[i,j],flat[i,j] = skewness_flatness( population )

    np.save(write_path + 'optimal_skew.npy', skew)
    np.save(write_path + 'optimal_flat.npy', flat)

    return


def gauss_kern(size, sizey=None):
    # Returns a normalized 2D gauss kernel array for convolutions
    from scipy import mgrid,exp
    size = int(size)
    if not sizey:
        sizey = size
    else:
        sizey = int(sizey)
    x, y = mgrid[-size:size+1, -sizey:sizey+1]
    g = exp(-(x**2/float(size)+y**2/float(sizey)))
    return g / g.sum()

def blur_image(im, n, ny=None) :
    # blurs the image by convolving with a gaussian kernel of typical
    # size n. The optional keyword argument ny allows for a different
    # size in the y direction.
    from scipy import signal
    g = gauss_kern(n, sizey=ny)
    improc = signal.convolve(im,g, mode='valid')
    return(improc)


def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        #raise ValueError "smooth only accepts 1 dimension arrays."
        print("\nERROR: smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        #raise ValueError "Input vector needs to be bigger than window size."
        print("\nERROR: Input vector needs to be bigger than window size.")

    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        #raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"
        print("\nERROR: Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y

def derivatives_1d( f , dx ):
    # second-order gradient & Laplacian,  cell centered grid
    N = len(f)-2
    grad = np.zeros([N])
    lap = np.zeros([N])
    dxm2 = dx**(-2.)
    for i in range(1,N+1):
        grad[i-1] = ( f[i+1] - f[i-1] ) / (2.0 * dx)
        lap[i-1] = (f[i-1] - 2.0*f[i] + f[i+1]) * dxm2
    return grad,lap

def derivatives_1d_variable_resolution( f , x ):
    # second-order 1d derivative,  cell centered grid, variable resolution
    Nx = len(x)-2
    grad = np.zeros([Nx])
    for i in range(1,Nx+1):
        dxi = x[i]-x[i-1]
        dxip1 = x[i+1]-x[i]
        #print('dxi,dxip1 = ',dxi,dxip1)
        #print(' f[i+1] coeff = ',  (dxi**2.) / (dxip1*dxi*(dxip1+dxi)))
        #print(' f[i] coeff = ',  (dxip1**2.-dxi**2.) / (dxip1*dxi*(dxip1+dxi)))
        #print(' f[i-1] coeff = ',  (dxip1**2.) / (dxip1*dxi*(dxip1+dxi)))
        grad[i-1] = ( f[i+1]*(dxi**2.) - f[i-1]*(dxip1**2.) + f[i]*(dxip1**2.-dxi**2.) ) / (dxip1*dxi*(dxip1+dxi))
    return grad

def derivatives_2vars( f , x , y ):
    # second-order gradient & Laplacian,  cell centered grid
    Nx = len(x)-2
    Ny = len(y)-2
    grad = np.zeros([Nx,Ny])
    for i in range(1,N+1):
        for j in range(1,N+1):
            #grad[i-1] = ( f[i+1] - f[i-1] ) / (2.0 * dx)
            dxi = x[i]-x[i-1]
            dxip1 = x[i+1]-x[i]
            dyi = y[i]-y[i-1]
            dyip1 = y[i+1]-y[i]   # now p51, eq.3.33
            #grad[i-1,j-1] = ( f[i+1,j+1] - f[i+1,j-1] - f[i-1,j+1] + f[i-1,j-1] ) / (4.0 * dx * dy)
            grad[i-1,j-1] = ( f[i+1,j] - f[i-1,j] + f[i,j] ) / (dxip1*dxi*(dxip1+dxi))
    return grad

def combine_by_area_threshold( data ):
    # the threshold is for combining all clusters that cover an area that is
    # less percent of the total spatio-temporal area.

    threshold = data.threshold
    #threshold = 0.05
    combined_labels = np.zeros([data.nc])*np.nan
    for j in range(0,data.nc):
        if data.max_area_weights[j] <= threshold:
            combined_labels[j] = j
    kept_labels = np.argwhere(np.isnan(combined_labels))[:,0]
    kept_balances = data.balance[kept_labels,:]
    new_nc = np.sum(np.isnan(combined_labels))+1
    new_labels = - np.ones([data.ng])
    for j in range(0,data.ng):
        for i in range(0,data.nc):
            if np.isnan(combined_labels[i]) == False: # point needs relabeling
                if int(data.labels[j]) == int(combined_labels[i]): # relabel
                    new_labels[j] = new_nc-1
                elif new_labels[j] == -1.: # if no need for relabeling, and the point has not been assigned, then:
                    for k in range(0,new_nc-1):
                        if kept_labels[k] == int(data.labels[j]):
                            new_labels[j] = k
    # get the balances too:
    for k in range(0,new_nc-1):
        kept_balances[k,:] = kept_balances[k,:]/(kept_labels[k]+1)
        kept_balances[k,:] = kept_balances[k,:]*(k+1)
    new_balances = np.zeros([new_nc,data.nf])
    new_balances[0:new_nc-1,:] = kept_balances
    new_balances[new_nc-1,:] = np.ones([data.nf])*new_nc

    data.nc = new_nc
    data.labels = new_labels
    data.refine_flag = True
    data.refined_labels = new_labels
    data.refined_balance = new_balances

    if data_set_name == 'ECCO' or data_set_name == 'ECCO_residual':
        feature_data = get_ECCO_features( features_path, standardize_flag, residual_flag, data.verbose )
    data.raw_features = feature_data.raw_features
    data.features = feature_data.features
    data.balance_combinations = generate_balances(data.nf,data.bias_flag)
    data.Nbootstrap = 1000
    data = get_scores( data , preassigned_balance_flag = True )

    print('\nnp.unique(data.labels) = ',np.unique(data.labels))
    print('data.refined_balance = ',data.refined_balance)
    print('score_max = ', data.score_max)
    print('area_weights = ', data.area_weights)

    return data

def get_Kr_score_stats( data ):

    K_flag = 0
    if data.cluster_method == 'KMeans' or data.cluster_method == 'GMM':
        K_flag = 1

    if K_flag == 1 and data.reduction_method == 'score':
        K = data.K; Kr = data.Kr; NE = data.NE; mean_score = data.mean_score
        Kr_mean = np.zeros([len(K),1])
        Kr_std = np.zeros([len(K),1])
        score_mean = np.zeros([len(K),1])
        score_std = np.zeros([len(K),1])
        if data.ic_flag == True:
            aic_mean = np.zeros([len(K),1])
            aic_std = np.zeros([len(K),1])
            bic_mean = np.zeros([len(K),1])
            bic_std = np.zeros([len(K),1])
        if NE > 1:
            for i in range(0,len(K)):
                Kr_bs = np.zeros([NE,1])
                Kr_bs[:,0] = Kr[i,:]
                Kr_mean[i,:],Kr_std[i,:] = bootstrap_mean( Kr_bs , Nbootstrap=1000 )
                score_bs = np.zeros([NE,1])
                score_bs[:,0] = mean_score[i,:]
                score_mean[i,:],score_std[i,:] = bootstrap_mean( score_bs , Nbootstrap=1000 )
                if data.ic_flag == True:
                    aic_bs = np.zeros([NE,1])
                    aic_bs[:,0] = data.aic[i,:]
                    aic_mean[i,:],aic_std[i,:] = bootstrap_mean( aic_bs , Nbootstrap=1000 )
                    bic_bs = np.zeros([NE,1])
                    bic_bs[:,0] = data.bic[i,:]
                    bic_mean[i,:],bic_std[i,:] = bootstrap_mean( bic_bs , Nbootstrap=1000 )
            data.Kr_mean = Kr_mean
            data.Kr_std = Kr_std
            data.score_mean = score_mean
            data.score_std = score_std
            if data.ic_flag == True:
                data.aic_mean = aic_mean
                data.aic_std = aic_std
                data.bic_mean = bic_mean
                data.bic_std = bic_std
        else:
            print('\n  ADD ic_flag to get_Kr_score_stats! \n')
            for i in range(0,len(K)):
                Kr_mean[i,:] = Kr[i,:]
                score_mean[i,:] = mean_score[i,:]
            data.Kr_mean = Kr_mean
            data.Kr_std = Kr_std*np.nan
            data.score_mean = score_mean
            data.score_std = score_std*np.nan
            #if data.ic_flag == True:
            #    data.aic_mean = aic_mean
            #    data.aic_std = aic_std
            #    data.bic_mean = bic_mean
            #    data.bic_std = bic_std
        return data

    elif K_flag == 1 and data.reduction_method == 'SPCA':

        K = data.K; Kr = data.Kr; NE = data.NE;
        mean_score = data.mean_score; alphas = data.alphas
        #print('np.shape(K),np.shape(alphas) = ',np.shape(K),np.shape(alphas))
        #print('np.shape(Kr) = ',np.shape(Kr))
        #print('np.shape(mean_score) = ',np.shape(mean_score))

        Kr_mean = np.zeros([len(K),1,len(alphas)])
        Kr_std = np.zeros([len(K),1,len(alphas)])
        score_mean = np.zeros([len(K),1,len(alphas)])
        score_std = np.zeros([len(K),1,len(alphas)])
        if NE > 1:
            for i in range(0,len(K)):
                for j in range(0,len(alphas)):
                    Kr_bs = np.zeros([NE,1])
                    Kr_bs[:,0] = Kr[i,:,j]
                    Kr_mean[i,:,j],Kr_std[i,:,j] = bootstrap_mean( Kr_bs , Nbootstrap=1000 )
                    score_bs = np.zeros([NE,1])
                    score_bs[:,0] = mean_score[i,:,j]
                    score_mean[i,:,j],score_std[i,:,j] = bootstrap_mean( score_bs , Nbootstrap=1000 )
            data.Kr_mean = Kr_mean
            data.Kr_std = Kr_std
            data.score_mean = score_mean
            data.score_std = score_std
        else:
            for i in range(0,len(K)):
                for j in range(0,len(alphas)):
                    Kr_mean[i,:,j] = Kr[i,:,j]
                    score_mean[i,:,j] = mean_score[i,:,j]
            data.Kr_mean = Kr_mean
            data.Kr_std = Kr_std*np.nan
            data.score_mean = score_mean
            data.score_std = score_std*np.nan
        return data

def get_dKr_convergence( data ):

    K_flag = 0
    if data.cluster_method == 'KMeans' or data.cluster_method == 'GMM':
        K_flag = 1

    if K_flag == 1 and data.reduction_method == 'score':
        # derivatives:
        dKr0 = np.zeros([len(data.K)-2,data.NE])
        d2Kr0 = np.zeros([len(data.K)-2,data.NE])
        if data.ic_flag == True:
            daic0 = np.zeros([len(data.K)-2,data.NE])
            dbic0 = np.zeros([len(data.K)-2,data.NE])
        for i in range(0,data.NE):
            #dKr0[:,i], d2Kr0[:,i] = derivatives_1d( data.Kr[:,i] , dx=1. )
            dKr0[:,i] = derivatives_1d_variable_resolution( data.Kr[:,i] , data.K )
            if data.ic_flag == True:
                daic0[:,i] = derivatives_1d_variable_resolution( data.aic[:,i] , data.K )
                dbic0[:,i] = derivatives_1d_variable_resolution( data.bic[:,i] , data.K )
        locs = np.argwhere(data.Kr_mean[:,0]>=np.mean(np.sum(dKr0,axis=0)))[:,0]
        Kselect = np.amin(data.K[locs])
        data.Kselect = Kselect

        dKr_mean_smooth = smooth(np.mean(dKr0,axis=1),window_len=data.window_length,window='flat')
        dKr_mean_smooth = dKr_mean_smooth[int(np.floor(data.window_length/2.)):len(dKr_mean_smooth)-int(np.floor(data.window_length/2.))]

        daic_mean_smooth = smooth(np.mean(daic0,axis=1),window_len=data.window_length,window='flat')
        daic_mean_smooth = daic_mean_smooth[int(np.floor(data.window_length/2.)):len(daic_mean_smooth)-int(np.floor(data.window_length/2.))]

        dbic_mean_smooth = smooth(np.mean(dbic0,axis=1),window_len=data.window_length,window='flat')
        dbic_mean_smooth = dbic_mean_smooth[int(np.floor(data.window_length/2.)):len(dbic_mean_smooth)-int(np.floor(data.window_length/2.))]

        data.dKr_mean_smooth = dKr_mean_smooth
        data.dKr0 = dKr0

        data.daic_mean_smooth = daic_mean_smooth
        data.daic0 = daic0

        data.dbic_mean_smooth = dbic_mean_smooth
        data.dbic0 = dbic0

        dKr_mean = np.zeros([len(data.K)-2,1])
        dKr_std = np.zeros([len(data.K)-2,1])
        if data.ic_flag == True:
            daic_mean = np.zeros([len(data.K)-2,1])
            daic_std = np.zeros([len(data.K)-2,1])
            dbic_mean = np.zeros([len(data.K)-2,1])
            dbic_std = np.zeros([len(data.K)-2,1])
        if data.NE > 1:
            for i in range(0,len(data.K)-2):
                dKr_bs = np.zeros([data.NE,1])
                dKr_bs[:,0] = dKr0[i,:]
                dKr_mean[i,:],dKr_std[i,:] = bootstrap_mean( dKr_bs , Nbootstrap=1000 )
                if data.ic_flag == True:
                    daic_bs = np.zeros([data.NE,1])
                    daic_bs[:,0] = daic0[i,:]
                    daic_mean[i,:],daic_std[i,:] = bootstrap_mean( daic_bs , Nbootstrap=1000 )
                    dbic_bs = np.zeros([data.NE,1])
                    dbic_bs[:,0] = dbic0[i,:]
                    dbic_mean[i,:],dbic_std[i,:] = bootstrap_mean( dbic_bs , Nbootstrap=1000 )

        if data.ic_flag == True:
            data.daic_mean = daic_mean
            data.daic_std = daic_std
            data.dbic_mean = dbic_mean
            data.dbic_std = dbic_std
        data.dKr_mean = dKr_mean
        data.dKr_std = dKr_std

        if data.ic_flag == True:
            locs = np.argwhere(daic_mean_smooth>=-200.)[:,0]
            Kaic_select = np.amin(data.K[locs])
            locs = np.argwhere(dbic_mean_smooth>=-200.)[:,0]
            Kbic_select = np.amin(data.K[locs])

            data.Kaic_select = Kaic_select
            data.Kbic_select = Kbic_select

        return data

    if K_flag == 1 and data.reduction_method == 'SPCA':
        # derivatives:
        dKr0 = np.zeros([len(data.K)-2,data.NE])
        d2Kr0 = np.zeros([len(data.K)-2,data.NE])
        for i in range(0,data.NE):
            dKr0[:,i], d2Kr0[:,i] = derivatives_2vars( data.Kr[:,i] , data.K , data.alphas )  # CHECK dx
        #print('KrR_select = mean integral dKrR/dK: np.mean(np.sum(dKr0,axis=0)) = ',np.mean(np.sum(dKr0,axis=0)))
        #print('KrR_select_std = std integral dKrR/dK: np.std(np.sum(dKr0,axis=0)) = ',np.std(np.sum(dKr0,axis=0)))
        locs = np.argwhere(data.Kr_mean[:,0]>=np.mean(np.sum(dKr0,axis=0)))[:,0]
        Kselect = np.amin(data.K[locs])
        dKr_mean_smooth = smooth(np.mean(dKr0,axis=1),window_len=data.window_length,window='flat')
        dKr_mean_smooth = dKr_mean_smooth[int(np.floor(data.window_length/2.)):len(dKr_mean_smooth)-int(np.floor(data.window_length/2.))]
        data.Kselect = Kselect
        data.dKr_mean_smooth = dKr_mean_smooth
        data.dKr0 = dKr0
        return data

def load_nterms_angio( output_path, q ):

    #t0 = get_restart_time( q , output_path )
    #print('t = ',t0)

    # hapto1, hapto2 ~ grad(n) dot grad(f) , n laplacian(f)

    filename_ddt = output_path + 'n_ddt_%i.npy' %(q)
    filename_diffusion = output_path + 'n_diffusion_%i.npy' %(q)
    filename_hapto1 = output_path + 'n_hapto1_%i.npy' %(q)
    filename_hapto2 = output_path + 'n_hapto2_%i.npy' %(q)
    filename_chemo1 = output_path + 'n_chemo1_%i.npy' %(q)
    filename_chemo2 = output_path + 'n_chemo2_%i.npy' %(q)
    filename_chemo3 = output_path + 'n_chemo3_%i.npy' %(q)

    nterms = easydict.EasyDict({
           "ddt": np.load(filename_ddt),
           "diffusion": np.load(filename_diffusion),
           "hapto1": np.load(filename_hapto1),
           "hapto2": np.load(filename_hapto2),
           "chemo1": np.load(filename_chemo1),
           "chemo2": np.load(filename_chemo2),
           "chemo3": np.load(filename_chemo3),
           "q": q,
           })

    return nterms

def load_vars_angio( output_path, q ):

    filename_c = output_path + 'c_%i.npy' %(q)
    filename_f = output_path + 'f_%i.npy' %(q)
    filename_n = output_path + 'n_%i.npy' %(q)

    c = np.load(filename_c)
    f = np.load(filename_f)
    n = np.load(filename_n)

    return c, f, n

def load_results( data ):

    # grid:
    if data.data_set_name == 'TBL': #'transitional_boundary_layer':
        TBL_file_path = data.features_path + 'Transition_BL_Time_Averaged_Profiles.h5'
        file = h5py.File( TBL_file_path , 'r')
        data.x = np.array(file['x_coor'])
        data.y = np.array(file['y_coor'])
        data.nx = len(data.x); data.ny = len(data.y)
    else:
        data.x = np.load(data.read_path + 'x.npy')
        data.y = np.load(data.read_path + 'y.npy')
        data.nx = len(data.x); data.ny = len(data.y)

    # convergence:
    if data.bias_flag == 'biased':
        data.mean_closure = np.load(data.read_path + 'conv_mean_closure.npy')
    if data.bias_flag == 'unbiased':
        data.mean_score_95 = np.load(data.read_path + 'conv_mean_score_95.npy')
    data.mean_score = np.load(data.read_path + 'conv_mean_score.npy')
    #data.fss = np.load(data.read_path + 'conv_fss.npy')
    data.Kr = np.load(data.read_path + 'conv_Kr.npy')
    data.E = np.load(data.read_path + 'conv_E.npy')
    data.NE = len(data.E)
    if data.cluster_method == 'GMM' or data.cluster_method == 'KMeans':
        data.K = np.load(data.read_path + 'conv_K.npy')
    elif data.cluster_method == 'HDBSCAN':
        data.ms = np.load(data.read_path + 'conv_ms.npy')
        data.mcs = np.load(data.read_path + 'conv_mcs.npy')
    if data.reduction_method == 'SPCA':
        data.alphas = np.load(data.read_path + 'conv_alphas.npy')
    if data.ic_flag == True:
        data.aic = np.load(data.read_path + 'conv_aic.npy')
        data.bic = np.load(data.read_path + 'conv_bic.npy')

    # optimal results:
    if data.cluster_method == 'GMM' or data.cluster_method == 'KMeans':
        data.nk = np.load(data.read_path + 'conv_nk.npy')
    elif data.cluster_method == 'HDBSCAN':
        data.nms = np.load(data.read_path + 'conv_nms.npy')
        data.nmcs = np.load(data.read_path + 'conv_nmcs.npy')
        data.retained_columns = np.load( data.read_path + 'optimal_retained_columns.npy' )
    if data.reduction_method == 'SPCA':
        data.na = np.load(data.read_path + 'conv_na.npy')
    if data.cluster_method != 'HDBSCAN': # and data.reduction_method != 'SPCA':
        data.ne = np.load(data.read_path + 'conv_ne.npy')
    data.nan_locs = np.load( data.read_path + 'nan_locs.npy' )
    data.feature_locs = np.load( data.read_path + 'feature_locs.npy' )
    data.labels = np.load( data.read_path + 'optimal_labels.npy' )
    data.nc = len(np.unique(data.labels))
    data.balance = np.load( data.read_path + 'optimal_balance.npy' )
    data.reduced_balance = data.balance
    data.max_scores = np.load( data.read_path + 'optimal_scores.npy' )
    if data.bias_flag == 'unbiased':
        data.max_scores_95 = np.load( data.read_path + 'optimal_scores_95.npy' )
    data.max_area_weights = np.load( data.read_path + 'optimal_max_area_weights.npy' )
    #max_scores_strings = get_max_scores_strings( max_scores )
    data.max_scores_strings = get_max_scores_and_areas_strings( data.max_scores, data.max_area_weights )

    return data
