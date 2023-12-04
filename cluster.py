import numpy as np
import math as ma
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rcParams.update({'font.size': 15})
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
import random
import h5py
import time

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

from utils import cluster, refine, get_TBL_features, generate_balances
from utils import coarsen_features, make_output_folders, get_feature_data
from utils import remove_nans_infs
import easydict

# problem with SPCA: is it getting normalized data?
# features_normalized: need to fix denom for -inf
# do synth with only 2 orders of mag difference

# data set choices:
# 'ECCO_residual', 'ECCO', 'tumor_angiogenesis', 'transitional_boundary_layer'

# compute location choices:
# 'home', 'darwin'

chosen_line = 1 # starts at 1
compute_location = 'home'
data_set_name = 'synth' #'TBL'  #'synth' #'ECCO' # 'Stokes'  #'TBL' #'tumor_angiogenesis' #


input_path = './input_parameters.txt'
lines = []
with open(input_path) as f:
    lines = f.readlines()
count = 0
data = []
for line in lines:
    count += 1
    data = np.append(data,f'{line}')
    #print(f'line {count}: {line}')

cluster_method = data[chosen_line].split()[0]
#cluster_method = 'GMM' #'KMeans' # 'HDBSCAN' #
reduction_method = data[chosen_line].split()[1]
#reduction_method = 'SPCA' #'score' #
if data[chosen_line].split()[2] != 'on':
    ic_flag = False
else:
    ic_flag = True
#ic_flag = False
standardize_flag = data[chosen_line].split()[3]
#standardize_flag = 'on'
bias_flag = data[chosen_line].split()[4]
#bias_flag = 'unbiased'
if data[chosen_line].split()[5] != 'on':
    save_flag = False
else:
    save_flag = True
#save_flag = True

# Number of ensemble realizations
E = np.arange(int(data[chosen_line].split()[6]),int(data[chosen_line].split()[7]),dtype=int)

# K-means / GMM number of prescribed clusters
K = np.arange(int(data[chosen_line].split()[8]),int(data[chosen_line].split()[9]),dtype=int)

# alpha l1 regression coefficient for SPCA
print('  float(data[chosen_line].split()[10]) = ',float(data[chosen_line].split()[10]))
print('  float(data[chosen_line].split()[11]) = ',float(data[chosen_line].split()[11]))
alphas = np.logspace(float(data[chosen_line].split()[10]),float(data[chosen_line].split()[11]),num=int(data[chosen_line].split()[12]),endpoint=True)
print('  alphas = ',alphas)
#print(' cluster_method = ',cluster_method)
#print(' reduction_method = ',reduction_method)


sample_pct = 1. #2375 # 0.25 # percent of sample for learning
Nbootstrap = 0 #1000
verbose = False #True #
skew_flag = False #True # depreciated
score_confidence = 'SEM'


selection_epsilon = 0.01 #1 # HDBSCAN, 0.0 for score, 0.01 for SPCA (for Stokes)
selection_method = 'leaf' #'eom' #'eom' #'leaf' # # HDBSCAN
selection_metric = 'euclidean' #'manhattan' #
alpha_factor = 100 # integer factor for file naming for SPCA

# turning down ms = 1 with eom = many little clusters with a decent score.
# turning ms = 10 with eom = fewer clusters with a better score.


residual_flag = 'off' # only matters for ECCO # depreciated
feature_removal = 'off' # depreciated

if compute_location == 'darwin':
    path = './'
elif compute_location == 'home':
    path = '/Users/bkaiser/Documents/data/m_score/' + data_set_name + '_' + cluster_method + '/'
    #if data_set_name == 'transitional_boundary_layer':
    #    path = '/Users/bkaiser/Documents/data/m_score/' + data_set_name + '_' + cluster_method + '/'
    #else:
    #    path = '/Users/bkaiser/Documents/data/m_score/' + data_set_name + '/'

write_path = path + 'output/'
figure_path = path + 'figures/'
read_path = path + 'features/'

#===============================================================================
# hyperparameters, etc

#if cluster_method == 'HDBSCAN':
#    NE = 1 # 1 HDBSCAN, 2 KMeans
#elif cluster_method == 'KMeans':

#NE = 1 # 1 HDBSCAN, 2 KMeans

# read these:

# Number of ensemble realizations
#E = np.arange(NE) # K-Means, SPCA
#E = np.arange(1) # HDBSCAN, SPCA

#K = np.arange(6,20,dtype=int)
#K = np.arange(4,100,dtype=int)
#K = np.array([50,51],dtype=int)
#K = np.array([9],dtype=int)


#alphas = np.logspace(-1,2.,num=10,endpoint=True) # KMEANS, HDBSCAN
#alphas = np.logspace(-2,-1.,num=301,endpoint=True) # FOR HDBSCAN, SYNTH
#alphas = np.array([10.])

# variable mcs and ms with eom
#mcs = np.array([5,10,15,50])
mcs = np.append( np.arange(10,101,10,dtype=int) , np.arange(200,1001,200,dtype=int) )
#ms = np.array([1])
ms = np.array([1,2,3,4,5,6,7,8,9,10,20,50])

#mcs = np.array([800,900,1000])
#mcs = np.arange(1000,5001,200,dtype=int)
#ms = np.array([100,150,200])

# (DBSCAN) epsilon
eps = np.logspace(-3.3979400086720376,-0.3979400086720376,num=16)

print('\n  data set: ',data_set_name)
print('  residual = ',residual_flag)
print('  standardization = ',standardize_flag)
if cluster_method == 'HDBSCAN':
    print('  mcs = ',mcs)
    print('  ms = ',ms)

#===============================================================================
# get features:

data = get_feature_data( read_path, data_set_name, standardize_flag, residual_flag, verbose )
data.cluster_method = cluster_method
data.reduction_method = reduction_method
data.data_set_name = data_set_name
data.read_path = read_path
data.figure_path = figure_path
data.write_path = write_path
data.save_flag = save_flag
data.standardize_flag = standardize_flag
data.residual_flag = residual_flag
data.feature_removal = feature_removal
data.sample_pct = sample_pct
data.preassigned_balance = False
data.Nbootstrap = Nbootstrap
data.E = E
data.K = K
data.eps = eps
data.ms = ms
data.mcs = mcs
data.alphas = alphas
data.nf_total = data.nf
data.verbose = verbose
data.ic_flag = ic_flag
data.selection_epsilon = selection_epsilon  #1 # HDBSCAN, 0.0 for score, 0.01 for SPCA (for Stokes)
data.selection_method = selection_method #'leaf' # # HDBSCAN
data.selection_metric = selection_metric #'manhattan' #
data.skew_flag = skew_flag
data.alpha_factor = alpha_factor
data.bias_flag = bias_flag
data.score_confidence = score_confidence


# worked for HDBSCAN + CHS:
#data.selection_epsilon = 0.325 #1 # HDBSCAN, 0.0 for score, 0.01 for SPCA (for Stokes)
#data.selection_method = 'leaf' #'eom' # # HDBSCAN
#data.selection_metric = 'manhattan' #'euclidean' #

# save metadata:
if save_flag == True:
    nan_locs_file_name =  write_path + 'map_nan_locs.npy'
    np.save(nan_locs_file_name, data.map_nan_locs)
    feature_locs_file_name =  write_path + 'feature_locs.npy'
    np.save(feature_locs_file_name, data.feature_locs)
    features_file_name =  write_path + 'features.npy'
    np.save(features_file_name, data.features)
    ne_file_name =  write_path + 'ne.npy'
    np.save(ne_file_name, data.E)
    if cluster_method == 'GMM' or cluster_method == 'KMeans':
        K_file_name =  write_path + 'K.npy'
        np.save(K_file_name, data.K)
        #np.save(K_file_name, data)
    elif cluster_method == 'DBSCAN':
        epsilon_file_name =  write_path + 'epsilon.npy'
        #np.save(epsilon_file_name, eps)
        np.save(epsilon_file_name, data.eps)
        min_samples_file_name =  write_path + 'min_samples.npy'
        #np.save(min_samples_file_name, ms)
        np.save(min_samples_file_name, data.ms)
        mcs_samples_file_name =  write_path + 'min_cluster_size.npy'
        #np.save(mcs_samples_file_name, mcs)
        np.save(mcs_samples_file_name, data.mcs)
    if reduction_method == 'SPCA':
        alphas_file_name =  write_path + 'alphas.npy'
        #np.save(alphas_file_name, alphas)
        np.save(alphas_file_name, data.alphas)


#===============================================================================

NA = len(alphas)
NE = len(E)
if cluster_method == 'GMM' or cluster_method == 'KMeans':
    Nj = len(K)
    if reduction_method == 'SPCA': # FIX THIS...
        Ni = len(alphas)
    else:
        Ni = 1
elif cluster_method == 'DBSCAN':
    Nj = len(ms)
    Ni = len(eps)
elif cluster_method == 'HDBSCAN':
    Nj = len(ms)
    Ni = len(mcs)

print('  clustering method = ',cluster_method)
print('  refinement method = ',reduction_method)
print('  score = ',bias_flag)

if cluster_method == 'HDBSCAN' and reduction_method == 'SPCA':
    NE = NA # no ensemble realizations required for HDBSCAN

start_total_time = time.time()
one_realization_time = np.zeros([NE,Nj,Ni])
for ee in range(0,NE): # number of ensemble realizations
    for j in range(0,Nj):
        for i in range(0,Ni):

            # 0) Add indices to dictionary (for saving output)
            print('\n*********************************************************************')
            if cluster_method == 'HDBSCAN' and reduction_method == 'SPCA':
                data.alpha_opt = data.alphas[ee]
                data.aa = ee
                print('\n  alpha = ',data.alpha_opt)
                print('  alpha index = ',ee)
            else:
                data.E = E[ee]
                print('\n  realization =',data.E)
            if verbose == True:
                print('  np.shape(data.features) = ',np.shape(data.features))
                print('  np.shape(data.raw_features) = ',np.shape(data.raw_features))
            if cluster_method == 'GMM' or cluster_method == 'KMeans':
                data.K = int(K[j])
                print('  number of prescribed clusters, K =',int(data.K))
                if reduction_method == 'SPCA':
                    data.alpha_opt = data.alphas[i]
                    data.aa = i
                    print('  alpha = ',data.alpha_opt)
                else:
                    print('  placeholder index i =',i)
            elif cluster_method == 'DBSCAN':
                data.ms = int(ms[j])
                data.eps = eps[i]
                print('  mininum number of samples =',data.ms)
                print('  epsilon =',data.eps)
            elif cluster_method == 'HDBSCAN':
                data.ms = int(ms[j])
                data.mcs = int(mcs[i])
                print('  mininum number of samples = ',data.ms)
                print('  mininum cluster size = ',data.mcs)

            if data.nf < data.nf_total:
                re_data = get_feature_data( read_path, data_set_name, standardize_flag, residual_flag )
                data.nf = re_data.nf
                data.balance_combinations = re_data.balance_combinations
                data.features = re_data.features
                data.raw_features = re_data.raw_features
                print('\n   Re-importing feature data:')
                print('   now: data.nf,data.nf_total = ',data.nf,data.nf_total)

            start_one_realization = time.time()

            # 1) Assign labels for the entire domain via clustering
            data = cluster( data , True )

            # 2) Find identical clusters, merge, score, save:
            refine( data )

            one_realization_time[ee,j,i] = time.time() - start_one_realization
            print('  wall time = ',one_realization_time[ee,j,i])

total_time = time.time() - start_total_time
print('\n  Calculation complete!')
print('  Total time: ',total_time)

file_name =  write_path + 'one_realization_time.npy'
np.save(file_name, one_realization_time)
file_name =  write_path + 'total_time.npy'
np.save(file_name, total_time)
