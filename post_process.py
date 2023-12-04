
# need to plot the chosen balances for the best scores (reduced_balances).
# plot the error of the feature mean (how well the clustering algorithm is working)
# need to plot recursion for transition in GMM
# need to plot noise ratio for DBSCAN

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
import easydict
from utils import get_convergence_and_optimal

compute_location = 'home' #'darwin'
data_set_name = 'synth' #'TBL' #'Stokes' #'ECCO' #
residual_flag = 'off' # only matters for ECCO
cluster_method = 'KMeans' #'GMM' # 'HDBSCAN' #
reduction_method = 'SPCA' # 'score' #
standardize_flag = 'on' #'off' # off for synth
verbose = False
ic_flag = False #  True #
bias_flag = 'unbiased'


# optimum
#idx_0 = 0
#idx_1 = 0
#idx_2 = 0 #  9 HDBSCAN (eps_0_ms1 results)

# optimum range (HDBSCAN, score, Stokes) total time = 1438.0803661346436

idx_0 = 0
idx_1 = 0
idx_2 = 0
#NE = 2 # K-Means CHS, SPCA

#if cluster_method == 'HDBSCAN':
#    NE = 1 # 1 HDBSCAN, 2 KMeans
#elif cluster_method == 'KMeans':
#    NE = 1 # 1 HDBSCAN, 2 KMeans

NE = 1
"""
# optimum range (K-means, score, Stokes)
idx_0 = 0 # K
idx_1 = 0 # realization
idx_2 = 0 # alpha or empty
NE = 2 #100
"""
# hyperparameters
E = np.arange(NE) # ensemble realizations

K = np.arange(2,11,dtype=int)
#K = np.arange(4,100,dtype=int)

alphas = np.logspace(-2,2.,num=100,endpoint=True) # KMEANS, HDBSCAN
#alphas = np.logspace(-2,-1.,num=301,endpoint=True) # FOR HDBSCAN, SYNTH
#alphas = np.array([0.001])


mcs = np.arange(100,1001,100,dtype=int)
ms = np.array([10,50,100])
#mcs = np.append(np.array([5,10,15,50]),np.arange(100,1001,100,dtype=int))
#ms = np.array([10,50,100])
#ms = np.array([1])
#mcs = np.array([800,900,1000])
#ms = np.array([100])

if compute_location == 'darwin':
    path = './'
elif compute_location == 'home':
    path = '/Users/bkaiser/Documents/data/m_score/' + data_set_name + '_' + cluster_method + '/'
    #if data_set_name == 'transitional_boundary_layer':
    #    path = '/Users/bkaiser/Documents/data/m_score/' + data_set_name + '_' + cluster_method + '/'
    #else:
    #    path = '/Users/bkaiser/Documents/data/m_score/' + data_set_name + '/'
write_path = path + 'results/'
figure_path = path + 'figures/'
read_path = path + 'output/'
feature_path = path + 'features/'

nan_locs = np.load(read_path + 'map_nan_locs.npy') #<------------------ NaN loc problem
feature_locs = np.load(read_path + 'feature_locs.npy')

if data_set_name == 'ECCO' or data_set_name == 'ECCO_residual':
    Nx = 720; Ny = 360
    x = np.linspace(0.5,float(Nx-0.5),num=Nx,endpoint=True)
    y = np.linspace(0.5,float(Ny-0.5),num=Ny,endpoint=True)
    x = x/2; y = y/2-90.;
elif data_set_name == 'tumor_angiogenesis':
    x = np.linspace(0.,1.,num=400,endpoint=True) # cell edges
    y = np.copy(x)
elif data_set_name == 'TBL':
    TBL_file_path = feature_path + 'Transition_BL_Time_Averaged_Profiles.h5'
    file = h5py.File( TBL_file_path , 'r')
    x = np.array(file['x_coor'])
    y = np.array(file['y_coor'])
elif data_set_name == 'Stokes':
    x = np.load(feature_path + 't.npy')
    y = np.load(feature_path + 'z.npy')
    #t = np.linspace((tf/Nt)/2.,tf-(tf/Nt)/2.,num=Nt,endpoint=True) # centers (input .npy)
    #z = -np.cos(((np.linspace(1., 2.*Nz, num=int(2*Nz)))*2.-1.)/(4.*Nz)*np.pi)*H+H # centers (input .npy)
    #z = z[0:Nz]
elif data_set_name == 'synth':
    x = np.load(feature_path + 'x.npy')
    y = np.load(feature_path + 'y.npy')

#if cluster_method == 'HDBSCAN':
#    standardize_flag = 'off'
#else:
#    standardize_flag = 'on'

print('\n  Data set: ',data_set_name)
print('  residual = ',residual_flag)
print('  standardization = ',standardize_flag)
print('  clustering method = ',cluster_method)
print('  refinement method = ',reduction_method)

data = easydict.EasyDict({
    "data_set_name": data_set_name,
    "cluster_method": cluster_method,
    "reduction_method": reduction_method,
    "write_path": write_path,
    "figure_path": figure_path,
    "feature_path": feature_path,
    "read_path": read_path,
    "standardize_flag": standardize_flag,
    "residual_flag": residual_flag,
    "nan_locs": nan_locs,
    "feature_locs": feature_locs,
    "min_convergence_idx_0":idx_0,
    "min_convergence_idx_1":idx_1,
    "min_convergence_idx_2":idx_2,
    "NE": NE,
    "K": K,
    "E": E,
    "mcs": mcs,
    "ms": ms,
    "alphas":alphas,
    "x": x,
    "y": y,
    "verbose": verbose,
    "ic_flag": ic_flag,
    "bias_flag": bias_flag,
    })

get_convergence_and_optimal( data, write_path, feature_path, data_set_name )

print('\n\n  Post processing complete! \n\n')
