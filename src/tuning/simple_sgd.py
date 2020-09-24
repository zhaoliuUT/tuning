import numpy as np
from tuning.cyMINoncyclic import mc_mean_grad_noncyclic # poisson model
from tuning.cyMINoncyclic import mc_mean_grad_gaussian, mc_mean_grad_gaussian_inhomo, \
mc_mean_grad_gaussian_inhomo_no_corr # gaussian models
import matplotlib.pyplot as plt


#============SGD algorithms for all models, ============
#============ with fp, fm constraints and regularity (1d/2d neighbours)============

def simple_sgd_with_laplacian(
    model, 
    tuning, weight, 
    eta, num_iter, fp, fm, mc_iter=1,
    inv_cov_mat=None, # only used for Gaussian
    laplacian_coeff=0, 
    laplacian_shape=None, # only used in 2d laplacian
    weighted_laplacian=False,
    conv=None, tau=1.0, num_threads=8,
):
    '''assume the points in 'tuning' and 'weight' are arranged in 1d or 2d neighbours. 
    (periodic boundary condition.)
    If 1d neighbour, laplacian_shape is None.
    If 2d neighbour, laplacian_shape: (numNeuro, numBin1, numBin2)
    (tuning, weight can be of numNeuro*(numBin1*numBin2) shape, or (numNeuro, numBin1, numBin2))
    '''
    if model not in ['Poisson', 'GaussianHomo', 'GaussianInhomo', 'GaussianInhomoNoCorr']:
        raise Exception('Model not implemented!')
    if model in ['GaussianHomo', 'GaussianInhomo', 'GaussianInhomoNoCorr'] and inv_cov_mat is None:
        raise Exception('Missing input for the inverse covariance matrix of Gaussian model!')
        
    
    nNeuro = tuning.shape[0] # whether 1d or 2d laplacian
    nBin = weight.size  # whether 1d or 2d laplacian
    if conv is None:
        conv = np.zeros(nBin)
        conv[0] = 1        
    
    x = tuning.reshape((nNeuro, nBin)).copy()
    x_grad = np.zeros((nNeuro, nBin))
    
    if laplacian_shape is not None: # 2d laplacian
        nNeuro, nBin1, nBin2 = laplacian_shape
        #nBin = nBin1*nBin2
    tuning_shape = tuning.shape
    
    # determine the monte carlo iteration function used 
  
    if model == 'Poisson':
        mc_iter_func = mc_mean_grad_noncyclic
    elif model == 'GaussianHomo':
        mc_iter_func = mc_mean_grad_gaussian
    elif model == 'GaussianInhomo':
        mc_iter_func = mc_mean_grad_gaussian_inhomo
    elif model == 'GaussianInhomoNoCorr':
        mc_iter_func = mc_mean_grad_gaussian_inhomo_no_corr        
        
    curve_list = []
    curve_grad_list = []            
    
    for i in range(num_iter):
        x_grad *= 0
        if model=='Poisson':
            x_mean = mc_iter_func(
                x_grad, x, weight, conv, tau, numIter=mc_iter, my_num_threads=num_threads)
        else:
            x_mean = mc_iter_func(
                x_grad, x, weight, inv_cov_mat, # for Gaussian Models
                conv, tau, numIter=mc_iter, my_num_threads=num_threads)
            
        x += eta*x_grad
        if (laplacian_coeff!=0) and (laplacian_shape is None): # 1d laplacian
            laplacian_term = np.zeros_like(x)
            if weighted_laplacian:                
                for j in range(nBin):
                    laplacian_term[:, j] = weight[(j-1)%nBin]*(x[:, (j-1)%nBin]-x[:, j])
                    laplacian_term[:, j] += weight[(j+1)%nBin]*(x[:, (j+1)%nBin]-x[:, j])
                laplacian_term *= weight #laplacian_term[k, j]*weight[j]
            else:
                for j in range(nBin):
                    laplacian_term[:, j] = x[:, (j-1)%nBin] + x[:, (j+1)%nBin] - 2*x[:, j]
            x += laplacian_coeff * laplacian_term
        
        
        if (laplacian_coeff!=0) and (laplacian_shape is not None): # 2d laplacian
            xs = x.reshape((nNeuro, nBin1, nBin2))
            ws = weight.reshape((nBin1, nBin2))
            laplacian_term = np.zeros((nNeuro, nBin1, nBin2))
            if weighted_laplacian:
                for j1 in range(nBin1):
                    for j2 in range(nBin2):
                        laplacian_term[:, j1, j2] = ws[(j1-1)%nBin1, j2]*(xs[:, (j1-1)%nBin1, j2] - xs[:, j1, j2])
                        laplacian_term[:, j1, j2] += ws[(j1+1)%nBin1, j2]*(xs[:, (j1+1)%nBin1, j2] - xs[:, j1, j2])
                        laplacian_term[:, j1, j2] += ws[j1, (j2-1)%nBin2]*(xs[:,j1,(j2-1)%nBin2] - xs[:, j1, j2])
                        laplacian_term[:, j1, j2] += ws[j1, (j2+1)%nBin2]*(xs[:,j1,(j2+1)%nBin2] - xs[:, j1, j2])
                laplacian_term *= ws #laplacian_term[k, j1, j2]*ws[j1, j2]
            else:
                for j1 in range(nBin1):
                    for j2 in range(nBin2):
                        laplacian_term[:, j1, j2] = xs[:, (j1-1)%nBin1, j2] + xs[:, (j1+1)%nBin1, j2] \
                        + xs[:,j1,(j2-1)%nBin2] + xs[:,j1,(j2+1)%nBin2]- 4*xs[:, j1, j2]
            xs += laplacian_coeff * laplacian_term
            x = xs.reshape((nNeuro, nBin1*nBin2))
        
        
        x[x>fp] = fp
        x[x<fm] = fm
        curve_list.append(x.reshape(tuning_shape).copy())
        curve_grad_list.append(x_grad.reshape(tuning_shape).copy())
        
    return curve_list, curve_grad_list