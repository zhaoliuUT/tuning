import numpy as np
from tuning.cyMINoncyclic import mc_mean_grad_noncyclic # poisson model
from tuning.cyMINoncyclic import mc_mean_grad_gaussian, mc_mean_grad_gaussian_inhomo, \
mc_mean_grad_gaussian_inhomo_no_corr # gaussian models
import matplotlib.pyplot as plt


#============Gradient Ascent and helper functions for maximizing the lower bound ============
#============ with fp, fm constraints and regularity (1d/2d neighbours)============

def lower_bound_compute_Q_matrix(tuning, var_diagonal):
    '''var_diagonal shape: (numNeuro, numBin) (same as tuning)'''
    nNeuro, nBin = tuning.shape
    dS = np.zeros((nBin, nBin))
    const = 1
    for k in range(nNeuro):
        dX = tuning[k,:][None, :] - tuning[k,:][:, None] # dX[i,j]=tuning[k,j]-tuning[k, i]
        sumVar = var_diagonal[k,:][None, :] + var_diagonal[k,:][:, None]
        #sumVar[i,j]=var_diagonal[k,j]+var_diagonal[k,i]
        dS += dX**2/sumVar
        const *= sumVar

    Q = np.exp(-0.5*dS)/np.sqrt(const)
    return Q

def lower_bound_evaluate(tuning, weight, var_diagonal):
    Q = lower_bound_compute_Q_matrix(tuning, var_diagonal)
#     MI_lower_bound = 0
#     # the first term
#     weighted_Q = np.sum(weight[None,:]*Q, axis = 1) # sum of weight[j]*Q[i,j] over j for fixed i
#     MI_lower_bound += (-1)*np.sum(weight*np.log(weighted_Q))
#     MI_lower_bound += (-0.5)*np.sum(weight*np.sum(np.log(var_diagonal), axis = 0))
#     # inside:  sum of log(var_diagonal[k,i]) over k for fixed i
    V = np.sum(np.log(var_diagonal), axis = 0) # sum_{k}(log(V_{k,i}))
    MI_lower_bound = -np.dot(weight, np.log(np.dot(Q, weight))) - 0.5*np.dot(weight, V)
    MI_lower_bound -= 0.5*tuning.shape[0]
    return MI_lower_bound

def lower_bound_grad(tuning, weight, var_diagonal, var_derivative):
    '''Gradient of the lower bound of MI with respective to f.
    If var_derivative is None: variance does not change with f
    If var_derivative is not None: variance changes with f,
    for Gaussian model with derivative of Variance - used to approximate Poisson model.
    (Poisson model: var_derivative is all ones matrix)
    '''
    # tuning.shape = (numNeuro, numBin)
    # weight: vector, sum =1, length = numBin
    # variance: same shape as tuning
    # var_derivative: same shape as tuning
    
    nNeuro, nBin = tuning.shape

    dS = np.zeros((nBin, nBin))
    const = 1
    for k in range(nNeuro):
        dX = tuning[k,:][None, :] - tuning[k,:][:, None] # dX[i,j]=tuning[k,j]-tuning[k, i]
        sumVar = var_diagonal[k,:][None, :] + var_diagonal[k,:][:, None] #sumVar[i,j]=var_diagonal[k,j] + var_diagonal[k,i]
        dS += dX**2/sumVar
        const *= sumVar

    Q = np.exp(-0.5*dS)/np.sqrt(const)
    weightQ = np.dot(Q, weight) # weightQ[i]=sum of weight[j]*Q[i,j] over j for fixed i
    #np.sum(weight[None,:]*Q, axis = 1)
    
    
    this_grad = np.zeros(tuning.shape)
    for k in range(nNeuro):
        dX = tuning[k,:][:, None] - tuning[k,:][None, :] # dX[i,l]=tuning[k,i]-tuning[k, l]
        sumVar = var_diagonal[k,:][:, None] + var_diagonal[k,:][None, :]
        #sumVar[i,l]=var_diagonal[k,i] + var_diagonal[k,l]
        u1 = (var_derivative[k,:][:,None] + 2*dX)/sumVar # u1[i,l]=(var_derivative[k,i]+2*dX[i,l])/sumVar[i,l]
        u2 = (var_derivative[k,:][:,None]*dX**2)/sumVar**2  
        frac = 1.0/weightQ[None, :] + 1.0/weightQ[:, None] # frac[i,l]=1.0/weightQ[l] + 1.0/weightQ[i]
    
        this_grad[k,:] = np.sum(weight[None,:]*frac*(u1-u2)*Q, axis = 1)
        # sum of weight[l]*frac[i,l]*(u1[i,l]-u2[i,l])*Q[i,l] over l
    
    this_grad -= var_derivative/var_diagonal # -var_derivative[k,i]/var_diagonal[k,i]    
    this_grad *= 0.5*weight[None,:] # multiply by 0.5*weight[i] for each k, i   
   
    
    return this_grad

def lower_bound_gd_with_laplacian(
    model, 
    tuning, 
    weight, 
    eta, 
    num_iter, 
    fp, fm, 
    var_diagonal, # if model is Poisson then this is equal to tuning (copied, not changed during iterations)
    laplacian_coeff=0, 
    laplacian_shape=None, # only used in 2d laplacian
    weighted_laplacian=False,
    conv=None, tau=1.0, num_threads=8,
):
    
    '''
    Gradient Ascent for the lower bound of Mutual Information, 
    currently only implemented the Poisson model and Inhomogeneous Gaussian (no correlation) model.
    Assume the points in 'tuning' and 'weight' are arranged in 1d or 2d neighbours. 
    (periodic boundary condition.)
    If 1d neighbour, laplacian_shape is None.
    If 2d neighbour, laplacian_shape: (numNeuro, numBin1, numBin2)
    (tuning, weight can be of numNeuro*(numBin1*numBin2) shape, or (numNeuro, numBin1, numBin2))
    '''
    if model not in ['Poisson', 'GaussianInhomoNoCorr']:
        raise Exception('Model not implemented!')
    elif var_diagonal.shape != tuning.shape:
        raise Exception('Wrong shape for the covariance matrix (diagonal form)!')

    
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
    
    curve_list = []
    curve_grad_list = []
            
    if model=='Poisson':
        this_var_diagonal = tuning.reshape((nNeuro, nBin)).copy()
        this_var_derivative = 1.0*np.ones((nNeuro, nBin))
    else: # var_diagonal already given in the input
        this_var_diagonal = var_diagonal.copy()
        this_var_derivative = np.zeros((nNeuro, nBin))
    
    for i in range(num_iter):
        # do one gradient ascent step
        x_grad = lower_bound_grad(tuning, weight, this_var_diagonal, this_var_derivative)
        x += eta*x_grad
        # apply regularity
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
        # project onto the constraints        
        x[x>fp] = fp
        x[x<fm] = fm
        curve_list.append(x.reshape(tuning_shape).copy())
        curve_grad_list.append(x_grad.reshape(tuning_shape).copy())
        # for poisson model: update the variance matrix after each iteration
        if model=='Poisson':
            this_var_diagonal = x.reshape((nNeuro, nBin)).copy()
        
    return curve_list, curve_grad_list