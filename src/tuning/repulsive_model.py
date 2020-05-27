import numpy as np
import time
import os
from scipy import special

def tuning_update_inhomo(tuning, weight, sigma_vec, alpha = 2, eta = 1, beta = 0, Lambda = 1,
                         elastic_term = 'sum',# other options: 'exp', 'expweight', 'rand'
                         elastic_term_periodic = True, # use closed curve
                         upper_bound = 1, lower_bound = 0):
    '''Gaussian Model, no periodic boundary condition
    '''
    # tuning.shape = (numNeuro, numBin)
    # weight: vector, sum =1, length = numBin
    # sigma_vec: vector, gaussian std, length = numBin
    # Lambda: number, std for the laplacian term
    # eta: number, coefficient of the repulsive term
    # alpha: number, exponent of \|x_k - x_l\| in the probability distribution
    # beta: number, coefficient of the laplacian term
    nNeuro, nBin = tuning.shape
    dS = np.zeros((nBin, nBin))
    for i in range(nNeuro):
        dX = tuning[i,:][None, :] - tuning[i,:][:, None] # dX[k, l]=tuning[i, l]-tuning[i, k]
        dS += dX**2
    dS = np.sqrt(dS)
    # dS[k,l]: euclidean distance between the k-th point and the l-th point    

    
    # negative derivatives
    u = np.zeros((nNeuro, nBin))
    F = (dS**(alpha-2))*np.exp(-dS**alpha/(2*sigma_vec[:,None]**alpha))/sigma_vec[:,None]**(alpha+1)
    F2 = (dS**(alpha-2))*np.exp(-dS**alpha/(2*sigma_vec[None,:]**alpha))/sigma_vec[None, :]**(alpha+1)
    for i in range(nNeuro):
        xi = tuning[i,:]
        dX = xi[None, :] - xi[:, None]
        u[i,:] = np.sum(weight[:,None]*dX*F,axis = 0)
        
        # inside the sum:
        # [k,l] = weight[k]*dX[k,l]*(dS[k,l]**(alpha-2))
        # *np.exp(-dS[k,l]**alpha/(2*sigma_vec[k]**alpha))/sigma_vec[k]**(alpha+1)
        # = weight[k]*(x[l]_i-x[k]_i)*(dS[k,l]**(alpha-2))
        # *np.exp(-dS[k,l]**alpha/(2*sigma_vec[k]**alpha))/sigma_vec[k]**(alpha+1)
        # dS[k,l] = dS[l,k]
        # sum over k for fixed l
        u[i,:] += np.sum(weight[:,None]*dX*F2, axis = 0)
          
        # inside the sum:
        # [k,l] = weight[k]*dX[k,l]*(dS[k,l]**(alpha-2))
        # *np.exp(-dS[k,l]**alpha/(2*sigma_vec[l]**alpha))/sigma_vec[l]**(alpha+1)
        # = weight[k]*(x1[l]_i-x1[k]_i)*(dS[k,l]**(alpha-2))
        # *np.exp(-dS[k,l]**alpha/(2*sigma_vec[l]**alpha))/sigma_vec[l]**(alpha+1)
        # sum over k for fixed l        
    u = weight*alpha*0.5*u # u[i,k] = weight[k]*u[i,k]*alpha*0.5
 
    # positive derivatives of the elastic term
    dl = np.zeros((nNeuro, nBin))
    
    if beta != 0:
        
        diffx = np.zeros((nNeuro, nBin))        
        diffx[:, 0:-1] = np.diff(tuning, axis = 1) # x[k+1]_i - x[k]_i
        if elastic_term_periodic:
            diffx[:, -1] = tuning[:, 0] - tuning[:, -1]
            
        revdiffx =-np.roll(diffx, 1, axis = 1) # x[k-1]_i - x[k]_i
        
        diffS = np.sqrt(np.sum(diffx**2, axis = 0)) # sum diffx[i,k]**2 from i=1 to nNeuro
        revdiffS = np.roll(diffS, 1)
        
        if elastic_term == 'sum':
            diffpow = diffS**(alpha-2)
            revdiffpow = revdiffS**(alpha-2)
            dl = diffx*diffpow + revdiffx*revdiffpow
            dl *= 0.5*alpha
        elif elastic_term == 'exp':
            expdiffpow = (diffS**(alpha-2))*np.exp(-diffS**alpha/(2*Lambda**alpha))
            exprevdiffpow = (revdiffS**(alpha-2))*np.exp(-revdiffS**alpha/(2*Lambda**alpha))
            dl = diffx*expdiffpow + revdiffx*exprevdiffpow
            dl *= 0.5*alpha/Lambda**3
        elif elastic_term == 'expweight':
            expdiffpow = (diffS**(alpha-2))*np.exp(-diffS**alpha/(2*Lambda**alpha))
            exprevdiffpow = (revdiffS**(alpha-2))*np.exp(-revdiffS**alpha/(2*Lambda**alpha))
            dl = weight*np.roll(weight,-1)*diffx*expdiffpow + \
            weight*np.roll(weight,1)*revdiffx*exprevdiffpow
            dl *= 0.5*alpha/Lambda**3
        elif elastic_term == 'rand':
            dl = np.random.randn((nNeuro, nBin))
           
        if elastic_term in ['sum', 'exp', 'expweight'] and (not elastic_term_periodic):
            dl[:,0] = 2*dl[:,0]
            dl[:, -1] = 2*dl[:, -1]

    tuningnew = tuning + eta*u + beta*dl

    tuningnew[tuningnew > upper_bound] = upper_bound
    tuningnew[tuningnew < lower_bound] = lower_bound
   
    
    return tuningnew

def conditional_probability_matrix_gaussian(tuning, sigma_vec, alpha = 2):
    nNeuro, nBin = tuning.shape
    dS = np.zeros((nBin, nBin))
    for i in range(tuning.shape[0]):
        dX = tuning[i,:][None, :] - tuning[i,:][:, None] # dX[k, l]=tuning[i, l]-tuning[i, k]
        dS += dX**2
    dS = np.sqrt(dS)
    D = np.exp(-dS**alpha/(2*sigma_vec[:,None]**alpha))/sigma_vec[:,None]
    return D

def evaluate_elastic_term(tuning, weight, Lambda, alpha = 2, beta = 0, elastic_term = 'sum', elastic_term_periodic = True):
  
    if beta == 0:
        return 0
    
    nNeuro, nBin = tuning.shape
    diffx = np.zeros((nNeuro, nBin))        
    diffx[:, 0:-1] = np.diff(tuning, axis = 1) # x[k+1]_i - x[k]_i
    if elastic_term_periodic:
        diffx[:, -1] = tuning[:, 0] - tuning[:, -1]
    diffS = np.sqrt(np.sum(diffx**2, axis = 0)) # sum diffx[i,k]**2 from i=1 to nNeuro

    elastic_value = 0
    if elastic_term == 'sum':
        elastic_value = -0.5*np.sum(diffS**alpha)
        
    elif elastic_term == 'exp':
        elastic_value = np.sum(np.exp(-diffS**alpha/(2*Lambda**alpha)))/Lambda
        
    elif elastic_term == 'expweight':
        elastic_value = np.sum(weight*np.roll(weight,-1)*np.exp(-diffS**alpha/(2*Lambda**alpha)))/Lambda
        # weight[i]*weight[i+1]
    return elastic_value*beta
    
def tuning_update_inhomo_periodic(tuning, weight, kappa_mat, alpha = 2, eta = 1, beta = 0, Lambda = 1,
                                  elastic_term = 'exp', # other options: 'expweight', 'rand'
                                  elastic_term_periodic = True, # use closed curve
                                 ):
    '''Von-Mises Model, periodic b.c. in [-1, 1]^k
    '''
    # tuning.shape = (numNeuro, numBin)
    # weight: vector, sum =1, length = numBin
    # kappa_mat: matrix, kappa_mat[:,l] = [kappa_{l,1},kappa_{l,2}] # shape = (2, numBin)
    # Lambda: number, 'kappa' parameter for the elastic term 
    # (theoretically we can also modify it to be a vector of size 2, not implemented yet)
    # when elastic_term takes 'rand', Lambda serves as 'kappa' parameter of noise.
    # alpha: number, exponent of 1-cos(x_k - x_l) term in the probability distribution
    # eta: number, coefficient of the repulsive term
    # beta: number, coefficient of the laplacian term

    nNeuro, nBin = tuning.shape
    P = np.zeros((nBin, nBin))
    for i in range(nNeuro):
        dX = tuning[i,:][None, :] - tuning[i,:][:, None] # dX[k, l]=tuning[i, l]-tuning[i, k]
        P += kappa_mat[i,:]*(1-np.cos(np.pi*dX))**(0.5*alpha)
    P = np.exp(-P)
    # inside exp:
    #[k,l] = kappa_mat[0,l]*(1-np.cos(np.pi*(x1[l] - x1[k])))**(alpha/2) 
    # + kappa_mat[1,l]*(1-np.cos(np.pi*(x2[l] - x2[k])))**(alpha/2)
    
    constants = np.exp(kappa_mat)/ special.iv(0, kappa_mat)/2
    P *= np.prod(constants, axis = 0)
    # at[k,l] position multiply by constants[0, l]*constants[1,l]
        
    # negative derivatives
    u = np.zeros((nNeuro, nBin))
    for i in range(nNeuro):
        xi = tuning[i,:]
        dX = xi[None, :] - xi[:, None]
        vi = (1-np.cos(np.pi*dX))**(0.5*alpha-1)
        u[i,:] = np.sum(weight[:,None]*vi*kappa_mat[i,:]*np.sin(np.pi*dX)*P, axis = 0)
        # inside the sum: 
        # [k,l] = weight[k]*(1-np.cos(np.pi*(x[l]_i-x[k]_i)))**(alpha/2-1)\
        # *kappa_mat[i,l]*np.sin(np.pi*(x[l]_i - x[k]_i) )*P[k,l]
        # sum over k for fixed l
        u[i,:] += np.sum(weight[:,None]*vi*kappa_mat[i,:][:,None]*np.sin(np.pi*dX)*P.T, axis = 0)
        # inside the sum: 
        #  [k,l] = weight[k]*(1-np.cos(np.pi*(x[l]_i-x[k]_i)))**(alpha/2-1)\
        # *kappa_mat[i,k]*np.sin(np.pi*(x[l]_i-x[k]_i))*P[l,k]
        # sum over k for fixed l
        
    u = np.pi*0.5*alpha*weight*u # u[i,k] = constant*weight[k]*u[i,k]
    
      
    # positive derivatives of the elastic term   
    dl = np.zeros((nNeuro, nBin))
    
    if beta != 0:
        diffx = np.zeros((nNeuro, nBin))        
        diffx[:, 0:-1] = np.diff(tuning, axis = 1) # x[k+1]_i - x[k]_i
        if elastic_term_periodic:
            diffx[:, -1] = tuning[:, 0] - tuning[:, -1]
        revdiffx =-np.roll(diffx, 1, axis = 1) # x[k-1]_i - x[k]_i
        
        diffP = np.sum(Lambda*(1-np.cos(np.pi*diffx))**(0.5*alpha), axis = 0) 
        # diffP[k] = sum Lambda*(1-cos(pi*diffx[i,k]))**(alpha/2) from i=0 to nNeuro-1
        diffP = np.exp(-diffP)
        revdiffP = np.roll(diffP, 1) # np.exp(-revdiffP)
        
                     
        Lambda_constant = (np.exp(Lambda)/ special.iv(0, Lambda)/2)**nNeuro #take power of nNeuro

        if elastic_term == 'exp':
            dl = (1-np.cos(np.pi*diffx))**(0.5*alpha-1)*np.sin(np.pi*diffx)*diffP
            dl += (1-np.cos(np.pi*revdiffx))**(0.5*alpha-1)*np.sin(np.pi*revdiffx)*revdiffP
            dl *= 0.5*alpha*np.pi*Lambda*Lambda_constant
        elif elastic_term == 'expweight':
            dl = weight*np.roll(weight,-1)*\
                (1-np.cos(np.pi*diffx))**(0.5*alpha-1)*np.sin(np.pi*diffx)*diffP
            dl += weight*np.roll(weight,1)*\
                (1-np.cos(np.pi*revdiffx))**(0.5*alpha-1)*np.sin(np.pi*revdiffx)*revdiffP
            dl *= 0.5*alpha*np.pi*Lambda*Lambda_constant
        elif elastic_term == 'rand':
            dl = np.random.vonmises(0, Lambda, size = (nNeuro, nBin))/np.pi # map to [-1, 1]
            
        if elastic_term in ['exp', 'expweight'] and (not elastic_term_periodic):
            dl[:, 0] = 2*dl[:, 0]
            dl[:, -1] = 2*dl[:, -1]

    
    # upper_bound, lower_bound = (-1, 1)
    tuningnew = tuning + eta*u + beta*dl
   
    tuningnew = tuningnew - 2*np.ceil((tuningnew-1)/2.0) # map to [-1, 1]^nNeuro

    return tuningnew

def conditional_probability_matrix_vonmises(tuning, kappa_mat, alpha = 2):
   
    nNeuro, nBin = tuning.shape
    P = np.zeros((nBin, nBin))
    for i in range(nNeuro):
        dX = tuning[i,:][None, :] - tuning[i,:][:, None] # dX[k, l]=tuning[i, l]-tuning[i, k]
        P += kappa_mat[i,:]*(1-np.cos(np.pi*dX))**(0.5*alpha)     
    P = np.exp(-P)
        
    constants = np.exp(kappa_mat)/ special.iv(0, kappa_mat)/2
    P *= np.prod(constants, axis = 0) # at[k,l] position multiply by constants[0, l]*constants[1,l]*...
    return P

def evaluate_elastic_term_periodic(tuning, weight, alpha = 2, beta = 0, elastic_term = 'exp', Lambda=1,
                                   elastic_term_periodic = True):
    '''Evaluate the elastic term in the Von Mises Model (including beta)'''
    
    nNeuro, nBin = tuning.shape
    if beta == 0:
        return 0
    
    diffx = np.zeros((nNeuro, nBin))        
    diffx[:, 0:-1] = np.diff(tuning, axis = 1) # x[k+1]_i - x[k]_i
    if elastic_term_periodic:
        diffx[:, -1] = tuning[:, 0] - tuning[:, -1]
        
    diffP = np.sum(Lambda*(1-np.cos(np.pi*diffx))**(0.5*alpha), axis = 0) 
    # diffP[k] = sum Lambda*(1-cos(pi*diffx[i,k]))**(alpha/2) from i=0 to nNeuro-1
    diffP = np.exp(-diffP)
    
    Lambda_constant = (np.exp(Lambda)/ special.iv(0, Lambda)/2)**nNeuro #take power of nNeuro

    if elastic_term == 'exp':
        elastic_value = np.sum(diffP)*Lambda_constant
        
    elif elastic_term == 'expweight':
        elastic_value = np.sum(weight*np.roll(weight,-1)*diffP)*Lambda_constant
    
    return elastic_value*beta


def tuning_update_poisson(tuning, weight, alpha = 2, eta = 1, beta = 0, Lambda = 1,
                          elastic_term = 'sum',# other options: 'exp', 'expweight', 'rand'
                          elastic_term_periodic = True, # use closed curve
                          upper_bound = 1, lower_bound = 0.01):
    '''Pseudo-Poisson Model (gaussian with variance=function value), no periodic boundary condition
    '''
    # tuning.shape = (numNeuro, numBin)
    # weight: vector, sum =1, length = numBin
    # Lambda: number, std for the laplacian term
    # eta: number, coefficient of the repulsive term
    # alpha: number, exponent of \|x_k - x_l\| in the probability distribution
    # beta: number, coefficient of the laplacian term
    nNeuro, nBin = tuning.shape
    dS = np.zeros((nBin, nBin))
    for i in range(nNeuro):
        dX = tuning[i,:][None, :] - tuning[i,:][:, None] # dX[k, l]=tuning[i, l]-tuning[i, k]
        dS += dX**2/tuning[i,:][None, :] # (tuning[i, l]-tuning[i, k])^2/tuning[i, l]
    dS = np.sqrt(dS)
    
    P = np.exp(-0.5*dS**alpha)
    P_constants = np.prod(tuning, axis = 0)**(0.5*alpha) # product of tuning[i,l] over all i
    P /= P_constants[None, :] # P[k,l] = P[k,l]/constants[l]    
    
    # negative derivatives
    u = np.zeros((nNeuro, nBin))
    F = dS**(alpha-2)*P
    F2 = (dS.T)**(alpha-2)*P.T

    for i in range(nNeuro):
        xi = tuning[i,:]
        dX = xi[None, :] - xi[:, None] # dX[k, l]=tuning[i, l]-tuning[i, k]
        temp = dX*F/xi[None,:] + dX*F2/xi[:,None] #dX[k,l]*F[k,l]/tuning[i,l]+dX[k,l]*F2[k,l]/tuning[i,k]
        temp += P/xi[None,:] # P[k,l]/tuning[i,l]
        temp -= 0.5*F*(dX**2)/(xi[None,:]**2) # 0.5*F[k,l]*dX[k,l]**2/(tuning[i,l]**2)
        
        u[i,:] = np.sum(weight[:,None]*temp, axis = 0)
        
        # inside the sum:
        # [k,l] = weight[k]*temp[k,l]
        # sum over k for fixed l
        
    u = weight*alpha*0.5*u # u[i,k] = weight[k]*u[i,k]*alpha*0.5
 
    # positive derivatives of the elastic term
    dl = np.zeros((nNeuro, nBin))
    
    if beta != 0:
        
        diffx = np.zeros((nNeuro, nBin))        
        diffx[:, 0:-1] = np.diff(tuning, axis = 1) # x[k+1]_i - x[k]_i
        if elastic_term_periodic:
            diffx[:, -1] = tuning[:, 0] - tuning[:, -1]
            
        revdiffx =-np.roll(diffx, 1, axis = 1) # x[k-1]_i - x[k]_i
        
        diffS = np.sqrt(np.sum(diffx**2, axis = 0)) # sum diffx[i,k]**2 from i=1 to nNeuro
        revdiffS = np.roll(diffS, 1)
        
        if elastic_term == 'sum':
            diffpow = diffS**(alpha-2)
            revdiffpow = revdiffS**(alpha-2)
            dl = diffx*diffpow + revdiffx*revdiffpow
            dl *= 0.5*alpha
        elif elastic_term == 'exp':
            expdiffpow = (diffS**(alpha-2))*np.exp(-diffS**alpha/(2*Lambda**alpha))
            exprevdiffpow = (revdiffS**(alpha-2))*np.exp(-revdiffS**alpha/(2*Lambda**alpha))
            dl = diffx*expdiffpow + revdiffx*exprevdiffpow
            dl *= 0.5*alpha/Lambda**3
        elif elastic_term == 'expweight':
            expdiffpow = (diffS**(alpha-2))*np.exp(-diffS**alpha/(2*Lambda**alpha))
            exprevdiffpow = (revdiffS**(alpha-2))*np.exp(-revdiffS**alpha/(2*Lambda**alpha))
            dl = weight*np.roll(weight,-1)*diffx*expdiffpow + \
            weight*np.roll(weight,1)*revdiffx*exprevdiffpow
            dl *= 0.5*alpha/Lambda**3
        elif elastic_term == 'rand':
            dl = np.random.randn((nNeuro, nBin))
           
        if elastic_term in ['sum', 'exp', 'expweight'] and (not elastic_term_periodic):
            dl[:,0] = 2*dl[:,0]
            dl[:, -1] = 2*dl[:, -1]

    tuningnew = tuning + eta*u + beta*dl

    tuningnew[tuningnew > upper_bound] = upper_bound
    tuningnew[tuningnew < lower_bound] = lower_bound   
    
    return tuningnew

def conditional_probability_matrix_poisson(tuning, alpha = 2):
    nNeuro, nBin = tuning.shape
    dS = np.zeros((nBin, nBin))
    for i in range(nNeuro):
        dX = tuning[i,:][None, :] - tuning[i,:][:, None] # dX[k, l]=tuning[i, l]-tuning[i, k]
        dS += dX**2/tuning[i,:][None, :] # (tuning[i, l]-tuning[i, k])^2/tuning[i, l]
    dS = np.sqrt(dS)
    
    P = np.exp(-0.5*dS**alpha)
    P_constants = np.prod(tuning, axis = 0)**(0.5*alpha) # product of tuning[i,l] over all i
    P /= P_constants[None, :] # P[k,l] = P[k,l]/constants[l]    
    return P


def mutual_distance(tuning):
    nNeuro, nBin = tuning.shape
    dS = np.zeros((nBin, nBin))
    for i in range(nNeuro):
        dX = tuning[i,:][None, :] - tuning[i,:][:, None] # dX[k, l]=tuning[i, l]-tuning[i, k]
        dS += dX**2
    dS = np.sqrt(dS)
    return dS