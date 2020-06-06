STUFF = "Hi"
##cython --annotate --cplus --link-args=-fopenmp --compile-args=-fopenmp --compile-args=-std=c++0x
#!python
#cython: boundscheck=False, wraparound=False, nonecheck=False, language_level=3

import numpy as np
from cython import floating
from libc.math cimport log, exp
import cython
from cython.parallel import parallel, prange, threadid

from itertools import product, combinations,permutations

#---------------------------------------
#---------- Poisson Model --------------
#---------------------------------------

# ----------Compute info and grad by Prtial Sum --------

cdef double log_factorial(int x) nogil:
    cdef double m = 0
    cdef int i
    if x <= 1:
        return 0
    else:
        for i in range(1, x+1):
            m += log(i)
        return m
    
#@cython.boundscheck(False)
#@cython.cdivision(True)
cdef double compute_prod_grad_noncyclic(int numBin, int numNeuro, double[:,:] rate, double[:] weight,
                                        int[:] count,\
                                        double[:,:] grad, double[:,:] texp, double[:] prod_p,
                                        double[:] log_prod_s) nogil:
    '''Given r = (r_1,...,r_P), compute I(r) = -sum_j[w_j*P_j(r)S_j(r)],
    and update gradI by adding the term corresponding to r.
    '''

    # rate: numNeuro*numBin
    # weight: numBin (sum up to 1)
    # count: numNeuro (same as r=(r_1, ..., r_P))
    
    
    # grad: numNeuro*numBin
    # texp: numNeuro*numBin (only temporary storage inside this function)
    # prod_p: numBin (only temporary storage inside this function)
    # prod_s: numBin (only temporary storage inside this function)
   

    cdef int i,j,k,l,p
    cdef double mymax = 0
    cdef int INT_MAX = <int>((<unsigned int>-1)>>1)
    cdef int INT_MIN = (-INT_MAX-1)
    # exp(INT_MIN) = 0

    for p in range(numNeuro):
        for j in range(numBin):
            if count[p] > 0:
                texp[p, j] = count[p]*log(rate[p,j]) - rate[p,j] if rate[p,j] else INT_MIN
            else:
                texp[p, j] = -rate[p,j]
            
    cdef double info = 0
    cdef double tmp_sum_l = 0
    
    for j in range(numBin):
        prod_p[j] = 0
        log_prod_s[j] = 0
        
        for p in range(numNeuro):
            prod_p[j] += texp[p,j] - log_factorial(count[p])
        prod_p[j] = exp(prod_p[j])   

        mymax = 0 # specific to j
        for l in range(numBin):
            tmp_sum_l = 0
            for k in range(numNeuro):
                tmp_sum_l += texp[k,l] - texp[k,j]
            if tmp_sum_l > mymax:
                mymax = tmp_sum_l

        for l in range(numBin):
            tmp_sum_l = 0
            for k in range(numNeuro):               
                tmp_sum_l += texp[k,l] - texp[k,j]
            log_prod_s[j] += weight[l]*exp(tmp_sum_l - mymax)
        log_prod_s[j] = log(log_prod_s[j]) + mymax
            
        info += weight[j]*prod_p[j]*log_prod_s[j] if prod_p[j] else 0
        
    info *= (-1)
    
    # update grad
    cdef double tmp_grad_kl = 0
    for k in range(numNeuro):
        for l in range(numBin):
            tmp_grad_kl = weight[l]*prod_p[l]*log_prod_s[l] if prod_p[l] else 0
            # the second term is zero
            grad[k,l] += (1 - count[k]/rate[k,l])*tmp_grad_kl if rate[k,l] else tmp_grad_kl
         
    return info


def partial_sum_mean_grad_noncyclic(double[:,:] MI_grad, double[:,:] tuning, double[:] weight,\
                     double[:] conv, double tau, int threshold = 50, int my_num_threads = 4):
    
    cdef int numNeuro = tuning.shape[0] # or cdef int numNeuro
    cdef int numBin = tuning.shape[1] # or cdef int numBin
    cdef int i,j,k,l,p
    
    cdef double[:,:] rate = np.zeros((numNeuro,numBin), dtype = np.float)
    for p in range(numNeuro):
        for j in range(numBin):
            for l in range(numBin):
                rate[p,j] += tuning[p, (numBin+j-l) % numBin]*conv[l]
            rate[p,j] *= tau
            

    cdef double info = 0
    cdef double this_info = 0
    
    cdef double[:,:] grad = np.zeros((numNeuro,numBin), dtype = np.float)
    cdef double[:,:,:] this_grad_all = np.zeros((my_num_threads,numNeuro,numBin), dtype = np.float)
    
    cdef double[:,:,:] texp_all = np.zeros((my_num_threads,numNeuro,numBin), dtype = np.float)
    cdef double[:,:] prod_p_all = np.zeros((my_num_threads,numBin), dtype = np.float)
    cdef double[:,:] prod_s_all = np.zeros((my_num_threads, numBin), dtype = np.float)
    
    count_arr_np = np.array(list(product(np.arange(0,threshold), repeat = numNeuro))).astype(np.intc)  # cartesian product
    cdef int[:,:] count_arr = count_arr_np
    cdef int num_comb
    num_comb = count_arr.shape[0]
    
    
    cdef int tid, idx
    
    with nogil, parallel(num_threads = my_num_threads):
        tid = threadid()
        for idx in prange(num_comb):
            #count = count_arr[idx,:]
            # grad is also updated 
            this_info = compute_prod_grad_noncyclic(numBin, numNeuro, rate, weight, count_arr[idx,:], \
                                                    this_grad_all[tid, :,:], \
                                                    texp_all[tid,:,:],prod_p_all[tid,:], prod_s_all[tid,:])
            info += this_info

    
    
    for k in range(numNeuro):
        for l in range(numBin):
            for tid in range(my_num_threads):
                grad[k,l] += this_grad_all[tid, k,l]
                        
    # compute tuning gradient
    for p in range(numNeuro):
        for l in range(numBin):
            MI_grad[p,l] = 0
            for j in range(numBin):
                MI_grad[p,l] += grad[p,j]*conv[(numBin+j-l) % numBin] 
            MI_grad[p,l] *= tau
            
    return info

# ----------Compute info and grad by Monte Carlo--------

cdef double compute_mean_grad_s_noncyclic(
    int s, int numBin, int numNeuro, double[:,:] rate, double[:] weight,int[:] count,
    double[:,:] lrate, double[:] mexp, double[:] dexp) nogil: 
    # for a fixed s in range(numBin)(s is same as m in the notes)
    # rate: numNeuro*numBin
    # weight: numBin (sum up to 1)
    # count: numNeuro (specific to s, Poisson distribution conditioning on s)
    
    # mexp: numBin,  (specific to s)
    # lrate: numNeuro*numBin,  (specific to s)
    # dexp: numBin (specific to s)
   

    cdef int i,j,k,l, p #, indj, indk
    cdef double mymax, mean_s, tmp_sum_s      

    for i in range(numBin):
        dexp[i] = 0
    #                     indj = (numBin+j-center[k]) % numBin
    #             indk = (numBin+s-center[k]) % numBin
        for p in range(numNeuro):
            lrate[p,i] = log(rate[p,i]/rate[p,s]) if rate[p,s] else 0
            dexp[i] += rate[p,s]-rate[p,i]        

    mymax = 1.0
    for i in range(numBin):
        mexp[i] = 0
        for p in range(numNeuro):
            mexp[i] += count[p]*lrate[p,i] if count[p] else 0
        mexp[i] += dexp[i]
        if mexp[i] > mymax:
            mymax = mexp[i] 
            
    for i in range(numBin):
        mexp[i] = exp(mexp[i] - mymax)
    

    mean_s = 0
    #     for i in range(numBin):      
    #         mean_s += weight[i]*mexp[i] #exp(mexp[i] - mymax)
    #     mean_s = mymax + log(mean_s)
    
    tmp_sum_s = 0
    for l in range(numBin):
        tmp_sum_s += weight[l]*mexp[l]
    mean_s = mymax + log(tmp_sum_s)
    
    return mean_s

#@cython.boundscheck(False)
#@cython.cdivision(True)
cdef void update_grad_s_noncyclic(int s, double[:,:] grad, int numBin, int numNeuro, double[:,:] rate,\
                                  double[:] weight, int[:] count, double mean_s) nogil:
    # s is now i in the notes..
    # count is sampled conditioning on s
    # derivative of the probability density term
    cdef int p
    for p in range(numNeuro):
        
        grad[p, s] += weight[s]*(count[p]/rate[p,s] - 1)*mean_s if rate[p,s] else (-mean_s)

            

# compute I and grad(I).
def mc_mean_grad_noncyclic(double[:,:] MI_grad, double[:,:] tuning, double[:] weight,\
                           double[:] conv, double tau, int numIter, int my_num_threads = 4):
    # conv is old stim...
    # tuing: numNeuro*numBin
    # MI_grad: numNeuro*numBin
    # weight: numBin, sum up to one
    # conv: numBin, weighted sum is one.
    # 
    cdef int numNeuro = tuning.shape[0] # or cdef int numNeuro
    cdef int numBin = tuning.shape[1] # or cdef int numBin
   
    cdef double[:,:] rate = np.zeros((numNeuro,numBin), dtype = np.float)
    cdef Py_ssize_t i,j,l,k,m,p,s
    cdef Py_ssize_t n_iter

    cdef double[:,:] grad = np.zeros((numNeuro,numBin), dtype = np.float)
    cdef double[:,:,:] grad_all = np.zeros((my_num_threads, numNeuro, numBin), dtype = np.float) 
    
    cdef double[:,:] dexp_all = np.zeros((my_num_threads, numBin), dtype = np.float)
    cdef double[:,:] mexp_all = np.zeros((my_num_threads, numBin), dtype = np.float)   
    cdef double[:,:,:] lrate_all = np.zeros((my_num_threads, numNeuro,numBin),dtype = np.float)
    
    cdef double mean = 0 
    cdef double this_mean_s = 0  
    
    for p in range(numNeuro):
        for j in range(numBin):
            for l in range(numBin):
                rate[p,j] += tuning[p, (numBin+j-l) % numBin]*conv[l]
            rate[p,j] *= tau
            
    # generate Poission samples for all s = 1,...,numBin(different s, different samples)
    #     np.random.seed(305)
    
    poisson_samples_np = np.zeros((numIter, numNeuro,numBin), dtype = np.intc)
    for p in range(numNeuro):
        for s in range(numBin):        
             poisson_samples_np[:, p, s] = np.random.poisson(lam = rate[p,s], size = numIter)
                
    cdef int[:,:,:] poisson_samples = poisson_samples_np
        
    cdef int tid

    with nogil, parallel(num_threads = my_num_threads):
        tid = threadid()
        for n_iter in prange(numIter):
            for s in range(numBin):
                # samples conditioning on s: poisson_samples[n_iter,:,s]
                # tmp_grad_all[tid, s,:,:] is updated
                this_mean_s = compute_mean_grad_s_noncyclic(
                    s, numBin, numNeuro, rate, weight, poisson_samples[n_iter,:,s],\
                    lrate_all[tid,:,:], mexp_all[tid,:], dexp_all[tid,:])
                
                # compute negative mean entropy
                mean += weight[s]*this_mean_s

                update_grad_s_noncyclic(s, grad_all[tid,:,:], numBin, numNeuro, rate, weight,\
                              poisson_samples[n_iter,:,s], this_mean_s)

    mean /= numIter
    mean *= (-1) # mutual information
    
    # compute the rate gradient
    
    for p in range(numNeuro):
        for s in range(numBin):
            # the 2nd term is zero
            # the first term
            for k in range(my_num_threads):
                grad[p,s] += grad_all[k,p,s] 
            # devide by numIter
            grad[p,s] /= numIter
            grad[p,s] *= (-1)

                        
    # compute tuning gradient
    for p in range(numNeuro):
        for l in range(numBin):
            MI_grad[p,l] = 0
            for j in range(numBin):
                MI_grad[p,l] += grad[p,j]*conv[(numBin+j-l) % numBin] 
            MI_grad[p,l] *= tau

    return mean#,mean_list_np,grad_all_np,poisson_samples_np,lrate_all_np, mexp_all_np,dexp_all_np


# ----------Blahut-Arimoto Algorithm by partial sum--------
#@cython.boundscheck(False)
#@cython.cdivision(True)
cdef double count_coeff_s_arimoto(int s, int numBin, int numNeuro, double[:,:] rate,
                                   double[:] weight,\
                                   int[:] count, double[:] rexp) nogil:
    '''
    For a fixed r=(r_1, ..., r_P) and fixed s = 1,...,numBin, 
    compute P_s(r) * log( 1 / S_s(r) )
    (part of c_j in the Blahut paper)
    '''
    # s is same as j = 1,...,numBin in the notes)
    # rate: numNeuro*numBin
    # weight: numBin (sum up to 1)
    # count: numNeuro (same as r=(r_1, ..., r_P))  
    
    # rexp: numBin  (specific to s) 
    # rexp[i] = sum_p [r_p*log(f_{r,i}) - f_{r,i} ]

    cdef int i, p
    cdef double prod_p_s, qexp_s, mymax
    cdef int INT_MAX = <int>((<unsigned int>-1)>>1)
    cdef int INT_MIN = (-INT_MAX-1)
    # exp(INT_MIN) = 0
    
    #     mymax = 1.0
    for i in range(numBin):
        rexp[i] = 0
        for p in range(numNeuro):
            if count[p] > 0:
                rexp[i] += count[p]*log(rate[p,i]) - rate[p,i] if rate[p,i] else INT_MIN
            else:
                rexp[i] += -rate[p,i]
    #         if rexp[i] > mymax:
    #             mymax = rexp[i]    
    
    prod_p_s = rexp[s]
    for p in range(numNeuro):
        prod_p_s -= log_factorial(count[p])
    prod_p_s = exp(prod_p_s) 
    
    qexp_s = 0
    for i in range(numBin):
        qexp_s += weight[i]*exp(rexp[i]) #weight[i]*exp(rexp[i] - mymax)
    # old version for probability: qexp_s = weight[s]*exp(rexp[s])/qexp_s
    qexp_s = exp(rexp[s])/qexp_s if qexp_s else 0
   
    return prod_p_s * log(qexp_s) if qexp_s else 0
    

def partial_sum_coeff_arimoto(double[:] coeff, double[:,:] tuning, double[:] weight,
                              double[:] slope,
                              double[:] conv, double tau,
                              int threshold = 50, int my_num_threads = 4):
    '''Compute arimoto update for one iteration.'''
    # conv is old stim...
    # weight_new: computed new weights(probability)
    # tuing: numNeuro*numBin
    # weight: numBin, sum up to one
    # slope: numNeuro, positive (same as 's' in Blahut paper)
    # conv: numBin, weighted sum is one.
    
    cdef int numNeuro = tuning.shape[0]
    cdef int numBin = tuning.shape[1]
   
    cdef double[:,:] rate = np.zeros((numNeuro,numBin), dtype = np.float)
    cdef Py_ssize_t i,j,l,k,m,p,s
    
    cdef double[:,:] coeff_all = np.zeros((my_num_threads, numBin), dtype = np.float)
    cdef double[:,:] rexp_all = np.zeros((my_num_threads, numBin), dtype = np.float)
    
    cdef double this_count_s = 0
    
    for p in range(numNeuro):
        for j in range(numBin):
            for l in range(numBin):
                rate[p,j] += tuning[p, (numBin+j-l) % numBin]*conv[l]
            rate[p,j] *= tau
            
    
    
    count_arr_np = np.array(list(product(np.arange(0,threshold), repeat = numNeuro))).astype(np.intc)  # cartesian product
    cdef int[:,:] count_arr = count_arr_np
    cdef int num_comb
    num_comb = count_arr.shape[0]
    
    cdef int tid, idx
    with nogil, parallel(num_threads = my_num_threads):
        tid = threadid()
        for idx in prange(num_comb):
            #count = count_arr[idx,:]
            for s in range(numBin):
                this_count_s = count_coeff_s_arimoto(s,numBin,numNeuro, rate, weight,\
                                                     count_arr[idx,:], rexp_all[tid, :])
                
                coeff_all[tid, s] += this_count_s
                
    for m in range(numBin): 
        coeff[m] = 0
        for tid in range(my_num_threads):
            coeff[m] += coeff_all[tid, m]
        coeff[m] = exp(coeff[m])
        for p in range(numNeuro):
            coeff[m] *= exp(-slope[p]*tuning[p,m])
    return

# ----------Blahut-Arimoto Algorithm by Monte Carlo--------

#@cython.boundscheck(False)
#@cython.cdivision(True)
cdef double compute_coeff_s_arimoto(int s, int numBin, int numNeuro, double[:,:] rate, double[:] weight,\
                                   int[:] count, double[:] rexp) nogil: 
    # for a fixed s in range(numBin)(s is same as m = 1,...,M in the notes)
    # rate: numNeuro*numBin
    # weight: numBin (sum up to 1)
    # count: numNeuro (specific to s, Poisson distribution conditioning on s)    
    
    # rexp: numBin  (specific to s)  

    cdef int i, p #, indj, indk
    cdef double mymax, qexp_s, mean_s
    
    #     mymax = 1.0
    for i in range(numBin):
        rexp[i] = 0
        for p in range(numNeuro):
            rexp[i] += count[p]*log(rate[p,i]) - rate[p,i] if rate[p,i] else 0           
    #         if rexp[i] > mymax:
    #             mymax = rexp[i]    
    qexp_s = 0
    for i in range(numBin):
        qexp_s += weight[i]*exp(rexp[i])
        
    # old version for probability: qexp_s = weight[s]*exp(rexp[s])/qexp_s
    qexp_s = exp(rexp[s])/qexp_s if qexp_s else 0
    #     qexp_s = 0
    #     for i in range(numBin):
    #         qexp_s += weight[i]*exp(rexp[i] - mymax)

    #     qexp_s = weight[s]*exp(rexp[s] - mymax)/qexp_s

    mean_s = log(qexp_s) if qexp_s else 0
    
    return mean_s

def mc_coeff_arimoto(double[:] coeff, double[:,:] tuning, double[:] weight, double[:] slope,
                     double[:] conv, double tau, int numIter, int my_num_threads = 4):
    '''Compute arimoto update for one iteration.'''
    # conv is old stim...
    # weight_new: computed new weights(probability)
    # tuing: numNeuro*numBin
    # weight: numBin, sum up to one
    # slope: numNeuro, positive
    # conv: numBin, weighted sum is one.
    # coeff: numBin
    
    cdef int numNeuro = tuning.shape[0]
    cdef int numBin = tuning.shape[1]
   
    cdef double[:,:] rate = np.zeros((numNeuro,numBin), dtype = np.float)
    cdef Py_ssize_t i,j,l,k,m,p,s
    cdef Py_ssize_t n_iter
    
    cdef double[:,:] coeff_all = np.zeros((my_num_threads, numBin), dtype = np.float)
    cdef double[:,:] rexp_all = np.zeros((my_num_threads, numBin), dtype = np.float)
    
    cdef double this_mean_s = 0
    
    for p in range(numNeuro):
        for j in range(numBin):
            for l in range(numBin):
                rate[p,j] += tuning[p, (numBin+j-l) % numBin]*conv[l]
            rate[p,j] *= tau
            
    # generate Poission samples for all s = 1,...,numBin(different s, different samples)
    #     np.random.seed(305)
    
    poisson_samples_np = np.zeros((numIter, numNeuro,numBin), dtype = np.intc)
    for p in range(numNeuro):
        for s in range(numBin):        
             poisson_samples_np[:, p, s] = np.random.poisson(lam = rate[p,s], size = numIter)
                
    cdef int[:,:,:] poisson_samples = poisson_samples_np
        
    cdef int tid

    with nogil, parallel(num_threads = my_num_threads):
        tid = threadid()
        for n_iter in prange(numIter):
            for s in range(numBin):
                # samples conditioning on s: poisson_samples[n_iter,:,s]
                this_mean_s = compute_coeff_s_arimoto(s,numBin,numNeuro, rate, weight,\
                                                     poisson_samples[n_iter,:,s], rexp_all[tid, :])
                
                coeff_all[tid, s] += this_mean_s
                
    for m in range(numBin): 
        coeff[m] = 0
        for tid in range(my_num_threads):
            coeff[m] += coeff_all[tid, m]
        coeff[m] /= numIter
        coeff[m] = exp(coeff[m])
        for p in range(numNeuro):
            coeff[m] *= exp(-slope[p]*tuning[p,m])
    return

#---------------------------------------
#---------- Gaussian Model -------------
#---------------------------------------

# ----------Compute info and grad by Monte Carlo--------
# int[:] count->double[：]　response
# add double[:,:] inv_cov_mat
cdef double compute_mean_grad_s_gaussian(int s, int numBin, int numNeuro, double[:,:] rate, double[:] weight,
                                         double[:,:] inv_cov_mat,
                                         double[:] response, double[:] quad) nogil:
    # for a fixed s in range(numBin)(s is same as m in the notes)
    # rate: numNeuro*numBin
    # weight: numBin (sum up to 1)
    # inv_cov_mat: numNeuro*numNeuro
    # response: numNeuro (specific to s, gaussian distribution conditioning on s)

    # tmp_grad: numBin*numNeuro*numBin, (will reuse in grad, so store for each s)
    # quad: numBin,  (specific to s)
       

    cdef int i,j,k,l, p #, indj, indk
    cdef double mean_s, tmp_sum, diff     #mymax

    for l in range(numBin):
        quad[l] = 0
        for i in range(numNeuro):
            for j in range(numNeuro):
                quad[l] += (response[i] - rate[i, l])*inv_cov_mat[i,j]*(response[j] - rate[j,l])
        
    mean_s = 0
    for l in range(numBin):
        mean_s += weight[l] * exp(-0.5*(quad[l] - quad[s]))
    mean_s = log(mean_s)
        
    return mean_s

#@cython.boundscheck(False)
#@cython.cdivision(True)
cdef void update_grad_s_gaussian(int s, double[:,:] grad, int numBin, int numNeuro, 
                                 double[:,:] rate, double[:] weight, double[:,:] inv_cov_mat, 
                                 double[:] response, double mean_s) nogil:
    # s is now i in the notes..
    # count is sampled conditioning on s
    # derivative of the probability density term
    cdef int p
    cdef double diff_p_s
    for p in range(numNeuro):
        diff_p_s = 0
        for k in range(numNeuro):
            diff_p_s += inv_cov_mat[p,k]*(response[k] - rate[k,s]) 
        
        grad[p, s] += weight[s]*diff_p_s*mean_s


# compute I and grad(I).
def mc_mean_grad_gaussian(double[:,:] MI_grad, double[:,:] tuning, double[:] weight,
                          double[:,:] inv_cov_mat,
                          double[:] conv, double tau, int numIter, int my_num_threads = 4):
    # conv is old stim...
    # tuing: numNeuro*numBin
    # MI_grad: numNeuro*numBin
    # weight: numBin, sum up to one
    # inv_cov_mat: inverse covariance matrix, numNeuro-by-numNeuro
    # conv: numBin, weighted sum is one.
    # 
    cdef int numNeuro = tuning.shape[0] # or cdef int numNeuro
    cdef int numBin = tuning.shape[1] # or cdef int numBin
   
    cdef double[:,:] rate = np.zeros((numNeuro,numBin), dtype = np.float)
    cdef Py_ssize_t i,j,l,k,m,p,s
    cdef Py_ssize_t n_iter

    cdef double[:,:] grad = np.zeros((numNeuro,numBin), dtype = np.float)
    cdef double[:,:,:] grad_all = np.zeros((my_num_threads, numNeuro, numBin), dtype = np.float) 
    
    cdef double[:,:] quad_all = np.zeros((my_num_threads, numBin), dtype = np.float)   

    cdef double mean = 0 
    cdef double this_mean_s = 0  
    
    for p in range(numNeuro):
        for j in range(numBin):
            for l in range(numBin):
                rate[p,j] += tuning[p, (numBin+j-l) % numBin]*conv[l]
            rate[p,j] *= tau
            
    # generate Gaussian samples for all s = 1,...,numBin(different s, different samples)
    #     np.random.seed(305)
    
    cov_mat = np.linalg.inv(inv_cov_mat)
    
    gaussian_samples_np = np.zeros((numIter, numNeuro,numBin), dtype = np.float)
    for s in range(numBin):
        gaussian_samples_np[:,:,s] = np.random.multivariate_normal(rate[:,s], cov_mat, size = numIter)
                
    cdef double[:,:,:] gaussian_samples = gaussian_samples_np
        
    cdef int tid

    with nogil, parallel(num_threads = my_num_threads):
        tid = threadid()
        for n_iter in prange(numIter):
            for s in range(numBin):
                # samples conditioning on s: gaussian_samples[n_iter,:,s]
                
                this_mean_s = compute_mean_grad_s_gaussian(
                    s, numBin, numNeuro, rate, weight, inv_cov_mat,
                    gaussian_samples[n_iter,:,s], quad_all[tid,:])
                
                # compute negative mean entropy
                mean += weight[s]*this_mean_s

                update_grad_s_gaussian(s, grad_all[tid,:,:], numBin, numNeuro, rate, weight,
                                       inv_cov_mat, gaussian_samples[n_iter,:,s], this_mean_s)

    mean /= numIter
    mean *= (-1) # mutual information
    
    # compute the rate gradient
    
    cdef double tmp_grad_term = 0
    for p in range(numNeuro):
        for s in range(numBin):
            # the 2nd term is zero
            # the first term
            for k in range(my_num_threads):
                grad[p,s] += grad_all[k,p,s] 
            # devide by numIter
            grad[p,s] /= numIter
            grad[p,s] *= (-1)

                        
    # compute tuning gradient
    for p in range(numNeuro):
        for l in range(numBin):
            MI_grad[p,l] = 0
            for j in range(numBin):
                MI_grad[p,l] += grad[p,j]*conv[(numBin+j-l) % numBin] 
            MI_grad[p,l] *= tau

    return mean#,mean_list_np,grad_all_np,poisson_samples_np,lrate_all_np, mexp_all_np,dexp_all_np



# ----------Blahut-Arimoto Algorithm for Gaussian by Monte Carlo--------

#@cython.boundscheck(False)
#@cython.cdivision(True)
cdef double compute_coeff_s_arimoto_gaussian(int s, int numBin, int numNeuro, 
                                             double[:,:] rate, double[:] weight,
                                             double[:,:] inv_cov_mat, 
                                             double[:] response, double[:] quad) nogil: 
    # for a fixed s in range(numBin)(s is same as m = 1,...,M in the notes)
    # rate: numNeuro*numBin
    # weight: numBin (sum up to 1)
    # count: numNeuro (specific to s, Poisson distribution conditioning on s)    
    
    # quad: numBin
    
    cdef int i, j, l, p #, indj, indk
    cdef double exp_sum
    
    for l in range(numBin):
        quad[l] = 0
        for i in range(numNeuro):
            for j in range(numNeuro):
                quad[l] += (response[i] - rate[i, l])*inv_cov_mat[i,j]*(response[j] - rate[j,l])
                
    exp_sum = 0
    for l in range(numBin):
        exp_sum += weight[l]*exp(-0.5*(quad[l]- quad[s]))
    
  
    return -log(exp_sum)
    

def mc_coeff_arimoto_gaussian(double[:] coeff, double[:,:] tuning, double[:] weight, 
                              double[:,:] inv_cov_mat, double[:] slope,
                              double[:] conv, double tau, int numIter, int my_num_threads = 4):
    '''Compute arimoto coefficients (exp of DKL).'''
    # coeff: numBin (store result)
    # tuing: numNeuro*numBin
    # weight: numBin, sum up to one
    # inv_cov_mat: numNeuro-by-numNeuro positive definite matrix
    # slope: numNeuro, >=0
    # conv: numBin, weighted sum is one.
   
    
    cdef int numNeuro = tuning.shape[0]
    cdef int numBin = tuning.shape[1]
   
    cdef double[:,:] rate = np.zeros((numNeuro,numBin), dtype = np.float)
    cdef Py_ssize_t i,j,l,k,m,p,s
    cdef Py_ssize_t n_iter
    
    cdef double[:,:] coeff_all = np.zeros((my_num_threads, numBin), dtype = np.float)
    cdef double[:,:] quad_all = np.zeros((my_num_threads, numBin), dtype = np.float)
    
    cdef double this_mean_s = 0
    
    for p in range(numNeuro):
        for j in range(numBin):
            for l in range(numBin):
                rate[p,j] += tuning[p, (numBin+j-l) % numBin]*conv[l]
            rate[p,j] *= tau
            
    # generate Poission samples for all s = 1,...,numBin(different s, different samples)
    #     np.random.seed(305)
    cov_mat = np.linalg.inv(inv_cov_mat)
    
    gaussian_samples_np = np.zeros((numIter, numNeuro,numBin), dtype = np.float)
    for s in range(numBin):
        gaussian_samples_np[:,:,s] = np.random.multivariate_normal(rate[:,s], cov_mat, size = numIter)
                
    cdef double[:,:,:] gaussian_samples = gaussian_samples_np

        
    cdef int tid

    with nogil, parallel(num_threads = my_num_threads):
        tid = threadid()
        for n_iter in prange(numIter):
            for s in range(numBin):
                # samples conditioning on s: gaussian_samples[n_iter,:,s]
                this_mean_s = compute_coeff_s_arimoto_gaussian(
                    s, numBin, numNeuro, rate, weight, inv_cov_mat,
                    gaussian_samples[n_iter,:,s], quad_all[tid, :])
                
                coeff_all[tid, s] += this_mean_s
                
    for m in range(numBin): 
        coeff[m] = 0
        for tid in range(my_num_threads):
            coeff[m] += coeff_all[tid, m]
        coeff[m] /= numIter
        coeff[m] = exp(coeff[m])
        for p in range(numNeuro):
            coeff[m] *= exp(-slope[p]*tuning[p,m])
    return

#------------------------------------------------------------
#-----------Inhomogeneous cases for Gaussian Model-----------
#------------------------------------------------------------

# ----------Compute info and grad by Monte Carlo: Inhomogeneous case--------
# double[:,:,:] inv_cov_mat, add double[:] rho
cdef double compute_mean_grad_s_gaussian_inhomo(
    int s, int numBin, int numNeuro, double[:,:] rate, double[:] weight,
    double[:,:,:] inv_cov_mat, double[:] rho, double[:] response, double[:] quad) nogil:
    # for a fixed s in range(numBin)(s is same as m in the notes)
    # rate: numNeuro*numBin
    # weight: numBin (sum up to 1)
    # inv_cov_mat: numNeuro*numNeuro*numBin, 
    # inv_cov_mat[:,:,j] is the inverse covariance matrix of p(r|theta_j)
    # rho: numBin, rho[j] = det(inv_cov_mat[:,:,j])^(1/2), can be pre-computed
    # response: numNeuro (specific to s, gaussian distribution conditioning on s)
    
    # quad: numBin,  (specific to s)
       

    cdef int i,j,k,l, p #, indj, indk
    cdef double mean_s, tmp_sum, diff     #mymax

    for l in range(numBin):
        quad[l] = 0
        for i in range(numNeuro):
            for j in range(numNeuro):
                quad[l] += (response[i] - rate[i, l])*inv_cov_mat[i,j,l]*(response[j] - rate[j,l])
        
    mean_s = 0
    for l in range(numBin):
        mean_s += weight[l] * rho[l]/rho[s] * exp(-0.5*(quad[l] - quad[s]))
    mean_s = log(mean_s)

    return mean_s


#@cython.boundscheck(False)
#@cython.cdivision(True)
cdef void update_grad_s_gaussian_inhomo(
    int s, double[:,:] grad, int numBin, int numNeuro,
    double[:,:] rate, double[:] weight, double[:,:,:] inv_cov_mat,
    double[:] response, double mean_s) nogil:
    # s is now i in the notes..
    # response is sampled conditioning on s
    # derivative of the probability density term
    cdef int p
    cdef double diff_p_s
    for p in range(numNeuro):
        diff_p_s = 0
        for k in range(numNeuro):
            diff_p_s += inv_cov_mat[p,k,s]*(response[k] - rate[k,s]) 
        
        grad[p, s] += weight[s]*diff_p_s*mean_s


# compute I and grad(I).
def mc_mean_grad_gaussian_inhomo(
    double[:,:] MI_grad, double[:,:] tuning, double[:] weight,
    double[:,:,:] inv_cov_mat,
    double[:] conv, double tau, int numIter, int my_num_threads = 4):
    # conv is old stim...
    # tuing: numNeuro*numBin
    # MI_grad: numNeuro*numBin
    # weight: numBin, sum up to one
    # inv_cov_mat: inverse covariance matrices, numNeuro*numNeuro*numBin, 
    #              where inv_cov_mat[:,:,j] is the inverse covariance matrix of p(r|theta_j)
    # conv: numBin, weighted sum is one.
    # 
    cdef int numNeuro = tuning.shape[0] # or cdef int numNeuro
    cdef int numBin = tuning.shape[1] # or cdef int numBin
   
    cdef double[:,:] rate = np.zeros((numNeuro,numBin), dtype = np.float)
    cdef Py_ssize_t i,j,l,k,m,p,s
    cdef Py_ssize_t n_iter

    cdef double[:,:] grad = np.zeros((numNeuro,numBin), dtype = np.float)
    cdef double[:,:,:] grad_all = np.zeros((my_num_threads, numNeuro, numBin), dtype = np.float) 
    
    cdef double[:,:] quad_all = np.zeros((my_num_threads, numBin), dtype = np.float)   

    cdef double mean = 0 
    cdef double this_mean_s = 0  
    
    for p in range(numNeuro):
        for j in range(numBin):
            for l in range(numBin):
                rate[p,j] += tuning[p, (numBin+j-l) % numBin]*conv[l]
            rate[p,j] *= tau
    
    # rho: numBin, rho[j] = det(inv_cov_mat[:,:,j])^(1/2)
    cdef double[:] rho = np.zeros(numBin, dtype = np.float)    
    for j in range(numBin):
        rho[j] = np.sqrt(np.linalg.det(inv_cov_mat[:,:,j]))
            
    # generate Gaussian samples for all s = 1,...,numBin(different s, different samples)
    #     np.random.seed(305)
        
    gaussian_samples_np = np.zeros((numIter, numNeuro,numBin), dtype = np.float)
    for s in range(numBin):
        cov_mat = np.linalg.inv(inv_cov_mat[:,:,s])
        gaussian_samples_np[:,:,s] = np.random.multivariate_normal(rate[:,s], cov_mat, size = numIter)
                
    cdef double[:,:,:] gaussian_samples = gaussian_samples_np
        
    cdef int tid

    with nogil, parallel(num_threads = my_num_threads):
        tid = threadid()
        for n_iter in prange(numIter):
            for s in range(numBin):
                # samples conditioning on s: gaussian_samples[n_iter,:,s]
                
                this_mean_s = compute_mean_grad_s_gaussian_inhomo(
                    s, numBin, numNeuro, rate, weight, inv_cov_mat, rho,
                    gaussian_samples[n_iter,:,s], quad_all[tid,:])
                
                # compute negative mean entropy
                mean += weight[s]*this_mean_s

                update_grad_s_gaussian_inhomo(
                    s, grad_all[tid,:,:], numBin, numNeuro, rate, weight,
                    inv_cov_mat, gaussian_samples[n_iter,:,s], this_mean_s)

    mean /= numIter
    mean *= (-1) # mutual information
    
    # compute the rate gradient
    
    cdef double tmp_grad_term = 0
    for p in range(numNeuro):
        for s in range(numBin):
            # the 2nd term is zero
            # the first term
            for k in range(my_num_threads):
                grad[p,s] += grad_all[k,p,s] 
            # devide by numIter
            grad[p,s] /= numIter
            grad[p,s] *= (-1)

                        
    # compute tuning gradient
    for p in range(numNeuro):
        for l in range(numBin):
            MI_grad[p,l] = 0
            for j in range(numBin):
                MI_grad[p,l] += grad[p,j]*conv[(numBin+j-l) % numBin] 
            MI_grad[p,l] *= tau

    return mean#,mean_list_np,grad_all_np,poisson_samples_np,lrate_all_np, mexp_all_np,dexp_all_np


# ----------Blahut-Arimoto Algorithm for Gaussian by Monte Carlo: Inhomogeneous case--------

#@cython.boundscheck(False)
#@cython.cdivision(True)
cdef double compute_coeff_s_arimoto_gaussian_inhomo(
    int s, int numBin, int numNeuro,
    double[:,:] rate, double[:] weight,
    double[:,:,:] inv_cov_mat, double[:] rho,
    double[:] response, double[:] quad) nogil: 
    # for a fixed s in range(numBin)(s is same as m = 1,...,M in the notes)
    # rate: numNeuro*numBin
    # weight: numBin (sum up to 1)    
    # inv_cov_mat: inverse covariance matrices, numNeuro*numNeuro*numBin, 
    #              where inv_cov_mat[:,:,j] is the inverse covariance matrix of p(r|theta_j)
    # rho: numBin, rho[j] = det(inv_cov_mat[:,:,j])^(1/2), pre-computed
    # response: numNeuro (specific to s, Gaussian distribution conditioning on s) 
    # quad: numBin
    
    cdef int i, j, l, p #, indj, indk
    cdef double exp_sum
    
    for l in range(numBin):
        quad[l] = 0
        for i in range(numNeuro):
            for j in range(numNeuro):
                quad[l] += (response[i] - rate[i, l])*inv_cov_mat[i,j,l]*(response[j] - rate[j,l])
                
    exp_sum = 0
    for l in range(numBin):
        exp_sum += weight[l]*rho[l]/rho[s]*exp(-0.5*(quad[l]- quad[s]))
    
  
    return -log(exp_sum)
    

def mc_coeff_arimoto_gaussian_inhomo(
    double[:] coeff, double[:,:] tuning, double[:] weight,
    double[:,:,:] inv_cov_mat,
    double[:] slope,
    double[:] conv, double tau, int numIter, int my_num_threads = 4):
    '''Compute arimoto coefficients (exp of DKL).'''
    # coeff: numBin (store result)
    # tuing: numNeuro*numBin
    # weight: numBin, sum up to one
    # inv_cov_mat: inverse covariance matrices, numNeuro*numNeuro*numBin, 
    #              where inv_cov_mat[:,:,j] is the inverse covariance matrix of p(r|theta_j)
    # slope: numNeuro, >=0
    # conv: numBin, weighted sum is one.
   
    
    cdef int numNeuro = tuning.shape[0]
    cdef int numBin = tuning.shape[1]
   
    cdef double[:,:] rate = np.zeros((numNeuro,numBin), dtype = np.float)
    cdef Py_ssize_t i,j,l,k,m,p,s
    cdef Py_ssize_t n_iter
    
    cdef double[:,:] coeff_all = np.zeros((my_num_threads, numBin), dtype = np.float)
    cdef double[:,:] quad_all = np.zeros((my_num_threads, numBin), dtype = np.float)
    
    cdef double this_mean_s = 0
    
    for p in range(numNeuro):
        for j in range(numBin):
            for l in range(numBin):
                rate[p,j] += tuning[p, (numBin+j-l) % numBin]*conv[l]
            rate[p,j] *= tau
            
    # rho: numBin, rho[j] = det(inv_cov_mat[:,:,j])^(1/2)
    cdef double[:] rho = np.zeros(numBin, dtype = np.float)    
    for j in range(numBin):
        rho[j] = np.sqrt(np.linalg.det(inv_cov_mat[:,:,j]))
            
    # generate Poission samples for all s = 1,...,numBin(different s, different samples)
    #     np.random.seed(305)
    
    gaussian_samples_np = np.zeros((numIter, numNeuro,numBin), dtype = np.float)
    for s in range(numBin):
        cov_mat = np.linalg.inv(inv_cov_mat[:,:,s])
        gaussian_samples_np[:,:,s] = np.random.multivariate_normal(rate[:,s], cov_mat, size = numIter)
                
    cdef double[:,:,:] gaussian_samples = gaussian_samples_np

        
    cdef int tid

    with nogil, parallel(num_threads = my_num_threads):
        tid = threadid()
        for n_iter in prange(numIter):
            for s in range(numBin):
                # samples conditioning on s: gaussian_samples[n_iter,:,s]
                this_mean_s = compute_coeff_s_arimoto_gaussian_inhomo(
                    s, numBin, numNeuro, rate, weight, inv_cov_mat, rho,
                    gaussian_samples[n_iter,:,s], quad_all[tid, :])
                
                coeff_all[tid, s] += this_mean_s
                
    for m in range(numBin): 
        coeff[m] = 0
        for tid in range(my_num_threads):
            coeff[m] += coeff_all[tid, m]
        coeff[m] /= numIter
        coeff[m] = exp(coeff[m])
        for p in range(numNeuro):
            coeff[m] *= exp(-slope[p]*tuning[p,m])
    return
