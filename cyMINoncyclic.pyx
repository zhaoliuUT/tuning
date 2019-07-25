STUFF = "Hi"
##cython --annotate --cplus --link-args=-fopenmp --compile-args=-fopenmp --compile-args=-std=c++0x
#cython: boundscheck=False, wraparound=False, nonecheck=False

import numpy as np
from cython import floating
from libc.math cimport log, exp
import cython
from cython.parallel import parallel, prange, threadid

from itertools import product, combinations,permutations

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
                                        double[:] prod_s) nogil: 
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

    for p in range(numNeuro):
        for j in range(numBin):
            texp[p, j] = count[p]*log(rate[p,j]) - rate[p,j] if rate[p,j] else 0
            
    cdef double info = 0
    cdef double tmp_sum_l = 0
    
    for j in range(numBin):
        prod_p[j] = 0
        prod_s[j] = 0
        
        for p in range(numNeuro):
            prod_p[j] += texp[p,j] - log_factorial(count[p])
        prod_p[j] = exp(prod_p[j])   
        
        for l in range(numBin):
            tmp_sum_l = 0
            for k in range(numNeuro):               
                tmp_sum_l += texp[k,l] - texp[k,j]
            prod_s[j] += weight[l]*exp(tmp_sum_l)
            
        info += weight[j]*prod_p[j]*log(prod_s[j])
        
    info *= (-1)
    
    # update grad
    cdef double tmp_grad_kl = 0
    for k in range(numNeuro):
        for l in range(numBin):
            tmp_grad_kl = weight[l]*prod_p[l]*log(prod_s[l])
            #grad[k,l] = weight[l]*prod_p[l]*log(prod_s[l])
            for j in range(numBin):
                tmp_grad_kl += weight[j]*prod_p[j]*weight[l]/prod_s[l]
                #grad[k,l] += weight[j]*prod_p[j]*weight[l]/prod_s[l]
            #grad[k,l] *= (1 - count[k]/rate[k,l]) if rate[k,l] else 1
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

cdef double compute_mean_grad_s_noncyclic(int s, int numBin, int numNeuro, double[:,:] rate, double[:] weight,int[:] count,\
                         double[:,:,:] tmp_grad, double[:,:] lrate, double[:] mexp, double[:] dexp) nogil: 
    # for a fixed s in range(numBin)(s is same as m in the notes)
    # rate: numNeuro*numBin
    # weight: numBin (sum up to 1)
    # count: numNeuro (specific to s, Poisson distribution conditioning on s)
    
    
    # tmp_grad: numBin*numNeuro*numBin, (will reuse in grad, so store for each s)
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
    
    for p in range(numNeuro):
        for i in range(numBin):
            tmp_grad[s, p, i] += (count[p]/rate[p,i] - 1)*weight[i]*mexp[i]/tmp_sum_s if rate[p,i] else (-weight[i]*mexp[i]/tmp_sum_s)
        
    return mean_s

#@cython.boundscheck(False)
#@cython.cdivision(True)
cdef void update_grad_s_noncyclic(int s, double[:,:] grad, int numBin, int numNeuro, double[:,:] rate, double[:] weight, \
                        int[:] count, double mean_s) nogil:
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
    
    # cdef double[:,:,:] tmp_grad = np.zeros((numBin, numNeuro,numBin),dtype = np.float)
    cdef double[:,:,:,:] tmp_grad_all = np.zeros((my_num_threads, numBin, numNeuro,numBin),dtype = np.float)


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
                this_mean_s = compute_mean_grad_s_noncyclic(s, numBin, numNeuro, rate, weight,\
                                             poisson_samples[n_iter,:,s],tmp_grad_all[tid,:,:,:],\
                                         lrate_all[tid,:,:], mexp_all[tid,:], dexp_all[tid,:])
                
                # compute negative mean entropy
                mean += weight[s]*this_mean_s

                update_grad_s_noncyclic(s, grad_all[tid,:,:], numBin, numNeuro, rate, weight,\
                              poisson_samples[n_iter,:,s], this_mean_s)

    mean /= numIter
    mean *= (-1) # mutual information
    
    # compute the rate gradient
    
    cdef double tmp_grad_term = 0
    for p in range(numNeuro):
        for s in range(numBin):
            # the 2nd term
            for m in range(numBin): 
                tmp_grad_term = 0
                for k in range(my_num_threads):
                    tmp_grad_term += tmp_grad_all[k, m,p, s]#tmp_grad[m, p,s] += tmp_grad_all[k, m,p, s]
                    # tmp_grad[m,p,s] is sampled conditioning on m
                grad[p,s] += weight[m]*tmp_grad_term #weight[m]*tmp_grad[m,p, s]
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
cdef double count_prob_s_arimoto(int s, int numBin, int numNeuro, double[:,:] rate,
                                   double[:] weight,\
                                   int[:] count, double[:] rexp) nogil: 
    '''
    For a fixed r=(r_1, ..., r_P) and fixed s = 1,...,numBin, 
    compute P_s(r) * log( w_s / S_s(r) )
    '''
    # s is same as j = 1,...,numBin in the notes)
    # rate: numNeuro*numBin
    # weight: numBin (sum up to 1)
    # count: numNeuro (same as r=(r_1, ..., r_P))  
    
    # rexp: numBin  (specific to s) 
    # rexp[i] = sum_p [r_p*log(f_{r,i}) - f_{r,i} ]

    cdef int i, p
    cdef double prod_p_s, qexp_s, mymax
    
    #     mymax = 1.0
    for i in range(numBin):
        rexp[i] = 0
        for p in range(numNeuro):
            rexp[i] += count[p]*log(rate[p,i]) - rate[p,i] if rate[p,i] else 0           
    #         if rexp[i] > mymax:
    #             mymax = rexp[i]    
    
    prod_p_s = rexp[s]
    for p in range(numNeuro):
        prod_p_s -= log_factorial(count[p])
    prod_p_s = exp(prod_p_s) 
    
    qexp_s = 0
    for i in range(numBin):
        qexp_s += weight[i]*exp(rexp[i]) #weight[i]*exp(rexp[i] - mymax)
    qexp_s = weight[s]*exp(rexp[s])/qexp_s #weight[s]*exp(rexp[s] - mymax)/qexp_s
   
    return prod_p_s * log(qexp_s) if qexp_s else 0
    

def partial_sum_prob_arimoto(double[:] weight_new, double[:,:] tuning, double[:] weight, 
                       double[:] conv, double tau, 
                       int threshold = 50, int my_num_threads = 4):
    '''Compute arimoto update for one iteration.'''
    # conv is old stim...
    # weight_new: computed new weights(probability)
    # tuing: numNeuro*numBin
    # weight: numBin, sum up to one
    # conv: numBin, weighted sum is one.
    
    cdef int numNeuro = tuning.shape[0]
    cdef int numBin = tuning.shape[1]
   
    cdef double[:,:] rate = np.zeros((numNeuro,numBin), dtype = np.float)
    cdef Py_ssize_t i,j,l,k,m,p,s
    
    cdef double[:,:] weight_new_all = np.zeros((my_num_threads, numBin), dtype = np.float)
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
                this_count_s = count_prob_s_arimoto(s,numBin,numNeuro, rate, weight,\
                                                     count_arr[idx,:], rexp_all[tid, :])
                
                weight_new_all[tid, s] += this_count_s
                
    
        
    cdef double prob_sum = 0
    for m in range(numBin): 
        weight_new[m] = 0
        for tid in range(my_num_threads):
            weight_new[m] += weight_new_all[tid, m]
        weight_new[m] = exp(weight_new[m])
        
    for m in range(numBin):
        prob_sum += weight_new[m] 

    for m in range(numBin): 
        weight_new[m] /= prob_sum
    return 0

# ----------Blahut-Arimoto Algorithm by Monte Carlo--------

#@cython.boundscheck(False)
#@cython.cdivision(True)
cdef double compute_prob_s_arimoto(int s, int numBin, int numNeuro, double[:,:] rate, double[:] weight,\
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
        
    qexp_s = weight[s]*exp(rexp[s])/qexp_s
    #     qexp_s = 0
    #     for i in range(numBin):
    #         qexp_s += weight[i]*exp(rexp[i] - mymax)

    #     qexp_s = weight[s]*exp(rexp[s] - mymax)/qexp_s

    mean_s = log(qexp_s) if qexp_s else 0
    
    return mean_s

def mc_prob_arimoto(double[:] weight_new, double[:,:] tuning, double[:] weight,\
                    double[:] conv, double tau, int numIter, int my_num_threads = 4):
    '''Compute arimoto update for one iteration.'''
    # conv is old stim...
    # weight_new: computed new weights(probability)
    # tuing: numNeuro*numBin
    # weight: numBin, sum up to one
    # conv: numBin, weighted sum is one.
    
    cdef int numNeuro = tuning.shape[0]
    cdef int numBin = tuning.shape[1]
   
    cdef double[:,:] rate = np.zeros((numNeuro,numBin), dtype = np.float)
    cdef Py_ssize_t i,j,l,k,m,p,s
    cdef Py_ssize_t n_iter
    
    cdef double[:,:] weight_new_all = np.zeros((my_num_threads, numBin), dtype = np.float)
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
                this_mean_s = compute_prob_s_arimoto(s,numBin,numNeuro, rate, weight,\
                                                     poisson_samples[n_iter,:,s], rexp_all[tid, :])
                
                weight_new_all[tid, s] += this_mean_s
                
    
        
    cdef double prob_sum = 0
    for m in range(numBin): 
        weight_new[m] = 0
        for tid in range(my_num_threads):
            weight_new[m] += weight_new_all[tid, m]
        weight_new[m] /= numIter
        weight_new[m] = exp(weight_new[m])
        
    for m in range(numBin):
        prob_sum += weight_new[m] 
#         print weight_new[m]
#     print prob_sum
    for m in range(numBin): 
        weight_new[m] /= prob_sum
    return 0


