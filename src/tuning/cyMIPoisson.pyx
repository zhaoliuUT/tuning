##cython --annotate --cplus --link-args=-fopenmp --compile-args=-fopenmp --compile-args=-std=c++0x
#!python
#cython: boundscheck=False, wraparound=False, nonecheck=False, language_level=3

import numpy as np
from cython import floating
from libc.math cimport log, exp
import cython
from cython.parallel import parallel, prange, threadid
# from libc.stdio cimport printf


@cython.boundscheck(False)
@cython.cdivision(True)
cdef double compute_mean_s(int s, int numCent, int delta, int numPop, int[:] center, double[:,:] rate, int[:,:] count,\
                         double[:,:,:] lrate, double[:] mexp, double[:] dexp) nogil: 
    # for a fixed s in range(delta)
    # lrate: numCent*numBin, mexp: numBin, dexp:numBin
    # count: numCent, rate:numBin
    # delta: 'stepPop' in MIred.
    cdef int i,j,k,p, indj, indk
    cdef int numBin
    numBin = numCent*delta
    cdef double mymax, mean_s       

    for j in range(numBin):
        dexp[j] = 0
        for k in range(numCent):
            indj = (numBin+j-center[k]) % numBin
            indk = (numBin+s-center[k]) % numBin
            for p in range(numPop):
                lrate[p,k,j] = log(rate[p,indj]/rate[p,indk])
                dexp[j] += rate[p,indk]-rate[p,indj]        

    mymax = 1.0
    for i in range(numBin):
        mexp[i] = 0
        for p in range(numPop):
            for k in range(numCent):
                mexp[i] += count[p,k]*lrate[p,k,i] if count[p,k] else 0
        mexp[i] += dexp[i]
        if mexp[i] > mymax:
            mymax = mexp[i] 

    mean_s = 0
    for i in range(numBin):      
        mean_s += exp(mexp[i] - mymax)
    mean_s = mymax + log(mean_s/numBin)
    return mean_s

@cython.boundscheck(False)
@cython.cdivision(True)
cdef void update_grad_s(int s, double[:,:] grad, int numCent, int delta, int numPop,\
                        int[:] center, double[:,:] rate, int[:,:] count, double mean_s) nogil:
    # update grad(I) (not -grad(I))
    cdef int p, k, indk, numBin
    numBin = numCent*delta    
    for k in range(numCent):
        indk = (numBin+s-center[k]) % numBin
        for p in range(numPop):
        #derivative probability density term
        # print 's=%d,k=%d,indk=%d'%(s,k,indk)
            grad[p,indk] += (1.0 - count[p,k]/rate[p,indk])*mean_s if count[p,k] else mean_s

# compute -I and -grad(I), for convenience of minimization.
def mc_mean_grad_red_pop(double[:,:] MI_grad,int numCent, int delta,\
                     double[:,:] tuning, double[:] stim, double tau, int numIter, int my_num_threads = 4):
    
    cdef int numPop = tuning.shape[0] # or cdef int numPop
    cdef int numBin = tuning.shape[1] # or cdef int numBin
    # numBin should be = numCent*delta. not checked here.
    cdef int numPopBin = numPop*numBin    
    cdef double[:,:] rate = np.zeros((numPop,numBin), dtype = np.float)
    cdef Py_ssize_t i,j,k,l, indi, inj, indk, p, q, s
    cdef Py_ssize_t n_iter    

    cdef double[:,:] grad = np.zeros((numPop,numBin), dtype = np.float)
    cdef double[:,:,:] grad_all = np.zeros((my_num_threads, numPop, numBin), dtype = np.float) 
    
    cdef double[:,:] dexp_all = np.zeros((my_num_threads, numBin), dtype = np.float)
    cdef double[:,:] mexp_all = np.zeros((my_num_threads, numBin), dtype = np.float)   
    cdef double[:,:,:,:] lrate_all = np.zeros((my_num_threads, numPop, numCent,numBin),dtype = np.float)
        
    cdef int[:] center = np.zeros(numCent, dtype = np.intc)
    for i in range(numCent):
        center[i] = i*delta
    # cdef int[:] center = np.arange(0,numBin,delta)

    cdef double mean = 0 
    cdef double this_mean_s = 0  
    
    for i in range(numPop):
        for j in range(numBin):
            for k in range(numBin):
                rate[i,j] += tuning[i, (numBin+j-k) % numBin]*stim[k]
            rate[i,j] = tau*rate[i,j]

    # generate Poission samples for all s = 1,...,delta(different s, different samples)
    #     np.random.seed(305)
    
    poisson_samples_np = np.zeros((numIter, numPop, delta, numCent), dtype = np.intc)
    for s in range(delta):
        for k in range(numCent):
            indk = (numBin+s-center[k]) % numBin
            for i in range(numPop):
                # count[k] = randPoi(rate[indk], &Seed);
                poisson_samples_np[:, i, s, k] = np.random.poisson(lam = rate[i,indk], size = numIter)
    cdef int[:,:,:,:] poisson_samples = poisson_samples_np
        
    cdef int tid

    with nogil, parallel(num_threads = my_num_threads):
        tid = threadid()
        for n_iter in prange(numIter):
            for s in range(delta):

                this_mean_s = compute_mean_s(s, numCent, delta, numPop, center, rate, poisson_samples[n_iter,:,s,:],
                                         lrate_all[tid,:,:,:], mexp_all[tid,:], dexp_all[tid,:])
                # compute negative mean entropy
                mean += this_mean_s

                update_grad_s(s, grad_all[tid,:,:], numCent,delta,numPop, center,rate,
                              poisson_samples[n_iter,:,s, :], this_mean_s)

    for i in range(numPop):
        for j in range(numBin):
            for k in range(my_num_threads):
                grad[i,j] += grad_all[k,i,j] 

    mean /= numIter*delta   
    # compute the rate gradient
    for p in range(numPop):
        for i in range(numBin):
            grad[p,i] /= numIter*delta
                        
    # compute tuning gradient
    for i in range(numPop):
        for j in range(numBin):
            MI_grad[i,j] = 0
            for k in range(numBin):
                MI_grad[i,j] += grad[i,k]*stim[(numBin+k-j) % numBin] 
            MI_grad[i,j] *= -tau

    return mean#,mean_list_np,grad_all_np,poisson_samples_np,lrate_all_np, mexp_all_np,dexp_all_np


# compute -I and -grad(I), for convenience of minimization.
def mc_mean_red_pop(int numCent, int delta,\
                     double[:,:] tuning, double[:] stim, double tau, int numIter, int my_num_threads = 4):

    cdef int numPop = tuning.shape[0] # or cdef int numPop
    cdef int numBin = tuning.shape[1] # or cdef int numBin
    cdef int numPopBin = numPop*numBin    
    cdef double[:,:] rate = np.zeros((numPop,numBin), dtype = np.float)
    cdef Py_ssize_t i,j,k,l, indi, inj, indk, p, q, s
    cdef Py_ssize_t n_iter    
    
    cdef double[:,:] dexp_all = np.zeros((my_num_threads, numBin), dtype = np.float)
    cdef double[:,:] mexp_all = np.zeros((my_num_threads, numBin), dtype = np.float)   
    cdef double[:,:,:,:] lrate_all = np.zeros((my_num_threads, numPop, numCent,numBin),dtype = np.float)
        
    cdef int[:] center = np.zeros(numCent, dtype = np.intc)
    for i in range(numCent):
        center[i] = i*delta
    # cdef int[:] center = np.arange(0,numBin,delta)

    cdef double mean = 0 
    cdef double this_mean_s = 0  
    
    for i in range(numPop):
        for j in range(numBin):
            for k in range(numBin):
                rate[i,j] += tuning[i, (numBin+j-k) % numBin]*stim[k]
            rate[i,j] = tau*rate[i,j]

    # generate Poission samples for all s = 1,...,delta(different s, different samples)
    #  np.random.seed(305)
    
    poisson_samples_np = np.zeros((numIter, numPop, delta, numCent), dtype = np.intc)
    for s in range(delta):
        for k in range(numCent):
            indk = (numBin+s-center[k]) % numBin
            for i in range(numPop):
                # count[k] = randPoi(rate[indk], &Seed);
                poisson_samples_np[:, i, s, k] = np.random.poisson(lam = rate[i,indk], size = numIter)
    cdef int[:,:,:,:] poisson_samples = poisson_samples_np
        
    cdef int tid

    with nogil, parallel(num_threads = my_num_threads):
        tid = threadid()
        for n_iter in prange(numIter):
            for s in range(delta):

                this_mean_s = compute_mean_s(s, numCent, delta, numPop, center, rate, poisson_samples[n_iter,:,s,:],
                                         lrate_all[tid,:,:,:], mexp_all[tid,:], dexp_all[tid,:])
                # compute negative mean entropy
                mean += this_mean_s

                # update_grad_s(s, grad_all[tid,:,:], numCent,delta,numPop, center,rate,
                              # poisson_samples[n_iter,:,s, :], this_mean_s)

            
    mean /= numIter*delta  
 
    return mean

def compute_rate_pop(double[:,:] tuning,double[:] stim, double tau):
    cdef int i,j,k
    cdef int numPop = tuning.shape[0] # or cdef int numPop
    cdef int numBin = tuning.shape[1] # or cdef int numBin
    rate_np = np.zeros((numPop,numBin), dtype = np.float)
    cdef double[:,:] rate = rate_np
    
    for i in range(numPop):
        for j in range(numBin):
            for k in range(numBin):
                rate[i,j] += tuning[i, (numBin+j-k) % numBin]*stim[k]
            rate[i,j] = tau*rate[i,j]
    
    return rate_np
