import time, sys, os, copy
import numpy as np
import matplotlib.pyplot as plt
import tuning
from tuning.tuning_curve import TuningCurve
from tuning.tuning_curve_optimizer import TuningCurveOptimizer
from tuning.tuning_curve_bn import TuningCurve_BN
from tuning.tuning_curve_optimizer_bn import TuningCurveOptimizer_BN
from tuning.tuning_curve_noncyclic import TuningCurve_Noncyclic
from tuning.tuning_curve_optimizer_noncyclic import TuningCurveOptimizer_Noncyclic

# -----------Noncyclic model with inequality constraints-----------
if __name__ == '__main__':    
    if len(sys.argv) >= 9:
        avg = float(sys.argv[1])
        fp = float(sys.argv[2])
        fm = float(sys.argv[3])
        numBin = int(sys.argv[4])
        ITER_NUM = int(sys.argv[5])
        ITER_CHANNEL = int(sys.argv[6])
        ITER_CAPACITY = int(sys.argv[7])
        NUM_THREADS = int(sys.argv[8])
       
        print 'weighted average = %.3f'%avg
        print 'FP = %.2f'%fp
        print 'FM = %.2f'%fm
        print 'number of bins = %d'%numBin
        print 'total number of iterations = %d'%ITER_NUM
        print 'channel iterations in each iteration = %d'%ITER_CHANNEL
        print 'capacity iterations in each iteration = %d'%ITER_CAPACITY
        print 'number of threads = %d'%NUM_THREADS

    else:
        raise Exception('Error: wrong input format!')

numNeuro = 3

weight = np.ones(numBin)/numBin

# tuning = np.linspace(fm, fp, numNeuro*numBin).reshape(numNeuro, numBin)
tuning = np.random.uniform(fm, fp, (numNeuro,numBin))
#conv = np.zeros(numBin)
#conv[0] = 1
tau = 1.0

curr_time = time.time()
tc_nc = TuningCurve_Noncyclic(tuning, weight, conv = None, tau = 1.0, MC_ITER = 1e6)
print("computation time using Monte Carlo 1e6 iterations = %.4f"%(time.time() - curr_time))
print("mutual information = %.4f"%tc_nc.info)

# tc_nc.plot()
print(np.dot(tuning, weight))
if np.any(np.dot(tuning, weight) > avg):
    raise Exception('Error: wrong input of average!')

tc_opt_nc = TuningCurveOptimizer_Noncyclic(tc_nc,fp,fm, average_cons = np.max(np.dot(tuning, weight)) + 0.1)

#tc_opt_nc = TuningCurveOptimizer_Noncyclic(tc_nc,fp,fm)

for k in range(ITER_NUM):
     # not saving results
     # use partial sum instead of Monte Carlo
    tc_opt_nc.channel_iterate(ITER_CHANNEL,INTER_STEPS = 1, ADD_TIME = False,
                              ADD_INEQ_CONS = True, 
                              USE_MC = False,
                              SUM_THRESHOLD_INFO = 50, SUM_THRESHOLD_GRAD = 50) #MC_ITER_INFO = 1e5, MC_ITER_GRAD = 1e5
    tc_opt_nc.capacity_iterate(ITER_CAPACITY, INTER_STEPS = 1, 
                               ADD_INEQ_CONS = True,
                               USE_MC = False,
                               SUM_THRESHOLD_BA = 50, 
                               ADD_TIME = False)


# tc_opt_nc.plot_info()

tc_opt_nc.save_res_list(FILE_NAME = 'data/test1_noncyclic', ADD_TIME = True)

tc_opt_nc.plot_animation(FILE_NAME = 'data/test1_noncyclic', ADD_TIME = True)

tc_opt_nc.plot_animation(FILE_NAME = 'data/test1_noncyclic_cube', ADD_TIME = True, IF_CUBE = True)

