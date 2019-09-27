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

# -----------Noncyclic model with NO constraints-----------
if __name__ == '__main__':    
    if len(sys.argv) >= 7:
        filename = sys.argv[1]
        ITER_NUM = int(sys.argv[2])
        ITER_CHANNEL = int(sys.argv[3])
        ITER_CAPACITY = int(sys.argv[4])
        INTER_STEPS = int(sys.argv[5])
        NUM_THREADS = int(sys.argv[6])
        #other possible parameters: USE_MC, MC_ITER (1e5 or 1e6) , SUM_THRESHOLD (50 or 100), numNeuro
        print 'filename = %s'%filename # no '.npy'
        print 'total number of iterations = %d'%ITER_NUM
        print 'channel iterations in each iteration = %d'%ITER_CHANNEL
        print 'capacity iterations in each iteration = %d'%ITER_CAPACITY
        print 'INTER_STEPS = %d'%INTER_STEPS
        print 'number of threads = %d'%NUM_THREADS

    else:
        raise Exception('Error: wrong input format!')

tc_opt_nc = TuningCurveOptimizer_Noncyclic.load_res_list(filename)

print('FP = ' + str(tc_opt_nc.fp))
print('FM = ' + str(tc_opt_nc.fm))

for k in range(ITER_NUM):
     # not saving results
     # use partial sum instead of Monte Carlo
    tc_opt_nc.channel_iterate(ITER_CHANNEL,INTER_STEPS = INTER_STEPS, ADD_TIME = False,
                              ADD_INEQ_CONS = False, 
                              USE_MC = False,
                              SUM_THRESHOLD_INFO = 50, SUM_THRESHOLD_GRAD = 50, NUM_THREADS = NUM_THREADS) #MC_ITER_INFO = 1e5, MC_ITER_GRAD = 1e5
    tc_opt_nc.capacity_iterate(ITER_CAPACITY, INTER_STEPS = INTER_STEPS,
                               ADD_INEQ_CONS = False,
                               USE_MC = False,
                               SUM_THRESHOLD_BA = 50, 
                               ADD_TIME = False, NUM_THREADS = NUM_THREADS)


# tc_opt_nc.plot_info()

tc_opt_nc.save_res_list(FILE_NAME = 'data/test0_noncyclic', ADD_TIME = True)

#tc_opt_nc.plot_animation(FILE_NAME = 'data/test0_noncyclic', ADD_TIME = True)

#tc_opt_nc.plot_animation(FILE_NAME = 'data/test0_noncyclic_cube', ADD_TIME = True, IF_CUBE = True)

