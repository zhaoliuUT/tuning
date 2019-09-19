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
    if len(sys.argv) >= 8:
        filename = sys.argv[1]
        avg = float(sys.argv[2])
        ITER_NUM = int(sys.argv[3])
        ITER_CHANNEL = int(sys.argv[4])
        ITER_CAPACITY = int(sys.argv[5])
        INTER_STEPS = int(sys.argv[6])
        NUM_THREADS = int(sys.argv[7])
        
        if ITER_CHANNEL%INTER_STEPS!=0 or ITER_CAPACITY%INTER_STEPS!=0:
            raise Exception('Error: wrong input of INTER_STEPS: must be factors of ITER_CHANNEL and ITER_CAPACITY.')
        
        print 'filename = %s'%filename # no '.npy'
        print 'input weighted average constraint = %.3f'%avg
        print 'total number of iterations = %d'%ITER_NUM
        print 'channel iterations in each iteration = %d'%ITER_CHANNEL
        print 'capacity iterations in each iteration = %d'%ITER_CAPACITY
        
        print 'INTER_STEPS = %d'%INTER_STEPS
        print 'number of threads = %d'%NUM_THREADS
        
        SUM_THRESHOLD = 50 # default
    if len(sys.argv) == 9:
        SUM_THRESHOLD = int(sys.argv[8])
        print 'input sum_threshold  = %d'%SUM_THRESHOLD
    else:
        raise Exception('Error: wrong input format!')

res_list = np.load(filename + ".npy", allow_pickle=True).item()
tc = TuningCurve_Noncyclic(res_list['x'][0], res_list['weight'][0], res_list['conv'], res_list['tau'], \
                           res_list['obj'][0], res_list['grad'][0]) 
tc_opt_nc = TuningCurveOptimizer_Noncyclic(tc, res_list['fp'], res_list['fm'],  average_cons = avg)
tc_opt_nc.res_list = res_list.copy()
tc_opt_nc.res_len = len(res_list['x'])

print 'FP = ' + str(res_list['fp'])
print 'FM = ' + str(res_list['fm'])
print 'number of bins = %d'%tc.numBin

for k in range(ITER_NUM):
     # not saving results
     # use partial sum instead of Monte Carlo
    tc_opt_nc.channel_iterate(ITER_CHANNEL,INTER_STEPS = INTER_STEPS, ADD_TIME = False,
                              ADD_INEQ_CONS = True, 
                              USE_MC = False,
                              SUM_THRESHOLD_INFO = 50, SUM_THRESHOLD_GRAD = 50, NUM_THREADS = NUM_THREADS) #MC_ITER_INFO = 1e5, MC_ITER_GRAD = 1e5
    tc_opt_nc.capacity_iterate(ITER_CAPACITY, INTER_STEPS = INTER_STEPS, 
                               ADD_INEQ_CONS = True,
                               USE_MC = False,
                               SUM_THRESHOLD_BA = 50, 
                               ADD_TIME = False, NUM_THREADS = NUM_THREADS)


# tc_opt_nc.plot_info()

tc_opt_nc.save_res_list(FILE_NAME = 'data/test1_noncyclic_conti', ADD_TIME = True)

#tc_opt_nc.plot_animation(FILE_NAME = 'data/test1_noncyclic_conti', ADD_TIME = True)

#tc_opt_nc.plot_animation(FILE_NAME = 'data/test1_noncyclic_conti_cube', ADD_TIME = True, IF_CUBE = True)
