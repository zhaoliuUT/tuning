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


numBin = 16
delta = 1

nu = 4
tau = 1.0
stimWidth = int(nu*tau) # tau maximum takes 4

FP = 0.9
FM = 0.1

# initialize stim function
stim = np.zeros(numBin)
stim[0:stimWidth] =  1./stimWidth
stim[stimWidth:] =  0
# stim[0:stimWidth] = np.exp(-np.arange(stimWidth))
# stim /=np.sum(stim)
# plt.plot(stim)
numPop = 1
tuning0 = np.random.uniform(FM, FP, numBin)

# numPop = 2
# tuning0 = np.zeros((numPop, numBin))
# tuning0[0] = np.linspace(FP, FM,numBin)
# tuning0[1] = np.linspace(FM, FP,numBin)

# -----------Poisson Model-----------
curr_time = time.time()
tc = TuningCurve(tuning0, stim, nu, tau, delta, MC_ITER = 1e5)
print("computation time = %.4f"%(time.time() - curr_time))
print("mutual information = %.4f"%tc.info)
# tc.plot()

tc_opt = TuningCurveOptimizer(tc, FP, FM, MC_ITER_INFO = 1e5, MC_ITER_GRAD = 1e5)
tc_opt.iterate(10, FILE_NAME = 'data/test0', ADD_TIME = False)

# Plot and save animations 
tc_opt.plot_animation(FILE_NAME = 'data/test0', ADD_TIME = False)

# -----------Binary Model-----------
curr_time = time.time()
bn_tc = TuningCurve_BN(tuning0, stim, nu, tau, delta, MC_ITER = 1e5)
print("computation time = %.4f"%(time.time() - curr_time))
print("mutual information = %.4f"%bn_tc.info)

bn_tc_opt = TuningCurveOptimizer_BN(bn_tc, FP, FM, MC_ITER_INFO = 1e5, MC_ITER_GRAD = 1e5)
bn_tc_opt.iterate(10, FILE_NAME = 'data/test0_bn',ADD_TIME = False)

# Plot and save animations 
bn_tc_opt.plot_animation(FILE_NAME = 'data/test0_bn', ADD_TIME = False)

# To continue iteration:
# tc_opt = TuningCurveOptimizer.load_res_list("data/test0")
# tc_opt.iterate(15, INTER_STEPS = 5, FILE_NAME = 'data/test0', ADD_TIME = False)
