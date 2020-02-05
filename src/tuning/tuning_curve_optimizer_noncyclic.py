import time, sys, os, copy
import numpy as np
import numpy.linalg as la
import scipy
from scipy import optimize
from functools import partial

from cyMINoncyclic import *
from anim_3dcube import *
from tuning.tuning_curve_noncyclic import *

class TuningCurveOptimizer_Noncyclic:
    """TuningCurveOptimizer_Noncyclic Class
    # Poisson Distribution
    Attributes:
        Properties: numNeuro, numBin, conv, tau
        Optimization Information: fp, fm, tuning0, grad0, info0, average_cons, bounds
        Result Storage: res_list, res_len, num_iter, inter_steps [these 4 attributes can change]
        
    Methods:
        __init__(self,
                 TuningCurve_init, # initial input (TuningCurve_Noncyclic object)
                 fp, # upper bound (vector of size numNeuro or constant)
                 fm, # lower bound (vector of size numNeuro or constant)
                 average_cons = None, # average constraints (vector of size numNeuro or constant) # upper bounds
                 )
                 
        # NUM_THREADS, USE_MC, MC_ITER_INFO, MC_ITER_GRAD , MC_ITER_BA, 
        # SUM_THRESHOLD_INFO, SUM_THRESHOLD_GRAD, SUM_THRESHOLD_BA: stored in res_list.
        channel_iterate(self,
                        NUM_ITER = 1, 
                        INTER_STEPS = 1, 
                        ADD_EQ_CONS = False,
                        ADD_INEQ_CONS = False,
                        NUM_THREADS = 8,
                        USE_MC = True,
                        MC_ITER_INFO = 1e5,
                        MC_ITER_GRAD = 1e5, 
                        SUM_THRESHOLD_INFO = 50, 
                        SUM_THRESHOLD_GRAD = 50,
                        PRINT = True, 
                        FILE_NAME = "", 
                        ADD_TIME = True,
                        ftol = 1e-15, 
                        disp = False)
                        
        capacity_iterate(self,
                         initial_slope = 0, #initialized slope, positive number or numpy array
                         capacity_tol = 1e-3, # tolerance for reaching capacity
                         slope_stepsize = 1e-2, # stepsize for slope iteration
                         constraint_tol = 1e-3, # tolerance for reaching the constraints
                         MAX_ITER = 1, # max number of iterations
                         EVAL_STEPS = 1, # mutual information evaluation every EVAL_STEPS steps.
                         ADD_EQ_CONS = False,
                         ADD_INEQ_CONS = False,
                         NUM_THREADS = 8,
                         USE_MC = True,
                         MC_ITER_BA = 1e5,
                         SUM_THRESHOLD_BA = 50,
                         PRINT = True,
                         FILE_NAME = "",
                         ADD_TIME = True)

        plot_res_id(self, i, ALL = True)
        
        plot_animation(self, FILE_NAME = "", ADD_TIME = True, XTICKS_IDX_LIST = [], \
                       ALIGN = "row", INCLUDE_RATE = False, INCLUDE_GRAD = False, INCLUDE_INFO = True, \
                       interval = 1000, dt = 1, IF_CUBE = False)
        
        plot_info(self, TITLE = "", maker = '.')
        save_res_list(self, FILE_NAME = "", ADD_TIME = True)
    Static Methods:
        load_res_list(filename) # Usage: TuningCurveOptimizer_Noncyclic.load_res_list(filename)
         
    """
    def __init__(self,
                 TuningCurve_init, # initial input (TuningCurve_Noncyclic object)
                 fp, # upper bound (vector of size numNeuro or constant)
                 fm, # lower bound (vector of size numNeuro or constant)
                 average_cons = None, # average constraints (vector of size numNeuro or constant) # upper bounds
                ):
        
        #self.TuningCurve_init = copy.copy(TuningCurve_init)
        self.numNeuro = TuningCurve_init.numNeuro # 2
        self.numBin = TuningCurve_init.numBin 
        self.conv = TuningCurve_init.conv.copy()
        self.tau = TuningCurve_init.tau
        
        if isinstance(fp, (int, float, complex)) or isinstance(fm, (int, float, complex)): # is number
            fp = fp*np.ones(self.numNeuro)
            fm = fm*np.ones(self.numNeuro)
        fp = np.array(fp)  # numpy array
        fm = np.array(fm)
        if fp.size and fm.size: # nonempty
            if fp.size != self.numNeuro or fm.size != self.numNeuro:
                raise Exception('Dimension mismatch for fp/fm!')
            if np.any(fp < 0) or np.any(fm <0) or np.any(fm > fp):
                raise Exception('Wrong input for fp/fm!')
        else:
            raise Exception('Missing input for fp, fm!')
            
        if average_cons is not None:
            if isinstance(average_cons, (int, float, complex)): # is number
                average_cons = average_cons*np.ones(self.numNeuro)
            average_cons = np.array(average_cons)  # numpy array
            if average_cons.size != self.numNeuro:
                raise Exception('Dimension mismatch for average constraints!')
            if np.any(average_cons < 0):
                raise Exception('Wrong input for average_cons: must be non-negative!')
            if np.any(average_cons < np.dot(TuningCurve_init.tuning, TuningCurve_init.weight)):
                raise Exception('Wrong input for average_cons: must be >= initial average!')


        self.fp = fp.copy() # might be None
        self.fm = fm.copy() # might be None
        
        self.tuning0 = TuningCurve_init.tuning.copy()         
        self.weight0 = TuningCurve_init.weight.copy()
        self.grad0 = TuningCurve_init.grad.copy()        
        self.info0 = TuningCurve_init.info
        # use total average instead of individual average for each neuro tuning curve
        self.average_cons = average_cons.copy() if average_cons is not None else None
        #np.dot(self.tuning0, self.weight0) # self.average_cons = np.average(TuningCurve_init.tuning)
        
        self.bounds = []
        for j in range(self.numNeuro):
            for i in range(self.numBin):
                self.bounds += [(fm[j], fp[j])]
        self.bounds = tuple(self.bounds)
        
        res_list = {'x':[self.tuning0.copy()],'grad':[self.grad0.copy()],'obj':[self.info0],\
                    'weight': [self.weight0.copy()],\
                    'slope': [None],\
                    'success':[1],'status':[1],'nfev':[0],'njev':[0]}
        res_list.update({'numNeuro': self.numNeuro, 'average': self.average_cons,\
                         'tau': self.tau, 'conv':self.conv.copy(), \
                         'fp': copy.copy(self.fp), 'fm': copy.copy(self.fm),\
                         'USE_MC': [None], 'NUM_THREADS': [0], \
                         'MC_ITER_INFO': [0], 'MC_ITER_GRAD': [0], 'MC_ITER_BA': [0], \
                         'SUM_THRESHOLD_INFO': [0], 'SUM_THRESHOLD_GRAD': [0], 'SUM_THRESHOLD_BA': [0],\
                         'num_iter': [0], 'inter_steps':[0]})
        self.res_list = res_list # result list
        self.res_len = 1 # result list length
        #self.num_iter = [0] # number of iterations
        #self.inter_steps = [0] # default
        
        
        
    def channel_iterate(self,
                        NUM_ITER = 1, 
                        INTER_STEPS = 1, 
                        ADD_EQ_CONS = False,
                        ADD_INEQ_CONS = False,
                        NUM_THREADS = 8, 
                        USE_MC = True,
                        MC_ITER_INFO = 1e5, MC_ITER_GRAD = 1e5,
                        SUM_THRESHOLD_INFO = 50, SUM_THRESHOLD_GRAD = 50,
                        PRINT = True, FILE_NAME = "", \
                        ADD_TIME = True, ftol = 1e-15, disp = False):
        # INTER_STEPS: intermediate steps for printing and saving
        # NUM_ITER: total number of iterations
        # number of plotting/saving: NUM_ITER/INTER_STEPS
        
        if self.average_cons is None and (ADD_EQ_CONS or ADD_INEQ_CONS):
            raise Exception('Must input average constraint!')

        tmpgrad = np.zeros_like(self.grad0) # just for temporary storage


        def opt_fun(x, weight, self):
            if USE_MC:
                return -mc_mean_grad_noncyclic(tmpgrad, x.reshape((self.numNeuro,self.numBin)), weight,
                                               self.conv, self.tau, 
                                               MC_ITER_INFO, NUM_THREADS)
            else:
                return -partial_sum_mean_grad_noncyclic(tmpgrad, x.reshape((self.numNeuro,self.numBin)), weight,
                                                       self.conv, self.tau,
                                                       SUM_THRESHOLD_INFO, NUM_THREADS)
        def grad_fun(x,weight, self): 
            x_grad = np.zeros((self.numNeuro,self.numBin), dtype = np.float)
            if USE_MC:
                x_mean = mc_mean_grad_noncyclic(x_grad, x.reshape((self.numNeuro,self.numBin)), weight,
                                                self.conv, self.tau,
                                                MC_ITER_GRAD, NUM_THREADS)
            else:
                x_mean = partial_sum_mean_grad_noncyclic(x_grad, x.reshape((self.numNeuro,self.numBin)), weight, 
                                                        self.conv, self.tau,
                                                        SUM_THRESHOLD_INFO, NUM_THREADS)
            return -x_grad.reshape(-1)
        

        
        curr_tuning = self.res_list['x'][-1].reshape(-1).copy()
        curr_weight = self.res_list['weight'][-1].copy()
        #plt.plot(self.res_list['x'][-1][0])
        #plt.plot(self.res_list['x'][-1][1])
        #plt.show()

        curr_num_iter = sum(self.res_list['num_iter'])
        
        if PRINT:
            print('{0:4s}   {1:9s}  {2:9s}  {3:9s}'.format('Iter', 'Mutual Information', 'Capacity Gap', 'Average'))
            curr_avg = np.dot(self.res_list['x'][-1], curr_weight)
            print('{0:4d}   {1: 3.6f} {2:15s}  {3:6s}'.format(curr_num_iter,self.res_list['obj'][-1], ' ', ' ')
                  + str(curr_avg)
                 )
            #curr_avg = np.average(np.dot(tc_opt_nc.res_list['x'][-1], curr_weight))
            #print('{0:4d}   {1: 3.6f} {2:15s}  {3:6s}  {4:3.6f}'.format(0,tc_opt_nc.res_list['obj'][-1], ' ', '+', curr_avg))


        my_cons = []
        if ADD_EQ_CONS:
            if self.numNeuro == 1:
                my_cons += [{'type':'eq', 'fun': lambda x: (self.average_cons - np.dot(x, curr_weight))}]
            else:
                def constraint_fun(x, n):
                    vec = x.reshape(self.numNeuro, -1)
                    return self.average_cons[n] - np.dot(vec[n], curr_weight)
                my_cons += [{'type':'eq', 'fun': partial(constraint_fun, n = i)} for i in range(self.numNeuro)]
        if ADD_INEQ_CONS:
            #inequality means that it is to be non-negative.
            # current average <= defined average constraint.
            if self.numNeuro == 1:
                my_cons += [{'type':'ineq', 'fun': lambda x: (self.average_cons - np.dot(x, curr_weight))}]
            else:
                def constraint_fun(x, n):
                    vec = x.reshape(self.numNeuro, -1)
                    return self.average_cons[n] - np.dot(vec[n], curr_weight)
                my_cons += [{'type':'ineq', 'fun': partial(constraint_fun, n = i)} for i in range(self.numNeuro)]

            
        for i in range(int(NUM_ITER/INTER_STEPS)):
            if ADD_EQ_CONS or ADD_INEQ_CONS:
                res = optimize.minimize(lambda x, self: opt_fun(x, curr_weight, self), \
                                        curr_tuning, method='SLSQP', args = self, \
                                        jac = lambda x, self:grad_fun(x, curr_weight, self), \
                                        bounds = self.bounds, constraints = my_cons, \
                                        options = {'maxiter':INTER_STEPS, 'ftol': ftol, 'disp': disp})
            else: # no averge constraints                
                 res = optimize.minimize(lambda x, self: opt_fun(x, curr_weight, self), \
                                        curr_tuning, method='SLSQP', args = self, \
                                        jac = lambda x, self:grad_fun(x, curr_weight, self), \
                                        bounds = self.bounds,\
                                        options = {'maxiter':INTER_STEPS, 'ftol': ftol, 'disp': disp})
        
            curr_tuning = res['x'].copy()
            # FIXME: judge whether the iteration stops before reaching maxiter.
            # actual number of iterations is res['nit'] - 1
            curr_num_iter += INTER_STEPS
            #print curr_tuning.shape
            # self.num_iter += INTER_STEPS
            self.res_list['x'].append(res['x'].reshape((self.numNeuro, self.numBin)).copy())
            self.res_list['grad'].append(-res['jac'][0:self.numNeuro*self.numBin].reshape((self.numNeuro, self.numBin)).copy())
            self.res_list['obj'].append(-res['fun'])            
            self.res_list['success'].append(res['success'])
            self.res_list['status'].append(res['status'])
            self.res_list['nfev'].append(res['nfev'])
            self.res_list['njev'].append(res['njev'])
            self.res_list['weight'].append(curr_weight.copy())
            self.res_list['slope'].append(None)
            if PRINT:
                curr_avg = np.dot(self.res_list['x'][-1], curr_weight)
                print('{0:4d}   {1: 3.6f} {2:15s}  {3:6s}'.format(curr_num_iter,-res['fun'], ' ', ' ')
                      + str(curr_avg)
                     )

                # print res['jac'].shape # 
        # save result list by default
        self.res_len = len(self.res_list['x'])
        #self.num_iter
        #self.inter_steps.append(INTER_STEPS)
        self.res_list['num_iter'].append(NUM_ITER)
        self.res_list['inter_steps'].append(INTER_STEPS)
        self.res_list['NUM_THREADS'].append(NUM_THREADS)
        self.res_list['USE_MC'].append(USE_MC)
        if USE_MC:
            self.res_list['MC_ITER_INFO'].append(MC_ITER_INFO)
            self.res_list['MC_ITER_GRAD'].append(MC_ITER_GRAD)
            self.res_list['SUM_THRESHOLD_INFO'].append(0)
            self.res_list['SUM_THRESHOLD_GRAD'].append(0)
        else:
            self.res_list['MC_ITER_INFO'].append(0)
            self.res_list['MC_ITER_GRAD'].append(0)
            self.res_list['SUM_THRESHOLD_INFO'].append(SUM_THRESHOLD_INFO)
            self.res_list['SUM_THRESHOLD_GRAD'].append(SUM_THRESHOLD_GRAD)
        self.res_list['MC_ITER_BA'].append(0)
        self.res_list['SUM_THRESHOLD_BA'].append(0)
        if FILE_NAME != "" or ADD_TIME == True:
            self.save_res_list(FILE_NAME, ADD_TIME)
        
   
    def capacity_iterate(self,
                         initial_slope = 0, #initialized slope, positive number or numpy array
                         capacity_tol = 1e-3, # tolerance for reaching capacity
                         slope_stepsize = 1e-2, # stepsize for slope iteration
                         constraint_tol = 1e-3, # tolerance for reaching the constraints
                         MAX_ITER = 1, # max number of iterations
                         EVAL_STEPS = 1, # mutual information evaluation every EVAL_STEPS steps.
                         ADD_EQ_CONS = False, ADD_INEQ_CONS = False,\
                         NUM_THREADS = 8, \
                         USE_MC = True,\
                         MC_ITER_BA = 1e5, SUM_THRESHOLD_BA = 50,\
                         PRINT = True, FILE_NAME = "", ADD_TIME = True):

        if isinstance(initial_slope, (int, float, complex)): # is number
            initial_slope = initial_slope*np.ones(self.numNeuro)
        initial_slope = np.array(initial_slope)  # numpy array
        if initial_slope.size != self.numNeuro:
            raise Exception('Dimension mismatch for the initial slope!')
        if np.any(initial_slope < 0):
            raise Exception('The initial slope must be non-negative!')

        if self.average_cons is None and (ADD_EQ_CONS or ADD_INEQ_CONS):
            raise Exception('Must input average constraint!')

        if ADD_EQ_CONS:
            raise Exception("Blahut-Arimoto with equality constraints: not implemented yet.")
        if ADD_INEQ_CONS:
            #inequality means that it is to be non-negative.
            # current average <= defined average constraint.
            if self.numNeuro == 1:
                constraint_ineq = lambda x, w: (self.average_cons - np.dot(x, w))
            else:
                constraint_ineq = lambda x, w: (self.average_cons - np.dot(x.reshape(self.numNeuro, -1), w))

        curr_tuning = self.res_list['x'][-1].copy()
        curr_weight = self.res_list['weight'][-1].copy()
        
        curr_num_iter = sum(self.res_list['num_iter'])
        tmpgrad = np.zeros_like(self.grad0) # just for temporary storage
        weight_new = np.zeros_like(curr_weight)
        
        if ADD_INEQ_CONS:
            curr_slope = initial_slope.copy()
        else:
            curr_slope = np.zeros(self.numNeuro)
        slope_new = np.zeros_like(curr_slope)
        coeff = np.zeros(self.numBin)

        curr_x = curr_tuning
        if PRINT:
            curr_avg = np.dot(self.res_list['x'][-1], curr_weight)
            print('{0:4d}   {1: 3.6f} {2:15s}  {3:6s}'.format(curr_num_iter,self.res_list['obj'][-1], ' ', ' ')
                  + str(curr_avg)
                 )
        
        halt_during_iter = False
        for k in range(MAX_ITER):
            # update coefficients
            if USE_MC:
                mc_coeff_arimoto(coeff, curr_x, curr_weight, curr_slope, 
                                 self.conv, self.tau, MC_ITER_BA, NUM_THREADS)
            else:
                partial_sum_coeff_arimoto(coeff, curr_x, curr_weight, curr_slope,
                                           self.conv, self.tau, SUM_THRESHOLD_BA, NUM_THREADS)
            # judge the capacity convergence, adjust weights and parameters(slope)

            weight_new = curr_weight.copy()
            slope_new = curr_slope.copy()

            capacity_gap = np.log(np.max(coeff)) - np.sum(curr_weight*np.log(coeff))
            change_slope = False
            if capacity_gap > capacity_tol:
                # update weights
                weight_new = curr_weight*coeff / np.sum(curr_weight*coeff)
            else:
                if ADD_INEQ_CONS:
                    for p in range(self.numNeuro):
                        if np.dot(curr_weight, curr_x[p,:]) > self.average_cons[p] + constraint_tol:
                            slope_new[p] = curr_slope[p] + slope_stepsize
                            change_slope = True
                        elif np.dot(curr_weight, curr_x[p,:]) < self.average_cons[p] - constraint_tol and \
                        curr_slope[p] > slope_stepsize:
                            slope_new[p] = curr_slope[p] - slope_stepsize
                            change_slope = True
                if not change_slope:
                    halt_during_iter = True
                    break

            # evaluate mutual information
            if (k+1)%EVAL_STEPS == 0 or k == MAX_ITER -1 or halt_during_iter:
                if USE_MC:
                    mean_new = mc_mean_grad_noncyclic(
                        tmpgrad, curr_x, weight_new, self.conv, self.tau, MC_ITER_BA, NUM_THREADS)
                else:
                    mean_new = partial_sum_mean_grad_noncyclic(
                        tmpgrad, curr_x, weight_new, self.conv, self.tau, SUM_THRESHOLD_BA, NUM_THREADS)
                if PRINT:
                    info_str = '%.2e'%capacity_gap
                    if change_slope: #np.any(np.fabs(slope_new - curr_slope) > 1e-4):
                        info_str += ' S ' # slope change
                    curr_avg = np.dot(curr_x, curr_weight)
                    print('{0:4d}   {1: 3.6f} {2:15s}{3:6s}'.format(curr_num_iter,mean_new, ' ', info_str)
                          + str(curr_avg)
                         )
            else:
                mean_new = None
                tmpgrad = None
            # save results
            self.res_list['x'].append(curr_x.copy())
            self.res_list['grad'].append(tmpgrad.copy())
            self.res_list['obj'].append(mean_new)
            self.res_list['success'].append(1)
            self.res_list['status'].append(1)
            self.res_list['nfev'].append(1)
            self.res_list['njev'].append(1)
            self.res_list['weight'].append(curr_weight.copy())
            self.res_list['slope'].append(curr_slope.copy())
            curr_weight = weight_new.copy()
            curr_slope = slope_new.copy()
            curr_num_iter += 1

        
        # save result list by default
        self.res_len = len(self.res_list['x'])
        if halt_during_iter:
            self.res_list['num_iter'].append(k) # actual number of iterations = k
        else:
            self.res_list['num_iter'].append(MAX_ITER)
        self.res_list['inter_steps'].append(1) # actual inter_steps taken.
        self.res_list['NUM_THREADS'].append(NUM_THREADS)
        self.res_list['USE_MC'].append(USE_MC)
        self.res_list['MC_ITER_INFO'].append(0)
        self.res_list['MC_ITER_GRAD'].append(0)
        self.res_list['SUM_THRESHOLD_INFO'].append(0)
        self.res_list['SUM_THRESHOLD_GRAD'].append(0)
        if USE_MC:
            self.res_list['MC_ITER_BA'].append(MC_ITER_BA)
            self.res_list['SUM_THRESHOLD_BA'].append(0)
        else:
            self.res_list['MC_ITER_BA'].append(0)
            self.res_list['SUM_THRESHOLD_BA'].append(SUM_THRESHOLD_BA)
        if FILE_NAME != "" or ADD_TIME == True:
            self.save_res_list(FILE_NAME, ADD_TIME)
            

    def save_res_list(self, FILE_NAME = "", ADD_TIME = True):
        if ADD_TIME:
            timestr = time.strftime("%m%d-%H%M%S")
        else:
            timestr = ""
        # e.g.filename = 'data1/test1/Pop=%d_1'%numNeuro
        filename = FILE_NAME + timestr
        # self.FILE_NAME = filename             
        directory = os.path.dirname(filename)
        if directory != '':
            try:
                os.stat(directory)
            except:
                os.makedirs(directory)    
        np.save(filename,self.res_list)
    
    def plot_res_id(self, i, ALL = True):
        # plot the i-th tuning curve in the  res_list
        if self.res_list['obj'][i] is not None:
            tuning_curve = TuningCurve_Noncyclic(self.res_list['x'][i], self.res_list['weight'][i],\
                                                 self.conv, self.tau,\
                                                 self.res_list['obj'][i], self.res_list['grad'][i])
        else:
            tuning_curve = TuningCurve_Noncyclic(self.res_list['x'][i], self.res_list['weight'][i],\
                                                 self.conv, self.tau)
        tuning_curve.plot(ALL)
        
        
        
    def plot_animation(self, FILE_NAME = "", ADD_TIME = True, interval = 1000, IF_CUBE = False, EVALUATE_ALL = False,
                       dt = 1, XTICKS_IDX_LIST = [], ALIGN = "row", # options for non-cube animation
                       INCLUDE_RATE = False, INCLUDE_GRAD = False, INCLUDE_INFO = True, # options for non-cube animation
                       INCLUDE_WEIGHT=True,# option for both
                       INCLUDE_FUN = True, INCLUDE_WEIGHT_BAR = True, # options for cube animation
                       weight_tol = 1e-3, weight_format = '%.2f',
                       cmap_name = 'nipy_spectral', shuffle_colors=False,# options for cube animation
                       ):
        tc_list = []
        # evaluate all the mutual information for every tuning curve during iterations
        for i in range(self.res_len):
            if self.res_list['obj'][i] is not None:
                tc = TuningCurve_Noncyclic(self.res_list['x'][i], self.res_list['weight'][i],\
                                           self.conv, self.tau,\
                                           self.res_list['obj'][i], self.res_list['grad'][i])
                tc_list.append(tc)
            elif EVALUATE_ALL:
                tc = TuningCurve_Noncyclic(self.res_list['x'][i], self.res_list['weight'][i],\
                                           self.conv, self.tau)
                tc_list.append(tc)
        # saving tc_list?
        
        steps_list = []
        pos = 0
        num_iter = self.res_list['num_iter']
        for i in range(1, len(num_iter)):
            steps_list += list(np.arange(pos, pos+num_iter[i], self.res_list['inter_steps'][i]))
            pos += num_iter[i]
        steps_list += [pos]
        steps_list = np.array(steps_list)
        if not EVALUATE_ALL:
            info_arr = np.array(self.res_list['obj'])
            steps_list = steps_list[info_arr != None]
    
        if IF_CUBE:
            anim = TuningCurve_Noncyclic.animation_tc_list_cube(
                tc_list, FILE_NAME = FILE_NAME, ADD_TIME = ADD_TIME, interval = interval,
                INCLUDE_FUN = INCLUDE_FUN, INCLUDE_WEIGHT=INCLUDE_WEIGHT,
                INCLUDE_WEIGHT_BAR = INCLUDE_WEIGHT_BAR, 
                weight_tol = weight_tol, weight_format = weight_format, 
                cmap_name = cmap_name, shuffle_colors = shuffle_colors,
            )
        else:
            anim = TuningCurve_Noncyclic.animation_tc_list(
                tc_list, FILE_NAME, ADD_TIME, FP = self.fp, FM = self.fm, interval= interval, dt = dt,
                index_list = list(steps_list), XTICKS_IDX_LIST = XTICKS_IDX_LIST, VAR_LABEL =  "",
                VAR_TXT_LIST = [], ALIGN = ALIGN, INCLUDE_RATE = INCLUDE_RATE,
                INCLUDE_GRAD = INCLUDE_GRAD, INCLUDE_INFO = INCLUDE_INFO, INCLUDE_WEIGHT=INCLUDE_WEIGHT,
            )
        return anim

        
        
    def plot_info(self, TITLE = "", marker = '.'):
        steps_list = []
        pos = 0
        num_iter = self.res_list['num_iter']
        for i in range(1, len(num_iter)):
            steps_list += list(np.arange(pos, pos+num_iter[i], self.res_list['inter_steps'][i]))
            pos += num_iter[i]
        steps_list += [pos]
        steps_list = np.array(steps_list)
        info_arr = np.array(self.res_list['obj'])
        plt.plot(steps_list[info_arr!=None], info_arr[info_arr !=None], marker = marker)
        plt.xlabel('number of steps')
        plt.ylabel('Mutual Information')
        plt.title(TITLE)
        
        
    @staticmethod 
    def load_res_list(filename):
        """
        Load res_list of a TuningCurveOptimizer_Noncyclic Object
        return a TuningCurveOptimizer_Noncyclic Object
        """
        
        
        res_list = np.load(filename+'.npy', allow_pickle=True).item()
        
        tc = TuningCurve_Noncyclic(res_list['x'][0], res_list['weight'][0], res_list['conv'], res_list['tau'], \
                                   res_list['obj'][0], res_list['grad'][0]) 
        
        tc_opt = TuningCurveOptimizer_Noncyclic(tc, res_list['fp'], res_list['fm'])
        
        tc_opt.res_list = res_list.copy()
        tc_opt.res_len = len(res_list['x'])
        return tc_opt
