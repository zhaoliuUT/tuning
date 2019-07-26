import time, sys, os, copy
import numpy as np
import numpy.linalg as la
import scipy
from scipy import optimize

from cyMINoncyclic import *
from anim_3dcube import *
from tuning_curve_noncyclic import *

class TuningCurveOptimizer_Noncyclic:
    """TuningCurveOptimizer_Noncyclic Class
    # Poisson Distribution
    Attributes:
        Properties: numNeuro, numBin, conv, tau
        Optimization Information: fp, fm, tuning0, grad0, info0, average, cons, bounds
        Result Storage: res_list, res_len, num_iter, inter_steps [these 4 attributes can change]
        
    Methods:
        __init__(self,
                 TuningCurve_init, # initial input (TuningCurve_Noncyclic object)
                 fp, # upper bound (vector of size numNeuro or constant)
                 fm) # lower bound (vector of size numNeuro or constant)
                 
        # NUM_THREADS, USE_MC, MC_ITER_INFO, MC_ITER_GRAD , MC_ITER_BA, 
        # SUM_THRESHOLD_INFO, SUM_THRESHOLD_GRAD, SUM_THRESHOLD_BA: stored in res_list.
                        
        channel_iterate(self, 
                        NUM_ITER = 1, 
                        INTER_STEPS = 1, 
                        ADD_CONS = False,
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
                        NUM_ITER = 1, 
                        INTER_STEPS = 1,
                        ADD_BA_CONS = False, 
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
        
        plot_info(self, TITLE = "")
        save_res_list(self, FILE_NAME = "", ADD_TIME = True)
    Static Methods:
        load_res_list(filename) # Usage: TuningCurveOptimizer_Noncyclic.load_res_list(filename)
         
    """
    def __init__(self,
                 TuningCurve_init, # initial input (TuningCurve_Noncyclic object)
                 fp, # upper bound (vector of size numNeuro or constant)
                 fm): # lower bound (vector of size numNeuro or constant)
        
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
            
            
        self.fp = fp.copy() # might be None
        self.fm = fm.copy() # might be None
        
        self.tuning0 = TuningCurve_init.tuning.copy()         
        self.weight0 = TuningCurve_init.weight.copy()
        self.grad0 = TuningCurve_init.grad.copy()        
        self.info0 = TuningCurve_init.info
        # use total average instead of individual average for each neuro tuning curve
        self.average = np.average(TuningCurve_init.tuning) # TuningCurve_init.average.copy()
        
        self.bounds = []
        for j in range(self.numNeuro):
            for i in range(self.numBin):
                self.bounds += [(fm[j], fp[j])]
        self.bounds = tuple(self.bounds)
        
        res_list = {'x':[self.tuning0.copy()],'grad':[self.grad0.copy()],'obj':[self.info0],\
                    'weight': [self.weight0.copy()],\
                    'success':[1],'status':[1],'nfev':[0],'njev':[0]}
        res_list.update({'numNeuro': self.numNeuro, 'average': self.average,\
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
                        ADD_CONS = False, 
                        NUM_THREADS = 8, 
                        USE_MC = True,
                        MC_ITER_INFO = 1e5, MC_ITER_GRAD = 1e5,
                        SUM_THRESHOLD_INFO = 50, SUM_THRESHOLD_GRAD = 50,
                        PRINT = True, FILE_NAME = "", \
                        ADD_TIME = True, ftol = 1e-15, disp = False):
        # INTER_STEPS: intermediate steps for printing and saving
        # NUM_ITER: total number of iterations
        # number of plotting/saving: NUM_ITER/INTER_STEPS
        
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
            print('{0:4s}   {1:9s}  {2:9s}'.format('Iter', 'Mutual Information', 'Weight Change'))
            print('{0:4d}   {1: 3.6f}'.format(curr_num_iter,self.res_list['obj'][-1]))

        if ADD_CONS:   
            def constraint_eq(x):
                return np.average(x) - self.average
            my_cons = [{'type':'eq', 'fun': constraint_eq}]
            
        for i in range(int(NUM_ITER/INTER_STEPS)):
            if ADD_CONS:
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
            if PRINT:
                print('{0:4d}   {1: 3.6f} '.format(curr_num_iter,-res['fun']))
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
        #self.save_res_list(FILE_NAME, ADD_TIME)
        
    

        
   
    def capacity_iterate(self, NUM_ITER = 1, INTER_STEPS = 1,\
                         ADD_BA_CONS = False, NUM_THREADS = 8, \
                         USE_MC = True,\
                         MC_ITER_BA = 1e5, SUM_THRESHOLD_BA = 50,\
                         PRINT = True, FILE_NAME = "", ADD_TIME = True):
        
        # NUM_ITER: number of iterations
        if ADD_BA_CONS == True:
            raise Exception("Blahut-Arimoto with constraints: not implemented yet.")

        curr_tuning = self.res_list['x'][-1].copy()
        curr_weight = self.res_list['weight'][-1].copy()
        
        curr_num_iter = sum(self.res_list['num_iter'])
        tmpgrad = np.zeros_like(self.grad0) # just for temporary storage
        weight_new = np.zeros_like(curr_weight)
        
        curr_x = curr_tuning
        if PRINT:
            #print  '{0:4s}   {1:9s}  {2:9s}'.format('Iter', 'Mutual Information', 'Weight Change') 
            print('{0:4d}   {1: 3.6f}'.format(curr_num_iter,self.res_list['obj'][-1]))
        
        for k in range(NUM_ITER):
            if USE_MC:
                mc_prob_arimoto(weight_new, curr_x, curr_weight, self.conv, self.tau, 
                                MC_ITER_BA, NUM_THREADS)
                mean_new = mc_mean_grad_noncyclic(tmpgrad, curr_x, weight_new,
                                                  self.conv, self.tau, MC_ITER_BA, NUM_THREADS)
            else:
                partial_sum_prob_arimoto(weight_new, curr_x, curr_weight, self.conv, self.tau, 
                                         SUM_THRESHOLD_BA, NUM_THREADS)
                mean_new = partial_sum_mean_grad_noncyclic(tmpgrad, curr_x, weight_new,
                                                           self.conv, self.tau, SUM_THRESHOLD_BA, NUM_THREADS)
            self.res_list['x'].append(curr_x.copy())
            self.res_list['grad'].append(tmpgrad.copy())
            self.res_list['obj'].append(mean_new)
            self.res_list['success'].append(1)
            self.res_list['status'].append(1)
            self.res_list['nfev'].append(1)
            self.res_list['njev'].append(1)
            self.res_list['weight'].append(weight_new.copy())
            curr_weight = weight_new.copy()
            curr_num_iter += 1
            if PRINT:
                if (k+1)%INTER_STEPS == 0:
                    print('{0:4d}   {1: 3.6f}             +'.format(curr_num_iter,mean_new))      
        
        # save result list by default
        self.res_len = len(self.res_list['x'])
        self.res_list['num_iter'].append(NUM_ITER)
        self.res_list['inter_steps'].append(1)
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
        tuning_curve = TuningCurve_Noncyclic(self.res_list['x'][i], self.res_list['weight'][i],\
                                             self.conv, self.tau,\
                                             self.res_list['obj'][i], self.res_list['grad'][i])
        tuning_curve.plot(ALL)
        
        
        
    def plot_animation(self, FILE_NAME = "", ADD_TIME = True, XTICKS_IDX_LIST = [], \
                       ALIGN = "row", INCLUDE_RATE = False, INCLUDE_GRAD = False, INCLUDE_INFO = True, \
                       interval = 1000, dt = 1, IF_CUBE = False):
        tc_list = []
        for i in range(self.res_len):
            tc = TuningCurve_Noncyclic(self.res_list['x'][i], self.res_list['weight'][i],\
                                       self.conv, self.tau,\
                                       self.res_list['obj'][i], self.res_list['grad'][i])
            tc_list.append(tc)
        
        steps_list = []
        pos = 0
        num_iter = self.res_list['num_iter']
        for i in range(1, len(num_iter)):
            steps_list += list(np.arange(pos, pos+num_iter[i], self.res_list['inter_steps'][i]))
            pos += num_iter[i]
        steps_list += [pos]
        
        if IF_CUBE:
            TuningCurve_Noncyclic.animation_tc_list_cube(tc_list, INCLUDE_FUN = True, \
                                                         FILE_NAME = FILE_NAME, ADD_TIME = ADD_TIME, \
                                                         interval = interval)
        else:
            TuningCurve_Noncyclic.animation_tc_list(tc_list, FILE_NAME, ADD_TIME, FP = self.fp, FM = self.fm, \
                                                XTICKS_IDX_LIST = XTICKS_IDX_LIST, \
                                                VAR_LABEL =  "", VAR_TXT_LIST = [], \
                                                ALIGN = ALIGN, INCLUDE_RATE = INCLUDE_RATE, \
                                                INCLUDE_GRAD = INCLUDE_GRAD, INCLUDE_INFO = INCLUDE_INFO,\
                                                index_list = steps_list, interval= interval, dt = dt)

        
        
        
        
    def plot_info(self, TITLE = ""):
        steps_list = []
        pos = 0
        num_iter = self.res_list['num_iter']
        for i in range(1, len(num_iter)):
            steps_list += list(np.arange(pos, pos+num_iter[i], self.res_list['inter_steps'][i]))
            pos += num_iter[i]
        steps_list += [pos]
        plt.plot(steps_list, self.res_list['obj'])
        plt.xlabel('number of steps')
        plt.ylabel('Mutual Information')
        plt.title(TITLE)
        #plt.show()
        
        
    @staticmethod 
    def load_res_list(filename):
        """
        Load res_list of a TuningCurveOptimizer_Noncyclic Object
        return a TuningCurveOptimizer_Noncyclic Object
        """
        
        
        res_list = np.load(filename+'.npy').item()
        
        tc = TuningCurve_Noncyclic(res_list['x'][0], res_list['weight'][0], res_list['conv'], res_list['tau'], \
                                   res_list['obj'][0], res_list['grad'][0]) 
        
        tc_opt = TuningCurveOptimizer_Noncyclic(tc, res_list['fp'], res_list['fm'])
        
        tc_opt.res_list = res_list.copy()
        tc_opt.res_len = len(res_list['x'])
        return tc_opt
    
