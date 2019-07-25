import time, sys, os, copy
import numpy as np
import numpy.linalg as la
import scipy
from scipy import optimize

from matplotlib import animation
import matplotlib.pyplot as plt
from cyMIPoisson import *
from TuningCurve import *

class TuningCurveOptimizer:
    """TuningCurveOptimizer Class (1 or 2 populations)
    # Poisson Distribution
    Attributes:
        Properties: numPop, numBin, numCent, delta, stim, stimWidth, nu, tau
        Optimization Information: fp, fm, tuning0, grad0, info0, average, cons, bounds
        Result Storage: res_list, res_len, num_iter, inter_steps [these 4 attributes can change]
        Optimizer Parameters: NUM_THREADS =8, NUM_ITER_INFO = 1e5,  NUM_ITER_GRAD = 1e5
    Methods:
        __init__(self,
                 TuningCurve_init, # initial input (TuningCurve object)
                 fp, # upper bound
                 fm, # lower bound
                 NUM_THREADS = 8,
                 MC_ITER_INFO = 1e5,
                 MC_ITER_GRAD = 1e5)
        iterate(self, 
                NUM_ITER = 1, 
                INTER_STEPS = 1, 
                PRINT = True, 
                FILE_NAME = "", 
                ADD_TIME = True,
                ftol = 1e-15, 
                disp = False)
        plot_res_id(self, i, ALL = True)
        plot_animation(self, FILE_NAME = "", ADD_TIME = True, XTICKS_IDX_LIST = [], color = False, interval = 1000, dt = 1)
        # old version: plot_animation_old(self, FILE_NAME = "", ADD_TIME = True, color = True, interval = 1000, dt = 1)
        plot_info(self, TITLE = "")
        save_res_list(self, FILE_NAME = "", ADD_TIME = True)
    Static Methods:
        load_res_list(filename) # Usage: TuningCurveOptimizer.load_res_list(filename)
         
    """
    def __init__(self,
                 TuningCurve_init, # initial input (TuningCurve object)
                 fp, # upper bound (vector of 2)
                 fm, # lower bound (vector of 2)
                 NUM_THREADS = 8,
                 MC_ITER_INFO = 1e5,
                 MC_ITER_GRAD = 1e5):  # # number of iterations for computing info and grad using MC method
        
        #self.TuningCurve_init = copy.copy(TuningCurve_init)
        self.numPop = TuningCurve_init.numPop # 2
        self.numBin = TuningCurve_init.numBin 
        self.numCent = TuningCurve_init.numCent
        self.delta = TuningCurve_init.delta
        self.stim = TuningCurve_init.stim.copy()
        self.nu = TuningCurve_init.nu
        self.tau = TuningCurve_init.tau
        if isinstance(fp, (int, float, complex)) or isinstance(fm, (int, float, complex)): # is number
            fp = fp*np.ones(self.numPop)
            fm = fm*np.ones(self.numPop)
        fp = np.array(fp)  # numpy array
        fm = np.array(fm)
        if fp.size != self.numPop or fm.size != self.numPop:
            raise Exception('Dimension mismatch for fp/fm!')
        if np.any(fp < 0) or np.any(fm <0) or np.any(fm > fp):
            raise Exception('Wrong input for fp/fm!')
        self.fp = fp.copy()
        self.fm = fm.copy()

        self.stimWidth = int(self.nu*self.tau)
        self.NUM_THREADS = NUM_THREADS
        self.MC_ITER_INFO = MC_ITER_INFO
        self.MC_ITER_GRAD = MC_ITER_GRAD
        
        self.tuning0 = TuningCurve_init.tuning.copy()       
        self.grad0 = TuningCurve_init.grad.copy()        
        self.info0 = TuningCurve_init.info
        self.average = TuningCurve_init.average
        
        self.bounds = []
        for j in range(self.numPop):
            for i in range(self.numBin):
                self.bounds += [(fm[j], fp[j])]
        self.bounds = tuple(self.bounds)
                
        def constraint_eq1(x):
            x_tuning = x.reshape((self.numPop, self.numBin))
            return np.average(x_tuning[0]) - self.average[0]
        def constraint_eq2(x):
            x_tuning = x.reshape((self.numPop, self.numBin))
            return np.average(x_tuning[1]) - self.average[1]
        if self.numPop == 1:
            my_cons = [{'type':'eq', 'fun': constraint_eq1}]
        elif self.numPop ==2:
            my_cons = [{'type':'eq', 'fun': constraint_eq1}, {'type':'eq', 'fun': constraint_eq2}]
        else:
            raise Exception('Wrong number of populations!')
        self.cons = my_cons
        
        res_list = {'x':[self.tuning0.copy()],'grad':[self.grad0.copy()],'obj':[self.info0],'success':[],'status':[],'nfev':[],'njev':[]}
        # add some atrributes to specify the parameters.
        res_list.update({'numPop': self.numPop, 'numCent': self.numCent, 'delta': self.delta, \
                         'average': self.average, 'nu': self.nu, 'tau': self.tau, 'stim':self.stim.copy(), \
                         'stimWidth': self.stimWidth,'fp': copy.copy(self.fp), 'fm': copy.copy(self.fm),\
                         'NUM_THREADS': self.NUM_THREADS, 'MC_ITER_INFO': self.MC_ITER_INFO, \
                         'MC_ITER_GRAD': self.MC_ITER_GRAD, 'num_iter': [0], 'inter_steps':[0]})  
        
        self.res_list = res_list # result list
        self.res_len = 1 # result list length
        self.num_iter = [0] # number of iterations
        self.inter_steps = [0] # default
        
        
    def iterate(self, NUM_ITER = 1, INTER_STEPS = 1, PRINT = True, FILE_NAME = "", ADD_TIME = True, ftol = 1e-15, disp = False):
        # INTER_STEPS: intermediate steps for printing and saving
        # NUM_ITER: total number of iterations
        # number of plotting/saving: NUM_ITER/INTER_STEPS
        MIgrad = np.zeros_like(self.grad0) # just for temporary storage

        def opt_fun(x, self):
            #             x_tuning = x.reshape((self.numPop, self.numBin))
            # mc_mean_grad_red_pop computes -I, -gradI.
            return mc_mean_grad_red_pop(MIgrad, self.numCent, self.delta,\
                                        x.reshape((self.numPop,self.numBin)), self.stim, self.tau,\
                                                  self.MC_ITER_INFO, self.NUM_THREADS)
        def grad_fun(x, self):   
            #         def grad_fun(x, stim, tau, numIter, my_num_threads):   
            #             x_tuning = x.reshape((self.numPop, self.numBin))
            # mc_mean_grad_red_pop computes -I, -gradI.
            x_grad = np.zeros((self.numPop, self.numBin), dtype = np.float)
            x_mean = mc_mean_grad_red_pop(x_grad, self.numCent, self.delta, \
                                          x.reshape((self.numPop, self.numBin)), self.stim, self.tau, \
                                          self.MC_ITER_GRAD, self.NUM_THREADS)
            # mc_mean_grad_red_pop(grad2, numCent, delta, tuning0, stim, tau, NUMITER2)
            # temp_mean = mc_mean_grad_pop(grad,tuning,stim,tau,numIter,my_num_threads)
            return x_grad.reshape(-1)
        
        
        curr_tuning = self.res_list['x'][-1].reshape(-1).copy()
        #plt.plot(self.res_list['x'][-1][0])
        #plt.plot(self.res_list['x'][-1][1])
        #plt.show()
                                        
        curr_num_iter = sum(self.num_iter)
        if PRINT:
            print('{0:4s}   {1:9s}'.format('Iter', 'Mutual Information'))
            print('{0:4d}   {1: 3.6f}'.format(curr_num_iter,self.res_list['obj'][-1]))
            
        for i in range(int(NUM_ITER/INTER_STEPS)):
            res = optimize.minimize(lambda x, self: opt_fun(x, self), \
                                    curr_tuning, args = self, method='SLSQP', \
                                    jac = lambda x,self:grad_fun(x,self), \
                                    bounds = self.bounds, constraints = self.cons, \
                                    options = {'maxiter':INTER_STEPS, 'ftol': ftol, 'disp': disp})
            
            curr_tuning = res['x'].copy()
            curr_num_iter += INTER_STEPS
            #print curr_tuning.shape
            # self.num_iter += INTER_STEPS
            self.res_list['x'].append(res['x'].reshape((self.numPop, self.numBin)).copy())
            self.res_list['grad'].append(-res['jac'][0:self.numPop*self.numBin].reshape((self.numPop, self.numBin)).copy())
            self.res_list['obj'].append(-res['fun'])
            self.res_list['success'].append(res['success'])
            self.res_list['status'].append(res['status'])
            self.res_list['nfev'].append(res['nfev'])
            self.res_list['njev'].append(res['njev'])
            if PRINT:
                print('{0:4d}   {1: 3.6f} '.format(curr_num_iter,-res['fun']))
                # print res['jac'].shape # 
        # save result list by default
        self.res_len = len(self.res_list['x'])
        self.num_iter.append(NUM_ITER)
        self.inter_steps.append(INTER_STEPS)
        self.res_list['num_iter'] = self.num_iter
        self.res_list['inter_steps'] = self.inter_steps
        if FILE_NAME != "" or ADD_TIME == True:
            self.save_res_list(FILE_NAME, ADD_TIME)
        #self.save_res_list(FILE_NAME, ADD_TIME)
        
    def save_res_list(self, FILE_NAME = "", ADD_TIME = True):
        if ADD_TIME:
            timestr = time.strftime("%m%d-%H%M%S")
        else:
            timestr = ""
        # e.g.filename = 'data1/test1/Pop=%d_1'%numPop
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
        tuning_curve = TuningCurve(self.res_list['x'][i], self.stim, self.nu,\
                                   self.tau, self.delta, self.res_list['obj'][i], self.res_list['grad'][i])
        tuning_curve.plot(ALL)
        
    def plot_animation(self, FILE_NAME = "", ADD_TIME = True, XTICKS_IDX_LIST = [], \
                       ALIGN = "row", INCLUDE_GRAD = False, INCLUDE_INFO = True, \
                       interval = 1000, color = False, dt = 1):
        tc_list = []
        for i in range(self.res_len):
            tc = TuningCurve(self.res_list['x'][i], self.stim, self.nu, self.tau, self.delta, \
                             self.res_list['obj'][i], self.res_list['grad'][i])
            tc_list.append(tc)
        
        steps_list = []
        pos = 0
        for i in range(1, len(self.num_iter)):
            steps_list += list(np.arange(pos, pos+self.num_iter[i], self.inter_steps[i]))
            pos += self.num_iter[i]
        steps_list += [pos]
        
        TuningCurve.animation_tc_list(tc_list, FILE_NAME, ADD_TIME, FP = self.fp, FM = self.fm, \
                                      XTICKS_IDX_LIST = XTICKS_IDX_LIST, \
                                      VAR_LABEL =  "", VAR_TXT_LIST = [], \
                                      ALIGN = ALIGN, INCLUDE_GRAD = INCLUDE_GRAD, INCLUDE_INFO = INCLUDE_INFO,\
                                      index_list = steps_list, interval= interval, color = color, dt = dt)
        
        
        
    def plot_info(self, TITLE = ""):
        steps_list = []
        pos = 0
        for i in range(1, len(self.num_iter)):
            steps_list += list(np.arange(pos, pos+self.num_iter[i], self.inter_steps[i]))
            pos += self.num_iter[i]
        steps_list += [pos]
        plt.plot(steps_list, self.res_list['obj'])
        plt.xlabel('number of steps')
        plt.ylabel('Mutual Information')
        plt.title(TITLE)
        #plt.show()
        
    @staticmethod    
    def load_res_list(filename):
        """
        Load res_list of a TuningCurveOptimizer Object
        return a TuningCurveOptimizer Object
        """
        res_list = np.load(filename+'.npy').item()
        tc = TuningCurve(res_list['x'][0], res_list['stim'], res_list['nu'], res_list['tau'], \
                         res_list['delta'], res_list['obj'][0], res_list['grad'][0])
        tc_opt = TuningCurveOptimizer(tc, res_list['fp'], res_list['fm'], res_list['NUM_THREADS'],\
                                              res_list['MC_ITER_INFO'], res_list['MC_ITER_GRAD'])
        tc_opt.res_list = res_list.copy()
        tc_opt.res_len = len(res_list['x'])
        tc_opt.num_iter = res_list['num_iter']
        tc_opt.inter_steps = res_list['inter_steps']
        return tc_opt            
