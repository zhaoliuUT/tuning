import time, sys, os, copy
import numpy as np
import numpy.linalg as la
import scipy
from scipy import optimize
from functools import partial

from tuning.tuning_curve_noncyclic import *

from tuning.cyMINoncyclic import mc_mean_grad_noncyclic, mc_coeff_arimoto # poisson model
from tuning.cyMINoncyclic import mc_mean_grad_gaussian, mc_mean_grad_gaussian_inhomo, \
mc_mean_grad_gaussian_inhomo_no_corr # gaussian models
from tuning.cyMINoncyclic import mc_coeff_arimoto_gaussian, mc_coeff_arimoto_gaussian_inhomo, \
mc_coeff_arimoto_gaussian_inhomo_no_corr
from tuning.simple_sgd import simple_sgd_with_laplacian
from tuning.anim_3dcube import pc_fun_weights, gen_mixed_anim


class TuningCurveOptimizer_Noncyclic:
    """TuningCurveOptimizer_Noncyclic Class
    # Poisson/Gaussian Distribution
    For old version with the average constraint (old algorithm),
    see commit: 8b58799
    
    Attributes:
        Properties: model, numNeuro, numBin, conv, tau, num_threads
        Current Tuning Information: tuning0, grad0, info0, average_cons, bounds
        Optimization Constraints: fp, fm, fp_vec, fm_vec (temporarily not used in SGD),
                                  laplacian_2d (true or false), laplacian_shape, weighted_laplacian
        Result Storage: res_len, and 13 lists:
            tuning_list, weight_list, info_list, grad_list, 
            inv_cov_list, laplacian_coeff_list, time_list, mark_list, 
            sgd_learning_rate_list, sgd_batch_size_list, sgd_iter_steps_list, 
            ba_batch_size_list, ba_iter_steps_list        
        
    Methods:
        __init__(self,
                 model,
                 tc_init, # initial input (TuningCurve_Noncyclic object)
                 fp, # upper bound (vector of size numNeuro or constant)
                 fm, # lower bound (vector of size numNeuro or constant)
                 num_threads=8, # number of threads in parallel computation
                )
        iterate(self,
                total_num_iter, # total number of iterations
                sgd_learning_rate,                 
                laplacian_coeff=0, # laplacian coefficient in regularization
                laplacian_2d=False, # laplacian on 2d neighbours or 1d neighbours
                laplacian_shape=None, # only for 2d laplacian
                weighted_laplacian=False, # whether laplacian is weighted
                
                sgd_decrease_lr=False, # decrease learning rate (according to sqrt(number of iterations))               
                sgd_batch_size=1000, # batch size for sgd
                sgd_iter_steps=1, # number of steps for sgd in every cycle
                ba_batch_size=1000, # batch size for Monte Carlo in Blahut-Arimoto
                ba_iter_steps=1, # number of steps for Blahut-Arimoto in every cycle
                
                alter_compute_info=1, # compute mutual information every several steps
                alter_compute_info_mc=1e4, # number of Monte Carlo iterations used
                print_info=True, # print mutual information every several steps
                plot_live=False, #used in live plotting, only available for 1d
                alter_plot_live=10, # plot every several steps
                live_fig=None, # figure used for live plotting
                live_ax_list=None, # axes used for live plotting                
               )
               
        check_list_len(self) # check the lists' lengths
        reset(self, pos) # reset the lists at the given position
        take_res_id(self, pos) # take the 'pos'-th tuning curve (TuningCurve_Noncyclic object)
        plot_res_id(self, pos, **kwargs)
            Plot the tuning curve at the given index.
            See TuningCurve_Noncyclic.plot for keyword arguments.
        plot_info(self, ax, alternate=False, color_sgd='r', color_ba='b')
             Simple plotting mutual information, or alternate the sgd/arimoto steps using different colors
             
        plot_animation(self, FILE_NAME = "", ADD_TIME = True, interval = 1000,
                       dt = 1, XTICKS_IDX_LIST = [], ALIGN = "row", # options for non-cube animation
                       INCLUDE_RATE = False, INCLUDE_GRAD = False, INCLUDE_INFO = True, 
                       # options for non-cube animation
                       INCLUDE_WEIGHT=True,# option for both
                       INCLUDE_FUN = True, INCLUDE_WEIGHT_BAR = True, # options for cube animation
                       **kwargs,# options for cube animation
                       )
            Old version for plotting animation
        plot_animation_cube(self,
                            path_vec_list=None,
                            color_arr_list=None,
                            radius=1, min_radius=0,
                            INCLUDE_FUN=True, INCLUDE_WEIGHT=True, INCLUDE_WEIGHT_BAR=True,
                            FILE_NAME="", ADD_TIME=True,
                            interval=1000,
                            **kwargs,
                           )
            Plot cube animation for 3d case. See 'gen_mixed_plots' in anim_3dcube.py
        save_res_dict(self, file_name="", add_time=True)
    Static Methods:
        load_res_dist(filename) 
        # Usage: TuningCurveOptimizer_Noncyclic.load_res_dist(filename)
         
    """
    def __init__(self,
                 model,
                 tc_init, # initial input (TuningCurve_Noncyclic object)
                 fp, # upper bound (vector of size numNeuro or constant)
                 fm, # lower bound (vector of size numNeuro or constant)
                 num_threads=8, # number of threads in parallel computation
                ):
        
        if model not in ['Poisson', 'GaussianHomo', 'GaussianInhomo', 'GaussianInhomoNoCorr']:
            raise Exception('Wrong input for the model!')
        if model != tc_init.model:
            raise Exception('Model mismatch tc_init.model!')
        
        self.model = model
        self.numNeuro = tc_init.numNeuro # 2
        self.numBin = tc_init.numBin 
        self.conv = tc_init.conv.copy()
        self.tau = tc_init.tau
        self.num_threads = num_threads

        # fp and fm constraints
        if isinstance(fp, (int, float, complex)) or isinstance(fm, (int, float, complex)): # is number
            self.fp = fp
            self.fm = fm
            fp_vec = fp*np.ones(self.numNeuro)
            fm_vec = fm*np.ones(self.numNeuro)
        else: # fp, fm can be arrays or lists for different neurons
            fp_vec = np.array(fp)  # numpy array
            fm_vec = np.array(fm)
            self.fp = np.max(fp_vec)
            self.fm = np.min(fm_vec)
        if fp_vec.size and fm_vec.size: # nonempty
            if fp_vec.size != self.numNeuro or fm_vec.size != self.numNeuro:
                raise Exception('Dimension mismatch for fp/fm!')
            if np.any(fp_vec < 0) or np.any(fm_vec <0) or np.any(fm_vec > fp_vec):
                raise Exception('Wrong input for fp/fm!')
        else:
            raise Exception('Missing input for fp, fm!') 
            
        self.fp_vec = fp_vec
        self.fm_vec = fm_vec
        
        self.bounds = []
        for j in range(self.numNeuro):
            for i in range(self.numBin):
                self.bounds += [(self.fm_vec[j], self.fm_vec[j])]
        self.bounds = tuple(self.bounds)
        
        # current tuning, weight, grad, info
        self.tuning = tc_init.tuning.copy()        
        self.weight = tc_init.weight.copy()
        self.grad = tc_init.grad.copy()
        self.info = tc_init.info
        
        # inverse covariance matrix        
        if self.model == 'Poisson':
            self.inv_cov_mat = None
        else:
            self.inv_cov_mat = tc_init.inv_cov_mat.copy()
        
        # lists for saving results
        self.tuning_list = [tc_init.tuning.copy()]
        self.weight_list = [tc_init.weight.copy()]
        self.info_list = [tc_init.info]
        self.grad_list = [tc_init.grad.copy()]
        if self.model == 'Poisson':
            self.inv_cov_list = [None]
        else:
            self.inv_cov_list = [tc_init.inv_cov_mat.copy()]
        self.time_list = [None]
        self.mark_list = [None]
        self.sgd_learning_rate_list = [None]
        self.sgd_batch_size_list = [None]
        self.sgd_iter_steps_list = [None]
        self.ba_batch_size_list = [None]
        self.ba_iter_steps_list = [None]
        self.laplacian_coeff_list = [None]

        self.res_len = 1 # result list length
        self.laplacian_2d = False # apply 2d laplacian or not
        self.laplacian_shape = None
        self.weighted_laplacian = False


    def iterate(self,
                total_num_iter, # total number of iterations
                sgd_learning_rate,                 
                laplacian_coeff=0, # laplacian coefficient in regularization
                laplacian_2d=False, # laplacian on 2d neighbours or 1d neighbours
                laplacian_shape=None, # only for 2d laplacian
                weighted_laplacian=False, # whether laplacian is weighted
                
                sgd_decrease_lr=False, # decrease learning rate (according to sqrt(number of iterations))               
                sgd_batch_size=1000, # batch size for sgd
                sgd_iter_steps=1, # number of steps for sgd in every cycle
                ba_batch_size=1000, # batch size for Monte Carlo in Blahut-Arimoto
                ba_iter_steps=1, # number of steps for Blahut-Arimoto in every cycle
                
                alter_compute_info=1, # compute mutual information every several steps
                alter_compute_info_mc=1e4, # number of Monte Carlo iterations used
                print_info=True, # print mutual information every several steps
                plot_live=False, #used in live plotting, only available for 1d
                alter_plot_live=10, # plot every several steps
                live_fig=None, # figure used for live plotting
                live_ax_list=None, # axes used for live plotting                
               ):
        
        self.laplacian_2d = laplacian_2d
        self.laplacian_shape = laplacian_shape
        self.weighted_laplacian = weighted_laplacian
        if (not laplacian_2d) and (laplacian_shape is not None):
            raise Exception('No input needed for laplacian shape for 1d neighbour case')
        if laplacian_2d and laplacian_shape is None:
            raise Exception('Missing input for laplacian shape for 2d neighbour case')
        if laplacian_shape is not None:
            if (len(laplacian_shape)!= 3 or laplacian_shape[0] != self.numNeuro 
                or laplacian_shape[1]*laplacian_shape[2]!= self.numBin):
                raise Exception('Wrong input for laplacian shape for 2d neighbour case')    
        
        curr_tuning = self.tuning.copy()
        curr_weight = self.weight.copy()
        if self.model == 'Poisson':
            curr_inv_cov_mat = None
        else:
            curr_inv_cov_mat = self.inv_cov_mat.copy()
        
        # determine the functions for Blahut-Arimoto iterations and information evaluations
        if self.model == 'Poisson':         
            ba_iter_func = mc_coeff_arimoto
            mc_iter_func = mc_mean_grad_noncyclic
        elif self.model == 'GaussianHomo':
            ba_iter_func = mc_coeff_arimoto_gaussian
            mc_iter_func = mc_mean_grad_gaussian
        elif self.model == 'GaussianInhomo':
            ba_iter_func = mc_coeff_arimoto_gaussian_inhomo
            mc_iter_func = mc_mean_grad_gaussian_inhomo
        elif self.model == 'GaussianInhomoNoCorr':
            ba_iter_func = mc_coeff_arimoto_gaussian_inhomo_no_corr
            mc_iter_func = mc_mean_grad_gaussian_inhomo_no_corr
            # when the variance has a derivative it is implemented,
            # but function arguments are not compatible with other sgd&arimoto functions.
                
        for num_iter in range(1, total_num_iter+1):
            if sgd_decrease_lr:
                curr_sgd_learning_rate = sgd_learning_rate/(np.sqrt(num_iter + 1))
            else:
                curr_sgd_learning_rate = sgd_learning_rate
   
            curr_sgd_batch_size = sgd_batch_size
            curr_sgd_iter_steps = sgd_iter_steps #int(1000/curr_sgd_batch_size)
            curr_ba_batch_size = ba_batch_size
            curr_ba_iter_steps = ba_iter_steps
            curr_laplacian_coeff = laplacian_coeff

            # ------SGD for updating the points' coordinates------
            curr_time = time.time()
            x_list, _ = simple_sgd_with_laplacian(
                self.model,
                curr_tuning, curr_weight,                
                inv_cov_mat=curr_inv_cov_mat,
                eta=curr_sgd_learning_rate,                
                mc_iter=curr_sgd_batch_size,
                num_iter=curr_sgd_iter_steps,
                fp=self.fp,
                fm=self.fm, # not working when fp_vec, fm_vec is not uniform...
                laplacian_coeff=laplacian_coeff,
                laplacian_shape=laplacian_shape,
                weighted_laplacian=weighted_laplacian,
                conv=self.conv,
                tau=self.tau,
                num_threads=self.num_threads,
            )
    

            curr_tuning = x_list[-1].copy()
            self.tuning_list.append(curr_tuning.copy())
            self.weight_list.append(curr_weight.copy())
            self.mark_list.append('sgd')

            spent_time1 = time.time() - curr_time
            self.time_list.append(spent_time1)


            # ------Monte Carlo based Arimoto Interation for updating the weights------                        
            curr_time = time.time()
            slope = np.zeros(self.numNeuro)
            new_prob_vec = curr_weight.copy()
            for k in range(curr_ba_iter_steps):    
                new_coeff = np.zeros(self.numBin)                    
                if self.model == 'Poisson':
                    ba_iter_func(
                        new_coeff, curr_tuning, new_prob_vec,
                        slope, self.conv, self.tau, 
                        curr_ba_batch_size, my_num_threads = self.num_threads)
                else:
                    ba_iter_func(
                        new_coeff, curr_tuning, new_prob_vec, 
                        curr_inv_cov_mat, 
                        slope, self.conv, self.tau, 
                        curr_ba_batch_size, my_num_threads = self.num_threads)
                new_prob_vec *= new_coeff
                new_prob_vec /= np.sum(new_prob_vec)

            spent_time2 = time.time() - curr_time
            self.time_list.append(spent_time2)

            curr_weight = new_prob_vec.copy()
            self.tuning_list.append(curr_tuning.copy())
            self.weight_list.append(curr_weight.copy())
            self.mark_list.append('ba')
            
            # ------save results------   
            self.sgd_learning_rate_list +=[curr_sgd_learning_rate, curr_sgd_learning_rate]
            self.sgd_batch_size_list += [curr_sgd_batch_size, curr_sgd_batch_size]
            self.ba_batch_size_list += [curr_ba_batch_size, curr_ba_batch_size]
            self.laplacian_coeff_list += [curr_laplacian_coeff, curr_laplacian_coeff]
            self.sgd_iter_steps_list += [curr_sgd_iter_steps, 0]
            self.ba_iter_steps_list += [0, curr_ba_iter_steps]
            
            self.inv_cov_list += [curr_inv_cov_mat, curr_inv_cov_mat]
            
            if num_iter%alter_compute_info == 0:
                grad_tc = np.zeros_like(curr_tuning)
                if self.model=='Poisson':
                    info_tc = mc_iter_func(
                        grad_tc, curr_tuning, curr_weight, 
                        self.conv, self.tau, 
                        numIter=int(alter_compute_info_mc), my_num_threads=self.num_threads)
                else:
                    info_tc = mc_iter_func(
                        grad_tc, curr_tuning, curr_weight, 
                        self.inv_cov_mat,
                        self.conv, self.tau, 
                        numIter=int(alter_compute_info_mc), my_num_threads=self.num_threads)
                if print_info:
                    print(num_iter, curr_sgd_batch_size, info_tc, spent_time1, spent_time2)
                self.info_list += [info_tc, info_tc]
                self.grad_list += [grad_tc.copy(), grad_tc.copy()]
            else:
                self.info_list += [None, None]
                self.grad_list += [None, None]

            if plot_live and num_iter%alter_plot_live==0:
                for i in range(self.numNeuro):
                    live_ax_list[i].clear()
                    xx, yy = pc_fun_weights(curr_tuning[i,:], curr_weight)
                    live_ax_list[i].plot(xx, yy)
                    live_ax_list[i].set_ylim([self.fm-0.1, self.fp+0.1])
        #         time.sleep(0.1)
                live_fig.canvas.draw()
        
                
        # update current tuning, weight, grad, info, res_len
        self.tuning = self.tuning_list[-1].copy() 
        self.weight = self.weight_list[-1].copy() 
        self.grad = self.grad_list[-1].copy() 
        self.info = self.info_list[-1]
        self.res_len = len(self.tuning_list)
        
    
    def check_list_len(self):
        # 13 lists same as self.res_len?
        print(len(self.tuning_list), len(self.weight_list), len(self.info_list), \
              len(self.grad_list), len(self.inv_cov_list), len(self.time_list), len(self.mark_list), \
              len(self.sgd_learning_rate_list), len(self.sgd_iter_steps_list), len(self.sgd_iter_steps_list), \
              len(self.ba_batch_size_list), len(self.ba_iter_steps_list), len(self.laplacian_coeff_list), \
              self.res_len, 
              )

    def reset(self, pos):
        # reset the lists at the given position
        # pos must be >= 0. if pos>=self.res_len, then do not reset.
        m = pos + 1
        self.tuning_list = self.tuning_list[0:m]
        self.weight_list = self.weight_list[0:m]
        self.info_list = self.info_list[0:m]
        self.grad_list = self.grad_list[0:m]
        self.time_list = self.time_list[0:m]
        self.mark_list = self.mark_list[0:m]
        
        self.sgd_learning_rate_list = self.sgd_learning_rate_list[0:m]
        self.sgd_batch_size_list = self.sgd_batch_size_list[0:m]
        self.sgd_iter_steps_list = self.sgd_iter_steps_list[0:m]
        self.ba_batch_size_list = self.ba_batch_size_list[0:m]
        self.ba_iter_steps_list = self.ba_iter_steps_list[0:m]
        self.laplacian_coeff_list = self.laplacian_coeff_list[0:m]
        self.inv_cov_list = self.inv_cov_list[0:m]

        self.tuning = self.tuning_list[-1].copy()
        self.weight = self.weight_list[-1].copy()
        self.grad = self.grad_list[-1].copy()
        self.info = self.info_list[-1]
        self.res_len = len(self.tuning_list)

        
    def take_res_id(self, pos):
        # pos must be in [0, self.len-1]
        if self.tuning_list[pos] is not None:
            tc = TuningCurve_Noncyclic(self.model, self.tuning_list[pos], self.weight_list[pos],
                                       inv_cov_mat=self.inv_cov_list[pos],
                                       conv=self.conv, tau=self.tau,
                                       info=self.info_list[pos],
                                       grad=self.grad_list[pos], 
                                       num_threads=self.num_threads,
                                      )
            return tc
        else:
            raise Exception('No result at index = %d!'%pos)
    
    #=========Plotting functions=========             
    def plot_res_id(self, pos, **kwargs):
        '''Plot the tuning curve at the given index. See TuningCurve_Noncyclic.plot for keyword arguments.'''
        tc = self.take_res_id(pos)
        tc.plot(**kwargs)
        
                  
    def plot_info(self, alternate=False, color_sgd='r', color_ba='b'):
        #plot_info(self, ax, alternate=False, color_sgd='r', color_ba='b'):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        if alternate:
            self.__class__.plot_info_alternate(ax, 
                                               self.info_list, self.mark_list, index_list=None, 
                                               color_sgd=color_sgd, color_ba=color_ba)
        else:
            ax.plot(self.info_list)
            
    def plot_animation(self, FILE_NAME = "", ADD_TIME = True, interval = 1000,
                       dt = 1, XTICKS_IDX_LIST = [], ALIGN = "row", # options for non-cube animation
                       INCLUDE_RATE = False, INCLUDE_GRAD = False, INCLUDE_INFO = True, # options for non-cube animation
                       INCLUDE_WEIGHT=True,# option for both
                       **kwargs,# options for cube animation
                       ):
        '''Older version'''
        tc_list = []
        # evaluate all the mutual information for every tuning curve during iterations
        index_list = []
        for i in range(self.res_len):
            if self.info_list[i] is not None:
                tc = TuningCurve_Noncyclic(self.model, self.tuning_list[i], self.weight_list[i],
                                           inv_cov_mat=self.inv_cov_list[i],
                                           conv=self.conv, tau=self.tau,
                                           info=self.info_list[i],
                                           grad=self.grad_list[i], 
                                           num_threads=self.num_threads,
                                          )
                tc_list.append(tc)
                index_list.append(i)
        anim = TuningCurve_Noncyclic.animation_tc_list(
            tc_list, FILE_NAME, ADD_TIME, FP = self.fp_vec, FM = self.fm_vec, 
            interval= interval, dt = dt,
            index_list = index_list, 
            XTICKS_IDX_LIST = XTICKS_IDX_LIST, VAR_LABEL =  "",
            VAR_TXT_LIST = [], ALIGN = ALIGN, INCLUDE_RATE = INCLUDE_RATE,
            INCLUDE_GRAD = INCLUDE_GRAD, INCLUDE_INFO = INCLUDE_INFO,
        )
        return anim
    
    def plot_animation_cube(self,
                            path_vec_list=None,
                            color_arr_list=None,
                            radius=1, min_radius=0,
                            INCLUDE_FUN=True, INCLUDE_WEIGHT=True, INCLUDE_WEIGHT_BAR=True,
                            FILE_NAME="", ADD_TIME=True,
                            interval=1000,
                            **kwargs,
                           ):
        '''See tuning.anim_3dcube gen_mixed_anim for keyword arguments'''

        anim = gen_mixed_anim(self.tuning_list, 
                              weights_list=self.weight_list, 
                              info_list=self.info_list,
                              path_vec_list=path_vec_list,
                              color_arr_list=color_arr_list,
                              radius=radius, min_radius=min_radius,
                              INCLUDE_FUN=INCLUDE_FUN, INCLUDE_WEIGHT=INCLUDE_WEIGHT, 
                              INCLUDE_WEIGHT_BAR=INCLUDE_WEIGHT_BAR,
                              FILE_NAME=FILE_NAME, ADD_TIME=ADD_TIME,
                              interval=interval,
                              **kwargs,
                             )
        return anim
        
    #=========Save and load results========= 
    
    
    def save_res_dict(self, file_name="", add_time=True):
        # save attributes
        res_dict = {}
        res_dict['model'] = self.model
        res_dict['conv'] = self.conv
        res_dict['tau'] = self.tau
        res_dict['num_threads'] = self.num_threads
        res_dict['fp'] = self.fp
        res_dict['fm'] = self.fm
        res_dict['fp_vec'] = self.fp_vec
        res_dict['fm_vec'] = self.fm_vec
        res_dict['laplacian_2d'] = self.laplacian_2d
        res_dict['laplacian_shape'] = self.laplacian_shape
        res_dict['weighted_laplacian'] = self.weighted_laplacian
        
        # save lists
        res_dict['tuning'] = self.tuning_list
        res_dict['weight'] = self.weight_list
        res_dict['grad'] = self.grad_list
        res_dict['info'] = self.info_list
        res_dict['inv_cov'] = self.inv_cov_list # for gaussian
        
        res_dict['mark'] = self.mark_list
        res_dict['time'] = self.time_list

        res_dict['sgd_learning_rate'] = self.sgd_learning_rate_list
        res_dict['sgd_batch_size'] = self.sgd_batch_size_list
        res_dict['sgd_iter_steps'] = self.sgd_iter_steps_list
        res_dict['ba_batch_size'] = self.ba_batch_size_list
        res_dict['ba_iter_steps'] = self.ba_iter_steps_list
        res_dict['laplacian_coeff'] = self.laplacian_coeff_list
        
        if add_time:
            timestr = time.strftime("%m%d-%H%M%S")
        else:
            timestr = ""
        # e.g.filename = 'data1/test1/Pop=%d_1'%numPop
        longfilename = file_name + timestr
        # self.FILE_NAME = filename             
        directory = os.path.dirname(longfilename)
        if directory != '':
            try:
                os.stat(directory)
            except:
                os.makedirs(directory)
        np.save(longfilename, res_dict)
        
    @staticmethod 
    def load_res_dict(file_name):
        """
        Load res_dict of a TuningCurveOptimizer_Noncyclic Object
        (filename including '.npy')
        return a TuningCurveOptimizer_Noncyclic Object
        """
        
        res_dict = np.load(file_name, allow_pickle=True, encoding = 'latin1').item()
                
        tc_init = TuningCurve_Noncyclic(
            res_dict['model'], 
            res_dict['tuning'][0],
            res_dict['weight'][0], 
            inv_cov_mat=res_dict['inv_cov'][0],
            conv=res_dict['conv'], 
            tau=res_dict['tau'],
            info=res_dict['info'][0], 
            grad=res_dict['grad'][0],
            num_threads=res_dict['num_threads'])
        
        tc_opt = TuningCurveOptimizer_Noncyclic(res_dict['model'], tc_init, res_dict['fp_vec'], 
                                                res_dict['fm_vec'])       
        
        # copy lists
        tc_opt.tuning_list = copy.copy(res_dict['tuning']) # or copy.deepcopy()
        tc_opt.weight_list = copy.copy(res_dict['weight'])
        tc_opt.info_list = copy.copy(res_dict['info'])
        tc_opt.grad_list = copy.copy(res_dict['grad'])

        tc_opt.inv_cov_list = copy.copy(res_dict['inv_cov'])

        tc_opt.time_list = copy.copy(res_dict['time'])
        tc_opt.mark_list = copy.copy(res_dict['mark'])
        tc_opt.sgd_learning_rate_list = copy.copy(res_dict['sgd_learning_rate']) 
        tc_opt.sgd_batch_size_list = copy.copy(res_dict['sgd_batch_size'])
        tc_opt.sgd_iter_steps_list = copy.copy(res_dict['sgd_iter_steps'])
        tc_opt.ba_batch_size_list = copy.copy(res_dict['ba_batch_size'])
        tc_opt.ba_iter_steps_list = copy.copy(res_dict['ba_iter_steps'])
        tc_opt.laplacian_coeff_list = copy.copy(res_dict['laplacian_coeff'])
        
        # current tuning, weight, grad, info and other attributes
        tc_opt.tuning = res_dict['tuning'][-1].copy()   
        tc_opt.weight = res_dict['weight'][-1].copy()   
        tc_opt.grad = res_dict['grad'][-1].copy()   
        tc_opt.info = res_dict['info'][-1]
        
        tc_opt.res_len = len(res_dict['info']) # result list length
        tc_opt.laplacian_2d = res_dict['laplacian_2d']
        tc_opt.laplacian_shape = res_dict['laplacian_shape']
        tc_opt.weighted_laplacian = res_dict['weighted_laplacian']
        return tc_opt
    
    @staticmethod
    def plot_info_alternate(ax, info_list, mark_list, index_list=None, color_sgd = 'r', color_ba = 'b'):
        if index_list is None:
            index_list = np.arange(len(info_list))
        else:
            index_list = np.array(index_list)

        sgd_idx = [i for i in index_list if mark_list[i]=='sgd']
        bandit_idx = [i for i in index_list if mark_list[i]=='ba']

        #plt.figure(figsize=(16,8))
        for idx in sgd_idx:
            ax.plot([idx-1, idx], [info_list[idx-1], info_list[idx]], c=color_sgd)

        for idx in bandit_idx:
            ax.plot([idx-1, idx], [info_list[idx-1], info_list[idx]], c=color_ba)