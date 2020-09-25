import time, sys, os, copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from tuning.cyMINoncyclic import mc_mean_grad_noncyclic # poisson model
from tuning.cyMINoncyclic import mc_mean_grad_gaussian, mc_mean_grad_gaussian_inhomo, \
mc_mean_grad_gaussian_inhomo_no_corr # gaussian models
from tuning.anim_3dcube import plot_funcs_in_figure, gen_mixed_plots, gen_mixed_anim, \
plot_funcs_bars_in_figure, plot_funcs_circles_in_figure

class TuningCurve_Noncyclic:
    """TuningCurve Class (one or multi-population)
    # Poisson/Gaussian Distribution (different models)
    Attributes:
        numNeuro, numBin, tuning, weight, conv, tau
        info, grad, 
    Methods:
       __init__(self,
                model, #'Poisson', 'GaussianHomo', 'GaussianInhomo', 'GaussianInhomoNoCorr' 
                tuning, # tuning curve with size numNeuro*numBin
                weight, # length of each bin (interval)   
                inv_cov_mat = None, # inverse covariance matrix for Gaussian Models (different shapes required)
                conv = None, # convolution kernel                               
                tau = 1.0, # tau
                info = None, # mutual information I(r,theta)
                grad = None,  # gradient of I (minus gradient of -I)                 
                mc_iter = 1e5, # number of iterations for computing info and grad using MC method,
                num_threads = 8, # number of threads for computing info and grad using MC.
                ):
        compute_info_grad(self,
                          mc_iter = 1e5, # number of iterations for computing info and grad using MC method,
                          num_threads = 8, # number of threads for computing info and grad using MC.
                          )
        plot(self, fig=None, 
             nrow=None, ncol=None, fp=None, fm=None, 
             data_axis = 0, path_vec = None, 
             )
             
        plot_bars(self, fig=None, 
                  nrow=None, ncol=None, 
                  fp=None, fm=None, 
                  path_vec=None, 
                  num_colors=21, cmap_name ='jet', add_colorbar=True,
                  **kwargs,
                 )
        plot_circles(self, grid_shape, fig=None, 
                     nrow=None, ncol=None, 
                     fp=None, fm=None,
                     path_vec=None, 
                     num_colors=21, cmap_name='jet', add_colorbar=True,
                     max_point_size=300,
                     **kwargs,
                    )
        plot_all() # old version of plot
        __copy__(self) # useage: copy.copy(tuning_curve_instance)
    Static Methods:
        # Usage: TuningCurve_Noncyclic.animation_tc_list(...)
        
        # plot tuning curves in row or column alignment
        animation_tc_list(tc_list, FILE_NAME = "", ADD_TIME = True, FP = [], FM = [],\
                          XTICKS_IDX_LIST = [], VAR_LABEL =  "", VAR_TXT_LIST = [], \
                          ALIGN = "row",\
                          INCLUDE_INFO = True, INCLUDE_RATE = False, INCLUDE_GRAD = False,\
                          index_list = [], interval=1000, dt = 1)
        (can choose to include rate, grad or info; 
        dropped the coloring gradient function as in TuningCurve class.)
                 
        # plot tuning curves with the same alignment as plot():
        animation_tc_list_all(tc_list, FILE_NAME = "", ADD_TIME = True, FP = [], FM = [], \
                              XTICKS_IDX_LIST = [], VAR_LABEL =  "", VAR_TXT_LIST = [], \
                              index_list = [], interval=1000, color = False, dt = 1)
                              
        # Cube Plot when numNeuro = 3:
        animation_tc_list_cube(tc_list, INCLUDE_FUN = True, \
                               FILE_NAME = "", ADD_TIME = True, interval = 1000)
        
           
    """
    def __init__(self,
                 model, #'Poisson', 'GaussianHomo', 'GaussianInhomo', 'GaussianInhomoNoCorr' 
                 tuning, # tuning curve with size (numNeuro, numBin)
                 weight, # weights with size (numBin, )
                 inv_cov_mat = None, # inverse covariance matrix for Gaussian Models (different shapes required)
                 conv = None, # convolution kernel                               
                 tau = 1.0, # tau
                 info = None, # mutual information I(r,theta)
                 grad = None,  # gradient of I (minus gradient of -I)                 
                 mc_iter = 1e5, # number of iterations for computing info and grad using MC method,
                 num_threads = 8, # number of threads for computing info and grad using MC or partial sum.
                ):
        '''Note:
        Different shapes for inv_cov_mat in different models:
        Poisson: inv_cov_mat is None
        GaussianHomo: inv_cov_mat: (numNeuro, numNeuro)
        GaussianInhomo: inv_cov_mat: (numNeuro, numNeuro, numBin)
        GaussianInhomoNoCorr: inv_cov_mat: (numNeuro, numBin)
        (in this case it is 'inv_cov_diag' argument in 'mc_mean_grad_gaussian_inhomo_no_corr')
        
        '''
        if model not in ['Poisson', 'GaussianHomo', 'GaussianInhomo', 'GaussianInhomoNoCorr']:
            raise Exception('Wrong input for the model!')
        if model in ['GaussianHomo', 'GaussianInhomo', 'GaussianInhomoNoCorr'] and inv_cov_mat is None:
            raise Exception('Missing input for the inverse covariance matrix of Gaussian model!')
            
        tuning = np.array(tuning)
        if len(tuning.shape) ==1:
            # tuning is a (numBin,) type array
            tuning = tuning.reshape((1,tuning.size))
        if np.any(tuning < 0):
            raise Exception('Wrong input for tuning function!')
        # hard to constrain on convolution now...
        if conv is None:
            conv = np.zeros(tuning.shape[1])
            conv[0] = 1
        if np.fabs(np.sum(conv) -1)>1e-5:
            raise Exception('Wrong input for convolution kernel! Must sum up to one.')        
        
        self.model = model
        self.numNeuro = tuning.shape[0]
        self.numBin = tuning.shape[1]
        self.tuning = tuning.copy() 
        self.weight = np.array(weight).copy()
        self.conv = np.array(conv).copy()
        self.tau = tau        
        
        # check inverse covarainace matrix shapes for Gaussian Model
        if model == 'GaussianHomo' and inv_cov_mat.shape != (self.numNeuro, self.numNeuro):
            raise Exception('Wrong shape for the inverse covariance matrix of Gaussian Homogeneous model! \
            Must be (numNeuro, numNeuro).')       
        if model == 'GaussianInhomo' and inv_cov_mat.shape != (self.numNeuro, self.numNeuro, self.numBin):
            raise Exception('Wrong shape for the inverse covariance matrix of Gaussian Inhomogeneous model! \
            Must be (numNeuro, numNeuro, numBin).')  
            # inv_cov_mat[:,:,j] is the inverse covariance matrix of p(r|theta_j)
        if model == 'GaussianInhomoNoCorr' and inv_cov_mat.shape != (self.numNeuro, self.numBin):
            raise Exception('Wrong shape for the inverse covariance matrix of Gaussian Inhomogeneous model \
            with no correlations! Must be (numNeuro, numBin).')  
            #inv_cov_mat[:,j] contains the diagonal entries of the inverse covariance matrix of p(r|theta_j) 
        
        if model != 'Poisson':
            self.inv_cov_mat = inv_cov_mat.copy()
         
        if info is None or grad is None:
            self.compute_info_grad(mc_iter, num_threads)
        else:
            self.info = info
            self.grad = grad.copy()

        
        #self.average = np.average(tuning, axis = 1) # integral average
        #self.num_iter = 0 # number of iterations
        
    def compute_info_grad(self,
                          mc_iter = 1e5, # number of iterations for computing info and grad using MC method,
                          num_threads = 8, # number of threads for computing info and grad using MC or partial sum.
                          ):
        """
        Compute mutual information and gradient of mutual information.
        (Since not for optimization purporses, +I and +gradI.)
        NUM_ITER: number of iterations in monte carlo method
        """
        grad0 = np.zeros((self.numNeuro, self.numBin))
        if self.model == 'Poisson':
            mean0 = mc_mean_grad_noncyclic(grad0, self.tuning, self.weight, 
                                           self.conv, self.tau, 
                                           mc_iter, num_threads)
        elif self.model == 'GaussianHomo':
            mean0 = mc_mean_grad_gaussian(grad0, self.tuning, self.weight, self.inv_cov_mat,
                                           self.conv, self.tau, 
                                           mc_iter, num_threads)
        elif self.model == 'GaussianInhomo':
            mean0 = mc_mean_grad_gaussian_inhomo(grad0, self.tuning, self.weight, self.inv_cov_mat,
                                                 self.conv, self.tau, 
                                                 mc_iter, num_threads)
        elif self.model == 'GaussianInhomoNoCorr':
            mean0 = mc_mean_grad_gaussian_inhomo_no_corr(grad0, self.tuning, self.weight, 
                                                         self.inv_cov_mat,
                                                         self.conv, self.tau, 
                                                         mc_iter, num_threads)            
            
        self.grad = grad0.copy()
        self.info = mean0
    
    def compute_rate(self):
        '''Compute the corresponding rate curve after convolution'''
        rate = np.zeros((self.numNeuro, self.numBin))
        for i in range(self.numNeuro):
            for j in range(self.numBin):
                for k in range(self.numBin):
                    rate[i,j] += self.tuning[i, (self.numBin+j-k) % self.numBin]*self.conv[k]
                rate[i,j] = self.tau*rate[i,j]
        return rate 
    
    def plot(self, fig=None, 
             nrow=None, ncol=None, fp=None, fm=None, 
             data_axis=0, path_vec=None, 
             ):
        
        '''
        Plot tuning curves in subplots in grid (nrow, ncol).
        Return the generated subplots' axes.
        See details in tuning.anim_3dcube:plot_funcs_in_figure'''
        if fig is None:
            fig = plt.figure()
        ax_list = plot_funcs_in_figure(fig, self.tuning, self.weight,
                                       nrow=nrow, ncol=ncol, fp=fp, fm=fm,
                                       path_vec=path_vec, data_axis=data_axis,
                                       )
        return ax_list
    
    def plot_bars(self, fig=None, 
                  nrow=None, ncol=None, 
                  fp=None, fm=None, 
                  path_vec=None, 
                  num_colors=21, cmap_name ='jet', add_colorbar=True,
                  **kwargs,
                 ):
        '''Plot tuning curves in subplots in grid (nrow, ncol), as colored bar plots.
        Return the generated subplots' axes (including the colorbar axis).
        especially suitable for high dimensional data (numNeuro>3).
        kwargs: same as usage in 'ax.plot'.
        See details in tuning.anim_3dcube:plot_funcs_bars_in_figure
        '''
        if fig is None:
            fig = plt.figure()
        ax_list = plot_funcs_bars_in_figure(fig, self.tuning, self.weight,
                                            nrow=nrow, ncol=ncol,
                                            fp=fp, fm=fm, 
                                            path_vec=path_vec,
                                            num_colors=num_colors, cmap_nam=cmap_name, 
                                            add_colorbar=add_colorbar,
                                            **kwargs,
                                           )
        return ax_list

    def plot_circles(self, grid_shape, fig=None, 
                     nrow=None, ncol=None, 
                     fp=None, fm=None,
                     path_vec=None, 
                     num_colors=21, cmap_name='jet', add_colorbar=True,
                     max_point_size=300,
                     **kwargs,
                    ):
        '''Plot scattered circles whose areas equal to weights on 2d grid.
        Return the generated subplots' axes (including the colorbar axis).
        Used in 2d-input cases.
        Require a grid_shape argument:  (nNeuro, nBin1, nBin2)
        (same as self.laplacian in TuningCurveOptimizer_Noncyclic)
        kwargs: same as usage in 'ax.scatter'.
        See details in tuning.anim_3dcube:plot_funcs_circles_in_figure
        '''
        if fig is None:
            fig = plt.figure()
        ax_list = plot_funcs_circles_in_figure(fig, self.tuning, self.weight, grid_shape, 
                                               nrow=nrow, ncol=ncol, 
                                               fp=fp, fm=fm,
                                               path_vec=path_vec, 
                                               num_colors=num_colors,
                                               color_map_name=cmap_name, 
                                               add_colorbar=add_colorbar,
                                               max_point_size=max_point_size,
                                               **kwargs,
                                              )
        return ax_list

    def plot_all(self):
        '''Old version of plotting: including tuning curve, weights, rate curve, gradient separately'''
        # plot tuning curve, weights, convolution, gradient separately.
        xtmp = np.cumsum(self.weight)
        xx = [ [xtmp[j]]*2 for j in range(0, self.numBin-1) ]  
        xx = [0]+list(np.array(xx).reshape(-1)) + [1.0]
        yy = []
        yy_rate = []
        yy_grad = []
        
        rate = self.compute_rate()
        for p in range(self.numNeuro):
            yy_p = [ [self.tuning[p][i], self.tuning[p][i+1]] for i in range(0, self.numBin-1) ] 
            yy_p = [self.tuning[p][0]] + list(np.array(yy_p).reshape(-1)) + [self.tuning[p][-1]]
            
            yy_rate_p = [ [rate[p][i], rate[p][i+1]] for i in range(0, self.numBin-1) ] 
            yy_rate_p = [rate[p][0]] + list(np.array(yy_rate_p).reshape(-1)) + [rate[p][-1]]
            yy_grad_p = [ [grad[p][i], self.grad[p][i+1]] for i in range(0, self.numBin-1) ] 
            yy_grad_p = [grad[p][0]] + list(np.array(yy_grad_p).reshape(-1)) + [self.grad[p][-1]]
            
            yy.append(yy_p)
            yy_rate.append(yy_rate_p)
            yy_grad.append(yy_grad_p)
        color_list = ['steelblue',  'seagreen', 'crimson', 'gray', 'm','gold', 'k']*10 # incase no enough colors..
            
        weight_max = np.max(self.weight)
        fun_max = np.max(self.tuning)
        fig = plt.figure(figsize = (16,8))
        ax_tuning = fig.add_subplot(2,2,1, ylim = (-0.01, fun_max + 0.01))
        ax_weight = fig.add_subplot(2,2,2, ylim = (-0.01, weight_max + 0.01))
        ax_rate = fig.add_subplot(2,2,3,ylim = (-0.01*self.tau, (fun_max + 0.01)*self.tau))
        ax_grad = fig.add_subplot(2,2,4)

        for p in range(self.numNeuro):
            ax_tuning.plot(xx, yy[p], lw = 1.5, color = color_list[p])
            # ax_tuning.plot(self.tuning[i])
        leg = ax_tuning.legend([r'$MI$ = %.4f'%(self.info)], handlelength=0, handletextpad=0, \
                         fancybox = True, loc = 'center right', bbox_to_anchor=(-0.05,0.5), fontsize = 15)
        for item in leg.legendHandles:
            item.set_visible(False)
        ax_tuning.set_title('Tuning Curve with %d neurons, %d bins'%(self.numNeuro,self.numBin))

        ax_weight.plot(self.weight, color = 'gray', lw = 1.5)
        ax_weight.set_title('Weights')

        for p in range(self.numNeuro):
            ax_rate.plot(xx, yy_rate[p],lw = 1.5, color = color_list[p],\
                         label = r'$\bar{f_{%d}}$ = %.1f'%(p,self.average[p]))
        ax_rate.set_title('Rate Curve')
        if self.numNeuro ==1:
            ax_rate.legend([r'$\bar{f}$ = %.1f'%self.average], loc='center right', \
                       fancybox = True,bbox_to_anchor=(-0.05,0.5), fontsize = 15)
        else:
            ax_rate.legend([r'$\bar{f_{%d}}$ = %.1f'%(i,self.average[i]) for i in range(self.numNeuro)], \
                       loc='center right',fancybox = True,bbox_to_anchor=(-0.05,0.5), fontsize = 15)

        for p in range(self.numNeuro):
            ax_grad.plot(xx, yy_grad[p],lw = 1.5, color = color_list[p], \
                         label = r'$(\nabla I)_{%d}$'%p)
        ax_grad.set_title('Gradient')
        ax_tuning.grid()
        ax_weight.grid()
        ax_rate.grid()
        ax_grad.grid()
        plt.show()            

    def plot_3dcube(self, 
                    path_vec=None,
                    color_arr=None,
                    radius=1, min_radius=0,
                    INCLUDE_FUN=True, INCLUDE_WEIGHT=True, INCLUDE_WEIGHT_BAR=True,
                    FILE_NAME="", ADD_TIME=False,# figure not saved if both FILE_NAME="" and ADD_TIME=False 
                     **kwargs, #kwargs in set_scatter_data_in_axis, set_data_in_figure.
                   ):
        '''See details in tuning.anim_3dcube gen_mixed_plots'''
        if self.numNeuro!= 3:
             raise Exception('Wrong dimension of tuning curve! The number of neurons must be 3.')
                
        figure_handles = gen_mixed_plots(self.tuning, weights=self.weight, info=self.info,
                                         path_vec=path_vec,
                                         color_arr=color_arr,
                                         radius=radius, min_radius=min_radius,
                                         INCLUDE_FUN=INCLUDE_FUN, 
                                         INCLUDE_WEIGHT=INCLUDE_WEIGHT, 
                                         INCLUDE_WEIGHT_BAR=INCLUDE_WEIGHT_BAR,
                                         FILE_NAME=FILE_NAME, ADD_TIME=ADD_TIME,
                                         **kwargs,
                                        )
        return figure_handles
    

    
    def plot_hist(self, ax, neuro_id=None, bins=50):
        '''Plot historgram of the tuning curve values with weights.
        If neuro_id is None, then plot histogram of all tuning values (numNeuro*numBin), 
        and for tuning[i,j] the weight will be weight[j]/numNeuro.'''
        if neuro_id is None:
            avg_weights = np.array(list(self.weight)*self.numNeuro)/self.numNeuro
            ax.hist(self.tuning.reshape(-1), bins=bins, 
                    weights = avg_weights)
        else:
            ax.hist(self.tuning[neuro_id,:], bins=bins, 
                    weights = self.weight)
        
    def __copy__(self):
        return TuningCurve_Noncyclic(self.tuning, self.weight, self.conv, self.tau, self.info, self.grad)

 
    # ---------Animation function for a list of tuning curves--------- 
    @staticmethod # same alignment as plot() , include tuning, weight, rate, grad   
    def animation_tc_list_all(tc_list, FILE_NAME = "", ADD_TIME = True, FP = [], FM = [], \
                              XTICKS_IDX_LIST = [], VAR_LABEL =  "", VAR_TXT_LIST = [], \
                              index_list = [], interval=1000, color = False, dt = 1):
        """Old version: Plot animation for a list of tuning curves. Same alignment as plot() function."""

        test_num = len(tc_list)
        # check dimensions
        numBin = tc_list[0].numBin
        numNeuro = tc_list[0].numNeuro
        for tc in tc_list:
            if tc.numNeuro!= numNeuro or tc.numBin != numBin:
                raise Exception('Dimension mismatch for different tuning curves!')
        # here use FP and FM as numbers
        FP = np.array(FP)
        FM = np.array(FM)
        if FP.size and FM.size: # nonempty 
            PLOT_FP_FM = True
            if not isinstance(FP, (int, float, complex)):
                FP = np.max(FP)
            if not isinstance(FM, (int, float, complex)): # is number
                FM = np.min(FM)
        else:
            PLOT_FP_FM = False
            FP = np.max(np.array([np.max(tc.tuning) for tc in tc_list]))
            FM = np.min(np.array([np.min(tc.tuning) for tc in tc_list]))  
        if not VAR_TXT_LIST: # empty parameters
            VAR_TXT_LIST = [""]*test_num
        #     else:
        #         VAR_TXT_LIST = ["%.4f"%var for var in VAR_LIST]
        if not index_list:
            index_list = np.arange(test_num)
        elif np.array(index_list).size != test_num:
            raise Exception('Wrong dimension of index list!')

        def index_curve_constant(x, c, tol = 1e-5):
            x_c = np.where(np.fabs(x - c) < tol)[0]
            if len(x_c) ==0:
                return []
            else:
                loc = list(np.where(np.diff(x_c)>1)[0]) + [x_c.size - 1]
                #print loc, x_c[loc]
                #print x_c, loc
                idx_list = []
                for i in range(len(loc)):
                    if i ==0:
                        idx_list.append([x_c[i], x_c[loc[i]] ])
                    else:
                        idx_list.append([x_c[loc[i-1] + 1], x_c[loc[i]] ])
                #print idx_list
                return idx_list


        # dt = 1
        line_tuning = []
        line_weight = []
        line_grad = []
        line_rate = []
        grad_lines_lists = []
        #         info_texts = []
        #         var_texts = []

        grad_max = np.max(np.array([np.max(np.fabs(tc.grad)) for tc in tc_list]))
        tau = np.max(np.array([np.max(tc.tau) for tc in tc_list]))

        #         FP = np.zeros(numNeuro)
        #         FM = np.zeros(numNeuro)
        #         for k in range(numNeuro):
        #             FP[k]  = np.max(np.array([np.max(tc.tuning[k]) for tc in tc_list]))
        #             FM[k]  = np.min(np.array([np.min(tc.tuning[k]) for tc in tc_list]))        
        rate_fp = np.max(np.array([tc.tau*np.max(tc.tuning) for tc in tc_list]))
        rate_fm = np.min(np.array([tc.tau*np.min(tc.tuning) for tc in tc_list]))
        weight_max = np.max(np.array([np.max(tc.weight) for tc in tc_list]))
        
        fig = plt.figure(figsize = (16,8))        
        ax_tuning = fig.add_subplot(2,2,1, xlim = (0,1), ylim = (FM-0.1*FP,FP+0.1*FP))
        ax_tuning.grid()               
        ax_weight = fig.add_subplot(2,2,2, xlim = (0,numBin), ylim=(-0.01,weight_max + 0.01))
        ax_weight.grid()
        ax_rate = fig.add_subplot(2,2,3, xlim = (0,1), ylim=(tau*(FM - 0.1*FP),tau*(FP+0.1*FP)))
        ax_rate.grid()
        ax_grad = fig.add_subplot(2,2,4, xlim = (0,1), ylim=(-1.1*grad_max,1.1*grad_max))
        ax_grad.grid()
        
        colors = ['steelblue',  'seagreen', 'crimson', 'gray', 'm','gold', 'k']*10
        for p in range(numNeuro):
            line, = ax_tuning.plot([], [], color = colors[p], lw = 1.5)# , 'o-', lw=2
            line_tuning.append(line)
            line, = ax_rate.plot([], [],color = colors[p], lw = 1.5)
            line_rate.append(line)
            line, = ax_grad.plot([], [],color = colors[p], lw = 1.5)
            line_grad.append(line)

            lines_list = []
            for j in range(numBin):
                l, = ax_grad.plot([], []) 
                lines_list.append(l)
            grad_lines_lists.append(lines_list)
        line_weight, = ax_weight.plot([], [], color = 'gray', lw = 1.5) # no population assotiated with weight
        
        # compute the piecewise functions
        
        
        

        def init():
            """initialize animation"""
            #             line_weight.set_data([], [])
            #             for p in range(numNeuro):
            #                 line_tuning[p].set_data([], [])
            #                 line_grad[p].set_data([], [])
            #                 line_rate[p].set_data([], [])
            if PLOT_FP_FM:
                ax_tuning.plot(np.linspace(0,1,numBin), np.ones(numBin)*FP,'--', color = 'r')
                ax_tuning.plot(np.linspace(0,1,numBin), np.ones(numBin)*FM,'--', color = 'c')
                ax_tuning.text(-0.08,FP/(FP - FM + 0.2), r'$f_{+}$', transform=ax_tuning.transAxes, fontsize = 15)
                ax_tuning.text(-0.08,FM/(FP - FM + 0.2), r'$f_{-}$', transform=ax_tuning.transAxes, fontsize = 15)
            #if numBin>10 and numBin%2 ==0:
            #    numBinwidth = 2*int(numBin/10)
            #else:
            #    numBinwidth = 1
            #ax_tuning.set_xticks(list(np.arange(0,numBin,numBinwidth)))   
            #ax_grad.set_xticks(list(np.arange(0,numBin,numBinwidth)))
            #ax_rate.set_xticks(list(np.arange(0,numBin,numBinwidth))) 
            #ax_rate.set_xticks(list(np.arange(0,numBin,numBinwidth)))
            return line_tuning, line_weight, line_rate, line_grad

        def animate(i):
            """perform animation step"""
            # global tuning_curve, dt
            # global tuning_list, grad_list, mean_list
            # n_iter = tuning_curve.num_iter
            ll = test_num
            curr_idx = index_list[i] # (i*dt)%ll
            curr_tuning = tc_list[i].tuning
            curr_weight =  tc_list[i].weight
            curr_rate = tc_list[i].compute_rate() #tc_listp[i].rate
            curr_grad = tc_list[i].grad
            curr_info = tc_list[i].info
            
            
            xtmp = np.cumsum(curr_weight)            
            xx = [ [xtmp[j]]*2 for j in range(0, numBin-1) ]  
            xx = [0]+list(np.array(xx).reshape(-1)) + [1.0]
            
            for p in range(numNeuro):          
                yy_p = [ [curr_tuning[p][j], curr_tuning[p][j+1]] for j in range(0, numBin-1) ] 
                yy_p = [curr_tuning[p][0]] + list(np.array(yy_p).reshape(-1)) + [curr_tuning[p][-1]]
            
                yy_rate_p = [ [curr_rate[p][j], curr_rate[p][j+1]] for j in range(0, numBin-1) ]  
                yy_rate_p = [curr_rate[p][0]] + list(np.array(yy_rate_p).reshape(-1)) + [curr_rate[p][-1]]
                yy_grad_p = [ [curr_grad[p][j], curr_grad[p][j+1]] for j in range(0, numBin-1) ] 
                yy_grad_p = [curr_grad[p][0]] + list(np.array(yy_grad_p).reshape(-1)) + [curr_grad[p][-1]]
                
                line_tuning[p].set_data(xx, yy_p) # tuning_curve.tuning[p]
                line_rate[p].set_data(xx, yy_rate_p) #tuning_curve.rate[p]              
                line_grad[p].set_data(xx, yy_grad_p) # tuning_curve.grad[p][:numBin]

                # adding colors...
                if color:
                    lines_list = grad_lines_lists[p]
                    idx_list1 = index_curve_constant(curr_tuning[p], FP) #tuning_curve.tuning[p]
                    k = 0
                    for idx in idx_list1:
                        lines_list[k].set_data(np.arange(idx[0],idx[1]+1), curr_grad[p][idx[0]:idx[1]+1])#tuning_curve.grad[p]
                        lines_list[k].set_color('r')
                        lines_list[k].set_linewidth(2.0)
                        k += 1
                    idx_list2 = index_curve_constant(curr_tuning[p], FM)#tuning_curve.tuning[p]
                    for idx in idx_list2:
                        lines_list[k].set_data(np.arange(idx[0],idx[1]+1), curr_grad[p][idx[0]:idx[1]+1])#tuning_curve.grad[p]
                        lines_list[k].set_color('g')
                        lines_list[k].set_linewidth(2.0)
                        k += 1
                    for j in range(k, numBin):
                        lines_list[j].set_data([], [])
                # adding colors finished
                # set xticks
                #if i in XTICKS_IDX_LIST:                    
                #    tuning = tc_list[i].tuning[p]
                #    diff = np.diff(tuning) # a[n+1] - a[n]
                #    pts = np.where(np.fabs(diff) > 1e-5)[0]
                #    ax_tuning.set_xticks([0] + list(pts) + [numBin-1])
                #    ax_grad.set_xticks([0] + list(pts) + [numBin-1])
                #    ax_rate.set_xticks([0] + list(pts) + [numBin-1])
                    
            line_weight.set_data(np.arange(numBin), curr_weight)
            ax_tuning.set_title('Tuning Curve: index = %d '%curr_idx + VAR_LABEL + VAR_TXT_LIST[i]) # tuning_curve.info
            ax_weight.set_title('Weights')
            ax_rate.set_title('Rate Curve')
            ax_grad.set_title('Gradient')
            
            # legends
            leg = ax_tuning.legend([r'$MI$ = %.4f'%(curr_info)], handlelength=0, \
                                   handletextpad=0, fancybox = True,\
                                   loc = 'center right', bbox_to_anchor=(-0.05,0.5), fontsize = 15)
            for item in leg.legendHandles:
                item.set_visible(False)
                
            if numNeuro ==1:
                ax_rate.legend([r'$\bar{f}$ = %.1f'%tc_list[i].average], loc='center right', \
                           fancybox = True,bbox_to_anchor=(-0.05,0.5), fontsize = 15)
            else:
                ax_rate.legend([r'$\bar{f_{%d}}$ = %.1f'%(p,tc_list[i].average[p]) for p in range(numNeuro)], \
                           loc='center right',fancybox = True,bbox_to_anchor=(-0.05,0.5), fontsize = 15)
                
            
            return line_tuning, line_weight, line_rate, line_grad

        # call the animator.  blit=True means only re-draw the parts that have changed.
        anim = animation.FuncAnimation(fig, animate, init_func = init,
                                       frames = test_num, interval=interval)#, blit=True      
        if ADD_TIME: 
            timestr = time.strftime("%m%d-%H%M%S")
        else:
            timestr = ""
        filename =  FILE_NAME + timestr + ".mp4" 
        directory = os.path.dirname(filename)
        if directory != "":
            try:
                os.stat(directory)
            except:
                os.makedirs(directory)  
        anim.save(filename, writer="ffmpeg")
        
        
               
    @staticmethod 
    def animation_tc_list(tc_list, FILE_NAME = "", ADD_TIME = True, FP = [], FM = [],\
                          XTICKS_IDX_LIST = [], VAR_LABEL =  "", VAR_TXT_LIST = [], \
                          ALIGN = "row",\
                          INCLUDE_INFO = True, INCLUDE_RATE = False, INCLUDE_GRAD = False,\
                          index_list = [], interval=1000, dt = 1):
        # ALIGN = "row" or "col" (default = row)
        """Old version: Plot animation for a list of tuning curves. Showing each population in 
        separate subplots aligned in rows or columns. Not including weight function. """

        test_num = len(tc_list)
        # check dimensions
        numBin = tc_list[0].numBin
        numNeuro = tc_list[0].numNeuro
        #delta = tc_list[0].delta
        #nu = tc_list[0].nu
        for tc in tc_list:
            if tc.numNeuro!= numNeuro or tc.numBin != numBin:
                raise Exception('Dimension mismatch for different tuning curves!')
                
        FP = np.array(FP) # numpy array
        FM = np.array(FM)        
        if FP.size and FM.size: # nonempty 
            PLOT_FP_FM = True
            if FP.size ==1:
                FP = FP[0]*np.ones(tc.numNeuro)
            if FM.size ==1:
                FM = FM[0]*np.ones(tc.numNeuro)
            elif FP.size != tc.numNeuro or FM.size != tc.numNeuro:
                raise Exception('Dimension mismatch for tuning curve f+/f-!')
            #if isinstance(FP, (int, float, complex)) and isinstance(FM, (int, float, complex)): # is number
            #    FP = FP*np.ones(tc.numNeuro)
            #    FM = FM*np.ones(tc.numNeuro)
        else:
            PLOT_FP_FM = False
            FP = np.zeros(numNeuro)
            FM = np.zeros(numNeuro)
            for k in range(numNeuro):
                FP[k]  = np.max(np.array([np.max(tc.tuning[k]) for tc in tc_list]))
                FM[k]  = np.min(np.array([np.min(tc.tuning[k]) for tc in tc_list]))  

        
        if not VAR_TXT_LIST: # empty parameters
            VAR_TXT_LIST = [""]*test_num
            #else:
            #    VAR_TXT_LIST = ["%.4f"%var for var in VAR_LIST]
        if not index_list:
            index_list = np.arange(test_num)
        elif np.array(index_list).size != test_num:
            raise Exception('Wrong dimension of index list!')
        
        #def index_curve_constant(x, c, tol = 1e-5):
        #    x_c = np.where(np.fabs(x - c) < tol)[0]

        #    if len(x_c) ==0:
        #        return []
        #    else:
        #        loc = list(np.where(np.diff(x_c)>1)[0]) + [x_c.size - 1]
        #        idx_list = []
        #        for i in range(len(loc)):
        #            if i ==0:
        #                idx_list.append([x_c[i], x_c[loc[i]] ])
        #            else:
        #                idx_list.append([x_c[loc[i-1] + 1], x_c[loc[i]] ])
        #        return idx_list
            
        def get_xlabels(grid_space, label_space):
            grids_num = int(numBin/grid_space)
            labels_num = int(numBin/label_space)
            w = int(label_space/grid_space)
            if w<=1:
                # use grid_space
                labels_str = [str(t) for t in np.arange(0,numBin, grid_space)] + [str(numBin-1)]
            else:
                labels_str = ['']*grids_num
                for i in range(0, grids_num, w):
                    labels_str[i] = str(i*grid_space)
                labels_str += [str(numBin-1)]
                labels_str[1] = str(grid_space)
            return labels_str
        

        # dt = 1

        ax_tuning = []
        ax_grad = []
        ax_rate = []
        line_tuning = []
        line_grad = []
        line_rate = []
        #grad_lines_lists = []
        #info_texts = []
        var_texts = []

        grad_max = np.max(np.array([np.max(np.fabs(tc.grad)) for tc in tc_list]))
      
        tau = np.max(np.array([np.max(tc.tau) for tc in tc_list]))  
        colors = ['steelblue',  'seagreen', 'crimson', 'gray', 'm','gold', 'k']*10
        
        if ALIGN == "row":
            n_rows = numNeuro
            n_cols = 1
            if INCLUDE_RATE:
                n_cols += 1
            if INCLUDE_GRAD:
                n_cols += 1
        else: #ALIGN == "col"
            n_cols = numNeuro
            n_rows = 1
            if INCLUDE_RATE:
                n_rows += 1
            if INCLUDE_GRAD:
                n_rows += 1               
        if INCLUDE_INFO:
            n_cols += 1
            
        rate_idx = 2   
        if INCLUDE_RATE:
            grad_idx = 3
        else:
            grad_idx = 2
        #print n_rows, n_cols
        
        fig = plt.figure(figsize = (n_cols*6,n_rows*6)) #(numNeuro*10,3*6)
         
        info_list = np.array([tc.info for tc in tc_list])
        
        if INCLUDE_INFO:
            info_max = np.max(info_list)
            info_min = np.min(info_list)
            #print info_max, info_min
            ax_info = fig.add_subplot(1,n_cols,n_cols, xlim = (0, index_list[-1]+1), ylim = (info_min-0.1, info_max+0.1))
            ax_info.grid()
            ax_info.set_title('Mutual Information', fontsize = 14)
            line_info, = ax_info.plot([],[])
        
        
        for k in range(numNeuro):
             # row or col alignment
            
            if ALIGN == "row":
                ax1 = fig.add_subplot(n_rows, n_cols, n_cols*k+1, xlim = (0,1), ylim = (FM[k]-0.1*FP[k],FP[k]+0.1*FP[k]))        
                if INCLUDE_RATE:
                    ax2 = fig.add_subplot(n_rows, n_cols,n_cols*k+rate_idx , xlim = (0,1), ylim=(tau*(FM[k]-0.1*FP[k]),tau*(FP[k]+0.1*FP[k])))
                if INCLUDE_GRAD:
                    ax3 = fig.add_subplot(n_rows, n_cols, n_cols*k+grad_idx , xlim = (0,1), ylim=(-1.1*grad_max,1.1*grad_max))
            else: #ALIGN == "col"
                ax1 = fig.add_subplot(n_rows, n_cols, k+1, xlim = (0,1), ylim = (FM[k]-0.1*FP[k],FP[k]+0.1*FP[k]))        
                if INCLUDE_RATE:
                    ax2 = fig.add_subplot(n_rows, n_cols,n_cols+k+1 , xlim = (0,1), ylim=(tau*(FM[k]-0.1*FP[k]),tau*(FP[k]+0.1*FP[k])))
                if INCLUDE_GRAD:
                    ax3 = fig.add_subplot(n_rows, n_cols, (grad_idx-1)*n_cols+k+1 , xlim = (0,1), ylim=(-1.1*grad_max,1.1*grad_max))

            if k ==0:
                ax1.set_title('Tuning Curve', fontsize = 14)
            ax1.grid()
            line1, = ax1.plot([], [], color = colors[k], lw = 1.5)
            ax_tuning.append(ax1)
            line_tuning.append(line1)  
            
            if INCLUDE_RATE:
                if k==0:
                    ax2.set_title('Rate Curve', fontsize = 14)
                ax2.grid()
                line2, = ax2.plot([], [], color = colors[k], lw = 1.5)
                ax_rate.append(ax2)
                line_rate.append(line2)
                
            if INCLUDE_GRAD:
                if k ==0:
                    ax3.set_title('Gradient', fontsize = 14)
                ax3.grid()
                line3, = ax3.plot([], [], color = colors[k], lw = 1.5) #line, = ax.plot([], [], 'o-', lw=2)
                ax_grad.append(ax3)                  
                line_grad.append(line3)     
                #lines_list = []
                #for j in range(numBin):
                #    l, = ax3.plot([], []) 
                #    lines_list.append(l)
                #grad_lines_lists.append(lines_list)
            
            vartxt = ax1.text(0.02,0.75,'',transform=ax1.transAxes, fontsize = 16)
            #info_texts.append(infotxt)
            var_texts.append(vartxt)
        if INCLUDE_INFO:
            infotxt = ax_info.text(0.35, 0.9, '', transform=ax_info.transAxes, fontsize = 16)
        else:
            infotxt = ax_tuning[0].text(0.02, 0.85, '', transform=ax_tuning[0].transAxes, fontsize = 16)
        
        
        #if numBin>10 and numBin%2 ==0:
        #    numBinwidth = 2*int(numBin/10)
        #else:
        #    numBinwidth = 1 
        # prepare for plotting piecewise constant curves

        
            
            
        def init():
            """initialize animation"""
            for p in range(numNeuro):      
                if PLOT_FP_FM:
                    ax_tuning[p].plot(np.linspace(0,1,numBin), np.ones(numBin)*FP[p],'--', linewidth=1.5)
                    ax_tuning[p].plot(np.linspace(0,1,numBin), np.ones(numBin)*FM[p],'--', linewidth=1.5)
                    # f+, f- labels
                    #ax_tuning[p].text(-0.1,(FP[p]-FM[p]+0.09*FP[p])/(FP[p]-FM[p]+0.2*FP[p]),"f+", \
                    #                  transform=ax_tuning[p].transAxes,fontsize = 15)
                    #ax_tuning[p].text(-0.1,0.09*FP[p]/(FP[p]-FM[p]+0.2*FP[p]), "f-", \
                    #                  transform=ax_tuning[p].transAxes,fontsize = 15)
                    # newly added
                    if INCLUDE_RATE:
                        ax_rate[p].plot(np.linspace(0,1,numBin), np.ones(numBin)*FP[p]*tau,'--', linewidth=1.5)
                        ax_rate[p].plot(np.linspace(0,1,numBin), np.ones(numBin)*FM[p]*tau,'--', linewidth=1.5)
                #ax_tuning[p].set_xticks(list(np.arange(0,numBin,8))+[numBin-1])
                #ax_rate[p].set_xticks(list(np.arange(0,numBin,8))+[numBin-1]) 
                #ax_tuning[p].set_xticks(list(np.arange(0,numBin,numBinwidth))+[numBin-1])
                #ax_rate[p].set_xticks(list(np.arange(0,numBin,numBinwidth))+[numBin-1])
                
  
                #xlabels = get_xlabels(1, int(numBin/numBinwidth)) # old: grid_space = nu, label_space = numBin/4
                
                #ax_tuning[p].set_xticklabels(xlabels)
                #ax_rate[p].set_xticklabels(xlabels)
                
                #neurons_arr = np.arange(0, numBin,1)# delta = 1
                #ax_tuning[p].plot(neurons_arr, (FM[p]-0.09*FP[p])*np.ones_like(neurons_arr),'o')
                #ax_rate[p].plot(neurons_arr, tau*(FM[p]-0.09*FP[p])*np.ones_like(neurons_arr),'o')
                if INCLUDE_GRAD:
                    line_grad[p].set_data([], [])
                    #ax_grad[p].set_xticks(list(np.arange(0,numBin,8))+[numBin-1])
                    #ax_grad[p].set_xticks(list(np.arange(0,numBin,nu))+[numBin-1])
                    #ax_grad[p].set_xticklabels(xlabels)
                    #ax_grad[p].plot(neurons_arr, (-1.09*grad_max)*np.ones_like(neurons_arr),'o')
            # set global title
            global_title = "Tuning Curve Optimization for %d neurons, %d bins"%(numNeuro,numBin)
            # adding avg firing rate (not very important)
            # if numNeuro ==1:
            #     global_title += r', $\bar{f}$ = %.1f'%tc_list[0].average
            # else:
            #     global_title += r', $\bar{f}$ = ['
            #     for i in range(numNeuro-1):
            #         global_title += "%.1f,"%tc_list[0].average[i]
            #     global_title += "%.1f]"%tc_list[0].average[numNeuro-1]
            #print global_title
            st = fig.suptitle(global_title, fontsize=17)
            #fig.tight_layout()

            #shift subplots down:
            st.set_y(0.95)
            fig.subplots_adjust(top=0.85)
            

            return line_tuning, line_grad, line_rate

        def animate(i):
            """perform animation step"""
            # global tuning_curve, dt
            # global tuning_list, grad_list, mean_list
            # n_iter = tuning_curve.num_iter
            ll = test_num
            curr_idx = index_list[i] # (i*dt)%ll
            curr_tuning = tc_list[i].tuning
            curr_grad = tc_list[i].grad
            curr_info = tc_list[i].info
            curr_rate =  tc_list[i].compute_rate() #tc_list[i].rate
            curr_weight = tc_list[i].weight
            
            xtmp = np.cumsum(curr_weight)        
            xx = [ [xtmp[j]]*2 for j in range(0, numBin-1) ]  
            xx = [0]+list(np.array(xx).reshape(-1)) + [1.0]

            if INCLUDE_INFO:
                line_info.set_data(index_list[0:i+1], info_list[0:i+1])
            infotxt.set_text('MI = %.4f' % curr_info)

            for p in range(numNeuro):
                yy_p = [ [curr_tuning[p][j], curr_tuning[p][j+1]] for j in range(0, numBin-1) ] 
                yy_p = [curr_tuning[p][0]] + list(np.array(yy_p).reshape(-1)) + [curr_tuning[p][-1]]            
                
                         
                
                line_tuning[p].set_data(xx, yy_p) # tuning_curve.tuning[p]
                if INCLUDE_RATE:
                    yy_rate_p = [ [curr_rate[p][j], curr_rate[p][j+1]] for j in range(0, numBin-1) ]  
                    yy_rate_p = [curr_rate[p][0]] + list(np.array(yy_rate_p).reshape(-1)) + [curr_rate[p][-1]]
                    line_rate[p].set_data(xx, yy_rate_p) #tuning_curve.rate[p] 
                #ax_tuning[p].set_title('Tuning Curve:index = %d'%curr_idx ) # tuning_curve.info

                #info_texts[p].set_text('MI = %.4f' % curr_info)
                
                var_texts[p].set_text( VAR_LABEL + VAR_TXT_LIST[i])
                if INCLUDE_GRAD:
                    yy_grad_p = [ [curr_grad[p][j], curr_grad[p][j+1]] for j in range(0, numBin-1) ] 
                    yy_grad_p = [curr_grad[p][0]] + list(np.array(yy_grad_p).reshape(-1)) + [curr_grad[p][-1]]
                    line_grad[p].set_data(xx, yy_grad_p) # tuning_curve.grad[p][:numBin]
                    # adding colors...
                    #if color:
                    #    lines_list = grad_lines_lists[p]
                    #    idx_list1 = index_curve_constant(curr_tuning[p], FP[p]) #tuning_curve.tuning[p]
                    #    k = 0
                    #    for idx in idx_list1:
                    #        lines_list[k].set_data(np.arange(idx[0],idx[1]+1), curr_grad[p][idx[0]:idx[1]+1])#tuning_curve.grad[p]
                    #        lines_list[k].set_color('r')
                    #        lines_list[k].set_linewidth(1.5)#2.0
                    #        k += 1
                    #    idx_list2 = index_curve_constant(curr_tuning[p], FM[p])#tuning_curve.tuning[p]
                    #    for idx in idx_list2:
                    #        lines_list[k].set_data(np.arange(idx[0],idx[1]+1), curr_grad[p][idx[0]:idx[1]+1])#tuning_curve.grad[p]
                    #        lines_list[k].set_color('g')
                    #        lines_list[k].set_linewidth(1.5) # 2.0
                    #        k += 1
                    #    for j in range(k, numBin):
                    #        lines_list[j].set_data([], [])
                    # adding colors finished
                # set xticks
                #if i in XTICKS_IDX_LIST: # e.g. XTICKS_IDX_LIST = [-1], i = -1
                #    tuning = tc_list[i].tuning[p]
                #    diff = np.diff(tuning) # a[n+1] - a[n]
                #    pts = np.where(np.fabs(diff) > 1e-5)[0]
                #    ax_tuning[p].set_xticks([0] + list(pts) + [numBin-1])
                #    ax_rate[p].set_xticks([0] + list(pts) + [numBin-1])
                #    if IF_GRAD:
                #        ax_grad[p].set_xticks([0] + list(pts) + [numBin-1])
                # adjust linewidth
                for p in range(numNeuro):
                    line_tuning[p].set_linewidth(1.5)
                    if INCLUDE_RATE:
                        line_rate[p].set_linewidth(1.5)
                    if INCLUDE_GRAD:
                        line_grad[p].set_linewidth(1.5)
                #ax_tuning[p].set_title('Tuning Curve:index = %d'%curr_idx ) # tuning_curve.info
                if INCLUDE_INFO:
                    line_info.set_linewidth(1.5)
            
            return line_tuning, line_grad, line_rate

        # call the animator.  blit=True means only re-draw the parts that have changed.
        anim = animation.FuncAnimation(fig, animate, init_func = init,
                                       frames = test_num, interval=interval)#, blit=True      
        if ADD_TIME: 
            timestr = time.strftime("%m%d-%H%M%S")
        else:
            timestr = ""
        filename =  FILE_NAME + timestr + ".mp4" 
        directory = os.path.dirname(filename)
        if directory != "":
            try:
                os.stat(directory)
            except:
                os.makedirs(directory)  
        anim.save(filename, writer="ffmpeg")
        
        
    @staticmethod 
    def animation_tc_list_cube(tc_list, 
                               path_vec_list=None,
                               color_arr_list=None,
                               radius=1, min_radius=0,
                               include_fun=True, include_weight=True, include_weight_bar=True,
                               file_name="", add_time=True,
                               interval=1000,
                               **kwargs,
                              ):
        tuning_list = []
        weights_list = []
        info_list = []
                    
        for tc in tc_list:
            tuning_list.append(tc.tuning)
            weights_list.append(tc.weight)
            info_list.append(tc.info)

        anim = gen_mixed_anim(tuning_list, weights_list=weights_list, info_list=info_list,
                              path_vec_list=path_vec_list,
                              color_arr_list=color_arr_list,
                              radius=radius, min_radius=min_radius,
                              INCLUDE_FUN=include_fun, INCLUDE_WEIGHT=include_weight, 
                              INCLUDE_WEIGHT_BAR=include_weight_bar,
                              FILE_NAME=file_name, ADD_TIME=add_time,
                              interval=interval,
                              **kwargs,
                             )
        return anim
    
    
    #----------A helper function from sigma to homogenous Gaussian inverse covariance matrix
    
    @staticmethod
    def sigma_to_inv_cov(sigma, nNeuro):
        inv_cov_matrix = np.diag(1.0*np.ones(nNeuro)/sigma**2)
        return inv_cov_matrix
       