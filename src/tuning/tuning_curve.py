import time, sys, os, copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from tuning.cyMIPoisson import *

class TuningCurve:
    """TuningCurve Class (one or multi-population)
    # Poisson Distribution
    Attributes:
        tuning, numPop, numBin, numCent, delta = 1, stim, stimWidth, 
        info, grad, rate, average, MC_ITER = 1e5
    Methods:
        __init__(self,
                 tuning, # tuning curve with size numPop*numBin
                 stim, # stimulus
                 nu, # stimulus width (nu)
                 tau, # tau
                 delta = 1, # distance between centers
                 info = None, # mutual information I(r,theta)
                 grad = None,  # gradient of I (minus gradient of -I)
                 MC_ITER = 1e5, # number of iterations for computing info and grad using MC method
                 NUM_THREADS = 8): # number of threads for computing info and grad using MC method
        compute_info_grad(self, NUM_ITER = 1e5)
        plot(self, ALL = True)
        __copy__(self) # useage: copy.copy(tuning_curve_instance)
    Static Method:
        animation_tc_list(tc_list, FILE_NAME = "", ADD_TIME = True, FP = [], FM = [], \
                          XTICKS_IDX_LIST = [], VAR_LABEL =  "", VAR_TXT_LIST = [], \
                          index_list = [], interval=1000, color = False, dt = 1)
        # Usage: TuningCurve.animation_tc_list(...)
        
        # old version: showing populations in columns.
        animation_tc_list_col(tc_list, FILE_NAME = "", ADD_TIME = True, FP = [], FM = [], \
                              XTICKS_IDX_LIST = [], VAR_LABEL =  "", VAR_TXT_LIST = [], \
                              index_list = [], interval=1000, color = False, dt = 1)        
    """
    def __init__(self,
                 tuning, # tuning curve with size numPop*numBin
                 stim, # stimulus
                 nu, # stimulus width (nu)
                 tau, # tau
                 delta = 1, # distance between centers
                 info = None, # mutual information I(r,theta)
                 grad = None,  # gradient of I (minus gradient of -I)
                 MC_ITER = 1e5, # number of iterations for computing info and grad using MC method
                 NUM_THREADS = 8): # number of threads for computing info and grad using MC method
        
        if len(tuning.shape) ==1:
            # tuning is a (numBin,) type array
            tuning = tuning.reshape((1,tuning.size))
        if np.any(tuning <0):
            raise Exception('Wrong input for tuning function!')
        
        self.numPop = tuning.shape[0]
        self.numBin = tuning.shape[1]
        if self.numBin % delta != 0:
            raise Exception('Wrong input: number of Bins is not an integer multiple of center distance!')
        
        self.delta = delta
        self.numCent = self.numBin/delta
        self.tuning = tuning.copy()        
        self.stim = stim.copy()
        self.nu = nu
        self.tau = tau
        self.stimWidth = int(nu*tau)
        #self.MC_ITER = MC_ITER
        
        if info is None or grad is None:
            self.compute_info_grad(MC_ITER, NUM_THREADS)
        else:
            self.info = info
            self.grad = grad.copy()
      
        
        rate = np.zeros_like(tuning)
        for i in range(self.numPop):
            for j in range(self.numBin):
                for k in range(self.numBin):
                    rate[i,j] += tuning[i, (self.numBin+j-k) % self.numBin]*stim[k]
                rate[i,j] = tau*rate[i,j]
        self.rate = rate # corresponding rate curve
        
        self.average = np.average(tuning, axis = 1) # integral average
        #self.num_iter = 0 # number of iterations
        
    def compute_info_grad(self, MC_ITER = 1e5, NUM_THREADS = 8):
        """
        Compute mutual information and gradient of mutual information.
        (Since not for optimization purporses, +I and +gradI.)
        NUM_ITER: number of iterations in monte carlo method
        """
        grad0 = np.zeros((self.numPop, self.numBin))
        mean0 = mc_mean_grad_red_pop(grad0, self.numCent, self.delta, self.tuning, self.stim, self.tau, MC_ITER,NUM_THREADS)
        self.grad = -grad0
        self.info = -mean0

    def plot(self, ALL = True):
        if ALL:
            fig = plt.figure(figsize = (16,8))
            ax_tuning = fig.add_subplot(2,2,1)
            stim_max = np.max(np.array(self.stim))
            ax_stim = fig.add_subplot(2,2,2, ylim = (-0.01, stim_max + 0.01))
            ax_rate = fig.add_subplot(2,2,3)
            ax_grad = fig.add_subplot(2,2,4)
            
            for i in range(self.numPop):
                ax_tuning.plot(self.tuning[i], linewidth=1.5)
            leg = ax_tuning.legend([r'$MI$ = %.4f'%(self.info)], handlelength=0, handletextpad=0, \
                             fancybox = True, loc = 'center right', bbox_to_anchor=(-0.05,0.5), fontsize = 15)
            for item in leg.legendHandles:
                item.set_visible(False)
            ax_tuning.set_title('Tuning Curve with %d bins, %d centers'%(self.numBin,self.numCent))
            
            ax_stim.plot(self.stim, color = 'r', linewidth=1.5)
            ax_stim.set_title('Stimulus with ' +  r'$\nu$ = %d, $\tau$ = %.1f'%(self.nu, self.tau))
           
            for i in range(self.numPop):
                ax_rate.plot(self.rate[i],linewidth=1.5, label = r'$\bar{f_{%d}}$ = %.2f'%(i,self.average[i]))
            ax_rate.set_title('Rate Curve')
            if self.numPop ==1:
                leg = ax_rate.legend([r'$\bar{f}$ = %.2f'%self.average],handlelength=0, handletextpad=0,\
                                     loc='center right', \
                                     fancybox = True,bbox_to_anchor=(-0.05,0.5), fontsize = 15)
                for item in leg.legendHandles:
                    item.set_visible(False)
            else:
                ax_rate.legend([r'$\bar{f_{%d}}$ = %.2f'%(i,self.average[i]) for i in range(self.numPop)], \
                               loc='center right',fancybox = True,bbox_to_anchor=(-0.05,0.5), fontsize = 15)
            
            for i in range(self.numPop):
                ax_grad.plot(self.grad[i],linewidth=1.5, label = r'$(\nabla I)_{%d}$'%i)
            ax_grad.set_title('Gradient of Mutual Information')
            ax_tuning.grid()
            ax_stim.grid()
            ax_rate.grid()
            ax_grad.grid()
            plt.show()
        else:
            # plot tuning curve and rate curve only
            fig = plt.figure(figsize = (10,8))
            ax_tuning = fig.add_subplot(2,1,1)
            ax_rate = fig.add_subplot(2,1,2)
            
            for i in range(self.numPop):
                ax_tuning.plot(self.tuning[i], linewidth=1.5)
            leg = ax_tuning.legend([r'$MI$ = %.4f'%(self.info)], handlelength=0, handletextpad=0, \
                             fancybox = True, loc = 'center right', bbox_to_anchor=(-0.05,0.5), fontsize = 15)
            for item in leg.legendHandles:
                item.set_visible(False)
            ax_tuning.set_title('Tuning Curve with %d bins, %d centers'%(self.numBin,self.numCent))           
            for i in range(self.numPop):
                ax_rate.plot(self.rate[i], linewidth=1.5, label = r'$\bar{f_{%d}}$ = %.2f'%(i,self.average[i]))
            ax_rate.set_title('Rate Curve')
            if self.numPop ==1:
                leg = ax_rate.legend([r'$\bar{f}$ = %.2f'%self.average],handlelength=0, handletextpad=0,\
                                     loc='center right', \
                                     fancybox = True,bbox_to_anchor=(-0.05,0.5), fontsize = 15)
                for item in leg.legendHandles:
                    item.set_visible(False)
            else:
                ax_rate.legend([r'$\bar{f_{%d}}$ = %.2f'%(i,self.average[i]) for i in range(self.numPop)], \
                               loc='center right',fancybox = True,bbox_to_anchor=(-0.05,0.5), fontsize = 15)
            ax_tuning.grid()
            ax_rate.grid()
            plt.show()
            
            
            

        
    def __copy__(self):        
        return TuningCurve(self.tuning, self.stim, self.nu, self.tau, self.delta, self.info, self.grad)


    # ---------Animation function for a list of tuning curves--------- 
    @staticmethod # same alignment as plot() , include tuning, stim, rate, grad   
    def animation_tc_list_all(tc_list, FILE_NAME = "", ADD_TIME = True, FP = [], FM = [], \
                              XTICKS_IDX_LIST = [], VAR_LABEL =  "", VAR_TXT_LIST = [], \
                              index_list = [], interval=1000, color = False, dt = 1):
        """Plot animation for a list of tuning curves. Same alignment as plot() function."""

        test_num = len(tc_list)
        # check dimensions
        numBin = tc_list[0].numBin
        numPop = tc_list[0].numPop
        for tc in tc_list:
            if tc.numPop!= numPop or tc.numBin != numBin:
                raise Exception('Dimension mismatch for different tuning curves!')
        # here use FP and FM as numbers
        FP = np.array(FP)
        FM = np.array(FM)
        if FP.any() and FM.any(): # nonempty 
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
        line_stim = []
        line_grad = []
        line_rate = []
        grad_lines_lists = []
        #         info_texts = []
        #         var_texts = []

        grad_max = np.max(np.array([np.max(np.fabs(tc.grad)) for tc in tc_list]))
        tau = np.max(np.array([np.max(tc.tau) for tc in tc_list]))

        #         FP = np.zeros(numPop)
        #         FM = np.zeros(numPop)
        #         for k in range(numPop):
        #             FP[k]  = np.max(np.array([np.max(tc.tuning[k]) for tc in tc_list]))
        #             FM[k]  = np.min(np.array([np.min(tc.tuning[k]) for tc in tc_list]))        
        rate_fp = np.max(np.array([np.max(tc.rate) for tc in tc_list]))
        rate_fm = np.min(np.array([np.min(tc.rate) for tc in tc_list]))
        stim_max = np.max(np.array([np.max(tc.stim) for tc in tc_list]))
        
        fig = plt.figure(figsize = (16,8))        
        ax_tuning = fig.add_subplot(2,2,1, xlim = (0,numBin), ylim = (FM-0.1*FP,FP+0.1*FP))
        ax_tuning.grid()               
        ax_stim = fig.add_subplot(2,2,2, xlim = (0,numBin), ylim=(-0.01,stim_max + 0.01))
        ax_stim.grid()
        ax_rate = fig.add_subplot(2,2,3, xlim = (0,numBin), ylim=(tau*(FM - 0.1*FP),tau*(FP+0.1*FP)))
        ax_rate.grid()
        ax_grad = fig.add_subplot(2,2,4, xlim = (0,numBin), ylim=(-1.1*grad_max,1.1*grad_max))
        ax_grad.grid()
        
        colors = ['b', 'g','y','k'] # at most 4 popoluations
        for p in range(numPop):
            line, = ax_tuning.plot([], [], color = colors[p])# , 'o-', lw=2
            line_tuning.append(line)
            line, = ax_rate.plot([], [],color = colors[p])
            line_rate.append(line)
            line, = ax_grad.plot([], [],color = colors[p])
            line_grad.append(line)

            lines_list = []
            for j in range(numBin):
                l, = ax_grad.plot([], []) 
                lines_list.append(l)
            grad_lines_lists.append(lines_list)
        line_stim, = ax_stim.plot([], [], color = 'r') # no population assotiated with stim

        def init():
            """initialize animation"""
            line_stim.set_data([], [])
            for p in range(numPop):
                line_tuning[p].set_data([], [])
                line_grad[p].set_data([], [])
                line_rate[p].set_data([], [])
            if PLOT_FP_FM:
                ax_tuning.plot(np.arange(numBin), np.ones(numBin)*FP,'--', color = 'r')
                ax_tuning.plot(np.arange(numBin), np.ones(numBin)*FM,'--', color = 'c')
                ax_tuning.text(-0.08,FP/(FP - FM + 0.2), r'$f_{+}$', transform=ax_tuning.transAxes, fontsize = 15)
                ax_tuning.text(-0.08,FM/(FP - FM + 0.2), r'$f_{-}$', transform=ax_tuning.transAxes, fontsize = 15)
            ax_tuning.set_xticks(list(np.arange(0,numBin,4)))   
            ax_grad.set_xticks(list(np.arange(0,numBin,4)))
            ax_rate.set_xticks(list(np.arange(0,numBin,4))) 
            ax_stim.set_xticks(list(np.arange(0,numBin,4)))
            return line_tuning, line_stim, line_rate, line_grad

        def animate(i):
            """perform animation step"""
            # global tuning_curve, dt
            # global tuning_list, grad_list, mean_list
            # n_iter = tuning_curve.num_iter
            ll = test_num
            curr_idx = index_list[i] # (i*dt)%ll
            curr_tuning = tc_list[i].tuning
            curr_stim =  tc_list[i].stim
            curr_rate = tc_list[i].rate
            curr_grad = tc_list[i].grad
            curr_info = tc_list[i].info
            
            for p in range(numPop):
                line_tuning[p].set_data(np.arange(numBin), curr_tuning[p]) # tuning_curve.tuning[p]
                line_rate[p].set_data(np.arange(numBin), curr_rate[p]) #tuning_curve.rate[p]                 
                line_grad[p].set_data(np.arange(numBin), curr_grad[p]) # tuning_curve.grad[p][:numBin]

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
                if i in XTICKS_IDX_LIST:                    
                    tuning = tc_list[i].tuning[p]
                    diff = np.diff(tuning) # a[n+1] - a[n]
                    pts = np.where(np.fabs(diff) > 1e-5)[0]
                    ax_tuning.set_xticks([0] + list(pts) + [numBin-1])
                    ax_grad.set_xticks([0] + list(pts) + [numBin-1])
                    ax_rate.set_xticks([0] + list(pts) + [numBin-1])
                    
            line_stim.set_data(np.arange(numBin), curr_stim)
            ax_tuning.set_title('Tuning Curve: index = %d '%curr_idx + VAR_LABEL + VAR_TXT_LIST[i]) # tuning_curve.info
            ax_stim.set_title('Stimulus with ' +  r'$\nu$ = %d, $\tau$ = %.1f'%(tc_list[i].nu, tc_list[i].tau))
            ax_rate.set_title('Rate Curve')
            ax_grad.set_title('Gradient of Mutual Information')
            
            # legends
            leg = ax_tuning.legend([r'$MI$ = %.4f'%(curr_info)], handlelength=0, \
                                   handletextpad=0, fancybox = True,\
                                   loc = 'center right', bbox_to_anchor=(-0.05,0.5), fontsize = 15)
            for item in leg.legendHandles:
                item.set_visible(False)
                
            if numPop ==1:
                ax_rate.legend([r'$\bar{f}$ = %.2f'%tc_list[i].average], loc='center right', \
                           fancybox = True,bbox_to_anchor=(-0.05,0.5), fontsize = 15)
            else:
                ax_rate.legend([r'$\bar{f_{%d}}$ = %.2f'%(p,tc_list[i].average[p]) for p in range(numPop)], \
                           loc='center right',fancybox = True,bbox_to_anchor=(-0.05,0.5), fontsize = 15)
                
            
            return line_tuning, line_stim, line_rate, line_grad

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
                               ALIGN = "row",INCLUDE_GRAD = False,INCLUDE_INFO = True,\
                               index_list = [], interval=1000, color = False, dt = 1):
        # ALIGN = "row" or "col" (default = row)
        """Plot animation for a list of tuning curves. Showing each population in 
        separate subplots aligned in rows or columns. Not including stim function. """

        test_num = len(tc_list)
        # check dimensions
        numBin = tc_list[0].numBin
        numPop = tc_list[0].numPop
        delta = tc_list[0].delta
        nu = tc_list[0].nu
        for tc in tc_list:
            if tc.numPop!= numPop or tc.numBin != numBin:
                raise Exception('Dimension mismatch for different tuning curves!')
        FP = np.array(FP) # numpy array
        FM = np.array(FM)
        
        if FP.size and FM.size: # nonempty 
            PLOT_FP_FM = True
            if FP.size ==1:
                FP = FP[0]*np.ones(tc.numPop)
            if FM.size ==1:
                FM = FM[0]*np.ones(tc.numPop)
            elif FP.size != tc.numPop or FM.size != tc.numPop:
                raise Exception('Dimension mismatch for tuning curve f+/f-!')
            #if isinstance(FP, (int, float, complex)) and isinstance(FM, (int, float, complex)): # is number
            #    FP = FP*np.ones(tc.numPop)
            #    FM = FM*np.ones(tc.numPop)
        else:
            PLOT_FP_FM = False
            FP = np.zeros(numPop)
            FM = np.zeros(numPop)
            for k in range(numPop):
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

        def index_curve_constant(x, c, tol = 1e-5):
            x_c = np.where(np.fabs(x - c) < tol)[0]

            if len(x_c) ==0:
                return []
            else:
                loc = list(np.where(np.diff(x_c)>1)[0]) + [x_c.size - 1]
                idx_list = []
                for i in range(len(loc)):
                    if i ==0:
                        idx_list.append([x_c[i], x_c[loc[i]] ])
                    else:
                        idx_list.append([x_c[loc[i-1] + 1], x_c[loc[i]] ])
                return idx_list
            
        def get_xlabels(grid_space, label_space): # default: grid_space = nu, label_space = int(numBin/4)
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
        grad_lines_lists = []
        #info_texts = []
        var_texts = []

        grad_max = np.max(np.array([np.max(np.fabs(tc.grad)) for tc in tc_list]))
      
        tau = np.max(np.array([np.max(tc.tau) for tc in tc_list]))  
        
        if ALIGN == "row":
            n_rows = numPop
            if INCLUDE_GRAD:
                n_cols = 3 # tuning, rate, grad
            else:
                n_cols = 2 # tuning, rate
        else:
            if INCLUDE_GRAD:
                n_rows = 3
            else:
                n_rows = 2
            n_cols = numPop
        if INCLUDE_INFO:
            n_cols += 1
        #print n_rows, n_cols
        
        fig = plt.figure(figsize = (n_cols*6,n_rows*6)) #(numPop*10,3*6)
         
        info_list = np.array([tc.info for tc in tc_list])
        
        if INCLUDE_INFO:
            info_max = np.max(info_list)
            info_min = np.min(info_list)
            #print info_max, info_min
            ax_info = fig.add_subplot(1,n_cols,n_cols, xlim = (0, index_list[-1]+1), ylim = (info_min-0.1, info_max+0.1))
            ax_info.grid()
            ax_info.set_title('Mutual Information', fontsize = 14)
            line_info, = ax_info.plot([],[])
        
        for k in range(numPop):
             # row or col alignment
            
            if ALIGN == "row":
                ax1 = fig.add_subplot(n_rows, n_cols, n_cols*k+1, xlim = (0,numBin-1), ylim = (FM[k]-0.1*FP[k],FP[k]+0.1*FP[k]))        
                ax2 = fig.add_subplot(n_rows, n_cols,n_cols*k+2 , xlim = (0,numBin-1), ylim=(tau*(FM[k]-0.1*FP[k]),tau*(FP[k]+0.1*FP[k])))
                if INCLUDE_GRAD:
                    ax3 = fig.add_subplot(n_rows, n_cols, n_cols*k+3 , xlim = (0,numBin-1), ylim=(-1.1*grad_max,1.1*grad_max))
            else:
                ax1 = fig.add_subplot(n_rows, n_cols, k+1, xlim = (0,numBin-1), ylim = (FM[k]-0.1*FP[k],FP[k]+0.1*FP[k]))        
                ax2 = fig.add_subplot(n_rows, n_cols,n_cols+k+1 , xlim = (0,numBin-1), ylim=(tau*(FM[k]-0.1*FP[k]),tau*(FP[k]+0.1*FP[k])))
                if INCLUDE_GRAD:
                    ax3 = fig.add_subplot(n_rows, n_cols, 2*n_cols+k+1 , xlim = (0,numBin-1), ylim=(-1.1*grad_max,1.1*grad_max))

            ax1.set_title('Tuning Curve', fontsize = 14)
            ax1.grid()
            line1, = ax1.plot([], []) #line, = ax.plot([], [], 'o-', lw=2)
            ax2.set_title('Rate Curve', fontsize = 14)
            ax2.grid()
            line2, = ax2.plot([], []) #line, = ax.plot([], [], 'o-', lw=2)

            ax_tuning.append(ax1)
            ax_rate.append(ax2)
            line_tuning.append(line1)       
            line_rate.append(line2)
            if INCLUDE_GRAD:
                ax3.set_title('Gradient of Mutual Information', fontsize = 14)
                ax3.grid()
                line3, = ax3.plot([], []) #line, = ax.plot([], [], 'o-', lw=2)
                ax_grad.append(ax3)                  
                line_grad.append(line3)     
                lines_list = []
                for j in range(numBin):
                    l, = ax3.plot([], []) 
                    lines_list.append(l)
                grad_lines_lists.append(lines_list)
            
            vartxt = ax1.text(0.02,0.75,'',transform=ax1.transAxes, fontsize = 16)
            #info_texts.append(infotxt)
            var_texts.append(vartxt)
        if INCLUDE_INFO:
            infotxt = ax_info.text(0.35, 0.9, '', transform=ax_info.transAxes, fontsize = 16)
        else:
            infotxt = ax_tuning[0].text(0.02, 0.85, '', transform=ax_tuning[0].transAxes, fontsize = 16)
        
        def init():
            """initialize animation"""
            for p in range(numPop):
                line_tuning[p].set_data([], [])
                line_rate[p].set_data([], [])                
                if PLOT_FP_FM:
                    ax_tuning[p].plot(np.arange(numBin), np.ones(numBin)*FP[p],'--', linewidth=1.5)
                    ax_tuning[p].plot(np.arange(numBin), np.ones(numBin)*FM[p],'--', linewidth=1.5)
                    # f+, f- labels
                    #ax_tuning[p].text(-0.1,(FP[p]-FM[p]+0.09*FP[p])/(FP[p]-FM[p]+0.2*FP[p]),"f+", \
                    #                  transform=ax_tuning[p].transAxes,fontsize = 15)
                    #ax_tuning[p].text(-0.1,0.09*FP[p]/(FP[p]-FM[p]+0.2*FP[p]), "f-", \
                    #                  transform=ax_tuning[p].transAxes,fontsize = 15)
                    # newly added
                    ax_rate[p].plot(np.arange(numBin), np.ones(numBin)*FP[p]*tau,'--', linewidth=1.5)
                    ax_rate[p].plot(np.arange(numBin), np.ones(numBin)*FM[p]*tau,'--', linewidth=1.5)
                #ax_tuning[p].set_xticks(list(np.arange(0,numBin,8))+[numBin-1])
                #ax_rate[p].set_xticks(list(np.arange(0,numBin,8))+[numBin-1]) 
                ax_tuning[p].set_xticks(list(np.arange(0,numBin,nu))+[numBin-1])
                ax_rate[p].set_xticks(list(np.arange(0,numBin,nu))+[numBin-1])
                xlabels = get_xlabels(nu, int(numBin/4)) # grid_space = nu, label_space = numBin/4
                ax_tuning[p].set_xticklabels(xlabels)
                ax_rate[p].set_xticklabels(xlabels)
                
                neurons_arr = np.arange(0, numBin,delta)
                ax_tuning[p].plot(neurons_arr, (FM[p]-0.09*FP[p])*np.ones_like(neurons_arr),'o')
                ax_rate[p].plot(neurons_arr, tau*(FM[p]-0.09*FP[p])*np.ones_like(neurons_arr),'o')
                if INCLUDE_GRAD:
                    line_grad[p].set_data([], [])
                    #ax_grad[p].set_xticks(list(np.arange(0,numBin,8))+[numBin-1])
                    ax_grad[p].set_xticks(list(np.arange(0,numBin,nu))+[numBin-1])
                    ax_grad[p].set_xticklabels(xlabels)
                    ax_grad[p].plot(neurons_arr, (-1.09*grad_max)*np.ones_like(neurons_arr),'o')
            # set global title
            global_title = "Tuning Curve Optimization for %d bins, %d neurons, stimulus width=%d"%(numBin,numBin/delta, nu)
            # adding avg firing rate (not very important)
            # if numPop ==1:
            #     global_title += r', $\bar{f}$ = %.1f'%tc_list[0].average
            # else:
            #     global_title += r', $\bar{f}$ = ['
            #     for i in range(numPop-1):
            #         global_title += "%.1f,"%tc_list[0].average[i]
            #     global_title += "%.1f]"%tc_list[0].average[numPop-1]
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
            curr_rate = tc_list[i].rate
            if INCLUDE_INFO:
                line_info.set_data(index_list[0:i+1], info_list[0:i+1])
            infotxt.set_text('MI = %.4f' % curr_info)

            for p in range(numPop):
                line_tuning[p].set_data(np.arange(numBin), curr_tuning[p]) # tuning_curve.tuning[p]
                line_rate[p].set_data(np.arange(numBin), curr_rate[p]) #tuning_curve.rate[p] 
                #ax_tuning[p].set_title('Tuning Curve:index = %d'%curr_idx ) # tuning_curve.info

                #info_texts[p].set_text('MI = %.4f' % curr_info)
                var_texts[p].set_text( VAR_LABEL + VAR_TXT_LIST[i])
                if INCLUDE_GRAD:
                    line_grad[p].set_data(np.arange(numBin), curr_grad[p]) # tuning_curve.grad[p][:numBin]
                    # adding colors...
                    if color:
                        lines_list = grad_lines_lists[p]
                        idx_list1 = index_curve_constant(curr_tuning[p], FP[p]) #tuning_curve.tuning[p]
                        k = 0
                        for idx in idx_list1:
                            lines_list[k].set_data(np.arange(idx[0],idx[1]+1), curr_grad[p][idx[0]:idx[1]+1])#tuning_curve.grad[p]
                            lines_list[k].set_color('r')
                            lines_list[k].set_linewidth(1.5)#2.0
                            k += 1
                        idx_list2 = index_curve_constant(curr_tuning[p], FM[p])#tuning_curve.tuning[p]
                        for idx in idx_list2:
                            lines_list[k].set_data(np.arange(idx[0],idx[1]+1), curr_grad[p][idx[0]:idx[1]+1])#tuning_curve.grad[p]
                            lines_list[k].set_color('g')
                            lines_list[k].set_linewidth(1.5) # 2.0
                            k += 1
                        for j in range(k, numBin):
                            lines_list[j].set_data([], [])
                    # adding colors finished
                # set xticks
                if i in XTICKS_IDX_LIST: # e.g. XTICKS_IDX_LIST = [-1], i = -1
                    tuning = tc_list[i].tuning[p]
                    diff = np.diff(tuning) # a[n+1] - a[n]
                    pts = np.where(np.fabs(diff) > 1e-5)[0]
                    ax_tuning[p].set_xticks([0] + list(pts) + [numBin-1])
                    ax_rate[p].set_xticks([0] + list(pts) + [numBin-1])
                    if IF_GRAD:
                        ax_grad[p].set_xticks([0] + list(pts) + [numBin-1])
                # adjust linewidth
                for p in range(numPop):
                    line_tuning[p].set_linewidth(1.5)
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
        

