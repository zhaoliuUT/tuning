import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import time
import os
import scipy
from scipy import optimize
from sklearn.cluster import KMeans
from scipy import special

import tuning
from tuning.cyMINoncyclic import mc_mean_grad_gaussian, mc_mean_grad_gaussian_inhomo # gaussian models
from tuning.cyMINoncyclic import mc_coeff_arimoto_gaussian, mc_coeff_arimoto_gaussian_inhomo # gaussian models
from tuning.repulsive_model import tuning_update_inhomo_periodic, conditional_probability_matrix_vonmises
from tuning.repulsive_model import evaluate_elastic_term_periodic, mutual_distance
from tuning.functions_for_analysis import gen_binary_hierachical_curves
from tuning.anim_3dcube import pc_fun_weights, plot_funcs_in_figure
# for 3d illustration:
# from tuning.anim_3dcube import create_figure_canvas, set_data_in_figure, get_color_array, gen_mixed_plots, gen_mixed_anim

# ----------initialization---------  
numNeuro = 5
tc = gen_binary_hierachical_curves(numNeuro, fp = 0.8, fm = -0.8)

# fig = plt.figure()
# _ = plot_funcs_in_figure(fig, tc, np.ones(tc.shape[1]), nrow=numNeuro, ncol=1, fp = 1, fm = -1)

numBin = 1024 #32*32
tuning = np.zeros((numNeuro, numBin))
num_pts = int(numBin/tc.shape[1])
vec = np.linspace(0, 1, num_pts + 1)[:num_pts]
for k in range(tc.shape[1]):
    tuning[:, k*num_pts:(k+1)*num_pts] = tc[:, k][:, None] + vec*(tc[:,(k+1)%tc.shape[1]] - tc[:, k])[:, None]
tuning[tuning < -1] = -1
tuning[tuning > 1] = 1
weight = 1.0*np.ones(numBin)/numBin
    
# fig = plt.figure()
# _ = plot_funcs_in_figure(fig, tuning, weight, nrow=numNeuro, ncol=1, fp = 1, fm = -1)
# from mpl_toolkits.mplot3d import Axes3D
# if numNeuro ==2 :
#     plt.figure()
#     plt.scatter(tuning[0,:], tuning[1,:])
#     plt.xlim([-1, 1])
#     plt.ylim([-1, 1])
# elif numNeuro == 3:
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.scatter(tuning[0,:], tuning[1,:], tuning[2,:])

#     ax.set_xlabel('x0')
#     ax.set_ylabel('x1')
#     ax.set_zlabel('x2')
#     ax.set_xlim([-1, 1])
#     ax.set_ylim([-1, 1])
#     ax.set_zlim([-1, 1])

#     plt.show()

# ----------optimization function, derivative, bounds, constraint---------  
# added self energy term
def opt_fun(w, A, gamma):
    return np.dot(w, (np.dot(A, w))) + gamma*np.sum(w**2)
            
def grad_fun(w, A, gamma):
    return np.dot(A + np.transpose(A), w) + 2*gamma*w

weight_bounds = []
for j in range(len(weight)): # number of bins
    weight_bounds += [(0, 1)]
weight_bounds = tuple(weight_bounds)

my_cons = [{'type':'eq', 'fun': lambda w: (np.sum(w) - 1)}]

# ----------parameters of the Von-mises model---------
kappa = 5
kappa_mat = np.zeros_like(tuning)
kappa_mat[:, :] = kappa
# print(kappa_mat)
alpha = 5
gamma = 0

# ---------initialization of results-saving-------
tuning_list = [tuning.copy()]
weight_list = [weight.copy()]

opt_list = []
info_list = []
EVALUATE_OPT = True
EVALUATE_EL = True
EVALUATE_MI = False
eta_list = [None]
beta_list = [None]
Lambda_list = [None]
elastic_term_list = [None]
elastic_value_list = [None]
gamma_list = [gamma]

tuning = tuning_list[0].copy()

if EVALUATE_OPT:
    P = conditional_probability_matrix_vonmises(tuning, kappa_mat, alpha = alpha)
    opt_list += [opt_fun(weight, P, gamma)]
    print(opt_list)
#--------start iterations-----------
ncol = 1
nrow = numNeuro
T = 100

eta = 20
beta = 0#5e-4

elastic_term = 'exp'
Lambda = 2

ALTER_WEIGHT = T+1

EVALUATE_OPT = True
EVALUATE_EL = True
EVALUATE_MI = False
PLOT = True
ALTER_PLOT = 10

fp = 1; fm = -1

if PLOT:
    %matplotlib notebook

    fig = plt.figure(figsize = (ncol*6, nrow*2))
    ax_list = []
    for i in range(numNeuro):
        ax = fig.add_subplot(nrow,ncol,i+1)
        ax_list.append(ax)

    for i in range(numNeuro):
        xx, yy = pc_fun_weights(tuning[i,:], weight)
        ax_list[i].plot(xx, yy)
        ax_list[i].set_ylim([fm, fp])
    #     ax_list[i].scatter(np.arange(numBin), tuning[i, :])

    fig.canvas.draw()

        
for num_iter in range(1, T+1):
    
    tuning_new = tuning_update_inhomo_periodic(
        tuning, weight, kappa_mat, alpha = alpha,
        eta = eta, beta = beta, Lambda = Lambda, elastic_term = elastic_term,
    )
    
    tuning = tuning_new.copy()

    
    if num_iter % ALTER_WEIGHT ==0:        
        D = conditional_probability_matrix_vonmises(tuning, kappa_mat, alpha=alpha)
        disp = False
        ftol = 1e-6
        res = optimize.minimize(opt_fun, weight, args = (D,gamma), \
                                method='SLSQP', jac = grad_fun, \
                                bounds = weight_bounds, constraints = my_cons, \
                                options = {'maxiter':WEIGHT_ITER_NUM, 'ftol': ftol, 'disp': disp})

        weight_new = res['x'].copy()
        print(res['fun'], opt_fun(weight, D, gamma))
        #print(weight_new)
        weight = weight_new.copy()
        
    tuning_list += [tuning.copy()]
    weight_list += [weight.copy()]
    
    eta_list += [eta]
    beta_list += [beta]
    Lambda_list += [elastic_term]
    elastic_term_list += [Lambda]
    gamma_list += [gamma]
    
    if EVALUATE_OPT:
        D = conditional_probability_matrix_vonmises(tuning, kappa_mat, alpha=alpha)   
        opt_list += [opt_fun(weight, D, gamma)]     
    if EVALUATE_EL:
        elastic_value_list += [
            evaluate_elastic_term_periodic(tuning, weight, alpha = alpha, 
                                           beta = beta, Lambda=Lambda, 
                                           elastic_term = elastic_term,
                                           elastic_term_periodic = True,
                                          )]

#     if EVALUATE_MI:
#         grad_tc = np.zeros_like(tuning)       
#         info_tc = mc_mean_grad_gaussian_inhomo(
#             grad_tc, tuning, weight,
#             inv_cov_matrix, conv, tau, int(1e4), my_num_threads = NUM_THREADS)
#         info_list += [info_tc/np.log(2)]

    if PLOT and num_iter%ALTER_PLOT==0:
        for i in range(numNeuro):
            ax_list[i].clear()
            xx, yy = pc_fun_weights(tuning[i,:], weight)
            ax_list[i].plot(xx, yy)
            ax_list[i].set_ylim([fm, fp])
#         time.sleep(0.1)
        fig.canvas.draw()
    if num_iter%10 == 0:
        print('num_iter=%d, opt = %.2f'%(num_iter, opt_list[-1]))
        
#-------showing and saving results----------
# plt.figure()
# plt.plot(opt_list)

# plt.figure()
# plt.plot(elastic_value_list)

# #--comparing histograms of mutual distances----
# tc = tuning_list[0]
# num_nearst_neighbours = 50
# dS = mutual_distance(tc)
# sorted_dS = np.sort(dS, axis = 1)
# plt.figure()
# _ = plt.hist(sorted_dS[:, 1:num_nearst_neighbours + 1].reshape(-1), bins = 100)

# tc = tuning_list[-1]
# dS = mutual_distance(tc)
# num_nearst_neighbours = 50
# sorted_dS = np.sort(dS, axis = 1)
# plt.figure()
# _ = plt.hist(sorted_dS[:, 1:num_nearst_neighbours + 1].reshape(-1), bins = 100)


FILE_NAME = "dim=%d_kappa=%.2f_alpha=%.2f_beta=%.1e_"%(
    numNeuro, kappa_mat[0,0],alpha, beta) + time.strftime("%m%d-%H%M%S")

res_dict = {}
res_dict['kappa'] = kappa_mat
res_dict['alpha'] = alpha

res_dict['eta'] = eta_list
res_dict['beta'] = beta_list
res_dict['gamma'] = gamma_list
res_dict['Lambda'] = Lambda_list
res_dict['elastic_term'] = elastic_term_list

res_dict['tuning'] = tuning_list
res_dict['weight'] = weight_list

res_dict['opt'] = opt_list
res_dict['info'] = info_list
res_dict['elastic'] = elastic_value_list

print(FILE_NAME)
np.save(FILE_NAME, res_dict)

# fig = plt.figure()
# ax_list = plot_funcs_in_figure(fig, tuning, weight, ncol = 1, nrow = numNeuro, fp = 1, fm = -1)
# ax_list[0].set_title(r'$\kappa = %.1f, \alpha = %.1f, \beta = %.1e$, F(x,w) = %.4f'%(kappa_mat[0, 0], alpha, beta, opt_list[-1]))
# plt.savefig('./figures/' + FILE_NAME+'.jpg')

# from mpl_toolkits.mplot3d import Axes3D
# if numNeuro ==2 :
#     plt.figure()
#     plt.scatter(tuning[0,:], tuning[1,:])
#     if beta != 0:
#         plt.plot(tuning[0, :], tuning[1, :], '--r', linewidth = 0.5)
#     plt.xlim([-1, 1])
#     plt.ylim([-1, 1])
#     plt.savefig('./figures/' + FILE_NAME+'_points.jpg')
# elif numNeuro == 3:
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.scatter(tuning[0,:], tuning[1,:], tuning[2,:])
#     if beta != 0:
#         ax.plot(tuning[0, :], tuning[1, :], tuning[2, :], '--r', linewidth = 0.5)

#     ax.set_xlabel('x0')
#     ax.set_ylabel('x1')
#     ax.set_zlabel('x2')
#     ax.set_xlim([-1, 1])
#     ax.set_ylim([-1, 1])
#     ax.set_zlim([-1, 1])

#     ax.set_title(r'$\kappa = %.1f, \alpha = %.1f, \beta = %.1f$, F(x,w) = %.4f'%(kappa_mat[0, 0], alpha, beta, opt_list[-1]))
#     plt.show()
#     plt.savefig('./figures/' + FILE_NAME+'_points.jpg')