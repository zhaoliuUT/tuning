import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import time
import os
import scipy
from scipy import optimize
from sklearn.cluster import KMeans
from scipy import special

# import tuning
# from tuning.cyMINoncyclic import mc_mean_grad_gaussian, mc_mean_grad_gaussian_inhomo # gaussian models
# from tuning.cyMINoncyclic import mc_coeff_arimoto_gaussian, mc_coeff_arimoto_gaussian_inhomo # gaussian models
from tuning.repulsive_model import tuning_update_inhomo, conditional_probability_matrix_gaussian
from tuning.repulsive_model import evaluate_elastic_term, mutual_distance
from tuning.functions_for_analysis import gen_binary_hierachical_curves
from tuning.anim_3dcube import pc_fun_weights, plot_funcs_in_figure
# for 3d illustration:
# from tuning.anim_3dcube import create_figure_canvas, set_data_in_figure, get_color_array, gen_mixed_plots, gen_mixed_anim


# # ----------initialization---------  
numNeuro = 2

numBin = 1020 

num_periods = 1
bn_tc = np.array([0.2, 0.8]*num_periods) # repetations of [0, 1, 0, 1, ...]
len_v = int(numBin/(2*num_periods))
vec = np.linspace(0, 1, len_v + 1)[:len_v]

tuning = np.zeros((numNeuro, numBin))
for k in range(len(bn_tc)):
    tuning[0, k*len_v:(k+1)*len_v] = bn_tc[k] + vec*(bn_tc[(k+1)%len(bn_tc)] - bn_tc[k])
    
len_shift = int(numBin/numNeuro)
for i in range(numNeuro):
    tuning[i,:] = np.roll(tuning[0,:], len_shift*i)



weight = 1.0*np.ones(numBin)/numBin

# fig = plt.figure()
# _ = plot_funcs_in_figure(fig, tuning, weight, nrow=numNeuro, ncol=1, fp = 1, fm = 0)

# # ----------optimization function, derivative, bounds, constraint---------  
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

# # ----------parameters of the Gaussian model---------
sigma=0.2
sigma_vec = np.array([sigma]*len(weight))
# print(kappa_mat)
alpha = 3
gamma = 0



# # ---------initialization of results-saving-------
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
    P = conditional_probability_matrix_gaussian(tuning, sigma_vec, alpha = alpha)
    opt_list += [opt_fun(weight, P, gamma)]
    print(opt_list)


# m = 1
# tuning_list = tuning_list[0:m]#[tuning_list[0].copy()][tuning_list[1].copy()]
# weight_list = weight_list[0:m]
# opt_list = opt_list[0:m]
# info_list = [] #
# eta_list = eta_list[0:m]
# beta_list = beta_list[0:m]
# Lambda_list = Lambda_list[0:m]
# elastic_term_list = elastic_term_list[0:m]
# gamma_list = gamma_list[0:m]
# elastic_value_list = elastic_value_list[0:m]

# tuning = tuning_list[-1].copy()
# weight = weight_list[-1].copy()

# #--------start iterations-----------
ncol = 1
nrow = numNeuro
T = 100

eta = 10
beta = 5e-3

elastic_term = 'exp'
Lambda = 0.2

ALTER_WEIGHT = T+1

EVALUATE_OPT = True
EVALUATE_EL = True
EVALUATE_MI = False
PLOT = True
ALTER_PLOT = 10

fp = 1; fm = 0

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
    
    tuning_new = tuning_update_inhomo(
        tuning, weight, sigma_vec, alpha = alpha,
        eta = eta, beta = beta, Lambda = Lambda, elastic_term = elastic_term,
    )
    
    tuning = tuning_new.copy()

    
    if num_iter % ALTER_WEIGHT ==0:        
        D = conditional_probability_matrix_gaussian(tuning, sigma_vec, alpha=alpha)
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
        D = conditional_probability_matrix_gaussian(tuning, sigma_vec, alpha=alpha)   
        opt_list += [opt_fun(weight, D, gamma)]     
    if EVALUATE_EL:
        elastic_value_list += [
            evaluate_elastic_term(tuning, weight, alpha = alpha, 
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
    if num_iter%100 == 0:
        print('num_iter=%d, opt = %.2f'%(num_iter, opt_list[-1]))
        

# #-------showing and saving results----------
plt.figure()
plt.plot(opt_list)

plt.figure()
plt.plot(elastic_value_list)


# plt.figure()
# _ = plt.hist(tuning.reshape(-1), bins = 500)

# plt.figure()
# _ = plt.hist2d(tuning[0,:], tuning[1,:], bins = 50)


# #--histograms of distances----

# tc = tuning_list[-1]
# dS = mutual_distance(tc)

# num_neighbours = 50
# dist_list = []
# for i in range(1, num_neighbours+1):
#     dist_list += list(np.diagonal(dS, offset = i))

# plt.figure()
# _ = plt.hist(dist_list, bins = 100)


FILE_NAME = "gauss_new/dim=%d_sigma=%.2f_alpha=%.2f_beta=%.1e_"%(
    numNeuro, sigma_vec[0],alpha, beta) + time.strftime("%m%d-%H%M%S")

res_dict = {}
res_dict['sigma'] = sigma_vec
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
# ax_list = plot_funcs_in_figure(fig, tuning, weight, ncol = 1, nrow = numNeuro, fp = 1, fm = 0)
# ax_list[0].set_title(r'$\sigma = %.1f, \alpha = %.1f, \beta = %.1e$, F(x,w) = %.4f'%(sigma_vec[0], alpha, beta, opt_list[-1]))
# plt.savefig(FILE_NAME+'.jpg')

# fig = plt.figure()
# ax_list = plot_funcs_in_figure(fig, tuning_list[0], weight_list[0], ncol = 1, nrow = numNeuro, fp = 1, fm = 0)
# ax_list[0].set_title(r'$\sigma = %.1f, \alpha = %.1f, \beta = %.1e$, F(x,w) = %.4f'%(sigma_vec[0], alpha, beta, opt_list[0]))
# plt.savefig(FILE_NAME+'_initial.jpg')


# from mpl_toolkits.mplot3d import Axes3D
# if numNeuro ==2 :
#     plt.figure()
#     plt.scatter(tuning[0,:], tuning[1,:])
#     if beta != 0:
#         plt.plot(tuning[0, :], tuning[1, :], '--r', linewidth = 0.5)
#     plt.xlim([-0.01, 1.01])
#     plt.ylim([-0.01, 1.01])
#     plt.savefig(FILE_NAME+'_points.jpg')
# elif numNeuro == 3:
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.scatter(tuning[0,:], tuning[1,:], tuning[2,:])
#     if beta != 0:
#         ax.plot(tuning[0, :], tuning[1, :], tuning[2, :], '--r', linewidth = 0.5)

#     ax.set_xlabel('x0')
#     ax.set_ylabel('x1')
#     ax.set_zlabel('x2')
#     ax.set_xlim([0, 1])
#     ax.set_ylim([0, 1])
#     ax.set_zlim([0, 1])

#     ax.set_title(r'$\sigma = %.1f, \alpha = %.1f, \beta = %.1f$, F(x,w) = %.4f'%(sigma_vec[0], alpha, beta, opt_list[-1]))
#     plt.show()
#     plt.savefig(FILE_NAME+'_points.jpg')

# #-------loading results----------
# FILE_NAME = ""

# res_dict = np.load(FILE_NAME+'.npy', allow_pickle=True, encoding = 'latin1').item()


# fig = plt.figure()
# ax_list = plot_funcs_in_figure(fig, res_dict['tuning'][-1], res_dict['weight'][-1], 
#                                ncol = 1, nrow = res_dict['tuning'][-1].shape[0], fp = 1, fm = 0)
# ax_list[0].set_title(r'$\sigma = %.1f, \alpha = %.1f, \beta = %.1e$, F(x,w) = %.4f'%(
#     res_dict['sigma'][0], res_dict['alpha'], res_dict['beta'][-1], res_dict['opt'][-1]))
# plt.savefig(FILE_NAME+'.jpg')

# fig = plt.figure()
# ax_list = plot_funcs_in_figure(fig, res_dict['tuning'][0], res_dict['weight'][0],  
#                                ncol = 1, nrow = res_dict['tuning'][0].shape[0], fp = 1, fm = 0)
# ax_list[0].set_title(r'$\sigma = %.1f, \alpha = %.1f, \beta = %.1e$, F(x,w) = %.4f'%(
#     res_dict['sigma'][0], res_dict['alpha'], res_dict['beta'][-1], res_dict['opt'][0]))
# plt.savefig(FILE_NAME+'_initial.jpg')

# tuning = res_dict['tuning'][-1].copy()
# from mpl_toolkits.mplot3d import Axes3D
# if numNeuro ==2 :
#     plt.figure()
#     plt.scatter(tuning[0,:], tuning[1,:])
#     if beta != 0:
#         plt.plot(tuning[0, :], tuning[1, :], '--r', linewidth = 0.5)
#     plt.xlim([-0.01, 1.01])
#     plt.ylim([-0.01, 1.01])
#     plt.savefig(FILE_NAME+'_points.jpg')
# elif numNeuro == 3:
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.scatter(tuning[0,:], tuning[1,:], tuning[2,:])
#     if beta != 0:
#         ax.plot(tuning[0, :], tuning[1, :], tuning[2, :], '--r', linewidth = 0.5)

#     ax.set_xlabel('x0')
#     ax.set_ylabel('x1')
#     ax.set_zlabel('x2')
#     ax.set_xlim([0, 1])
#     ax.set_ylim([0, 1])
#     ax.set_zlim([0, 1])

#     ax.set_title(r'$\sigma = %.1f, \alpha = %.1f, \beta = %.1f$, F(x,w) = %.4f'%(
#         res_dict['sigma'][0], res_dict['alpha'], res_dict['beta'][-1], res_dict['opt'][-1]))
#     plt.show()
#     plt.savefig(FILE_NAME+'_points.jpg')

