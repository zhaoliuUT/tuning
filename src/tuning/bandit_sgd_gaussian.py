import numpy as np
from tuning.cyMINoncyclic import mc_mean_grad_gaussian
import matplotlib.pyplot as plt


#============Gaussian Noncyclic model with NO constraints============

# -----------Bandit Algorithm helper functions-----------

def gaussian_log_ratio(bin_index, output_value, input_set, input_prob_vec, inverse_cov_matrix):
    # bin_index: i = 0,1,...,numBin-1
    # output_value: response r (vector with length=numNeuro)
    # input_set: {\lambda_{k,j}} (vector shape=(numNeuro,numBin)) (same as tuning_curve) (all nonzero)
    # input_prob_vec: {w_j} (list or 1-d vector with length = numBin)
    # inverse_cov_matrix: numNeuro-by-numNeuro numpy array.
    if input_set.ndim == 1:
        input_set = input_set.reshape((1,-1))
    numNeuro, numBin = input_set.shape
    
    sum_exp = 0
    for l in range(numBin):
        vec_l = output_value - input_set[:,l]
        vec_bin_index = output_value - input_set[:, bin_index]
        quad_l = np.dot(vec_l, np.dot(inverse_cov_matrix, vec_l))
        quad_bin_index = np.dot(vec_bin_index, np.dot(inverse_cov_matrix, vec_bin_index))
        sum_exp += input_prob_vec[l]*np.exp(-0.5*(quad_l - quad_bin_index))
        
    return -np.log(sum_exp)


# -----------Bandit Algorithm for Arimoto iterations-----------

def gaussian_bandit_iteration(input_set, inverse_cov_matrix,
                              initial_prob_vec = None, max_iter = 1000, batch_size = 1, 
                              dkl_discount_factor = "decrease", epsilon=0,                               
                              update_rule = "additive",#"direct", "multiply"
                              initial_learning_rate = 0.01, learning_rate_decrease_rate = 0):
    # 'temporal difference scheme'
    # dkl_discount_factor = "decrease" or [0,1) e.g. 0.9
    # dkl_discount_factor = 0 is the same as using argmax(rewards) ignoring path history
    # epsilon probability on selecting other choices
    
    if not (dkl_discount_factor == "decrease" or (dkl_discount_factor >=0 and dkl_discount_factor <=1)):
        raise Exception("dkl_discount_factor must be in [0,1] or 'decrease'!")
    if not (update_rule in ["additive", "direct", "multiply"]):
        raise Exception("update_rule must be in ['additive', 'direct','multiply']!")
    
    if input_set.ndim == 1:
        input_set = input_set.reshape((1,-1))
    numNeuro, numBin = input_set.shape
    #num_inputs = numBin
    if initial_prob_vec is None:
        prob_vec = 1.0*np.ones(numBin)/numBin # initalize with equal weights
    else:
        prob_vec = initial_prob_vec.copy()

    rewards = np.zeros(numBin)
    DKL_estimates = np.zeros(numBin)
    
    num_iter = 0
    rewards_list = []
    DKL_estimates_list = []
    prob_vec_list = [prob_vec.copy()]
    
    # generate gaussian samples in advance
    cov_matrix = np.linalg.inv(inverse_cov_matrix)
    
    gaussian_samples = np.zeros((max_iter*batch_size, numNeuro, numBin))
    for j in range(numBin):
        gaussian_samples[:,:,j] = np.random.multivariate_normal(input_set[:,j], 
                                                                cov_matrix, size = max_iter*batch_size)
    
    while(num_iter < max_iter):
        rewards *= 0
        for k in range(batch_size):
            for i in range(numBin):
                # compute rewards by sampling
                output_sample = gaussian_samples[num_iter*batch_size+k, :, i]
                rewards[i] += gaussian_log_ratio(i, output_sample, input_set, 
                                                 prob_vec, inverse_cov_matrix)
        rewards /= batch_size
        
        
        # update the DKL_estimates with discounting factor       
        if dkl_discount_factor == "decrease":
            DKL_estimates = 1.0/(num_iter+1)*rewards + 1.0*num_iter/(num_iter+1)*DKL_estimates                
        else:
            DKL_estimates = (1-dkl_discount_factor)*rewards + dkl_discount_factor*DKL_estimates
              
        alpha = initial_learning_rate/(learning_rate_decrease_rate*num_iter + 1)

        if update_rule == "additive":
            u = np.random.uniform()
            if u < epsilon:
                optimal_index = np.random.choice(numBin)
            else:
                optimal_index = np.argmax(DKL_estimates)
            prob_vec *= 1.0 - alpha
            prob_vec[optimal_index] += alpha
        elif update_rule == "multiply":
            prob_vec *= np.exp(rewards)**alpha
            prob_vec /= np.sum(prob_vec)
        elif update_rule == "direct":
            prob_vec *= np.exp(DKL_estimates)
            prob_vec /= np.sum(prob_vec)
            
        rewards_list.append(rewards.copy())
        DKL_estimates_list.append(DKL_estimates.copy())
        prob_vec_list.append(prob_vec.copy())
        
        num_iter += 1
        
    return prob_vec, rewards_list, DKL_estimates_list, prob_vec_list

# -----------SGD Algorithm-----------

def simple_sgd_gaussian(tuning, weight, inv_cov_mat, eta, NUM_ITER, fp, fm, MC_ITER = 1, conv = None, tau = 1.0, NUM_THREADS = 4):
    curve_list = []
    grad_list = []
    numBin = len(weight)
    if conv is None:
        conv = np.zeros(numBin)
        conv[0] = 1
        
    x = tuning.copy()
    x_grad = np.zeros_like(tuning)
    
    for i in range(NUM_ITER):
        x_grad *= 0
        x_mean = mc_mean_grad_gaussian(x_grad, x, weight, inv_cov_mat, conv, tau, numIter=MC_ITER, my_num_threads=NUM_THREADS)
        x += eta*x_grad
        x[x>fp] = fp
        x[x<fm] = fm
        curve_list.append(x.copy())
        grad_list.append(x_grad.copy())
        
    return curve_list, grad_list

#------------Plotting function------------

def plot_info_alternate(ax, info_list, mark_list, index_list=None, color_sgd = 'r', color_bandit = 'b'):
    if index_list is None:
        index_list = np.arange(len(info_list))
    else:
        index_list = np.array(index_list)

    sgd_idx = [i for i in index_list if mark_list[i]=='sgd']
    bandit_idx = [i for i in index_list if mark_list[i]=='bandit']

    #plt.figure(figsize=(16,8))
    for idx in sgd_idx:
        ax.plot([idx-1, idx], [info_list[idx-1], info_list[idx]], c=color_sgd)

    for idx in bandit_idx:
        ax.plot([idx-1, idx], [info_list[idx-1], info_list[idx]], c=color_bandit)


def plot_bandit_iteration_results(DKL_estimates_list, prob_vec_list, rewards_list):
    DKL_estimates_arr = np.array(DKL_estimates_list)
    prob_vec_arr = np.array(prob_vec_list)
    rewards_arr = np.array(rewards_list)
    numBin = prob_vec_arr.shape[1]
    
    plt.subplot(1,3,1)
    for i in range(numBin):
        if(new_prob_vec[i] > 1e-3):
            plt.plot(DKL_estimates_arr[:,i], label = '%d'%i)
        else:
            plt.plot(DKL_estimates_arr[:,i], '--', label = '%d'%i)
    plt.legend(ncol = 5)
    plt.title('DKL_estimates')
    
    plt.subplot(1,3,2)
    for i in range(numBin):
        if(new_prob_vec[i] > 1e-3):
            plt.plot(rewards_arr[:,i], label = '%d'%i)
        else:
            plt.plot(rewards_arr[:,i], '--', label = '%d'%i)
    plt.legend(ncol = 5)
    plt.title('Rewards')
    
    plt.subplot(1,3,3)
    for i in range(numBin):
        plt.plot(prob_vec_arr[:,i],  label = '%d'%i)
    plt.legend(ncol = 5)
    plt.title('Probability vectors')

#=========Gaussian Noncyclic model, Inhomogeneous ('inverse_cov_matrix' is 3d)============

def gaussian_log_ratio_inhomo(bin_index, output_value, input_set, input_prob_vec, 
                       inverse_cov_matrix, rho):
    # bin_index: i = 0,1,...,numBin-1
    # output_value: response r (vector with length=numNeuro)
    # input_set: {\lambda_{k,j}} (vector shape=(numNeuro,numBin)) (same as tuning_curve) (all nonzero)
    # input_prob_vec: {w_j} (list or 1-d vector with length = numBin)
    # inverse_cov_matrix: numNeuro*numNeuro*numBin numpy array.
    # rho: numBin numpy array
    if input_set.ndim == 1:
        input_set = input_set.reshape((1,-1))
    numNeuro, numBin = input_set.shape
    
    vec_bin_index = output_value - input_set[:, bin_index]
    quad_bin_index = np.dot(vec_bin_index, np.dot(inverse_cov_matrix[:,:,bin_index], vec_bin_index))
    
    sum_exp = 0
    for l in range(numBin):
        vec_l = output_value - input_set[:,l]        
        quad_l = np.dot(vec_l, np.dot(inverse_cov_matrix[:,:,l], vec_l))
        sum_exp += input_prob_vec[l]*rho[l]/rho[bin_index]*np.exp(-0.5*(quad_l - quad_bin_index))
        
    return -np.log(sum_exp)

def gaussian_DKL_inhomo(bin_index, gaussian_samples, input_set, input_prob_vec, inverse_cov_matrix, rho):
    """
    Compute D_KL(p(y|x) || p(y)) given the value of x and the samples of y according to p(y|x_value).
    log_ratio_fun: log(p(y|x)/p(y))
                    format: log_ratio_fun(x_value, y_value, args)
    x_value: the value of x
    y_samples: a list of samples from p(y|x_value)
               Note that should be a list, ( [sample1, sample2, ..., sample10,...]),
               or a numpy array with dimension number_of_samples * y_dimension. Store y in each row.
    """
    dkl = 0
    if gaussian_samples.ndim == 1:
        gaussian_samples = gaussian_samples.reshape(1, -1)        
    num_samples, numNeuro = gaussian_samples.shape
    
    for i in range(num_samples):
        dkl += gaussian_log_ratio_inhomo(bin_index, gaussian_samples[i,:], 
                                  input_set, input_prob_vec, inverse_cov_matrix, rho)
    return dkl/num_samples

def gaussian_bandit_iteration_inhomo(input_set, inverse_cov_matrix,
                              initial_prob_vec = None, max_iter = 1000, batch_size = 1, 
                              dkl_discount_factor = "decrease", epsilon=0,                               
                              update_rule = "additive",#"direct", "multiply"
                              initial_learning_rate = 0.01, learning_rate_decrease_rate = 0):
    # 'temporal difference scheme'
    # dkl_discount_factor = "decrease" or [0,1) e.g. 0.9
    # dkl_discount_factor = 0 is the same as using argmax(rewards) ignoring path history
    # epsilon probability on selecting other choices
    
    if not (dkl_discount_factor == "decrease" or (dkl_discount_factor >=0 and dkl_discount_factor <=1)):
        raise Exception("dkl_discount_factor must be in [0,1] or 'decrease'!")
    if not (update_rule in ["additive", "direct", "multiply"]):
        raise Exception("update_rule must be in ['additive', 'direct','multiply']!")
    
    if input_set.ndim == 1:
        input_set = input_set.reshape((1,-1))
    numNeuro, numBin = input_set.shape
    #num_inputs = numBin
    if initial_prob_vec is None:
        prob_vec = 1.0*np.ones(numBin)/numBin # initalize with equal weights
    else:
        prob_vec = initial_prob_vec.copy()

    # rho: numBin, rho[j] = det(inv_cov_mat[:,:,j])^(1/2)
    rho = np.zeros(numBin)
    for j in range(numBin):
        rho[j] = np.sqrt(np.linalg.det(inverse_cov_matrix[:,:,j]))

    rewards = np.zeros(numBin)
    DKL_estimates = np.zeros(numBin)
    
    num_iter = 0
    rewards_list = []
    DKL_estimates_list = []
    prob_vec_list = [prob_vec.copy()]
    
    # generate gaussian samples in advance
        
    gaussian_samples = np.zeros((max_iter*batch_size, numNeuro, numBin))
    for j in range(numBin):
        cov_matrix = np.linalg.inv(inverse_cov_matrix[:,:,j])
        gaussian_samples[:,:,j] = np.random.multivariate_normal(input_set[:,j], 
                                                                cov_matrix, size = max_iter*batch_size)
    
    while(num_iter < max_iter):
        rewards *= 0
        for k in range(batch_size):
            for i in range(numBin):
                # compute rewards by sampling
                output_sample = gaussian_samples[num_iter*batch_size+k, :, i]
                rewards[i] += gaussian_log_ratio_inhomo(i, output_sample, input_set, 
                                                 prob_vec, inverse_cov_matrix, rho)
        rewards /= batch_size
        
        
        # update the DKL_estimates with discounting factor       
        if dkl_discount_factor == "decrease":
            DKL_estimates = 1.0/(num_iter+1)*rewards + 1.0*num_iter/(num_iter+1)*DKL_estimates                
        else:
            DKL_estimates = (1-dkl_discount_factor)*rewards + dkl_discount_factor*DKL_estimates
              
        alpha = initial_learning_rate/(learning_rate_decrease_rate*num_iter + 1)

        if update_rule == "additive":
            u = np.random.uniform()
            if u < epsilon:
                optimal_index = np.random.choice(numBin)
            else:
                optimal_index = np.argmax(DKL_estimates)
            prob_vec *= 1.0 - alpha
            prob_vec[optimal_index] += alpha
        elif update_rule == "multiply":
            prob_vec *= np.exp(rewards)**alpha
            prob_vec /= np.sum(prob_vec)
        elif update_rule == "direct":
            prob_vec *= np.exp(DKL_estimates)
            prob_vec /= np.sum(prob_vec)
            
        rewards_list.append(rewards.copy())
        DKL_estimates_list.append(DKL_estimates.copy())
        prob_vec_list.append(prob_vec.copy())
        
        num_iter += 1
        
    return prob_vec, rewards_list, DKL_estimates_list, prob_vec_list
