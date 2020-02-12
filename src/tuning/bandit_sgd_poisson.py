import numpy as np
from tuning.cyMINoncyclic import mc_mean_grad_noncyclic
import matplotlib.pyplot as plt

# -----------Poissonian Noncyclic model with NO constraints-----------

# -----------Bandit Algorithm helper functions-----------

def poisson_log_ratio(bin_index, output_count, input_set, input_prob_vec):
    # bin_index: i = 0,1,...,numBin-1
    # output_count: c (vector with length=numNeuro)
    # input_set: {\lambda_{k,j}} (vector shape=(numNeuro,numBin)) (same as tuning_curve) (all nonzero)
    # input_prob_vec: {w_j} (list or 1-d vector with length = numBin)
    if input_set.ndim == 1:
        input_set = input_set.reshape((1,-1))
    numNeuro, numBin = input_set.shape
    
    phi = np.zeros(numBin)
    max_phi = -1e10
    
    for j in range(numBin):
        phi[j] = 0
        for p in range(numNeuro):
            phi[j] += output_count[p]*(np.log(input_set[p,j]) - np.log(input_set[p, bin_index]))
            phi[j] += input_set[p, bin_index] - input_set[p,j]
        if phi[j] > max_phi:
            max_phi = phi[j]
    sum_log = 0
    for i in range(numBin):
        sum_log += input_prob_vec[i]*np.exp(phi[i] - max_phi)
    sum_log = np.log(sum_log) + max_phi
    return -sum_log

def poisson_DKL(bin_index, poisson_samples, input_set, input_prob_vec):
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
    if poisson_samples.ndim == 1:
        poisson_samples = poisson_samples.reshape(1, -1)        
    num_samples, numNeuro = poisson_samples.shape
    
    for i in range(num_samples):
        dkl += poisson_log_ratio(bin_index, poisson_samples[i,:], input_set, input_prob_vec)
    return dkl/num_samples

# -----------Bandit Algorithm for Arimoto iterations-----------

def poisson_bandit_iteration(input_set,
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
    
    # generate poisson samples in advance
    poisson_samples = np.zeros((max_iter*batch_size, numNeuro, numBin), dtype=np.int)
    for p in range(numNeuro):
        for j in range(numBin):
            poisson_samples[:, p,j] = np.random.poisson(lam = input_set[p,j], size = max_iter*batch_size)

    while(num_iter < max_iter):
        rewards *= 0
        for k in range(batch_size):
            for i in range(numBin):
                # compute rewards by sampling
                output_sample = poisson_samples[num_iter*batch_size+k, :, i]
                rewards[i] += poisson_log_ratio(i, output_sample, input_set, prob_vec)
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

def simple_sgd_poisson(tuning, weight, eta, NUM_ITER, fp, fm, MC_ITER = 1, conv = None, tau = 1.0, NUM_THREADS=1):
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
        x_mean = mc_mean_grad_noncyclic(x_grad, x, weight, conv, tau, numIter=MC_ITER, my_num_threads=NUM_THREADS)
        x += eta*x_grad
        x[x>fp] = fp
        x[x<fm] = fm
        curve_list.append(x.copy())
        grad_list.append(x_grad.copy())
        
    return curve_list, grad_list


