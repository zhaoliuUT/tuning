import numpy as np
import matplotlib.pyplot as plt

#========= Generate hierachical binary tuning curves =========
def gen_binary_hierachical_curves(numNeuro, fp = 1, fm = 0):
    x_ = np.array([0, 1]).reshape((1, 2))
    curr_dim = 1

    while curr_dim < numNeuro:
        if len(x_.shape) ==1:
            old_dim = 1
            old_len = len(x_)
        else:
            old_dim, old_len = x_.shape

        curr_dim = old_dim+1
        curr_len = 2*old_len#x_.shape[1]
        y = np.zeros((curr_dim, curr_len)).astype(int)
        y[0, :old_len] = 0
        y[0, old_len:] = 1
        y[1:, :old_len]= x_.copy()
        y[1:, old_len:]= np.flip(x_, axis = 1)
        x_ = y.copy()
        
    corners = x_.copy()
    corners[x_ == 0] = fm
    corners[x_ == 1] = fp    
    return corners

## example:
# tc = gen_binary_hierachical_curves(5, fp = 1, fm = 0.01)
# from tuning.anim_3dcube import plot_funcs_in_figure
# fig = plt.figure()
# _ = plot_funcs_in_figure(fig, tc, np.ones(tc.shape[1]), nrow=5, ncol=1)

def compute_period(yy, noise_tol = 0.15, period_tol = 0.6):
    yy_diff = np.diff(np.concatenate((yy, [yy[0]])))
    # find the positions where yy_diff changes sign
    sign_diff = 1.0*(yy_diff > 0) - 1.0*(yy_diff < 0)
    sign_change = 1 + np.where(np.diff(np.concatenate((sign_diff, [sign_diff[0]])))!=0)[0]
    sign_change = list(sign_change)
    if len(yy) in sign_change:
        sign_change.remove(len(yy))
        sign_change.append(0)
    
#     plt.figure()
#     plt.plot(sign_diff)
#     plt.plot(yy)
#     print(sign_change)

    func_variations = []
    func_variations_sign = []
    func_variations_index = []
    for i, idx in enumerate(sign_change):
        if i != len(sign_change) - 1:
            next_idx = sign_change[i+1]
        else:
            next_idx = sign_change[0]
#         print(next_idx, np.arange(idx, next_idx+1), yy[idx:next_idx+1], yy[next_idx] - yy[idx])
#         print(next_idx,  yy[next_idx] - yy[idx])
        curr_variation = yy[next_idx] - yy[idx]
        if i ==0 or (np.fabs(curr_variation) > noise_tol and curr_variation*func_variations_sign[-1] <= 0):
            # includes the case when sign=0 (can happen at i=0)
            func_variations.append(curr_variation)
            func_variations_sign.append(np.sign(curr_variation))
            func_variations_index.append(next_idx)
        else:
            func_variations[-1] += curr_variation
            func_variations_index[-1] = next_idx
#         print(func_variations)
#         print(func_variations_sign)
    func_variations = np.array(func_variations)
    func_variations_index = [func_variations_index[-1]]+func_variations_index[0:-1]
    #print(func_variations)
    #print(func_variations_sign)
    #print(func_variations_index)
    
#     for k in range(len(func_variations_index)):
#         print(all_index[k], all_index[(k-1)%len(all_index)])
#         curr_index = func_variations_index[k]
#         prev_index = func_variations_index[(k-1)%len(func_variations_index)]
#         print(yy[curr_index] - yy[prev_index])
    # should be the same as func_variations
    increase_num = np.sum(func_variations > period_tol) # 0.6
    decrease_num = np.sum(func_variations < -period_tol)
    if increase_num == decrease_num:
        return increase_num
    else:
        print('different number of increasing intervals and decreasing intervals: %d and %d!'%(increase_num, decrease_num))
        return max(increase_num, decrease_num)