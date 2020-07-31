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
    if fp != 1 or fm != 0:
        xnew = np.zeros_like(x_).astype(float)
        xnew[x_ == 0] = fm
        xnew[x_ == 1] = fp
        return xnew
    else:
        return x_

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




def find_unique_points(points_data, tol = 1e-3, return_index = False):
    #     points_data: (numDimension, numPoints)
    # each column is a point
    # return value has dimension (numDimension, smaller number of points)
    # for the return index: points_data[:, returnedindex] == result.
    point_dim, point_num = points_data.shape

    if point_dim == 1:
        points_data = points_data.reshape(-1)
        ind = np.argsort(points_data)
        xx = points_data[ind]
        xxdiff = np.append(1, np.diff(xx))
        result = xx[xxdiff > tol]
        result = result.reshape(1, len(result))
        if return_index:
            return result, ind[xxdiff>tol]
        else:
            return result

    xx = points_data.T
    sort_keys = (xx[:,0], xx[:,1])
    for k in range(2, point_dim):
        sort_keys = (*sort_keys, xx[:,k])
    ind = np.lexsort(sort_keys) # sort using multiple keys
    xx = xx[ind, :]
    xxdiff = np.diff(xx, axis = 0)
    errors = np.append(1, np.sum(xxdiff**2, axis = 1))
    result = xx[errors>tol, :].T
    if return_index:
        return result, ind[errors > tol]
    else:
        return result

def find_unique_points_weights(points_data, points_weights=None, tol = 1e-3, return_index = False):
    #     points_data: (numDimension, numPoints)
    # each column is a point
    # return value has dimension (numDimension, smaller number of points)
    # for the return index: points_data[:, returnedindex] == result.
    # also sum up the weights according to the unique indices
    point_dim, point_num = points_data.shape

    if point_dim == 1:
        points_data = points_data.reshape(-1)
        ind = np.argsort(points_data)
        xx = points_data[ind]
        errors = np.append(1, np.diff(xx))
        result = xx[errors > tol]
        result = result.reshape(1, len(result))
    else:
        xx = points_data.T
        sort_keys = (xx[:,0], xx[:,1])
        for k in range(2, point_dim):
            sort_keys = (*sort_keys, xx[:,k])
        ind = np.lexsort(sort_keys) # sort using multiple keys
        xx = xx[ind, :]
        xxdiff = np.diff(xx, axis = 0)
        errors = np.append(1, np.sum(xxdiff**2, axis = 1))
        result = xx[errors>tol, :].T

    if points_weights is not None:
        # sum up the weights according to the unique indices
        newweights = np.zeros(result.shape[1])

        errors_ind = np.where(errors > tol)[0]
        for j, start_idx in enumerate(errors_ind[:-1]):
            #start_idx = errors_ind[j]
            end_idx = errors_ind[j+1]
            newweights[j] = np.sum(points_weights[ind[start_idx:end_idx]])
        newweights[-1] = np.sum(points_weights[ind[errors_ind[-1]:]])
    # return results
    if points_weights is None:
        if return_index:
            return result, ind[errors > tol]
        else:
            return result
    else:
        if return_index:
            return result, newweights, ind[errors > tol]
        else:
            return result, newweights
        
def compute_bump_widths(one_dim_tc, weights, fm = None):
    # compute widths of bumps (which are continuous parts != fm)
    # one_dim_tc: (numBin, ) numpy array, same shape as weights
    # circulated.
    if fm is None:
        fm = np.min(one_dim_tc)
    nBin = len(one_dim_tc)
    fmindices = np.where(one_dim_tc==fm)[0]
    diff_fmindices = np.diff(list(fmindices)+[fmindices[0]+100]) # diff_fmindices[i]=fmindices[i+1]-fmindices[i]
    diff_fmindices2 = np.roll(diff_fmindices, 1) #diff_fmindices2[i] = fmindices[i] - fmindices[i-1]
    
    bump_start_indices = fmindices[diff_fmindices>1] 
    bump_end_indices = fmindices[diff_fmindices2>1]
    if diff_fmindices[-1]!=1: #one_dim_tc[-1]!=fm or one_dim_tc[0]!=fm
        bump_start_indices = np.roll(bump_start_indices, 1)
        
    bump_widths_with_weights = np.zeros(len(bump_start_indices))
    for k in range(len(bump_start_indices)):
        j1 = bump_start_indices[k]
        j2 = bump_end_indices[k]
        if(j1 > j2):
            bump_widths_with_weights[k] = np.sum(ww[j1+1:]) + np.sum(ww[0:j2])
        else:
            bump_widths_with_weights[k] = np.sum(ww[j1+1:j2]) # sum from ww[j1+1] to ww[j2-1]
    return bump_widths_with_weights
        