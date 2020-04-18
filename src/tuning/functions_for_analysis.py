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