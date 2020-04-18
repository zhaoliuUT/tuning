import matplotlib.pyplot as plt
# plot 2d piecewise constant functions and weights as 2d colormesh or 3d surfaces
def plot_colormesh_in_axis(ax, f_values, weights, cmap = 'RdBu'):
    
    if np.ndim(f_values) != 2 or np.ndim(weights) != 2:
        raise Exception('f_values, weights must be 2d arrays!')
    if f_values.shape != weights.shape:
        raise Exception('Dimension Mismatch!')
        
    w1 = np.sum(weights, axis = 1)
    w2 = np.sum(weights, axis = 0)

    x1 = [0]+list(np.cumsum(w1))
    x2 = [0]+list(np.cumsum(w2))
    xx, yy = np.meshgrid(x1, x2)
    zz = f_values.T
    
    ax.clear()    
    colormesh = ax.pcolormesh(xx, yy, zz, cmap = cmap, vmin=zz.min(), vmax=zz.max())
    #print(xx)
    #print(yy)
    #print(zz)
    #print(xx.shape)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks(x1)
    ax.set_yticks(x2)
    ax.set_aspect("equal")

    #     fig.colorbar(c, ax=ax)
    return colormesh

def plot_surfaces_in_axis(ax, f_values, weights):
    # 'ax' must be 3d axis
    
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')

    if np.ndim(f_values) != 2 or np.ndim(weights) != 2:
        raise Exception('f_values, weights must be 2d arrays!')
    if f_values.shape != weights.shape:
        raise Exception('Dimension Mismatch!')
    if not hasattr(ax, 'get_zlim'):
        raise Exception('Wrong Dimension of plot: axis must be 3D!')
        
    w1 = np.sum(weights, axis = 1)
    w2 = np.sum(weights, axis = 0)

    n1 = len(weight1)
    n2 = len(weight2)
    
    x1 = [0]+list(np.cumsum(weight1))
    x2 = [0]+list(np.cumsum(weight2))
    
    surf_list = []
    for i in range(n1):
        for j in range(n2):
            xx, yy = np.meshgrid([x1[i], x1[i+1]], [x2[j], x2[j+1]])
            zz = f_values[i,j]*np.ones_like(xx)
            surf = ax.plot_surface(xx, yy, zz)
            surf_list += [surf]
            
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.set_xticks(x1)
    ax.set_yticks(x2)
    return surf_list

# plot 2d piecewise constant functions and weights if the weights are not product distributions
def plot_colormesh_in_axis_non_product(ax, f_values, weights, cmap = 'RdBu'):
    
    if np.ndim(f_values) != 2 or np.ndim(weights) != 2:
        raise Exception('f_values, weights must be 2d arrays!')
    if f_values.shape != weights.shape:
        raise Exception('Dimension Mismatch!')
        
    first_dim, second_dim = weights.shape
    xx_temp = np.zeros((second_dim, first_dim))
    y_temp = np.zeros(second_dim)
    
    for i in range(second_dim):
        #print('i=', i)
        coeff = np.zeros((first_dim, first_dim))
        for j in range(1, first_dim):
            coeff[j-1, 0] = weights[j, i]
            coeff[j-1, j] = -weights[0, i]

        coeff[-1, :] = 1
        #print(coeff)
        rhs = np.zeros(first_dim)
        rhs[-1] = 1
        #print(rhs)
        # solve for the ratios of the current column of weights such that they sum up to one
        new_ratios = np.linalg.solve(coeff, rhs)
        #print(new_ratios)
        #print(weights[0, i]/new_ratios[0])
        y_temp[i] = weights[0, i]/new_ratios[0]
        xx_temp[i, :] = np.cumsum(new_ratios)
    #print(xx_temp)

    xx = np.zeros((2*second_dim, first_dim + 1))
    for i in range(second_dim):
        xx[2*i,1:] = xx_temp[i,:]
        xx[2*i+1, 1:] = xx_temp[i,:]
    #print(xx)

    yy = np.zeros((2*second_dim, first_dim + 1))
    y_sum0 = np.cumsum(y_temp)
    y_list = np.zeros(2*len(y_sum0))
    y_list[0::2] = y_sum0
    y_list[1::2] = y_sum0
    y_list = [0] + list(y_list[:-1])
    #print(y_list)
    _, yy = np.meshgrid(np.zeros(first_dim+1), y_list)
    #print(yy)


    zz = np.zeros((2*second_dim-1, first_dim))
    zz_temp = f_values.T
    for i in range(second_dim - 1):
        zz[2*i,:] = zz_temp[i,:].copy()
        zz[2*i +1, :] = zz_temp[i,:].copy()
    zz[-1,:] = zz_temp[-1,:].copy()
    #print(zz_temp0)
    #print(zz)

    ax.clear()    
    colormesh = ax.pcolormesh(xx, yy, zz, cmap = 'RdBu', vmin=zz.min(), vmax=zz.max())
#     print(xx)
#     print(yy)
#     print(zz)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks(xx[0,:])
    ax.set_yticks(yy[:,0])
    ax.set_aspect("equal")
    # fig.colorbar(colormesh, ax=ax)
    return colormesh

def get_color_array_2d_periodic(num_pts1, num_pts2, cmap_name = 'coolwarm', num_pts_color_range = None):
    '''
    Return a color array of shape (num_pts1*num_pts2, 4), with 2d periodic boundary.
    Note: other commonly used colormap names: nipy_spectral, cubehelix,viridis,...
    '''
    color_arr = np.zeros((num_pts1*num_pts2, 4))
    half_pts = int(np.ceil(num_pts1/2.0))
    if num_pts_color_range is None or num_pts_color_range < half_pts:
        num_pts_color_range = half_pts

#     print(num_pts_color_range, half_pts)
    
    colormap = plt.cm.get_cmap('coolwarm', num_pts_color_range)
    temp_arr = np.array([colormap(i) for i in range(half_pts)])
    if num_pts1 %2 == 0:
        temp_arr = np.concatenate((temp_arr, np.flip(temp_arr, axis = 0)), axis = 0)
    else:
        temp_arr = np.concatenate((temp_arr, np.flip(temp_arr, axis = 0)[1:]), axis = 0)

#     print(temp_arr, temp_arr.shape)

    for i in range(num_pts2):
        color_arr[num_pts1*i:num_pts1*(i+1), :] = np.roll(temp_arr, i, axis = 0)
#     print(color_arr, color_arr.shape)
    return color_arr

##----------------example: plot tuning, weights in 2d square using gen_mixed_plots, and connect the 'grids'--------
## (here numNeuro = 2, numBin1 = 5, numBin2 = 5)
# color_array = get_color_array_2d_periodic(numBin1, numBin2, num_pts_color_range = 5)
# figure_handles = gen_mixed_plots(tuning.reshape(numNeuro, numBin1*numBin2), weights=np.arange(25), path_vec=None,
#                     radius=1, min_radius=0.01,
#                     INCLUDE_FUN=True, INCLUDE_WEIGHT=True, INCLUDE_WEIGHT_BAR=True,
#                     FILE_NAME="", ADD_TIME=False,
#                     color_arr = color_array,
#                     weight_tol = -1, weight_format = '%d',
#                    )
# ax_points = figure_handles['ax_points']
# for i in range(numBin1):
#     ax_points.plot(tuning[0, i, :], tuning[1, i, :], color = 'r', linestyle='--')
#     ax_points.plot([tuning[0,i,-1], tuning[0,i,0]], [tuning[1,i,-1], tuning[1,i,0]], color = 'r', linestyle='--')
# for j in range(numBin2):
#     ax_points.plot(tuning[0, :, j], tuning[1,: , j], color = 'g', linestyle='--')
#     ax_points.plot([tuning[0,-1,j], tuning[0,0,j]], [tuning[1,-1,j], tuning[1,0,j]], color = 'g', linestyle='--')
