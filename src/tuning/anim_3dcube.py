## new version
import time, sys, os, copy
import numpy as np
import matplotlib.pyplot as plt

from itertools import product, combinations
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.art3d import juggle_axes
from mpl_toolkits.mplot3d import Axes3D, proj3d
import matplotlib.gridspec as gridspec
from matplotlib import animation

def pc_fun_weights(f_vec, weights):
    '''For 1-d plotting purpose, generate the x coordinates and y coordinates of a piecewise constant
    function (including the jump discontinuities).
    '''
    if weights.size != f_vec.size:
        raise Exception("Dimension mismatch!")
#     if np.fabs(np.sum(weights) - 1)>1e-5:
#         raise Exception("Weight sum not equal to one!")
    numBin = weights.size
    xtmp = np.cumsum(weights)
    xx = [ [xtmp[i]]*2 for i in range(0, numBin-1) ]  
    xx = [0]+list(np.array(xx).reshape(-1)) + [xtmp[-1]]
    yy = [ [f_vec[i], f_vec[i+1]] for i in range(0, numBin-1) ] 
    yy = [f_vec[0]] + list(np.array(yy).reshape(-1)) + [f_vec[-1]]
    return xx, yy

class Arrow3D(FancyArrowPatch):

    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)

def find_best_path(points_data):
    '''Find the shortest closed path (in euclidean distance)
    using Elastic Net Algorithm in mlrose package.'''
    # tuning curve size: (numNeruo, numBin)
    # numBin = num_pts
    points_data = np.array(points_data)
    if len(points_data.shape) == 1:
        points_data = points_data.reshape((1,points_data.size))
    num_pts = points_data.shape[1] # number of points

    def euclidean_distance(x,y):
        return np.sqrt(np.sum((x-y)**2))

    dist_list = []
    for i in range(num_pts):
        for j in range(num_pts):
            if i!=j:
                dist_list.append((i, j, euclidean_distance(points_data[:,i], points_data[:,j])))

    # Initialize fitness function object using dist_list
    fitness_dists = mlrose.TravellingSales(distances = dist_list)

    problem_fit = mlrose.TSPOpt(length = num_pts, fitness_fn = fitness_dists,
                                maximize=False)

    # Solve problem using the genetic algorithm
    best_path, best_path_len = mlrose.genetic_alg(problem_fit, random_state = 2)

    return best_path, best_path_len # length of closed curve

def draw_cube_in_axis(ax, radius, min_radius):
    '''Draw a cube in 3d axis with corrdinates: [min_radius, radius]^3.
    The plotted boundaries of the cube are labeled as 'boundary_0', ..., 'boundary_11'.
    '''
    r = [min_radius, radius]
    boundary_idx = 0
    for s, e in combinations(np.array(list(product(r, r, r))), 2):
        if np.sum(np.abs(s-e)) == r[1]-r[0]:
            ax.plot3D(*zip(s, e), color="k", alpha = 0.5, lw = 1.5,
                     label='boundary_%d'%boundary_idx)#linestyle='--'
            boundary_idx += 1

    r2 = 1.4*radius
    arrowendpts =  [[r2, min_radius, min_radius],
                    [min_radius, min_radius, r2],
                    [min_radius, r2, min_radius]]
    for s in arrowendpts:
        a = Arrow3D(*zip([min_radius,min_radius,min_radius], s), mutation_scale=20,
                    lw=1, arrowstyle="-|>", color="k")
        ax.add_artist(a)
    ax._axis3don = False
    ax.grid(True)
    ax.set_xlim([min_radius,1.5*radius])
    ax.set_ylim([min_radius,1.5*radius])
    ax.set_zlim([min_radius,1.5*radius])

    endpts = [[1.45*radius,-0.1*radius+min_radius,min_radius],
              [-0.1*radius+min_radius,1.45*radius,min_radius],
              [min_radius,min_radius,1.45*radius]]
    for k in range(len(endpts)):
        ax.text(endpts[k][0],endpts[k][1],endpts[k][2],r'$f_%d$'%(k+1),color = 'k',fontsize = 16)
    ax.view_init(azim = 30)

def draw_square_in_axis(ax, radius, min_radius):
    '''Draw a square in (2d) axis with corrdinates: [min_radius, radius]^2.
    The plotted boundary of the cube is labeled as 'boundary'.
    Also set appropriate axis limits and draw the arrows with labels 'f_1', 'f_2'.
    '''
    #ax.grid(True)
    ax.set_xlim([min_radius-radius*0.1, radius*1.1])
    ax.set_ylim([min_radius-radius*0.1,radius*1.1])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    ax.plot([min_radius, radius, radius, min_radius, min_radius], 
           [min_radius, min_radius, radius, radius, min_radius], color='k', 
            label='boundary',
           )

    ax.arrow(min_radius, min_radius, radius*1.05-min_radius, 0, # x,y, dx, dy
             head_width=0.01*radius, head_length=0.05*radius, fc='k', ec='k'
            )
    ax.arrow(min_radius, min_radius, 0, radius*1.05-min_radius,
             head_width=0.01*radius, head_length=0.05*radius, fc='k', ec='k'
            )
    # arrow ?
    endpts = [[1.05*radius, min_radius-0.05*radius],
              [min_radius-0.05*radius, 1.05*radius]]
    for k in range(len(endpts)):
        ax.text(endpts[k][0],endpts[k][1],r'$f_%d$'%(k+1),color ='k',fontsize = 15)

def draw_line_in_axis(ax, radius, min_radius):
    '''Draw a horizontal line in (2d) axis with x-coordinate in [min_radius, radius],
    y-coordinate = 0.
    The plotted line is labeled as 'boundary'.
    Also set appropriate axis limits and draw an arrow on the x-axis with label 'f'.
    '''
    #ax.grid(True)
    ax.set_xlim([min_radius-radius*0.1, radius*1.1]) # plot data in the x direction
    # y-direction always zero
    half_len = 0.5*(radius*1.2 - min_radius)
    ax.set_ylim([-half_len, half_len])
    ax.set_yticks([])
    ax.set_xticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.plot([min_radius, radius], [0, 0], 'k', '--',
            label='boundary',
           )
    ax.text(1.1*radius-half_len, -0.15*half_len, r'$f$', color='k', fontsize=15)

def get_color_array(num_pts, cmap_name = 'nipy_spectral', shuffle_colors = False):
    '''
    Return a color array of shape (num_pts, 4), 
    which option of whether randomly shuffle the colors or not.
    Note: other commonly used colormap names: cubehelix,viridis,...
    '''

    #color_arr = np.arange(num_pts)
    cmap = plt.cm.get_cmap(cmap_name, num_pts)
    color_arr = np.array([cmap(i) for i in range(num_pts)])

    if shuffle_colors:
        np.random.shuffle(color_arr)

    return color_arr


def set_scatter_data_in_axis(ax, scat, points_data, weights=None, path_vec=None,
                             color_arr=None,
                             radius=1,
                             weights_txt_list=[],
                             weight_tol=1e-3, weight_format='%.2f',
                             point_size=100,
                             path_close=True, path_color='crimson', linestyle='dashed',
                            ):

    '''Set data (1/2/3-dim) in a given matplotlib axis, including the scatter points, weights' texts,
    and the line connecting the path.

    Main arguments:
        ax:
            matplotlib axis (usual 2d or 3d)
        scat:
            matplotlib.collections.PathCollection object, the return value of 'plt.scatter(...)'
        points_data:
            numpy array with shape (point dimension, number of points), specifying the coordinates of points
            in columns (same shape as a tuning curve.)
            Note: point dimension <= 3.
    NO Return,
        but the contents in the list 'weights_txt_list' is changed during the execution of the function
        (since it is mutable type).
    Main Keyword Arguments:
        weights:
            numpy array, 1-dimensional, with length = number of points.
            Though usually assumed to be posive and sum up to one,
            negative weights or weights with sum != 1 can also be plotted in the scatter plots.
            If weights=None, no weight labels in the scatter plot.
        path_vec:
            numpy array, 1-dimensional, with length = number of points.
            Contains integers in [0,number of points-1], specifying the order in the path.
            If path_vec is not None, no path will be plotted.
        color_arr:
            color array with dimension (number of colors, 4).
	    The number of colors must be the same as number of points.
            Note: can be generated using get_color_arr(num_pts).
        radius:
            maximum value of the function.
        weights_txt_list:
            the list of texts which are weights' labels.
            (note that not all texts in the axis are weights' labels, e.g. the 'f_1', 'f_2'.)
    Other Keyword Arguments:
        weight_tol = 1e-3, weight_format = '%.2f',
            weight properties in the scatter plot.
            Any point with weight absolute value < weight_tol will have empty weight label.
            To set all weights label empty, simply use weight_format="".
        point_size = 100,
            size of points in the scatter plot.
            Note: default point_size=100, appropriate for 3d. Choose smaller point_size for 1d/2d. e.g. 50.
        path_close = True, path_color='crimson', linestyle='dashed',
            path properties in the scatter plot: whether it is closed, the color and linestyle.
    '''

    # check inputs
    points_data = np.array(points_data)
    if len(points_data.shape) == 1:
        # tuning is a (num_pts,) type array
        points_data = points_data.reshape((1,points_data.size))
    num_pts = points_data.shape[1] # number of points
    data_dimension = points_data.shape[0] # dimension
    if data_dimension == 1:
        X = points_data[0,:] # plot data in the x direction
        Y = np.zeros(num_pts) # y-direction always zero
    elif data_dimension == 2:
        X = points_data[0,:]
        Y = points_data[1,:]
    elif data_dimension == 3:
        X = points_data[0,:]
        Y = points_data[1,:]
        Z = points_data[2,:]
    else:
        raise Exception("Wrong dimension of input points_data: must be <= 3!")

    if (weights is not None) and len(weights) != num_pts:
        raise Exception("Wrong dimension of input weights: inconsitent with points_data!")

    path_line = None
    for line in ax.lines:
        if line.get_label()=='path':
            path_line = line
    if path_line is None:
        raise Excpetion('Wrong input: no path line is found in the ax.lines!')

    if data_dimension == 1 and path_vec is not None:
        raise Exception("Wrong dimension of input points_data: \
        unable to plot the path in 1-dimension!")

    if path_vec is not None and len(path_vec)!= num_pts:
        raise Exception('Wrong input of the path vector: \
        the length must be equal to the number of points in point_data!')
    elif path_vec is not None and np.any(path_vec >= num_pts):
        raise Exception('Wrong input of the path vector: \
        each number must be an interger in [0, number of points-1]!')

    if color_arr is not None and color_arr.shape[0]!=num_pts:
        raise Exception("Wrong dimension of color array: inconsistent with the number of points!")

    # set scatter points' data
    if data_dimension <= 2:
        scat.set_offsets(np.vstack([X.reshape(-1),Y.reshape(-1)]).T) # only works for 2d
    else:
        scat._offsets3d = juggle_axes(X, Y, Z, 'z')

    # set scatter points' colors, alpha, sizes
    if color_arr is None:
        color_arr = get_color_array(num_pts)

    if data_dimension <= 2:
        scat.set_facecolor(color_arr) # only works for 2d
    else:
        scat._facecolor3d = color_arr
    scat.set_edgecolor('k')
    scat.set_alpha(1.0)
    sizes = np.ones(num_pts)*point_size
    if weights is not None:
        sizes[np.fabs(weights) < weight_tol] = 0
    scat.set_sizes(sizes)

    # set weight texts of the scatter points

    if len(weights_txt_list) < num_pts:
        # add empty texts
        if data_dimension <=2:
            weights_txt_list += [ax.text(0,0,"",fontsize = 14, color = 'k')
                         for _ in range(num_pts - len(weights_txt_list))] # 'steelblue'
        else:
            weights_txt_list += [ax.text2D(0,0,"",fontsize = 14, color = 'k') # 3d axis
                         for _ in range(num_pts - len(weights_txt_list))] # 'steelblue'

    if weights is not None:
        if data_dimension <=2:
            for txt, new_x, new_y, w in zip(weights_txt_list[:num_pts], X, Y, weights):
                if np.fabs(w) > weight_tol:
                    txt.set_position((new_x+0.01*radius,new_y+0.01*radius))
                    txt.set_text(weight_format%w)
                else:
                    txt.set_text("")
        else:
            for txt, new_x, new_y, new_z, w in zip(weights_txt_list[:num_pts], X, Y, Z, weights):
            # animating Text in 3D proved to be tricky. Tip of the hat to @ImportanceOfBeingErnest
            # for this answer https://stackoverflow.com/a/51579878/1356000
                if np.fabs(w) > weight_tol:
                    x_, y_, _ = proj3d.proj_transform(new_x+0.1*radius, new_y+0.1*radius, new_z+0.1*radius, \
                                                      ax.get_proj())
                    txt.set_position((x_,y_))
                    txt.set_text(weight_format%w)
                else:
                    txt.set_text("")

    # set path data
    ## get the coordinates in the order of path_vec
    if path_vec is not None:
        x_cords = [X[path_vec[i]] for i in range(len(path_vec))]
        y_cords = [Y[path_vec[i]] for i in range(len(path_vec))]
        if path_close:
            x_cords += [X[path_vec[0]]]
            y_cords += [Y[path_vec[0]]]

        if data_dimension == 3:
            z_cords = [Z[path_vec[i]] for i in range(len(path_vec))]
            if path_close:
                z_cords += [Z[path_vec[0]]]
        ##  set data of the line
        if data_dimension == 2:
            path_line.set_data(x_cords, y_cords)
        else:
            path_line.set_xdata(x_cords)
            path_line.set_ydata(y_cords)
            path_line.set_3d_properties(z_cords)
            #path_line.set_data_3d(x_cords, y_cords, z_cords)  #python bug, currently not working
        ## set color and linestyle
        path_line.set_color(path_color)
        path_line.set_linestyle(linestyle)
    return

def set_func_data_in_axis_list(ax_list, points_data, weights=None, path_vec=None):
    '''
    Set data in a list of axes where functions are plotted.
    The length of list = the number of functions = the dimension of points.
    Each function is a function with y values = points_data[j,:], x values = cummulative sum of weights.
    If weights=None, the functions' line plots use equal weights from 0 to 1 to show only the function values.
    If path_vec is not None, then re-arrange the function values and weights according to path_vec.
    '''

    points_data = np.array(points_data)
    if len(points_data.shape) == 1:
        points_data = points_data.reshape((1,points_data.size))
    num_pts = points_data.shape[1] # number of points
    data_dimension = points_data.shape[0] # dimension

    if path_vec is not None:
        ordered_points_data = points_data[:, path_vec]
        ordered_weights = weights[path_vec]
    else:
        ordered_points_data = points_data
        ordered_weights = weights

    if len(ax_list) != data_dimension:
        raise Exception('Wrong input of function lines in axis: \
        dimension inconsistent with points_data!')

    func_lines_list = []
    for ax_f in ax_list:
        func_line = [line for line in ax_f.lines if 'func' in line.get_label()][0]
        func_lines_list += [func_line]

    for j, line in enumerate(func_lines_list):
        if weights is not None:
            xx, yy = pc_fun_weights(ordered_points_data[j, :], ordered_weights)
            line.set_data(xx, yy)
        else:
            line.set_data(1.0*np.arange(num_pts)/num_pts, ordered_points_data[j, :])
    return

def set_hist_data_in_axis(ax, weights, color_arr, path_vec=None, weight_max=1.0):
    '''
    Set data in the histogram plot of weights. (no points_data needed).
    The y-limit of the axis can be adjusted using weight_max.
    If path_vec is not None, the order of the bars in the histogram are re-arranged according to path_vec.
    '''

    num_pts =len(weights) # number of points

    if color_arr.shape[0]!=num_pts:
        raise Exception("Wrong dimension of color array: inconsistent with the number of points!")

    if path_vec is not None:
        ordered_weights = weights[path_vec]
        ordered_color_arr = color_arr[path_vec,:]
        ordered_xticks = path_vec
    else:
        ordered_weights = weights
        ordered_color_arr = color_arr
        ordered_xticks = np.arange(num_pts)

    ax.clear()
    barcollection = ax.bar(np.arange(num_pts), ordered_weights, color = ordered_color_arr)
    ax.set_ylim(0, weight_max)
    ax.set_xticks(np.arange(num_pts))
    ax.set_xticklabels(['%d'%idx for idx in ordered_xticks])
    return

def set_weight_bar_data_in_axis(ax, weights, color_arr, path_vec=None, ):
    '''
    Set data in the horizontal colored bar of weights. (no points_data needed).
    If path_vec is not None, the order of the colors of weights are re-arranged according to path_vec.
    '''

    num_pts =len(weights) # number of points

    if color_arr.shape[0]!=num_pts:
        raise Exception("Wrong dimension of color array: inconsistent with the number of points!")

    if path_vec is not None:
        ordered_weights = weights[path_vec]
        ordered_color_arr = color_arr[path_vec,:]
    else:
        ordered_weights = weights
        ordered_color_arr = color_arr

    ax.clear()
    ax.set_xlim([0, 1])
    ax.set_ylim([-0.1, 0.1])
    ax.set_aspect(0.2)
    ax.set_yticks([])

    for j in range(num_pts):
        rect = plt.Rectangle((np.sum(ordered_weights[:j]), -0.1), ordered_weights[j], 0.2,
                             facecolor=ordered_color_arr[j])
        ax.add_artist(rect)
        #rect_list.append(rect)
    return

def create_figure_canvas(data_dimension=3, radius=1, min_radius=0,
                         INCLUDE_INFO=True,
                         INCLUDE_FUN=True, INCLUDE_WEIGHT=True, INCLUDE_WEIGHT_BAR=True,
                        ):
    '''
    When numNeuro <= 2, create a square figure. When numNeuro == 3, create a cube figure.
    with options INCLUDE_INFO, INCLUDE_FUN, INCLUDE_WEIGHT, INCLUDE_WEIGHT_BAR for the different sub-figures (or information txt)
    to be included.
    min_radius, radius are min, max of point coordinate values, used to set the limits of axis.
    (acts the same way as f+, f-).
    '''
    
    if data_dimension > 3:
        raise Exception("Wrong input: data_dimension can't be more than 3!")

    # figure setup
    if INCLUDE_FUN or INCLUDE_WEIGHT or INCLUDE_WEIGHT_BAR:
        fig = plt.figure(figsize = (12, 6 + 1.0*INCLUDE_WEIGHT_BAR))
        gs0 = gridspec.GridSpec(1, 2, width_ratios=[2, 1])
        n_cols = data_dimension*INCLUDE_FUN + INCLUDE_WEIGHT + INCLUDE_WEIGHT_BAR
        ax_w_bar_idx = data_dimension*INCLUDE_FUN
        ax_w_idx = data_dimension*INCLUDE_FUN + INCLUDE_WEIGHT_BAR

        height_ratios = np.ones(n_cols)
        if INCLUDE_WEIGHT_BAR:
            height_ratios[ax_w_bar_idx] = 0.5
        gs01 = gridspec.GridSpecFromSubplotSpec(n_cols,1,subplot_spec=gs0[1], height_ratios=height_ratios)

        # ax_points is used for the scatter plot of the data points
        if data_dimension == 3:
            ax_points = fig.add_subplot(gs0[0], projection='3d')
        else: # data_dimension < =2
            ax_points = fig.add_subplot(gs0[0])
        ax_points.set_aspect("equal")

        # ax_f_list is used for the line plots of the tuning curves (functions),
        # with x values being the weight, y values being the function values (points' coordinates)
        if INCLUDE_FUN:
            ax_f_list = []
            for j in np.arange(data_dimension):
                ax_f_list += [fig.add_subplot(gs01[j])]
        else:
            ax_f_list = None

        # ax_w_bar is used for displaying a colorbar specifying the colors of the points,
        # arranged according to the x corrdinates in ax_f_list
        if INCLUDE_WEIGHT_BAR:
            ax_w_bar = fig.add_subplot(gs01[ax_w_bar_idx])
            ax_w_bar.set_xlim([0, 1])
            ax_w_bar.set_ylim([-0.1, 0.1])
            ax_w_bar.set_aspect(0.2)
            ax_w_bar.set_yticks([])
        else:
            ax_w_bar = None

        # ax_w is used for plotting a histogram of the weights
        if INCLUDE_WEIGHT:
            ax_w = fig.add_subplot(gs01[ax_w_idx])
            #weight_max = np.max(np.array(weights_list))
            #ax_w.set_ylim(0, weight_max + 0.05)
        else:
            ax_w = None
    else:
        fig = plt.figure(figsize = (8,8))
        if data_dimension == 3:
            ax_points = fig.gca(projection='3d')
        else:
            ax_points = fig.gca()
        ax_points.set_aspect("equal")
        ax_f_list = None
        ax_w_bar = None
        ax_w = None

    # draw cube/square
    # Note: the lines that are plotted for the boundaries are labeled as 'boundary' (1d or 2d)
    # or 'boundary_0',...''boundary_11' (3d)
    if data_dimension == 3:
        draw_cube_in_axis(ax_points, radius, min_radius)
    elif data_dimension == 2:
        draw_square_in_axis(ax_points, radius, min_radius)
    else:
        draw_line_in_axis(ax_points, radius, min_radius)
    
    # plot scatter points
    if data_dimension == 3:
        scat = ax_points.scatter([radius], [radius], [radius], s= 0)#, edgecolor='k', alpha=1)
    elif data_dimension == 2:
        scat = ax_points.scatter([radius], [radius], s= 0)#, edgecolor='k', alpha=1)
    else:
        scat = ax_points.scatter([radius], [0], s= 0)#, edgecolor='k', alpha=1)

    # plot path
    if data_dimension == 3:
        path_line = ax_points.plot3D([], [], [], lw = 1.5, alpha=0.7, label='path')[0]
    else:
        path_line = ax_points.plot([], [], lw = 1.5, alpha=0.7, label='path')[0]

    
    # plot info txt
    # Note: the path line has label 'path'.
    if INCLUDE_INFO:
        if data_dimension == 3:
            x0, y0, _ = proj3d.proj_transform(1.6*radius, 1.25*radius, 0, ax_points.get_proj())
            info_txt = ax_points.text2D(x0,y0,"", fontsize = 15,
                                       ha='center', va='center',#horizontalalignment, verticalalignment
                                       )
        elif data_dimension <= 2:
            xmin, xmax = ax_points.get_xlim()
            ymin, ymax = ax_points.get_ylim()
            x0 = xmin + 0.5*(xmax - xmin)
            y0 = ymin - 0.02*(ymax - ymin)
            info_txt = ax_points.text(x0, y0, "", fontsize = 14,
                                      ha='center', va='center', #horizontalalignment, verticalalignment
                                     )
            #info_txt = ax_points.set_title( "", fontsize = 14)
    else:
        info_txt = None

    if data_dimension == 1:
        func_colors = ['steelblue']
    elif data_dimension == 2:
        func_colors = ['seagreen','steelblue']
    else:
        func_colors = ['crimson', 'seagreen', 'steelblue']

    # plot functions
    # Note: line labels are 'func_0', 'func_1', 'func_2' e.g. in 3-d.
    # the lines can also be retrived by e.g. ax_f_list[0].lines
    if INCLUDE_FUN:
        func_lines_list = []
        for j, ax_f in enumerate(ax_f_list):
            func_line = ax_f.plot([], [], color = func_colors[j],
                                  lw = 2, label='func_%d'%j)[0]
            func_lines_list += [func_line]
        #line1, = ax_f1.plot([], [], color = 'crimson', lw = 2)
        #line2, = ax_f2.plot([], [], color = 'seagreen', lw = 2)
        #line3, = ax_f3.plot([], [], color = 'steelblue', lw = 2)
        #ax_f_list = [ax_f1, ax_f2, ax_f3]
        #lines_list = [line1, line2, line3]
        for ax_f in ax_f_list:
            ax_f.set_xlim([0,1])
            ax_f.set_ylim([0,radius*1.01])
            ax_f.set_yticks([0,radius])
            ax_f.spines['top'].set_visible(False)
            ax_f.spines['right'].set_visible(False)
            ax_f.yaxis.set_ticks_position('left')
            ax_f.xaxis.set_ticks_position('bottom')
            #ax_f.arrow(0,0,1,0,fc='k', ec='k', lw =0.1,head_width=0.05, head_length=0.02,overhang = 0.1,\
            #           length_includes_head= False, clip_on = False)
            #ax_f.arrow(0,0,0,radius,fc='k', ec='k', lw =0.1,head_width=0.01*radius, head_length=0.1*radius,\
            #           overhang = 0.1*radius,\
            #           length_includes_head= False, clip_on = False)
    else:
        func_lines_list = None
        
    figure_handles = {'fig':fig,
                      'data_dimension':data_dimension,
                      'ax_points':ax_points,
                      'ax_f_list':ax_f_list, 'ax_w':ax_w, 'ax_w_bar':ax_w_bar,
                      'scat': scat,
                      'path_line': path_line,
                      'func_lines_list':func_lines_list,
                      'info_txt':info_txt,
                      'weights_txt_list':[],
                      'radius':radius, 'min_radius':min_radius,
                     }
    return figure_handles

def set_data_in_figure(cube_fig_setup, X, Y, Z, weight = None, info = None,
                       weight_tol = 1e-3, weight_format = '%.2f',
                       color_arr = None, cmap_name = 'nipy_spectral', shuffle_colors = False,
                       point_size = 100,
                       weight_max = 1.0,
                       info_format = 'MI = %.4f',
                      ):
    '''Set data in cube_fig_setup (all axis)'''
    
    num_pts = len(X)
    
    # generate color array if color_arr is None
    if color_arr is None:
            #color_arr = np.arange(num_pts)
        cmap = plt.cm.get_cmap(cmap_name, num_pts) # other colormap names: cubehelix,viridis,...
        color_arr = np.array([cmap(i) for i in range(num_pts)])
    if shuffle_colors:
        np.random.shuffle(color_arr)
    if color_arr.shape[0]!=num_pts:
        raise Exception("Wrong dimension of color array!")

    # set data in the cube axis
    ax_cube = cube_fig_setup['ax_cube']
    scat = cube_fig_setup['scat']
    weight_txt_list = cube_fig_setup['weight_txt_list']
    radius = cube_fig_setup['radius']

    weight_txt_list = set_scatter_data_in_axis(
        ax_cube, scat, X, Y, Z, weight=weight,
        weight_txt_list=weight_txt_list, radius=radius,
        weight_tol=weight_tol, weight_format=weight_format,
        color_arr=color_arr, cmap_name=cmap_name, shuffle_colors=shuffle_colors,point_size=point_size,
    )
    cube_fig_setup['weight_txt_list'] = weight_txt_list
                
    # set lines' data
    if cube_fig_setup['lines_list'] is not None:
        line1, line2, line3 = cube_fig_setup['lines_list']
        if weight is not None:
            x1, y1 = pc_fun_weights(X, weight)#pc_fun_weights(X_list[i],weights_list[i])
            x2, y2 = pc_fun_weights(Y, weight)
            x3, y3 = pc_fun_weights(Z, weight)

            line1.set_data(x1, y1)
            line2.set_data(x2, y2)
            line3.set_data(x3, y3)
        else:
            line1.set_data(1.0*np.arange(num_pts)/num_pts, X)
            line2.set_data(1.0*np.arange(num_pts)/num_pts, Y)
            line3.set_data(1.0*np.arange(num_pts)/num_pts, Z)
            
    # set weight histogram' data
    ax_w = cube_fig_setup['ax_w']
    if ax_w is not None:
        ax_w.clear()
        barcollection = ax_w.bar(np.arange(num_pts), weight, color = color_arr)
        # set axis limit
        ax_w.set_ylim(0, weight_max)

    # set weight bar's data
    ax_w_bar = cube_fig_setup['ax_w_bar']
    if ax_w_bar is not None:
        #rect_list = []
        ax_w_bar.clear()
        ax_w_bar.set_xlim([0, 1])
        ax_w_bar.set_ylim([-0.1, 0.1])
        ax_w_bar.set_aspect(0.2)
        ax_w_bar.set_yticks([])
        for j in range(num_pts):
            rect = plt.Rectangle((np.sum(weight[:j]), -0.1), weight[j], 0.2,
                                 facecolor=color_arr[j])
            ax_w_bar.add_artist(rect)
            #rect_list.append(rect)

    # set info data
    if (cube_fig_setup['info_txt'] is not None) and (info is not None):
        cube_fig_setup['info_txt'].set_text(info_format%info)

    return cube_fig_setup


def cube3dplots(X, Y, Z, weight = None, info = None,
                radius = 1, min_radius = 0,
                weight_tol = 1e-3, weight_format = '%.2f',
                weight_max = None, info_format = 'MI = %.4f',
                color_arr = None, cmap_name = 'nipy_spectral', shuffle_colors = False,
                point_size = 100,
                path_vec = None, path_close = True,
                path_color='crimson', linestyle='dashed',
                INCLUDE_FUN = True, INCLUDE_WEIGHT = True, INCLUDE_WEIGHT_BAR = True,
                FILE_NAME = "", ADD_TIME = True):
    '''Plot a single cube figure.'''
    
    # check inputs
    
    if len(X)!=len(Y) or len(Y)!=len(Z):
        raise Exception('Wrong dimension of inputs!')    
    if weight is not None and len(weight)!= len(X):
        raise Exception('Wrong dimension of inputs: weight')
#     elif np.fabs(np.sum(weight) - 1)>1e-5:
#         raise Exception('Wrong input of weight: sum error!')

    if info is None:
        INCLUDE_INFO = False
    else:
        INCLUDE_INFO = True
        
    num_pts = len(X) # number of points
    if weight_max is not None:
        curr_weight_max = weight_max
    elif weight is not None:
        curr_weight_max = np.max(np.array(weight)) # maximum weight
    else:
        curr_weight_max = 1.0

    curr_radius = max(np.max(np.array([X,Y,Z])), radius) # radius
    curr_min_radius = min(np.min(np.array([X,Y,Z])), min_radius) # min_radius
    
    # figure setup
    
    cube_fig_setup =  setup_cube3d_figure(
        radius = curr_radius, min_radius = curr_min_radius,
        #weight_max = weight_max, #weight_tol = weight_tol,
        #cmap_name = cmap_name, shuffle_colors = shuffle_colors,
        INCLUDE_INFO = INCLUDE_INFO,
        INCLUDE_FUN = INCLUDE_FUN, INCLUDE_WEIGHT = INCLUDE_WEIGHT, INCLUDE_WEIGHT_BAR = INCLUDE_WEIGHT_BAR,
    )
    
    # set data
    cube_fig_setup = set_data_in_figure(
        cube_fig_setup, X, Y, Z, weight = weight, info = info,
        weight_tol = weight_tol, weight_format = weight_format,
        weight_max = curr_weight_max, info_format = info_format,
        color_arr = color_arr,
        cmap_name = cmap_name, shuffle_colors = shuffle_colors,
        point_size = point_size,
    )

    # plot path
    
    if path_vec is not None:
        plot_path_in_axis(
            cube_fig_setup['ax_cube'], X, Y, Z,
            path_vec, path_close=path_close,
            path_color=path_color, linestyle=linestyle,
        )
    
    # save figure
    
    if ADD_TIME: 
        timestr = time.strftime("%m%d-%H%M%S")
    else:
        timestr = ""
    filename =  FILE_NAME + timestr + ".png" 
    directory = os.path.dirname(filename)
    if directory != "":
        try:
            os.stat(directory)
        except:
            os.makedirs(directory)  
    plt.savefig(filename)
    return cube_fig_setup

def anim3dplots(X_list, Y_list, Z_list, weights_list = None, info_list = None, 
                radius = 1, min_radius = 0,
                weight_tol = 1e-3, weight_format = '%.2f',
                weight_max = None, info_format = 'MI = %.4f',
                color_arr_list = None,
                cmap_name = 'nipy_spectral', shuffle_colors = False,
                point_size = 100,
                INCLUDE_FUN = True, INCLUDE_WEIGHT = True, INCLUDE_WEIGHT_BAR = True,
                FILE_NAME = "", ADD_TIME = True, interval = 1000):
    
    '''Plot animation in 3d cube.'''
    
    # check inputs    
        
    if len(X_list) != len(Y_list) or len(Y_list)!= len(Z_list):
        raise Exception('Wrong dimension of inputs!')
    
    elif len(X_list[0])!=len(Y_list[0]) or len(Y_list[0])!=len(Z_list[0]):
        raise Exception('Wrong dimension of inputs!')   
    
    if weights_list is not None:
        if len(weights_list) != len(X_list) or len(weights_list[0])!= len(X_list[0]):
            raise Exception('Wrong dimension of inputs: weight_list')
        elif np.fabs(np.sum(weights_list[0]) - 1)>1e-5:
            raise Exception('Wrong input of weights: sum error!')

    if info_list is None:
        INCLUDE_INFO = False
    else:
        INCLUDE_INFO = True
    
    
    num_pts = len(X_list[0])#np.max([len(X) for X in X_list]) # number of points

    num_frames = len(X_list) # number of frames
    if weight_max is not None:
        curr_weight_max = weight_max
    elif weight is not None:
        curr_weight_max = np.max([np.max(w) for w in weights_list]) # maximum weight
    else:
        curr_weight_max = 1.0

    x_max = np.max([np.max(X_list[i]) for i in range(num_frames)])
    y_max = np.max([np.max(Y_list[i]) for i in range(num_frames)])
    z_max = np.max([np.max(Z_list[i]) for i in range(num_frames)])
    x_min = np.min([np.min(X_list[i]) for i in range(num_frames)])
    y_min = np.min([np.min(Y_list[i]) for i in range(num_frames)])
    z_min = np.min([np.min(Z_list[i]) for i in range(num_frames)])
    curr_radius = max(x_max, y_max, z_max, radius)#max(np.max(np.array([X_list,Y_list,Z_list])), radius) # radius
    curr_min_radius = min(x_min, y_min, z_min, min_radius)#min(np.min(np.array([X_list,Y_list,Z_list])), min_radius) # min_radius
    # figure setup
    
    cube_fig_setup =  setup_cube3d_figure(
        radius = curr_radius, min_radius = curr_min_radius,
        #weight_max = weight_max, #weight_tol = weight_tol,
        #cmap_name = cmap_name, shuffle_colors = shuffle_colors,
        INCLUDE_INFO = INCLUDE_INFO,
        INCLUDE_FUN = INCLUDE_FUN, INCLUDE_WEIGHT = INCLUDE_WEIGHT, INCLUDE_WEIGHT_BAR = INCLUDE_WEIGHT_BAR,
    )

    # animation function.  This is called sequentially
    
    def animate(i):
        if weights_list is not None:
            curr_weight = weights_list[i]
        else:
            curr_weight = None
        if info_list is not None:
            curr_info = info_list[i]
        else:
            curr_info = None
        if color_arr_list is not None:
            curr_color_arr = color_arr_list[i]
        else:
            curr_color_arr = None
        _ = set_data_in_figure(
            cube_fig_setup, X_list[i], Y_list[i], Z_list[i], weight = curr_weight, info = curr_info,
            weight_tol = weight_tol, weight_format = weight_format,
            weight_max = curr_weight_max, info_format = info_format,
            color_arr = curr_color_arr,
            cmap_name = cmap_name, shuffle_colors = shuffle_colors,
            point_size = point_size,
        )
        return cube_fig_setup

    # call the animator.  blit=True means only re-draw the parts that have changed.
    fig = cube_fig_setup['fig']
    anim = animation.FuncAnimation(fig, animate,
                                   frames = num_frames, 
                                   interval=interval)#, blit=True  
    # save the animator.
    
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
    return anim


# plot cube in axis
def cube3dplots_in_axis(
    ax_cube, X, Y, Z, weight = None, info = None, 
    radius = 1, min_radius = 0,
    weight_tol = 1e-3, weight_format = '%.2f',
    info_format = 'MI = %.4f',
    color_arr = None, cmap_name = 'nipy_spectral', shuffle_colors = False,
    point_size = 100,
    path_vec = None, path_close = True,
    path_color='crimson', linestyle='dashed'):

    '''Plot cube and points data in a given matplotlib axis.'''

    # check inputs

    if len(X)!=len(Y) or len(Y)!=len(Z):
        raise Exception('Wrong dimension of inputs!')
    if weight is not None and len(weight)!= len(X):
        raise Exception('Wrong dimension of inputs: weight')

    num_pts = len(X) # number of points

    curr_radius = max(np.max(np.array([X,Y,Z])), radius) # radius
    curr_min_radius = min(np.min(np.array([X,Y,Z])), min_radius) # min_radius

    # draw cube

    draw_cube_in_axis(ax_cube, curr_radius, curr_min_radius)

    # plot points and weights in cube

    scat = ax_cube.scatter([radius], [radius], [radius], s= 0)#, edgecolor='k', alpha=1)

    weight_txt_list = set_scatter_data_in_axis(
        ax_cube, scat, X, Y, Z, weight=weight,
        weight_txt_list=None, radius=curr_radius,
        weight_tol=weight_tol, weight_format=weight_format,
        color_arr=color_arr, cmap_name=cmap_name, shuffle_colors=shuffle_colors,point_size=point_size,
    )

    # set info text

    if info is not None:
        x0, y0, _ = proj3d.proj_transform(0,-0.3*curr_radius, -1.5*curr_radius, ax_cube.get_proj())
        info_txt = ax_cube.text2D(x0,y0,"", fontsize = 15)
        info_txt.set_text(info_format%info)

    # plot path

    if path_vec is not None:
        plot_path_in_axis(
            ax_cube, X, Y, Z,
            path_vec, path_close=path_close,
            path_color=path_color, linestyle=linestyle,
        )
