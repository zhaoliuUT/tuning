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
    r = [min_radius, radius]
    for s, e in combinations(np.array(list(product(r, r, r))), 2):
        if np.sum(np.abs(s-e)) == r[1]-r[0]:
            ax.plot3D(*zip(s, e), color="k", alpha = 0.5, lw = 1.5)#linestyle='--'

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

def set_scatter_data_in_axis(ax, scat, X, Y, Z, weight = None,
                             weight_txt_list=None, radius = 1,
                             weight_tol = 1e-3, weight_format = '%.2f',
                             color_arr = None,
                             cmap_name = 'nipy_spectral', shuffle_colors = False,
                             point_size = 100,
                            ):

    '''Set data in cube axis, including the scatter points, and weights' texts.'''

    num_pts = len(X)

    # set scatter points' data
    scat._offsets3d = juggle_axes(X, Y, Z, 'z')

    # set scatter points' colors, alpha, sizes
    if color_arr is None:
            #color_arr = np.arange(num_pts)
        cmap = plt.cm.get_cmap(cmap_name, num_pts) # other colormap names: cubehelix,viridis,...
        color_arr = np.array([cmap(i) for i in range(num_pts)])
    if shuffle_colors:
        np.random.shuffle(color_arr)
    if color_arr.shape[0]!=num_pts:
        raise Exception("Wrong dimension of color array!")

    scat._facecolor3d = color_arr
    #scat.set_facecolor(color_arr) #python bug: this only works for 2d
    scat.set_edgecolor('k')
    scat.set_alpha(1.0)
    sizes = np.ones(num_pts)*point_size
    if weight is not None:
        sizes[np.fabs(weight) < weight_tol] = 0
    scat.set_sizes(sizes)


    # set weight texts of the scatter points
    if weight_txt_list is None:
        txt_list = []
    else:
        txt_list = weight_txt_list

    #radius = cube_fig_setup['radius']
    #ax_cube = cube_fig_setup['ax_cube']
    if len(txt_list) < num_pts:
        # add empty texts
        txt_list += [ax.text2D(0,0,"",fontsize = 14, color = 'k')
                     for _ in range(num_pts - len(txt_list))] # 'steelblue'

    if weight is not None:
        for txt, new_x, new_y, new_z, w in zip(txt_list[:num_pts], X, Y, Z, weight):
            # animating Text in 3D proved to be tricky. Tip of the hat to @ImportanceOfBeingErnest
            # for this answer https://stackoverflow.com/a/51579878/1356000
            if np.fabs(w) > weight_tol:
                x_, y_, _ = proj3d.proj_transform(new_x+0.1*radius, new_y+0.1*radius, new_z+0.1*radius, \
                                                  ax.get_proj())
                txt.set_position((x_,y_))
                txt.set_text(weight_format%w)
            else:
                txt.set_text("")

    return txt_list

def plot_path_in_axis(ax, X, Y, Z, path_vec,
                      path_close=True, path_color='crimson', linestyle='dashed',
                     ):
    '''connect the path of points in cube axis:
    the path is a vector of integers, each in [0,num_pts-1], specifying the order.
    '''

    num_pts = len(X)
    if np.any(path_vec >= num_pts):
        raise Exception('Wrong input of the path vector!')

    # connect the points by path in the order of path_vec
    for i in range(len(path_vec) - 1):
        ax.plot3D([X[path_vec[i]], X[path_vec[i+1]]],
                  [Y[path_vec[i]], Y[path_vec[i+1]]],
                  [Z[path_vec[i]], Z[path_vec[i+1]]],
                  color=path_color, linestyle=linestyle,
                  lw = 1.5, alpha=0.7)
    if path_close:
        ax.plot3D([X[path_vec[-1]], X[path_vec[0]]],
                  [Y[path_vec[-1]], Y[path_vec[0]]],
                  [Z[path_vec[-1]], Z[path_vec[0]]],
                  color=path_color,linestyle=linestyle,
                  lw = 1.5, alpha=0.7)

def setup_cube3d_figure(radius = 1, min_radius = 0,
                        INCLUDE_INFO = True,
                        INCLUDE_FUN = True, INCLUDE_WEIGHT = True, INCLUDE_WEIGHT_BAR = True,
                       ):
    
    '''When numNeuro = 3, plot points in a cube.'''
    ''' the 'radius' argument acts the same way as maxium FP.'''
    

    # figure setup
    if INCLUDE_FUN or INCLUDE_WEIGHT or INCLUDE_WEIGHT_BAR:
        fig = plt.figure(figsize = (12, 6 + 1.0*INCLUDE_WEIGHT_BAR))
        gs0 = gridspec.GridSpec(1, 2, width_ratios=[2, 1])
        n_cols = 3*INCLUDE_FUN + INCLUDE_WEIGHT + INCLUDE_WEIGHT_BAR
        ax_w_bar_idx = 3*INCLUDE_FUN
        ax_w_idx = 3*INCLUDE_FUN + INCLUDE_WEIGHT_BAR

        height_ratios = np.ones(n_cols)
        if INCLUDE_WEIGHT_BAR:
            height_ratios[ax_w_bar_idx] = 0.5
        gs01 = gridspec.GridSpecFromSubplotSpec(n_cols,1,subplot_spec=gs0[1], height_ratios=height_ratios)

        ax_cube = fig.add_subplot(gs0[0], projection='3d')
        ax_cube.set_aspect("equal")
        if INCLUDE_FUN:
            ax_f1 = fig.add_subplot(gs01[0])
            ax_f2 = fig.add_subplot(gs01[1])
            ax_f3 = fig.add_subplot(gs01[2])
            ax_f_list = [ax_f1, ax_f2, ax_f3]
        else:
            ax_f_list = None
        if INCLUDE_WEIGHT_BAR:
            ax_w_bar = fig.add_subplot(gs01[ax_w_bar_idx])
            ax_w_bar.set_xlim([0, 1])
            ax_w_bar.set_ylim([-0.1, 0.1])
            ax_w_bar.set_aspect(0.2)
            ax_w_bar.set_yticks([])
        else:
            ax_w_bar = None
        if INCLUDE_WEIGHT:
            ax_w = fig.add_subplot(gs01[ax_w_idx])
            #weight_max = np.max(np.array(weights_list))
            #ax_w.set_ylim(0, weight_max + 0.05)
        else:
            ax_w = None
    else:
        fig = plt.figure(figsize = (8,8))
        ax_cube = fig.gca(projection='3d')
        ax_cube.set_aspect("equal")
        ax_f_list = None
        ax_w_bar = None
        ax_w = None

    # draw cube
    draw_cube_in_axis(ax_cube, radius, min_radius)
    
    # plot scatter points
    scat = ax_cube.scatter([radius], [radius], [radius], s= 0)#, edgecolor='k', alpha=1)
    
    # plot info txt
    if INCLUDE_INFO:
        x0, y0, _ = proj3d.proj_transform(0,-0.3*radius, -1.5*radius,ax_cube.get_proj())
        info_txt = ax_cube.text2D(x0,y0,"", fontsize = 15)
    else:
        info_txt = None

    # plot weight txt list
    weight_txt_list = []#[ax_cube.text2D(0,0,"",fontsize = 14, color = 'k') for _ in range(18)]  # 'steelblue'
    if INCLUDE_FUN:
        line1, = ax_f1.plot([], [], color = 'crimson', lw = 2)
        line2, = ax_f2.plot([], [], color = 'seagreen', lw = 2)
        line3, = ax_f3.plot([], [], color = 'steelblue', lw = 2)
        ax_f_list = [ax_f1, ax_f2, ax_f3]
        lines_list = [line1, line2, line3]
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
        lines_list = None
        
    cube_fig_setup = {'fig':fig, 'ax_cube':ax_cube,
                      'ax_f_list':ax_f_list, 'ax_w':ax_w, 'ax_w_bar':ax_w_bar,
                      'scat': scat, 'lines_list':lines_list,
                      'info_txt':info_txt,
                      'weight_txt_list':weight_txt_list,
                      'radius':radius, 'min_radius':min_radius,
                     }
    return cube_fig_setup

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
