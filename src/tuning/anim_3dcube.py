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
    if weights.size != f_vec.size or np.fabs(np.sum(weights) - 1)>1e-5:
        raise Exception("Dimension mismatch!")
    numBin = weights.size
    xtmp = np.cumsum(weights)
    xx = [ [xtmp[i]]*2 for i in range(0, numBin-1) ]  
    xx = [0]+list(np.array(xx).reshape(-1)) + [1.0]
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

def setup_cube3d_figure(num_pts,
                        radius = 1, weight_max = 1.0, weight_tol = 1e-3, 
                        cmap_name = 'nipy_spectral', shuffle_colors = False,
                        PLOT_WEIGHTS = True, INCLUDE_INFO = True,
                        INCLUDE_FUN = True, INCLUDE_WEIGHT = True, INCLUDE_WEIGHT_BAR = True,
                       ):
    
    '''When numNeuro = 3, plot points in a cube.'''
    ''' the 'radius' argument acts the same way as maxium FP.'''
    
    X0 = np.zeros(num_pts)
    Y0 = np.zeros(num_pts)
    Z0 = np.zeros(num_pts)
    W0 = 1.0*np.ones(num_pts)/num_pts
    
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
            ax_w.set_ylim(0, weight_max + 0.05)
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
    r = [0, radius]
    for s, e in combinations(np.array(list(product(r, r, r))), 2):
        if np.sum(np.abs(s-e)) == r[1]-r[0]:
            ax_cube.plot3D(*zip(s, e), color="k", alpha = 0.5, lw = 1.5)#linestyle='--'

    r2 = [0, 1.4*radius]
    for s in np.array(list(product(r2, r2, r2))):
        if np.sum(s> 0) == 1:
            a = Arrow3D(*zip([0,0,0], s), mutation_scale=20,
                lw=1, arrowstyle="-|>", color="k")
            ax_cube.add_artist(a)
    ax_cube._axis3don = False
    ax_cube.grid(True)
    ax_cube.set_xlim([0,1.5*radius])
    ax_cube.set_ylim([0,1.5*radius])
    ax_cube.set_zlim([0,1.5*radius])
    endpts = [[1.45*radius,-0.1*radius,0],[0.1*radius,1.45*radius,0],[0,0,1.45*radius]]
    
    
    for k in range(len(endpts)):
        ax_cube.text(endpts[k][0],endpts[k][1],endpts[k][2],r'$f_%d$'%(k+1),color = 'k',fontsize = 16)

    ax_cube.view_init(azim = 30)

    # color_arr = np.arange(num_pts)
    cmap = plt.cm.get_cmap(cmap_name, num_pts) # other colormap names: cubehelix,viridis,...
    color_arr = np.array([cmap(i) for i in range(num_pts)])
    if shuffle_colors:
        np.random.shuffle(color_arr)
    
    scat = ax_cube.scatter(X0, Y0, Z0,  c = color_arr,s= 100, edgecolor='k', alpha=1)
    if INCLUDE_INFO:
        x0, y0, _ = proj3d.proj_transform(0,-0.3*radius, -1.5*radius,ax_cube.get_proj())
        info_txt = ax_cube.text2D(x0,y0,"", fontsize = 15)
    else:
        info_txt = None
    
    
    txt_list = [ax_cube.text2D(0,0,"",fontsize = 14, color = 'k') for _ in range(num_pts)]  # 'steelblue'
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
        
    if INCLUDE_WEIGHT:
        barcollection = ax_w.bar(np.arange(num_pts), W0, color = color_arr)
    else:
        barcollection = None
    if INCLUDE_WEIGHT_BAR:
        rect_list = []
        for j in range(num_pts):
            rect = plt.Rectangle((np.sum(W0[:j]), -0.1), W0[j], 0.2,
                                 facecolor=color_arr[j])
            ax_w_bar.add_artist(rect)
            rect_list.append(rect)
    else:
        rect_list = None
        
    cube_fig_setup = {'fig':fig, 'ax_cube':ax_cube,
                     'ax_f_list':ax_f_list, 'ax_w':ax_w, 'ax_w_bar':ax_w_bar,
                     'scat': scat, 'lines_list':lines_list, 'barcollection':barcollection, 'rect_list':rect_list,
                     'info_txt':info_txt, 'weight_txt_list':txt_list,
                     'color_arr':color_arr, 'radius':radius,
                     }
    return cube_fig_setup

def plot_cube3d_data(cube_fig_setup, X, Y, Z, weight = None, info = None,
                     weight_tol = 1e-3, weight_format = '%.2f',
                    ): 
    '''Set data in cube_fig_setup'''
    
    num_pts = len(X)
    # plot points and label weights
    cube_fig_setup['scat']._offsets3d = juggle_axes(X, Y, Z, 'z')    
    sizes = np.ones(num_pts)*100
    if weight is not None:
        sizes[weight < weight_tol] = 0
    cube_fig_setup['scat'].set_sizes(sizes)
    
    txt_list = cube_fig_setup['weight_txt_list']
    radius = cube_fig_setup['radius']
    ax_cube = cube_fig_setup['ax_cube']
    if weight is not None:
        for txt, new_x, new_y, new_z, w in zip(txt_list, X, Y, Z, weight):
            # animating Text in 3D proved to be tricky. Tip of the hat to @ImportanceOfBeingErnest
            # for this answer https://stackoverflow.com/a/51579878/1356000
            if w > weight_tol:
                x_, y_, _ = proj3d.proj_transform(new_x+0.1*radius, new_y+0.1*radius, new_z+0.1*radius, \
                                                  ax_cube.get_proj())
                txt.set_position((x_,y_))
                txt.set_text(weight_format%w)
            else:
                txt.set_text("")
                
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
            

    if cube_fig_setup['barcollection'] is not None:
        for j, b in enumerate(cube_fig_setup['barcollection']):
            b.set_height(weight[j])
    if cube_fig_setup['rect_list'] is not None:
        for j in range(num_pts):
            cube_fig_setup['rect_list'][j].set_x(np.sum(weight[:j]))
            cube_fig_setup['rect_list'][j].set_width(weight[j])

    if (cube_fig_setup['info_txt'] is not None) and (info is not None):
        cube_fig_setup['info_txt'].set_text("MI = %.4f"%info)

    return cube_fig_setup

def cube3dplots(X, Y, Z, weight = None, info = None, 
                radius = 1, weight_tol = 1e-3, weight_format = '%.2f',
                cmap_name = 'nipy_spectral', shuffle_colors = False,
                INCLUDE_FUN = True, INCLUDE_WEIGHT = True, INCLUDE_WEIGHT_BAR = True,
                FILE_NAME = "", ADD_TIME = True):
    '''Plot a single cube figure.'''
    
    # check inputs
    
    if len(X)!=len(Y) or len(Y)!=len(Z):
        raise Exception('Wrong dimension of inputs!')    
    if weight is None:
        PLOT_WEIGHTS = False
        weight_max = 1.0
    elif len(weight)!= len(X):
        raise Exception('Wrong dimension of inputs: weight')
    elif np.fabs(np.sum(weight) - 1)>1e-5:
        raise Exception('Wrong input of weight: sum error!')
    else:
        PLOT_WEIGHTS = True        
    if info is None:
        INCLUDE_INFO = False
    else:
        INCLUDE_INFO = True
        
    num_pts = len(X) # number of points
    if weight is not None:
        weight_max = np.max(np.array(weight)) # maximum weight
    else:
        weight_max = None
    curr_radius = max(np.max(np.array([X,Y,Z])), radius) # radius
    
    # figure setup
    
    cube_fig_setup =  setup_cube3d_figure(
        num_pts, radius = curr_radius, weight_max = weight_max, weight_tol = weight_tol, 
        cmap_name = cmap_name, shuffle_colors = shuffle_colors,
        PLOT_WEIGHTS = PLOT_WEIGHTS, INCLUDE_INFO = INCLUDE_INFO,
        INCLUDE_FUN = INCLUDE_FUN, INCLUDE_WEIGHT = INCLUDE_WEIGHT, INCLUDE_WEIGHT_BAR = INCLUDE_WEIGHT_BAR,
    )
    
    # set data
    
    cube_fig_setup = plot_cube3d_data(cube_fig_setup, X, Y, Z, weight = weight, info = info,
                                      weight_tol = weight_tol, weight_format = weight_format,
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
                radius = 1, weight_tol = 1e-3, weight_format = '%.2f',
                cmap_name = 'nipy_spectral', shuffle_colors = False,
                INCLUDE_FUN = True, INCLUDE_WEIGHT = True, INCLUDE_WEIGHT_BAR = True,
                FILE_NAME = "", ADD_TIME = True, interval = 1000):
    
    '''Plot animation in 3d cube.'''
    
    # check inputs    
        
    if len(X_list) != len(Y_list) or len(Y_list)!= len(Z_list):
        raise Exception('Wrong dimension of inputs!')
    
    elif len(X_list[0])!=len(Y_list[0]) or len(Y_list[0])!=len(Z_list[0]):
        raise Exception('Wrong dimension of inputs!')   
    
    if weights_list is None:
        PLOT_WEIGHTS = False
    elif len(weights_list) != len(X_list) or len(weights_list[0])!= len(X_list[0]):
        raise Exception('Wrong dimension of inputs: weight_list')
    elif np.fabs(np.sum(weights_list[0]) - 1)>1e-5:
        raise Exception('Wrong input of weights: sum error!')
    else:
        PLOT_WEIGHTS = True
    if info_list is None:
        INCLUDE_INFO = False
    else:
        INCLUDE_INFO = True
    
    
    num_pts = len(X_list[0]) # number of points
    num_frames = len(X_list) # number of frames
    if weights_list is not None:
        weight_max = np.max(np.array(weights_list)) # maximum weight
    else:
        weight_max = None
    curr_radius = max(np.max(np.array([X_list,Y_list,Z_list])), radius) # radius
    
    # figure setup
    
    cube_fig_setup =  setup_cube3d_figure(
        num_pts, radius = curr_radius, weight_max = weight_max, weight_tol = weight_tol, 
        cmap_name = cmap_name, shuffle_colors = shuffle_colors,
        PLOT_WEIGHTS = PLOT_WEIGHTS, INCLUDE_INFO = INCLUDE_INFO,
        INCLUDE_FUN = INCLUDE_FUN, INCLUDE_WEIGHT = INCLUDE_WEIGHT, INCLUDE_WEIGHT_BAR = INCLUDE_WEIGHT_BAR,
    )

    # animation function.  This is called sequentially
    
    def animate(i):
        _ = plot_cube3d_data(
            cube_fig_setup, X_list[i], Y_list[i], Z_list[i], weight = weights_list[i], info = info_list[i],
            weight_tol = weight_tol, weight_format = weight_format,
            FILE_NAME = "", ADD_TIME = False, # not saving image
        )  
        return cube_fig_setup
        #return cube_fig_setup

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